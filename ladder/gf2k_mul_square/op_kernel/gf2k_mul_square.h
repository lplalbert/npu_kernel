/*!
 * \file gf2k_mul_square.h
 * \brief Gf2kMulSquare kernel class definition
 *
 * Each row encodes one GF(2^k) field element (1 byte for k=8, 16 bytes for
 * k=128, little-endian: byte j bit r is the coefficient of x^(8j+r)). Rows are
 * independent:
 *   mul    : c = a * b  (carryless GF(2) product followed by reduction mod P)
 *   square : c = a * a
 * P_8 = x^8 + x^4 + x^3 + x + 1 (0x11B), P_128 = x^128 + x^7 + x^2 + x + 1.
 *
 * Implementation: rows are split across cores (no workspace, no cross-core
 * synchronisation); each core streams tiles of rows with DataCopyPad into UB,
 * does the carryless multiply / reduction with scalar integer bit operations
 * and writes the tile back with DataCopyPad. All arithmetic is exact.
 *
 * The k=8 path reads/writes 8 bytes at a time through a uint64 view of UB to
 * cut UB scalar accesses per element by roughly 8x; the k=128 path reads two
 * uint64 per row instead of 16 bytes.
 */

#ifndef GF2KMULSQUARE_H
#define GF2KMULSQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gf2k_mul_square_tiling_data.h"
#include "gf2k_mul_square_tiling_key.h"

namespace NsGf2kMulSquare {

using namespace AscendC;

// GF(2^8) multiply: carryless product reduced by 0x11B; square uses v == u.
__aicore__ inline uint8_t Gf2Mul8(uint8_t u, uint8_t v)
{
    uint8_t r = 0;
    for (int i = 0; i < 8; ++i) {
        if (v & 1u) {
            r ^= u;
        }
        uint8_t hi = (uint8_t)(u & 0x80u);
        u = (uint8_t)(u << 1);
        if (hi) {
            u ^= 0x1Bu;
        }
        v = (uint8_t)(v >> 1);
    }
    return r;
}

// 128x128 carryless multiply -> 256 bits in z[0..3] (z[0] is the lowest word).
// a=(a0,a1), b=(b0,b1). 4-bit window method: build the 16 multiples T[k]=a*k
// (a has up to 131 bits -> 3 words), then for each of the 32 nibbles of b do
// table lookup + shift by 4*i + XOR accumulate.
__aicore__ inline void Gf2MulWords(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1, uint64_t z[4])
{
    // s[j] = a << j (j = 0..3), 192 bits (3 words)
    uint64_t s0[4], s1[4], s2[4];
    uint64_t c0 = a0, c1 = a1, c2 = 0;
    for (int j = 0; j < 4; ++j) {
        s0[j] = c0; s1[j] = c1; s2[j] = c2;
        c2 = (c2 << 1) | (c1 >> 63);
        c1 = (c1 << 1) | (c0 >> 63);
        c0 <<= 1;
    }
    // T[k] = a * k (k = 0..15), 3 words
    uint64_t T0[16], T1[16], T2[16];
    T0[0] = 0; T1[0] = 0; T2[0] = 0;
    for (int j = 0; j < 4; ++j) {
        int pw = 1 << j;
        for (int k = 0; k < pw; ++k) {
            T0[pw + k] = T0[k] ^ s0[j];
            T1[pw + k] = T1[k] ^ s1[j];
            T2[pw + k] = T2[k] ^ s2[j];
        }
    }
    uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    for (int i = 0; i < 32; ++i) {
        uint64_t nib = (i < 16) ? ((b0 >> (4 * i)) & 0xFULL) : ((b1 >> (4 * (i - 16))) & 0xFULL);
        uint64_t t0 = T0[nib], t1 = T1[nib], t2 = T2[nib];
        int sh = 4 * i;
        int wl = sh >> 6;
        int s = sh & 63;
        uint64_t v0, v1, v2, v3;
        if (s == 0) {
            v0 = t0; v1 = t1; v2 = t2; v3 = 0;
        } else {
            v0 = t0 << s;
            v1 = (t1 << s) | (t0 >> (64 - s));
            v2 = (t2 << s) | (t1 >> (64 - s));
            v3 = t2 >> (64 - s);
        }
        if (wl == 0) {
            r0 ^= v0; r1 ^= v1; r2 ^= v2; r3 ^= v3;
        } else {
            // a << 4*i is below 2^255, so the v3 spill is always zero here.
            r1 ^= v0; r2 ^= v1; r3 ^= v2;
        }
    }
    z[0] = r0; z[1] = r1; z[2] = r2; z[3] = r3;
}

// Fold the high 128 bits: T = H*0x87 = H ^ (H<<1) ^ (H<<2) ^ (H<<7), H = (h0,h1).
// Returns the low 128 bits (lo0,lo1) and the overflow bits outHi (at most 7).
__aicore__ inline void Gf2FoldTop(uint64_t h0, uint64_t h1, uint64_t& lo0, uint64_t& lo1, uint32_t& outHi)
{
    uint64_t t0 = h0;
    uint64_t t1 = h1;
    uint32_t tov = 0;
    t0 ^= (h0 << 1);
    t1 ^= (h1 << 1) | (h0 >> 63);
    tov ^= (uint32_t)(h1 >> 63);
    t0 ^= (h0 << 2);
    t1 ^= (h1 << 2) | (h0 >> 62);
    tov ^= (uint32_t)(h1 >> 62);
    t0 ^= (h0 << 7);
    t1 ^= (h1 << 7) | (h0 >> 57);
    tov ^= (uint32_t)(h1 >> 57);
    lo0 = t0;
    lo1 = t1;
    outHi = tov;
}

// Reduce the 256-bit z[0..3] modulo x^128 + x^7 + x^2 + x + 1 (x^128 == 0x87)
// and return 128 bits in out[0..1].
__aicore__ inline void Gf2Reduce128(uint64_t z[4], uint64_t out[2])
{
    uint64_t l0 = z[0];
    uint64_t l1 = z[1];
    uint64_t t0, t1;
    uint32_t th = 0;
    Gf2FoldTop(z[2], z[3], t0, t1, th);
    l0 ^= t0;
    l1 ^= t1;
    // Second fold: th < 2^7, so th*0x87 < 2^14 and no overflow bits remain.
    uint64_t s0, s1;
    uint32_t sh = 0;
    Gf2FoldTop((uint64_t)th, 0ULL, s0, s1, sh);
    l0 ^= s0;
    l1 ^= s1;
    out[0] = l0;
    out[1] = l1;
}

template <typename T>
class Gf2kMulSquare {
public:
    __aicore__ inline Gf2kMulSquare(){};

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const Gf2kMulSquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoTile(int64_t rowStart, int64_t rows, int64_t bytes);

private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> aBuf;
    TBuf<QuePosition::VECCALC> bBuf;
    TBuf<QuePosition::VECCALC> cBuf;

    GlobalTensor<uint8_t> aGm;
    GlobalTensor<uint8_t> bGm;
    GlobalTensor<uint8_t> cGm;

    int64_t totalRows_ = 0;
    int64_t width_ = 1;
    int64_t k_ = 8;
    int64_t op_ = 0;
    int64_t rowsPerCore_ = 1;
    int64_t tileRows_ = 1;
    int64_t tileBytes_ = 0;
};

template <typename T>
__aicore__ inline void Gf2kMulSquare<T>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                              const Gf2kMulSquareTilingData* tilingData)
{
    totalRows_ = tilingData->totalRows;
    width_ = tilingData->width;
    k_ = tilingData->k;
    op_ = tilingData->op;
    rowsPerCore_ = tilingData->rowsPerCore;
    tileRows_ = tilingData->tileRows;
    if (width_ < 1) {
        width_ = 1;
    }
    if (tileRows_ < 1) {
        tileRows_ = 1;
    }
    if (rowsPerCore_ < 0) {
        rowsPerCore_ = 0;
    }
    tileBytes_ = tileRows_ * width_;

    // Declare at least one byte so that empty tensors still get a valid range.
    const int64_t totalBytes = totalRows_ * width_;
    const int64_t gmBytes = totalBytes > 0 ? totalBytes : 1;
    aGm.SetGlobalBuffer((__gm__ uint8_t*)a, gmBytes);
    bGm.SetGlobalBuffer((__gm__ uint8_t*)b, gmBytes);
    cGm.SetGlobalBuffer((__gm__ uint8_t*)c, gmBytes);

    pipe.InitBuffer(aBuf, (uint32_t)tileBytes_);
    pipe.InitBuffer(bBuf, (uint32_t)tileBytes_);
    pipe.InitBuffer(cBuf, (uint32_t)tileBytes_);
}

template <typename T>
__aicore__ inline void Gf2kMulSquare<T>::DoTile(int64_t rowStart, int64_t rows, int64_t bytes)
{
    if (rows <= 0 || bytes <= 0) {
        return;
    }
    LocalTensor<uint8_t> inA = aBuf.Get<uint8_t>();
    LocalTensor<uint8_t> inB = bBuf.Get<uint8_t>();
    LocalTensor<uint8_t> outC = cBuf.Get<uint8_t>();
    const int64_t gOff = rowStart * width_;

    DataCopyExtParams pIn;
    pIn.blockCount = 1;
    pIn.blockLen = (uint32_t)bytes;
    pIn.srcStride = 0;
    pIn.dstStride = 0;
    pIn.rsv = 0;
    DataCopyPad(inA, aGm[gOff], pIn, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    DataCopyPad(inB, bGm[gOff], pIn, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    if (k_ == 8) {
        // One row is one byte. Handle 8 rows per uint64 UB access.
        LocalTensor<uint64_t> inA64 = inA.template ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> inB64 = inB.template ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> outC64 = outC.template ReinterpretCast<uint64_t>();
        const int64_t groups = rows >> 3;
        for (int64_t g = 0; g < groups; ++g) {
            const uint64_t wa = inA64.GetValue((uint32_t)g);
            const uint64_t wb = (op_ == 1) ? wa : inB64.GetValue((uint32_t)g);
            uint64_t wr = 0;
            for (int j = 0; j < 8; ++j) {
                const uint8_t u = (uint8_t)((wa >> (8 * j)) & 0xFFULL);
                const uint8_t v = (uint8_t)((wb >> (8 * j)) & 0xFFULL);
                wr |= ((uint64_t)Gf2Mul8(u, v)) << (8 * j);
            }
            outC64.SetValue((uint32_t)g, wr);
        }
        for (int64_t i = (groups << 3); i < rows; ++i) {
            const uint8_t u = inA.GetValue((uint32_t)i);
            const uint8_t v = (op_ == 1) ? u : inB.GetValue((uint32_t)i);
            outC.SetValue((uint32_t)i, Gf2Mul8(u, v));
        }
    } else {
        // 16 bytes per row = 2 uint64 (little endian).
        LocalTensor<uint64_t> inA64 = inA.template ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> inB64 = inB.template ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> outC64 = outC.template ReinterpretCast<uint64_t>();
        for (int64_t i = 0; i < rows; ++i) {
            const uint32_t w = (uint32_t)(i * 2);
            const uint64_t a0 = inA64.GetValue(w);
            const uint64_t a1 = inA64.GetValue(w + 1);
            uint64_t b0, b1;
            if (op_ == 1) {
                b0 = a0;
                b1 = a1;
            } else {
                b0 = inB64.GetValue(w);
                b1 = inB64.GetValue(w + 1);
            }
            uint64_t z[4];
            Gf2MulWords(a0, a1, b0, b1, z);
            uint64_t ow[2];
            Gf2Reduce128(z, ow);
            outC64.SetValue(w, ow[0]);
            outC64.SetValue(w + 1, ow[1]);
        }
    }

    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

    DataCopyExtParams pOut;
    pOut.blockCount = 1;
    pOut.blockLen = (uint32_t)bytes;
    pOut.srcStride = 0;
    pOut.dstStride = 0;
    pOut.rsv = 0;
    DataCopyPad(cGm[gOff], outC, pOut);

    // MTE3 is asynchronous: drain the pipeline before the next tile overwrites
    // the same UB buffer.
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Gf2kMulSquare<T>::Process()
{
    const int64_t blockId = (int64_t)GetBlockIdx();
    int64_t startRow = blockId * rowsPerCore_;
    if (startRow > totalRows_) {
        startRow = totalRows_;
    }
    int64_t endRow = startRow + rowsPerCore_;
    if (endRow > totalRows_) {
        endRow = totalRows_;
    }

    int64_t r = startRow;
    while (r < endRow) {
        int64_t rows = endRow - r;
        if (rows > tileRows_) {
            rows = tileRows_;
        }
        DoTile(r, rows, rows * width_);
        r += rows;
    }
}

} // namespace NsGf2kMulSquare
#endif // GF2KMULSQUARE_H

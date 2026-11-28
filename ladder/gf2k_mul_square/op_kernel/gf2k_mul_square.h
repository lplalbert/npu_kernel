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
 * Schedules:
 *  - k=8 vector: lane = row. Two rows are packed into one int16 lane (low/high
 *    byte); GF(2^8) product is an 8-step shift-and-xor (xtime) run with int16
 *    And/Or/Xor/Shift + Muls(bit,-1) masks. Covers even chunk sizes.
 *  - k=128 vector: lane = row, 8 int16 word-columns per row; square is a bit
 *    spread, mul a bit-serial carryless product; shared 256->128 fold.
 *  - Any scalar tail / small shape uses scalar integer bit operations.
 * All arithmetic is exact; output is bit-identical to the reference.
 */

#ifndef GF2KMULSQUARE_H
#define GF2KMULSQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gf2k_mul_square_tiling_data.h"
#include "gf2k_mul_square_tiling_key.h"

namespace NsGf2kMulSquare {

using namespace AscendC;

constexpr int32_t GF2K_VEC_NW16 = 8;   // 16-byte row = 8 int16 words (k=128)
constexpr int32_t GF2K_VEC_SW = 24;    // scratch words per row (k=128 reduce)

// ------------------------------- scalar helpers -------------------------------

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

__aicore__ inline void Gf2MulWords(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1, uint64_t z[4])
{
    uint64_t s0[4], s1[4], s2[4];
    uint64_t c0 = a0, c1 = a1, c2 = 0;
    for (int j = 0; j < 4; ++j) {
        s0[j] = c0; s1[j] = c1; s2[j] = c2;
        c2 = (c2 << 1) | (c1 >> 63);
        c1 = (c1 << 1) | (c0 >> 63);
        c0 <<= 1;
    }
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
            r1 ^= v0; r2 ^= v1; r3 ^= v2;
        }
    }
    z[0] = r0; z[1] = r1; z[2] = r2; z[3] = r3;
}

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

__aicore__ inline void Gf2Reduce128(uint64_t z[4], uint64_t out[2])
{
    uint64_t l0 = z[0];
    uint64_t l1 = z[1];
    uint64_t t0, t1;
    uint32_t th = 0;
    Gf2FoldTop(z[2], z[3], t0, t1, th);
    l0 ^= t0;
    l1 ^= t1;
    uint64_t s0, s1;
    uint32_t sh = 0;
    Gf2FoldTop((uint64_t)th, 0ULL, s0, s1, sh);
    l0 ^= s0;
    l1 ^= s1;
    out[0] = l0;
    out[1] = l1;
}

// ---------------------------- vector helpers (int16 lanes) ----------------------------

__aicore__ inline void VShlWords(const LocalTensor<int16_t>& dst, const LocalTensor<int16_t>& src,
                                 int nw, int R, uint32_t off, const LocalTensor<int16_t>& tmp,
                                 const LocalTensor<int16_t>& lm)
{
    if (off == 0u) {
        for (int w = 0; w < nw; ++w) {
            Adds(dst[w * R], src[w * R], (int16_t)0, R);
        }
        Duplicate(dst[nw * R], (int16_t)0, R);
        return;
    }
    Duplicate(lm, (int16_t)((1u << off) - 1u), R);
    ShiftLeft(dst[0], src[0], (int16_t)off, R);
    for (int w = 1; w < nw; ++w) {
        ShiftRight(tmp, src[(w - 1) * R], (int16_t)(16 - off), R);
        And(tmp, tmp, lm, R);
        ShiftLeft(dst[w * R], src[w * R], (int16_t)off, R);
        Or(dst[w * R], dst[w * R], tmp, R);
    }
    ShiftRight(dst[nw * R], src[(nw - 1) * R], (int16_t)(16 - off), R);
    And(dst[nw * R], dst[nw * R], lm, R);
}

__aicore__ inline void VClMul(const LocalTensor<int16_t>& a, const LocalTensor<int16_t>& b,
                              const LocalTensor<int16_t>& z, const LocalTensor<int16_t>& s,
                              int nw, int R)
{
    for (int w = 0; w < 2 * nw; ++w) {
        Duplicate(z[w * R], (int16_t)0, R);
    }
    LocalTensor<int16_t> sh = s;
    LocalTensor<int16_t> tmp = s[(nw + 1) * R];
    LocalTensor<int16_t> maskT = s[(nw + 2) * R];
    LocalTensor<int16_t> tmp2 = s[(nw + 3) * R];
    LocalTensor<int16_t> lm = s[(nw + 4) * R];
    LocalTensor<int16_t> onesT = s[(nw + 5) * R];
    Duplicate(onesT, (int16_t)1, R);
    const int totalBits = 16 * nw;
    for (int t = 0; t < totalBits; ++t) {
        const int bt = t >> 4;
        const uint32_t off = (uint32_t)(t & 15);
        if (off == 0u) {
            Adds(maskT, b[bt * R], (int16_t)0, R);
        } else {
            ShiftRight(maskT, b[bt * R], (int16_t)off, R);
        }
        And(maskT, maskT, onesT, R);
        Muls(maskT, maskT, (int16_t)-1, R);
        VShlWords(sh, a, nw, R, off, tmp, lm);
        int wmax = 2 * nw - bt;
        if (wmax > nw + 1) {
            wmax = nw + 1;
        }
        for (int w = 0; w < wmax; ++w) {
            And(tmp2, sh[w * R], maskT, R);
            Xor(z[(bt + w) * R], z[(bt + w) * R], tmp2, R);
        }
    }
}

__aicore__ inline void VSquareSpread(const LocalTensor<int16_t>& a, const LocalTensor<int16_t>& z,
                                     const LocalTensor<int16_t>& s, int nw, int R)
{
    LocalTensor<int16_t> sh = s;
    LocalTensor<int16_t> mFF = s[1 * R];
    LocalTensor<int16_t> m0F = s[2 * R];
    LocalTensor<int16_t> m33 = s[3 * R];
    LocalTensor<int16_t> m55 = s[4 * R];
    LocalTensor<int16_t> lo = s[5 * R];
    LocalTensor<int16_t> hi = s[6 * R];
    Duplicate(mFF, (int16_t)0x00FF, R);
    Duplicate(m0F, (int16_t)0x0F0F, R);
    Duplicate(m33, (int16_t)0x3333, R);
    Duplicate(m55, (int16_t)0x5555, R);
    for (int w = 0; w < nw; ++w) {
        And(lo, a[w * R], mFF, R);
        ShiftRight(hi, a[w * R], (int16_t)8, R);
        And(hi, hi, mFF, R);
        ShiftLeft(sh, lo, (int16_t)4, R);
        Or(sh, lo, sh, R);
        And(sh, sh, m0F, R);
        ShiftLeft(z[(2 * w) * R], sh, (int16_t)2, R);
        Or(z[(2 * w) * R], sh, z[(2 * w) * R], R);
        And(z[(2 * w) * R], z[(2 * w) * R], m33, R);
        ShiftLeft(sh, z[(2 * w) * R], (int16_t)1, R);
        Or(z[(2 * w) * R], z[(2 * w) * R], sh, R);
        And(z[(2 * w) * R], z[(2 * w) * R], m55, R);
        ShiftLeft(sh, hi, (int16_t)4, R);
        Or(sh, hi, sh, R);
        And(sh, sh, m0F, R);
        ShiftLeft(z[(2 * w + 1) * R], sh, (int16_t)2, R);
        Or(z[(2 * w + 1) * R], sh, z[(2 * w + 1) * R], R);
        And(z[(2 * w + 1) * R], z[(2 * w + 1) * R], m33, R);
        ShiftLeft(sh, z[(2 * w + 1) * R], (int16_t)1, R);
        Or(z[(2 * w + 1) * R], z[(2 * w + 1) * R], sh, R);
        And(z[(2 * w + 1) * R], z[(2 * w + 1) * R], m55, R);
    }
}

__aicore__ inline void VReduce128(const LocalTensor<int16_t>& z, const LocalTensor<int16_t>& s,
                                  int nw, int R)
{
    LocalTensor<int16_t> sh = s;
    LocalTensor<int16_t> tmp = s[(nw + 1) * R];
    LocalTensor<int16_t> lm = s[(nw + 2) * R];
    LocalTensor<int16_t> tw = s[(nw + 3) * R];
    LocalTensor<int16_t> tov = s[(2 * nw + 3) * R];
    LocalTensor<int16_t> t2 = s[(2 * nw + 4) * R];
    LocalTensor<int16_t> ht = s[(2 * nw + 5) * R];
    LocalTensor<int16_t> hz = z[nw * R];
    for (int w = 0; w < nw; ++w) {
        Adds(tw[w * R], hz[w * R], (int16_t)0, R);
    }
    Duplicate(tov, (int16_t)0, R);
    const int ss[3] = {1, 2, 7};
    for (int k = 0; k < 3; ++k) {
        VShlWords(sh, hz, nw, R, (uint32_t)ss[k], tmp, lm);
        for (int w = 0; w < nw; ++w) {
            Xor(tw[w * R], tw[w * R], sh[w * R], R);
        }
        Xor(tov, tov, sh[nw * R], R);
    }
    for (int w = 0; w < nw; ++w) {
        Xor(z[w * R], z[w * R], tw[w * R], R);
    }
    Adds(t2, tov, (int16_t)0, R);
    for (int k = 0; k < 3; ++k) {
        ShiftLeft(ht, tov, (int16_t)ss[k], R);
        Xor(t2, t2, ht, R);
    }
    Xor(z[0], z[0], t2, R);
}

// GF(2^8) vector multiply (r = a*b mod 0x11B), lane = row. 8-step shift-and-xor.
__aicore__ inline void VGf8Mul(const LocalTensor<int16_t>& r, const LocalTensor<int16_t>& a,
                               const LocalTensor<int16_t>& b, const LocalTensor<int16_t>& s, int L)
{
    LocalTensor<int16_t> aa = s;
    LocalTensor<int16_t> bb = s[L];
    LocalTensor<int16_t> t = s[2 * L];
    LocalTensor<int16_t> mk = s[3 * L];
    LocalTensor<int16_t> hi = s[4 * L];
    LocalTensor<int16_t> one = s[5 * L];
    LocalTensor<int16_t> mff = s[6 * L];
    Adds(aa, a, (int16_t)0, L);
    Adds(bb, b, (int16_t)0, L);
    Duplicate(r, (int16_t)0, L);
    Duplicate(one, (int16_t)1, L);
    Duplicate(mff, (int16_t)0x00FF, L);
    for (int i = 0; i < 8; ++i) {
        And(mk, bb, one, L);
        Muls(mk, mk, (int16_t)-1, L);
        And(t, aa, mk, L);
        Xor(r, r, t, L);
        ShiftRight(hi, aa, (int16_t)7, L);
        And(hi, hi, one, L);
        Muls(hi, hi, (int16_t)0x1B, L);
        ShiftLeft(aa, aa, (int16_t)1, L);
        Xor(aa, aa, hi, L);
        And(aa, aa, mff, L);
        ShiftRight(bb, bb, (int16_t)1, L);
    }
}

// ------------------------------------ class ------------------------------------

template <typename T>
class Gf2kMulSquare {
public:
    __aicore__ inline Gf2kMulSquare(){};

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const Gf2kMulSquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoTile(int64_t rowStart, int64_t rows, int64_t bytes);
    __aicore__ inline void DoTileVec(int64_t rowStart, int64_t rows);
    __aicore__ inline void DoTileVec8(int64_t rowStart, int64_t rows);

private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> aBuf;
    TBuf<QuePosition::VECCALC> bBuf;
    TBuf<QuePosition::VECCALC> cBuf;
    TBuf<QuePosition::VECCALC> vA;
    TBuf<QuePosition::VECCALC> vB;
    TBuf<QuePosition::VECCALC> vZ;
    TBuf<QuePosition::VECCALC> vS;
    TBuf<QuePosition::VECCALC> rawIn;
    TBuf<QuePosition::VECCALC> rawOut;
    TBuf<QuePosition::VECCALC> b8Raw;
    TBuf<QuePosition::VECCALC> b8A;
    TBuf<QuePosition::VECCALC> b8B;
    TBuf<QuePosition::VECCALC> b8R;
    TBuf<QuePosition::VECCALC> b8O;
    TBuf<QuePosition::VECCALC> b8S;
    TBuf<QuePosition::VECCALC> b8M;

    GlobalTensor<uint8_t> aGm;
    GlobalTensor<uint8_t> bGm;
    GlobalTensor<uint8_t> cGm;
    GlobalTensor<int16_t> aGm16;
    GlobalTensor<int16_t> bGm16;
    GlobalTensor<int16_t> cGm16;

    int64_t totalRows_ = 0;
    int64_t width_ = 1;
    int64_t k_ = 8;
    int64_t op_ = 0;
    int64_t rowsPerCore_ = 1;
    int64_t tileRows_ = 1;
    int64_t tileBytes_ = 0;
    int nw16_ = 0;
    int vR_ = 0;
    bool vec_ = false;
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

    const int64_t totalBytes = totalRows_ * width_;
    const int64_t gmBytes = totalBytes > 0 ? totalBytes : 1;
    aGm.SetGlobalBuffer((__gm__ uint8_t*)a, gmBytes);
    bGm.SetGlobalBuffer((__gm__ uint8_t*)b, gmBytes);
    cGm.SetGlobalBuffer((__gm__ uint8_t*)c, gmBytes);
    const int64_t gmW16 = (totalBytes / 2) > 0 ? (totalBytes / 2) : 1;
    aGm16.SetGlobalBuffer((__gm__ int16_t*)a, gmW16);
    bGm16.SetGlobalBuffer((__gm__ int16_t*)b, gmW16);
    cGm16.SetGlobalBuffer((__gm__ int16_t*)c, gmW16);

    pipe.InitBuffer(aBuf, (uint32_t)tileBytes_);
    pipe.InitBuffer(bBuf, (uint32_t)tileBytes_);
    pipe.InitBuffer(cBuf, (uint32_t)tileBytes_);

    nw16_ = GF2K_VEC_NW16;
    if (k_ == 128 && width_ == 16 && op_ == 1) {
        // Vector path is profitable only for square (bit-spread); the carryless
        // mul vector path is not faster than the scalar tail, so mul stays scalar.
        vec_ = true;
        const int wordsPerRow = (op_ == 0)
            ? (nw16_ + nw16_ + 2 * nw16_ + GF2K_VEC_SW + nw16_ + nw16_)
            : (nw16_ + 2 * nw16_ + GF2K_VEC_SW + nw16_ + nw16_);
        int r = (int)((150u * 1024u) / (uint32_t)(wordsPerRow * 2));
        if (r > 512) r = 512;
        if (r < 64) r = 64;
        r = (r / 64) * 64;
        vR_ = r;
        const uint32_t RR = (uint32_t)vR_;
        pipe.InitBuffer(vA, (uint32_t)nw16_ * RR * 2u);
        if (op_ == 0) {
            pipe.InitBuffer(vB, (uint32_t)nw16_ * RR * 2u);
        }
        pipe.InitBuffer(vZ, (uint32_t)(2 * nw16_) * RR * 2u);
        pipe.InitBuffer(vS, (uint32_t)GF2K_VEC_SW * RR * 2u);
        pipe.InitBuffer(rawIn, (uint32_t)nw16_ * RR * 2u);
        pipe.InitBuffer(rawOut, (uint32_t)nw16_ * RR * 2u);
    } else if (k_ == 8 && width_ == 1 && rowsPerCore_ > 0 && (rowsPerCore_ % 2 == 0)) {
        vec_ = true;
        int r = 256;    // rows per vector tile (chunk cap; validated safe)
        vR_ = r;
        const uint32_t L = (uint32_t)(vR_ / 2);
        pipe.InitBuffer(b8Raw, L * 2u);
        pipe.InitBuffer(b8A, 2u * L * 2u);
        pipe.InitBuffer(b8B, 2u * L * 2u);
        pipe.InitBuffer(b8R, 2u * L * 2u);
        pipe.InitBuffer(b8O, L * 2u);
        pipe.InitBuffer(b8S, 7u * L * 2u);
        pipe.InitBuffer(b8M, L * 2u);
    }
}

// ------------------------------------ scalar tile ------------------------------------

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

    PipeBarrier<PIPE_ALL>();
}

// ------------------------------------ k=128 vector tile ------------------------------------

template <typename T>
__aicore__ inline void Gf2kMulSquare<T>::DoTileVec(int64_t rowStart, int64_t rows)
{
    const int nw = nw16_;
    const int R = (int)rows;
    const uint32_t nv = (uint32_t)rows;
    const int64_t W = width_;

    LocalTensor<int16_t> a = vA.Get<int16_t>();
    LocalTensor<int16_t> z = vZ.Get<int16_t>();
    LocalTensor<int16_t> s = vS.Get<int16_t>();
    LocalTensor<int16_t> rawI = rawIn.Get<int16_t>();
    LocalTensor<int16_t> rawO = rawOut.Get<int16_t>();

    DataCopyExtParams pin;
    pin.blockCount = 1;
    pin.blockLen = (uint32_t)(nv * (uint32_t)W);
    pin.srcStride = 0;
    pin.dstStride = 0;
    pin.rsv = 0;

    DataCopyPad(rawI, aGm16[rowStart * nw], pin, DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t r = 0; r < nv; ++r) {
        const uint32_t base = r * (uint32_t)nw;
        for (int w = 0; w < nw; ++w) {
            a.SetValue((uint32_t)(w * R) + r, rawI.GetValue(base + (uint32_t)w));
        }
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);

    if (op_ == 0) {
        LocalTensor<int16_t> b = vB.Get<int16_t>();
        DataCopyPad(rawI, bGm16[rowStart * nw], pin, DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t r = 0; r < nv; ++r) {
            const uint32_t base = r * (uint32_t)nw;
            for (int w = 0; w < nw; ++w) {
                b.SetValue((uint32_t)(w * R) + r, rawI.GetValue(base + (uint32_t)w));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        VClMul(a, b, z, s, nw, R);
    } else {
        VSquareSpread(a, z, s, nw, R);
    }

    VReduce128(z, s, nw, R);

    // V -> S : gather result word columns (scalar) then DMA out
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t r = 0; r < nv; ++r) {
        const uint32_t base = r * (uint32_t)nw;
        for (int w = 0; w < nw; ++w) {
            rawO.SetValue(base + (uint32_t)w, z.GetValue((uint32_t)(w * R) + r));
        }
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

    DataCopyExtParams po;
    po.blockCount = 1;
    po.blockLen = (uint32_t)(nv * (uint32_t)W);
    po.srcStride = 0;
    po.dstStride = 0;
    po.rsv = 0;
    DataCopyPad(cGm16[rowStart * nw], rawO, po);
    // MTE3 -> S : drain the async store before the next tile reuses UB
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// ------------------------------------ k=8 vector tile ------------------------------------

template <typename T>
__aicore__ inline void Gf2kMulSquare<T>::DoTileVec8(int64_t rowStart, int64_t rows)
{
    const uint32_t nv = (uint32_t)rows;    // rows (even)
    const int L = (int)(rows / 2);          // int16 lanes: 2 rows per lane
    const uint32_t Lu = (uint32_t)L;

    LocalTensor<int16_t> raw = b8Raw.Get<int16_t>();
    LocalTensor<int16_t> alo = b8A.Get<int16_t>();
    LocalTensor<int16_t> ahi = b8A.Get<int16_t>()[L];
    LocalTensor<int16_t> blo = b8B.Get<int16_t>();
    LocalTensor<int16_t> bhi = b8B.Get<int16_t>()[L];
    LocalTensor<int16_t> rlo = b8R.Get<int16_t>();
    LocalTensor<int16_t> rhi = b8R.Get<int16_t>()[L];
    LocalTensor<int16_t> o = b8O.Get<int16_t>();
    LocalTensor<int16_t> s = b8S.Get<int16_t>();
    LocalTensor<int16_t> mff = b8M.Get<int16_t>();

    // Load a: rows bytes contiguous = L int16 (low byte = even row, high = odd).
    // DMA in <=256B chunks (blockCount=8 x blockLen=32) to stay within the
    // per-DataCopyPad size limit observed on this core; nv is a multiple of 256.
    DataCopyExtParams p;
    p.blockCount = 8;
    p.blockLen = 32;
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    const int64_t basePair = rowStart / 2;
    for (uint32_t off = 0; off < nv; off += 256u) {
        DataCopyPad(raw[(off / 2)], aGm16[basePair + (int64_t)(off / 2)], p,
                    DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Duplicate(mff, (int16_t)0x00FF, L);
    And(alo, raw, mff, L);
    ShiftRight(ahi, raw, (int16_t)8, L);
    And(ahi, ahi, mff, L);

    if (op_ == 0) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (uint32_t off = 0; off < nv; off += 256u) {
            DataCopyPad(raw[(off / 2)], bGm16[basePair + (int64_t)(off / 2)], p,
                        DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        And(blo, raw, mff, L);
        ShiftRight(bhi, raw, (int16_t)8, L);
        And(bhi, bhi, mff, L);
    } else {
        Adds(blo, alo, (int16_t)0, L);
        Adds(bhi, ahi, (int16_t)0, L);
    }

    VGf8Mul(rlo, alo, blo, s, L);
    VGf8Mul(rhi, ahi, bhi, s, L);

    ShiftLeft(o, rhi, (int16_t)8, L);
    Or(o, o, rlo, L);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    for (uint32_t off = 0; off < nv; off += 256u) {
        DataCopyPad(cGm16[basePair + (int64_t)(off / 2)], o[(off / 2)], p);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

// ------------------------------------ process ------------------------------------

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
    if (vec_) {
        const int64_t step = (k_ == 8) ? 256 : 64;
        while (endRow - r >= step) {
            int64_t chunk = endRow - r;
            if (chunk > (int64_t)vR_) {
                chunk = vR_;
            }
            chunk = (chunk / step) * step;
            if (k_ == 8) {
                DoTileVec8(r, chunk);
            } else {
                DoTileVec(r, chunk);
            }
            r += chunk;
        }
    }
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

/*!
 * \file gf2_poly.h
 * \brief Gf2Poly kernel class definition
 *
 * Row-wise GF(2) polynomial arithmetic over uint8 byte vectors. Encoding is
 * little-endian: byte j, bit r encodes the coefficient of x^(8j+r).
 *   mul        : y[i] = A_i(x) * B_i(x)          (full carryless product, 2W bytes)
 *   reduce     : y[i] = T_i(x) mod P(x)          (W bytes)
 *   mul_reduce : y[i] = (A_i(x) * B_i(x)) mod P(x) (W bytes)
 *
 * All rows are independent and share one runtime modulus P (degree m, monic,
 * P = mod_poly). Multi-core split is per row: no workspace, no cross-core
 * synchronisation, no atomics. Each core streams tiles of rows through UB with
 * DataCopyPad and does the bit arithmetic with scalar ops only (no vector /
 * Gather instructions).
 */

#ifndef GF2POLY_H
#define GF2POLY_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gf2_poly_tiling_data.h"
#include "gf2_poly_tiling_key.h"

namespace NsGf2Poly {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int GF2P_MAX_NW = 4;  // max uint64 words for a/b row (m <= 256)
constexpr int GF2P_MAX_PW = 5;  // max uint64 words for P (m + 1 bits)
constexpr int GF2P_MAX_ZW = 8;  // max uint64 words for product (2m bits)

// Carryless (GF(2)) multiply: a (nw words) * b (nw words) -> z (2*nw words).
// 4-bit windowed table method, pure integer / bit operations.
__aicore__ inline void Gf2ClMul(const uint64_t* a, const uint64_t* b, int nw, uint64_t* z)
{
    const int total = 2 * nw;
    for (int i = 0; i < total; ++i) {
        z[i] = 0ULL;
    }
    const int tw = nw + 1;           // a<<3 can spill into one extra word
    uint64_t tab[16][GF2P_MAX_PW];   // tab[k] = a * k over GF(2), k = 0..15
    for (int w = 0; w < tw; ++w) {
        tab[0][w] = 0ULL;
    }
    uint64_t sh[4][GF2P_MAX_PW];     // sh[j] = a << j, j = 0..3
    for (int w = 0; w < tw; ++w) {
        sh[0][w] = (w < nw) ? a[w] : 0ULL;
    }
    for (int j = 1; j < 4; ++j) {
        for (int w = tw - 1; w > 0; --w) {
            sh[j][w] = (sh[j - 1][w] << 1) | (sh[j - 1][w - 1] >> 63);
        }
        sh[j][0] = sh[j - 1][0] << 1;
    }
    for (int j = 0; j < 4; ++j) {
        const int pw = 1 << j;
        for (int k = 0; k < pw; ++k) {
            for (int w = 0; w < tw; ++w) {
                tab[pw + k][w] = tab[k][w] ^ sh[j][w];
            }
        }
    }
    const int nibbles = nw * 16;  // 4 bits per nibble over nw*64 bits of b
    for (int i = 0; i < nibbles; ++i) {
        const int bit = 4 * i;
        const int wi = bit >> 6;
        const int sft = bit & 63;
        const uint64_t nib = (b[wi] >> sft) & 0xFULL;
        for (int w = 0; w < tw; ++w) {
            uint64_t val = tab[nib][w];
            if (val == 0ULL) {
                continue;
            }
            const int dst = wi + w;
            if (sft == 0) {
                if (dst < total) {
                    z[dst] ^= val;
                }
            } else {
                if (dst < total) {
                    z[dst] ^= (val << sft);
                }
                if (dst + 1 < total) {
                    z[dst + 1] ^= (val >> (64 - sft));
                }
            }
        }
    }
}

// Polynomial long division: reduce z (2*nw words, degree < 2m) modulo monic
// P (pnw words, degree m). Result r: nw words.
__aicore__ inline void Gf2Reduce(const uint64_t* z, int nw, const uint64_t* P, int pnw, int m,
                                 uint64_t* r)
{
    uint64_t rem[GF2P_MAX_PW];
    for (int w = 0; w < pnw; ++w) {
        rem[w] = 0ULL;
    }
    const int mw = m >> 6;
    const int mb = m & 63;
    const int totalBits = 2 * m;
    for (int pos = totalBits - 1; pos >= 0; --pos) {
        const uint64_t inbit = (z[pos >> 6] >> (pos & 63)) & 1ULL;
        for (int w = pnw - 1; w > 0; --w) {
            rem[w] = (rem[w] << 1) | (rem[w - 1] >> 63);
        }
        rem[0] = (rem[0] << 1) | inbit;
        if ((rem[mw] >> mb) & 1ULL) {
            for (int w = 0; w < pnw; ++w) {
                rem[w] ^= P[w];
            }
        }
    }
    for (int w = 0; w < nw; ++w) {
        r[w] = rem[w];
    }
}

// Polynomial Barrett: precompute mu = floor(x^(2m) / P); muLow = low m bits of mu
// (the implicit x^m term is dropped). Only valid when m is a multiple of 64.
__aicore__ inline void Gf2ComputeMu(const uint64_t* P, int pnw, int m, int nw, uint64_t* muLow)
{
    const int nz = 2 * nw + 1;
    uint64_t cur[GF2P_MAX_ZW + 1];
    for (int w = 0; w < nz; ++w) {
        cur[w] = 0ULL;
    }
    const int topbit = 2 * m;
    cur[topbit >> 6] |= (1ULL << (topbit & 63));
    uint64_t quo[GF2P_MAX_PW];
    for (int w = 0; w < pnw; ++w) {
        quo[w] = 0ULL;
    }
    for (int pos = 2 * m; pos >= m; --pos) {
        if ((cur[pos >> 6] >> (pos & 63)) & 1ULL) {
            const int qb = pos - m;
            quo[qb >> 6] |= (1ULL << (qb & 63));
            const int sh = pos - m;
            const int wo = sh >> 6;
            const int bo = sh & 63;
            for (int w = 0; w < pnw; ++w) {
                const uint64_t pv = P[w];
                const int d = wo + w;
                if (d < nz) {
                    cur[d] ^= (pv << bo);
                }
                if (bo != 0 && d + 1 < nz) {
                    cur[d + 1] ^= (pv >> (64 - bo));
                }
            }
        }
    }
    for (int w = 0; w < nw; ++w) {
        muLow[w] = quo[w];
    }
}

// Barrett reduction for m multiple of 64: z (2*nw words, deg < 2m) mod P.
__aicore__ inline void Gf2ReduceBarrett(const uint64_t* z, int nw, const uint64_t* P,
                                        const uint64_t* muLow, uint64_t* r)
{
    // q1 = floor(z / x^m)  -> words [nw .. 2nw-1]
    uint64_t q1[GF2P_MAX_NW];
    for (int w = 0; w < nw; ++w) {
        q1[w] = z[nw + w];
    }
    // q2 = q1 * mu = (q1 << m) ^ clmul(q1, muLow)
    uint64_t t2[GF2P_MAX_ZW];
    Gf2ClMul(q1, muLow, nw, t2);
    for (int w = 0; w < nw; ++w) {
        t2[nw + w] ^= q1[w];
    }
    // q3 = floor(q2 / x^m)
    uint64_t q3[GF2P_MAX_NW];
    for (int w = 0; w < nw; ++w) {
        q3[w] = t2[nw + w];
    }
    // q3*P = (q3 << m) ^ clmul(q3, Q)   where Q = P low m bits (== P[0..nw-1])
    uint64_t t3[GF2P_MAX_ZW];
    Gf2ClMul(q3, P, nw, t3);
    for (int w = 0; w < nw; ++w) {
        t3[nw + w] ^= q3[w];
    }
    // r = low m bits of (z ^ q3*P)
    for (int w = 0; w < nw; ++w) {
        r[w] = z[w] ^ t3[w];
    }
}

template <typename T>
class Gf2Poly {
public:
    __aicore__ inline Gf2Poly(){};

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR t, GM_ADDR mod_poly, GM_ADDR y,
                                const Gf2PolyTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWords(LocalTensor<uint8_t>& src, int64_t byteOff, int nBytes,
                                     uint64_t* dst, bool aligned8);
    __aicore__ inline void StoreWords(LocalTensor<uint8_t>& dst, int64_t byteOff, int nBytes,
                                      const uint64_t* src, bool aligned8);
    __aicore__ inline void DoTile(int64_t rowStart, int64_t rows, int64_t bytesAB, int64_t bytesT,
                                  int64_t bytesOut);

private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> aBuf;
    TBuf<QuePosition::VECCALC> bBuf;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    TBuf<QuePosition::VECCALC> modBuf;

    GlobalTensor<uint8_t> aGm;
    GlobalTensor<uint8_t> bGm;
    GlobalTensor<uint8_t> tGm;
    GlobalTensor<uint8_t> yGm;
    GlobalTensor<uint8_t> modGm;

    uint64_t P_[GF2P_MAX_PW];
    uint64_t muLow_[GF2P_MAX_NW];

    int64_t totalRows_ = 0;
    int64_t width_ = 1;
    int64_t outWidth_ = 1;
    int64_t m_ = 8;
    int64_t mode_ = 2;
    int64_t rowsPerCore_ = 1;
    int64_t tileRows_ = 1;
    int64_t tileBytes_ = 0;
    int nw_ = 1;
    int pnw_ = 1;
    bool alignAB_ = false;  // width_ % 8 == 0
    bool alignT_ = false;   // (2*width_) % 8 == 0
    bool alignOB_ = false;  // outWidth_ % 8 == 0
    bool barrett_ = false;  // use Barrett reduction (m multiple of 64)
};

// Copy nBytes from a UB byte tensor starting at byteOff into word array dst.
template <typename T>
__aicore__ inline void Gf2Poly<T>::LoadWords(LocalTensor<uint8_t>& src, int64_t byteOff, int nBytes,
                                             uint64_t* dst, bool aligned8)
{
    const int nwds = (nBytes + 7) / 8;
    if (aligned8) {
        LocalTensor<uint64_t> s64 = src.ReinterpretCast<uint64_t>();
        const uint32_t base = (uint32_t)(byteOff >> 3);
        for (int w = 0; w < nwds; ++w) {
            dst[w] = s64.GetValue(base + (uint32_t)w);
        }
    } else {
        for (int w = 0; w < nwds; ++w) {
            uint64_t v = 0ULL;
            const int64_t b0 = byteOff + (int64_t)w * 8;
            for (int r = 0; r < 8; ++r) {
                if (w * 8 + r < nBytes) {
                    v |= (uint64_t)src.GetValue((uint32_t)(b0 + r)) << (8 * r);
                }
            }
            dst[w] = v;
        }
    }
}

// Write nBytes of word array src into a UB byte tensor starting at byteOff.
template <typename T>
__aicore__ inline void Gf2Poly<T>::StoreWords(LocalTensor<uint8_t>& dst, int64_t byteOff, int nBytes,
                                              const uint64_t* src, bool aligned8)
{
    const int nwds = (nBytes + 7) / 8;
    if (aligned8) {
        LocalTensor<uint64_t> d64 = dst.ReinterpretCast<uint64_t>();
        const uint32_t base = (uint32_t)(byteOff >> 3);
        for (int w = 0; w < nwds; ++w) {
            d64.SetValue(base + (uint32_t)w, src[w]);
        }
    } else {
        for (int w = 0; w < nwds; ++w) {
            const int64_t b0 = byteOff + (int64_t)w * 8;
            for (int r = 0; r < 8; ++r) {
                if (w * 8 + r < nBytes) {
                    dst.SetValue((uint32_t)(b0 + r), (uint8_t)(src[w] >> (8 * r)));
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Gf2Poly<T>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR t, GM_ADDR mod_poly,
                                       GM_ADDR y, const Gf2PolyTilingData* tilingData)
{
    totalRows_ = tilingData->totalRows;
    width_ = tilingData->width;
    outWidth_ = tilingData->outWidth;
    m_ = tilingData->m;
    mode_ = tilingData->mode;
    rowsPerCore_ = tilingData->rowsPerCore;
    tileRows_ = tilingData->tileRows;
    if (width_ < 1) {
        width_ = 1;
    }
    if (outWidth_ < 1) {
        outWidth_ = width_;
    }
    if (tileRows_ < 1) {
        tileRows_ = 1;
    }
    if (rowsPerCore_ < 0) {
        rowsPerCore_ = 0;
    }
    nw_ = (int)((width_ + 7) / 8);
    if (nw_ < 1) {
        nw_ = 1;
    }
    if (nw_ > GF2P_MAX_NW) {
        nw_ = GF2P_MAX_NW;
    }
    pnw_ = (int)((m_ + 1 + 63) / 64);
    if (pnw_ > GF2P_MAX_PW) {
        pnw_ = GF2P_MAX_PW;
    }
    alignAB_ = (width_ % 8 == 0);
    alignT_ = ((2 * width_) % 8 == 0);
    alignOB_ = (outWidth_ % 8 == 0);
    barrett_ = (m_ >= 64 && (m_ % 64 == 0));
    tileBytes_ = tileRows_ * width_;

    // Declare at least one byte so that empty tensors still get a valid range.
    const int64_t abBytesGm = totalRows_ * width_ > 0 ? totalRows_ * width_ : 1;
    const int64_t tBytesGm = totalRows_ * 2 * width_ > 0 ? totalRows_ * 2 * width_ : 1;
    const int64_t yBytesGm = totalRows_ * outWidth_ > 0 ? totalRows_ * outWidth_ : 1;
    aGm.SetGlobalBuffer((__gm__ uint8_t*)a, abBytesGm);
    bGm.SetGlobalBuffer((__gm__ uint8_t*)b, abBytesGm);
    tGm.SetGlobalBuffer((__gm__ uint8_t*)t, tBytesGm);
    yGm.SetGlobalBuffer((__gm__ uint8_t*)y, yBytesGm);
    modGm.SetGlobalBuffer((__gm__ uint8_t*)mod_poly, width_ + 1);

    const uint32_t abBytes = (uint32_t)(((tileBytes_) + 31) / 32 * 32);
    const uint32_t tBytes = (uint32_t)(((tileRows_ * 2 * width_) + 31) / 32 * 32);
    const uint32_t oBytes = (uint32_t)(((tileRows_ * outWidth_) + 31) / 32 * 32);
    pipe.InitBuffer(aBuf, abBytes);
    pipe.InitBuffer(bBuf, abBytes);
    pipe.InitBuffer(tBuf, tBytes);
    pipe.InitBuffer(outBuf, oBytes);
    pipe.InitBuffer(modBuf, (uint32_t)(((width_ + 1) + 31) / 32 * 32));

    // Load mod_poly once per core and build the modulus words P_.
    LocalTensor<uint8_t> pm = modBuf.Get<uint8_t>();
    DataCopyExtParams pIn;
    pIn.blockCount = 1;
    pIn.blockLen = (uint32_t)(width_ + 1);
    pIn.srcStride = 0;
    pIn.dstStride = 0;
    pIn.rsv = 0;
    DataCopyPad(pm, modGm, pIn, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (int w = 0; w < pnw_; ++w) {
        uint64_t v = 0ULL;
        for (int r = 0; r < 8; ++r) {
            const int64_t bidx = (int64_t)w * 8 + r;
            if (bidx <= width_) {
                v |= (uint64_t)pm.GetValue((uint32_t)bidx) << (8 * r);
            }
        }
        P_[w] = v;
    }
    if (barrett_) {
        Gf2ComputeMu(P_, pnw_, (int)m_, nw_, muLow_);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Gf2Poly<T>::DoTile(int64_t rowStart, int64_t rows, int64_t bytesAB,
                                          int64_t bytesT, int64_t bytesOut)
{
    if (rows <= 0) {
        return;
    }
    LocalTensor<uint8_t> inA = aBuf.Get<uint8_t>();
    LocalTensor<uint8_t> inB = bBuf.Get<uint8_t>();
    LocalTensor<uint8_t> inT = tBuf.Get<uint8_t>();
    LocalTensor<uint8_t> outY = outBuf.Get<uint8_t>();

    if (mode_ != 1) {  // mul / mul_reduce need a and b
        DataCopyExtParams pAB;
        pAB.blockCount = 1;
        pAB.blockLen = (uint32_t)bytesAB;
        pAB.srcStride = 0;
        pAB.dstStride = 0;
        pAB.rsv = 0;
        DataCopyPad(inA, aGm[rowStart * width_], pAB, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        DataCopyPad(inB, bGm[rowStart * width_], pAB, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    }
    if (mode_ == 1) {  // reduce needs t
        DataCopyExtParams pT;
        pT.blockCount = 1;
        pT.blockLen = (uint32_t)bytesT;
        pT.srcStride = 0;
        pT.dstStride = 0;
        pT.rsv = 0;
        DataCopyPad(inT, tGm[rowStart * 2 * width_], pT, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t z[GF2P_MAX_ZW];
        for (int w = 0; w < 2 * nw_; ++w) {
            z[w] = 0ULL;
        }
        if (mode_ == 1) {
            LoadWords(inT, i * 2 * width_, (int)(2 * width_), z, alignT_);
        } else {
            uint64_t aw[GF2P_MAX_NW];
            uint64_t bw[GF2P_MAX_NW];
            LoadWords(inA, i * width_, (int)width_, aw, alignAB_);
            LoadWords(inB, i * width_, (int)width_, bw, alignAB_);
            Gf2ClMul(aw, bw, nw_, z);
        }
        if (mode_ == 0) {
            StoreWords(outY, i * outWidth_, (int)outWidth_, z, alignOB_);
        } else {
            uint64_t r[GF2P_MAX_NW];
            if (barrett_) {
                Gf2ReduceBarrett(z, nw_, P_, muLow_, r);
            } else {
                Gf2Reduce(z, nw_, P_, pnw_, (int)m_, r);
            }
            StoreWords(outY, i * width_, (int)width_, r, alignAB_);
        }
    }

    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

    DataCopyExtParams pOut;
    pOut.blockCount = 1;
    pOut.blockLen = (uint32_t)bytesOut;
    pOut.srcStride = 0;
    pOut.dstStride = 0;
    pOut.rsv = 0;
    DataCopyPad(yGm[rowStart * outWidth_], outY, pOut);

    // MTE3 is async: drain before the next tile reuses the same UB buffer.
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Gf2Poly<T>::Process()
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
        const int64_t bytesAB = rows * width_;
        const int64_t bytesT = rows * 2 * width_;
        const int64_t bytesOut = rows * outWidth_;
        DoTile(r, rows, bytesAB, bytesT, bytesOut);
        r += rows;
    }
}

}  // namespace NsGf2Poly
#endif  // GF2POLY_H

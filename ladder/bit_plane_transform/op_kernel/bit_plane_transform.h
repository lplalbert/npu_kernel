/*!
 * \file bit_plane_transform.h
 * \brief BitPlaneTransform kernel class definition.
 *
 *  transpose: x[N,C] uint8 -> y[ceil(N/32), 8C, 4] uint8
 *    y[g,8j+r,d] = sum_s bit_r(x[32g+8d+s, j]) << s
 *  inverse:   x[G,8C,4] uint8 -> y[n,C] uint8
 *    y[i,j] = sum_r bit_{i%8}( x[i/32,8j+r,(i%32)/8] ) << r
 *
 *  Vector fast path for transpose (C in {4,8,16,32}):
 *    - batch G = 256/C groups so G*(C/4) = 64 uint32 lanes = one vector.
 *    - UB tile layout [s][64 lanes]; lane p = g*(C/4)+cg holds 4 bytes of
 *      columns 4cg..4cg+3 of row s of group (g0+g). 32 strided DataCopyPad.
 *    - D[r,d] = OR_{s'} ((tile[8d+s']>>r)&0x01010101)<<s'
 *      T[q,r] = OR_{d} ((D[r,d]>>8q)&0xFF)<<8d
 *    - T[q,r] lane p is the output uint32 word for (column, plane r); global
 *      uint32 index = 32*p + 8*q + r, stored as 64 x 4-byte blocks stride 124.
 *
 *  inverse / C==1 / remainder / partial groups use the exact scalar SWAR path.
 */

#ifndef BITPLANETRANSFORM_H
#define BITPLANETRANSFORM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bit_plane_transform_tiling_data.h"
#include "bit_plane_transform_tiling_key.h"

namespace NsBitPlaneTransform {

using namespace AscendC;

__aicore__ inline uint64_t Transpose8x8(uint64_t x)
{
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    x = x ^ t ^ (t << 28);
    return x;
}

template <typename T>
class BitPlaneTransform {
public:
    __aicore__ inline BitPlaneTransform(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BitPlaneTransformTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoTransposeTile(const LocalTensor<uint8_t>& inU8, const LocalTensor<uint32_t>& outU32,
                                           int64_t b, int64_t C, int64_t gb);
    __aicore__ inline void DoInverseTile(const LocalTensor<uint32_t>& inU32, const LocalTensor<uint8_t>& outU8,
                                         int64_t b, int64_t C, int64_t gb);
    __aicore__ inline void DoPartialTranspose(int64_t g, int64_t V, int64_t C, int64_t gb);
    __aicore__ inline void DoPartialInverse(int64_t g, int64_t V, int64_t C, int64_t gb);
    __aicore__ inline void DoScalarGroupTranspose(int64_t g, int64_t C, int64_t gb);
    __aicore__ inline void DoVecBatchTranspose(int64_t bi, int64_t C, int64_t gb);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;
    TBuf<QuePosition::VECCALC> tileBuf_;
    TBuf<QuePosition::VECCALC> dBuf_;
    TBuf<QuePosition::VECCALC> tBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> gOff_;
    TBuf<QuePosition::VECCALC> gDst_;
    TBuf<QuePosition::VECCALC> bStage_;

    GlobalTensor<uint8_t> inGm_;
    GlobalTensor<uint8_t> outGm_;

    BitPlaneTransformTilingData t_;
    int64_t tileBytes_ = 0;
    bool vecPath_ = false;
};

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::Init(GM_ADDR x, GM_ADDR y,
                                                  const BitPlaneTransformTilingData* tilingData)
{
    t_ = *tilingData;
    const int64_t C = (t_.cols > 0) ? t_.cols : 1;
    const int64_t gb = 32 * C;
    const bool isInv = (t_.dirFlag != 0);
    vecPath_ = (!isInv) && (C == 32);

    int64_t B = (t_.batchGroups > 0) ? t_.batchGroups : 1;
    tileBytes_ = B * gb;
    if (tileBytes_ < 32) {
        tileBytes_ = 32;
    }
    const int64_t totalBytes = t_.totalGroups * gb;
    inGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x), totalBytes);
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), totalBytes);

    if (vecPath_) {
        pipe.InitBuffer(inBuf_, static_cast<uint32_t>(gb));
        pipe.InitBuffer(outBuf_, static_cast<uint32_t>(gb));
        pipe.InitBuffer(tileBuf_, 8192);
        pipe.InitBuffer(dBuf_, 8192);
        pipe.InitBuffer(tBuf_, 8192);
        pipe.InitBuffer(tmpBuf_, 256);
        pipe.InitBuffer(maskBuf_, 512);
        pipe.InitBuffer(gOff_, 8192);
        pipe.InitBuffer(gDst_, 8192);
        pipe.InitBuffer(bStage_, 65536);
    } else {
        pipe.InitBuffer(inBuf_, static_cast<uint32_t>(tileBytes_));
        pipe.InitBuffer(outBuf_, static_cast<uint32_t>(tileBytes_));
        pipe.InitBuffer(tileBuf_, 32);
        pipe.InitBuffer(dBuf_, 32);
        pipe.InitBuffer(tBuf_, 32);
        pipe.InitBuffer(tmpBuf_, 32);
        pipe.InitBuffer(maskBuf_, 32);
        pipe.InitBuffer(gOff_, 32);
        pipe.InitBuffer(gDst_, 32);
        pipe.InitBuffer(bStage_, 32);
    }
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::DoTransposeTile(const LocalTensor<uint8_t>& inU8,
                                                             const LocalTensor<uint32_t>& outU32,
                                                             int64_t b, int64_t C, int64_t gb)
{
    for (int64_t gg = 0; gg < b; ++gg) {
        const int64_t ib = gg * gb;
        const int64_t ob = gg * gb / 4;
        for (int64_t j = 0; j < C; ++j) {
            uint64_t Q[4];
            for (int i = 0; i < 4; ++i) {
                uint64_t v = 0;
                for (int tt = 0; tt < 8; ++tt) {
                    uint64_t byteVal = static_cast<uint64_t>(inU8.GetValue(ib + (8 * i + tt) * C + j));
                    v |= (byteVal << (8 * tt));
                }
                Q[i] = v;
            }
            uint32_t w[8];
            for (int r = 0; r < 8; ++r) {
                w[r] = 0;
            }
            for (int i = 0; i < 4; ++i) {
                uint64_t Tr = Transpose8x8(Q[i]);
                for (int r = 0; r < 8; ++r) {
                    w[r] |= static_cast<uint32_t>((Tr >> (8 * r)) & 0xFFULL) << (8 * i);
                }
            }
            const int64_t wbase = ob + j * 8;
            for (int r = 0; r < 8; ++r) {
                outU32.SetValue(wbase + r, w[r]);
            }
        }
    }
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::DoInverseTile(const LocalTensor<uint32_t>& inU32,
                                                           const LocalTensor<uint8_t>& outU8,
                                                           int64_t b, int64_t C, int64_t gb)
{
    for (int64_t gg = 0; gg < b; ++gg) {
        const int64_t wb0 = gg * gb / 4;
        const int64_t ob = gg * gb;
        for (int64_t j = 0; j < C; ++j) {
            const int64_t wbase = wb0 + j * 8;
            uint32_t w[8];
            for (int r = 0; r < 8; ++r) {
                w[r] = inU32.GetValue(wbase + r);
            }
            for (int d = 0; d < 4; ++d) {
                uint64_t tw = 0;
                for (int r = 0; r < 8; ++r) {
                    uint64_t byteVal = static_cast<uint64_t>((w[r] >> (8 * d)) & 0xFFU);
                    tw |= (byteVal << (8 * r));
                }
                uint64_t Qd = Transpose8x8(tw);
                for (int s = 0; s < 8; ++s) {
                    outU8.SetValue(ob + (8 * d + s) * C + j, static_cast<uint8_t>((Qd >> (8 * s)) & 0xFFULL));
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::DoPartialTranspose(int64_t g, int64_t V, int64_t C, int64_t gb)
{
    const int64_t base = g * gb;
    for (int64_t j = 0; j < C; ++j) {
        for (int r = 0; r < 8; ++r) {
            for (int d = 0; d < 4; ++d) {
                uint32_t val = 0;
                for (int s = 0; s < 8; ++s) {
                    int64_t lane = 8 * d + s;
                    if (lane < V) {
                        uint8_t bv = inGm_.GetValue(base + lane * C + j);
                        val |= static_cast<uint32_t>((bv >> r) & 1U) << s;
                    }
                }
                outGm_.SetValue(base + (8 * j + r) * 4 + d, static_cast<uint8_t>(val));
            }
        }
    }
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::DoPartialInverse(int64_t g, int64_t V, int64_t C, int64_t gb)
{
    const int64_t ob = g * gb;
    for (int64_t j = 0; j < C; ++j) {
        uint32_t w[8];
        for (int r = 0; r < 8; ++r) {
            uint32_t word = 0;
            for (int d = 0; d < 4; ++d) {
                uint8_t bb = inGm_.GetValue(ob + j * 32 + 4 * r + d);
                word |= static_cast<uint32_t>(bb) << (8 * d);
            }
            w[r] = word;
        }
        for (int64_t i = 0; i < V; ++i) {
            int d = static_cast<int>((i >> 3) & 3);
            int s = static_cast<int>(i & 7);
            uint32_t val = 0;
            for (int r = 0; r < 8; ++r) {
                uint32_t bit = (((w[r] >> (8 * d)) & 0xFFU) >> s) & 1U;
                val |= bit << r;
            }
            outGm_.SetValue(ob + i * C + j, static_cast<uint8_t>(val));
        }
    }
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::DoScalarGroupTranspose(int64_t g, int64_t C, int64_t gb)
{
    LocalTensor<uint8_t> inU8 = inBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> outU8 = outBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> outU32 = outBuf_.Get<uint32_t>();
    DataCopy(inU8, inGm_[g * gb], static_cast<uint32_t>(gb));
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    DoTransposeTile(inU8, outU32, 1, C, gb);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(outGm_[g * gb], outU8, static_cast<uint32_t>(gb));
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::DoVecBatchTranspose(int64_t bi, int64_t C, int64_t gb)
{
    const int64_t G = 8;
    const int64_t L = C / 4;
    const int64_t g0 = bi * G;

    LocalTensor<uint8_t> tileU8 = tileBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> tile = tileBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> dU32 = dBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> tU32 = tBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> tmp = tmpBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> maskAll = maskBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> mask01 = maskAll;
    LocalTensor<uint32_t> maskFF = maskAll[64];

    const uint32_t gmGap = static_cast<uint32_t>(32 * C - C);
    for (int64_t s = 0; s < 32; ++s) {
        DataCopyExtParams p(static_cast<uint16_t>(G), static_cast<uint32_t>(C), gmGap, 0, 0);
        DataCopyPadExtParams<uint8_t> pad;
        DataCopyPad(tileU8[s * 256], inGm_[g0 * gb + s * C], p, pad);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    for (int r = 0; r < 8; ++r) {
        for (int d = 0; d < 4; ++d) {
            LocalTensor<uint32_t> D = dU32[(r * 4 + d) * 64];
            ShiftRight(D, tile[(8 * d + 0) * 64], static_cast<uint32_t>(r), 64);
            And(D, D, mask01, 128);
            for (int sp = 1; sp < 8; ++sp) {
                ShiftRight(tmp, tile[(8 * d + sp) * 64], static_cast<uint32_t>(r), 64);
                And(tmp, tmp, mask01, 128);
                ShiftLeft(tmp, tmp, static_cast<uint32_t>(sp), 64);
                Or(D, D, tmp, 128);
            }
        }
    }

    for (int q = 0; q < 4; ++q) {
        for (int r = 0; r < 8; ++r) {
            LocalTensor<uint32_t> Tq = tU32[(q * 8 + r) * 64];
            ShiftRight(Tq, dU32[(r * 4 + 0) * 64], static_cast<uint32_t>(8 * q), 64);
            And(Tq, Tq, maskFF, 128);
            for (int d = 1; d < 4; ++d) {
                ShiftRight(tmp, dU32[(r * 4 + d) * 64], static_cast<uint32_t>(8 * q), 64);
                And(tmp, tmp, maskFF, 128);
                ShiftLeft(tmp, tmp, static_cast<uint32_t>(8 * d), 64);
                Or(Tq, Tq, tmp, 128);
            }
        }
    }

    // permute tBuf [k=8q+r][p] -> final contiguous layout out word index 32*p + 8*q + r
    // gather permute: out word o = 32*p + k  <- src word k*64 + p  (byte offsets x4)
    LocalTensor<uint32_t> oU32 = gDst_.Get<uint32_t>();
    LocalTensor<uint32_t> gOff = gOff_.Get<uint32_t>();
    Gather(oU32, tU32, gOff, 0, 2048);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(outGm_[32 * C * g0], gDst_.Get<uint8_t>(), 8192);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t C = (t_.cols > 0) ? t_.cols : 1;
    const int64_t gb = 32 * C;
    const int64_t full = t_.fullGroups;
    const bool isInv = (t_.dirFlag != 0);

    if (vecPath_) {
        const int64_t G = 8;
        const int64_t nBatch = full / G;
        const int64_t remStart = nBatch * G;

        if (nBatch > 0) {
            int64_t base = nBatch / blockNum;
            int64_t rem = nBatch % blockNum;
            int64_t bstart = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
            int64_t bcnt = base + (blockIdx < rem ? 1 : 0);
            LocalTensor<uint32_t> maskAll = maskBuf_.Get<uint32_t>();
            Duplicate(maskAll, static_cast<uint32_t>(0x01010101U), 64);
            Duplicate(maskAll[64], static_cast<uint32_t>(0x000000FFU), 64);
            LocalTensor<uint32_t> gOff = gOff_.Get<uint32_t>();
            for (int32_t o = 0; o < 2048; ++o) {
                gOff.SetValue(o, static_cast<uint32_t>((((o & 31) * 64) + (o >> 5)) * 4));
            }
            PipeBarrier<PIPE_ALL>();
            for (int64_t k = 0; k < bcnt; ++k) {
                DoVecBatchTranspose(bstart + k, C, gb);
            }
            PipeBarrier<PIPE_ALL>();
        }

        for (int64_t g = remStart + blockIdx; g < full; g += blockNum) {
            DoScalarGroupTranspose(g, C, gb);
        }
        PipeBarrier<PIPE_ALL>();

        if (blockIdx == 0 && t_.totalGroups > full) {
            int64_t V = t_.rows - 32 * full;
            if (V > 0) {
                DoPartialTranspose(full, V, C, gb);
            }
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    LocalTensor<uint8_t> inU8 = inBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> inU32 = inBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> outU8 = outBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> outU32 = outBuf_.Get<uint32_t>();

    int64_t base = (blockNum > 0) ? (full / blockNum) : 0;
    int64_t rem = (blockNum > 0) ? (full % blockNum) : 0;
    int64_t start = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
    int64_t cnt = base + (blockIdx < rem ? 1 : 0);
    int64_t end = start + cnt;

    int64_t B = (t_.batchGroups > 0) ? t_.batchGroups : 1;
    bool first = true;
    int64_t g = start;
    while (g < end) {
        int64_t b = B;
        if (g + b > end) {
            b = end - g;
        }
        if (!first) {
            PipeBarrier<PIPE_ALL>();
        }
        first = false;
        const int64_t bytes = b * gb;
        DataCopy(inU8, inGm_[g * gb], static_cast<uint32_t>(bytes));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (!isInv) {
            DoTransposeTile(inU8, outU32, b, C, gb);
        } else {
            DoInverseTile(inU32, outU8, b, C, gb);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(outGm_[g * gb], outU8, static_cast<uint32_t>(bytes));
        g += b;
    }

    PipeBarrier<PIPE_ALL>();

    if (blockIdx == 0 && t_.totalGroups > full) {
        int64_t V = t_.rows - 32 * full;
        if (V > 0) {
            if (!isInv) {
                DoPartialTranspose(full, V, C, gb);
            } else {
                DoPartialInverse(full, V, C, gb);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

} // namespace NsBitPlaneTransform
#endif // BITPLANETRANSFORM_H

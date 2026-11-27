/*!
 * \file bit_plane_transform.h
 * \brief BitPlaneTransform 算子 kernel 类定义
 *
 *  Semantics (per problem spec):
 *    transpose: x[N,C] uint8 -> y[ceil(N/32), 8C, 4] uint8
 *      y[g,8j+r,d] = sum_s bit_r(x[32g+8d+s, j]) << s   (invalid lanes -> 0)
 *    inverse:   x[G,8C,4] uint8 -> y[n,C] uint8  (G=ceil(n/32))
 *      y[i,j] = sum_r bit_{i%8}( x[i/32,8j+r,(i%32)/8] ) << r
 *
 *  Implementation: each group (32 lanes x C bytes = 32*C contiguous bytes) is
 *  transformed with an exact 8x8 bit-matrix transpose (SWAR, 64-bit words).
 *  For transpose, 4 chunks of 8 lane-bytes are bit-transposed to the 8 plane
 *  bytes (one 32-bit word per plane r) -> 8 words per column written contiguously.
 *  Inverse is the exact reverse (transpose is self-inverse).  Trailing partial
 *  group (rows % 32 != 0) handled by scalar definition on block 0.
 *
 *  NOTE: this vector-core ISA (dav_c220) does not support uint8 for
 *  And/Or/Sub/Shift/Duplicate/Gather, so a byte-lane vector network plus
 *  byte-granular Gather is not available; the 64-bit SWAR scalar path below is
 *  the fastest correct formulation measured on this platform.
 */

#ifndef BITPLANETRANSFORM_H
#define BITPLANETRANSFORM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bit_plane_transform_tiling_data.h"
#include "bit_plane_transform_tiling_key.h"

namespace NsBitPlaneTransform {

using namespace AscendC;

// Exact 8x8 bit-matrix transpose of a 64-bit word (byte i = matrix row i).
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

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;

    GlobalTensor<uint8_t> inGm_;
    GlobalTensor<uint8_t> outGm_;

    BitPlaneTransformTilingData t_;
    int64_t tileBytes_ = 0;
};

template <typename T>
__aicore__ inline void BitPlaneTransform<T>::Init(GM_ADDR x, GM_ADDR y,
                                                  const BitPlaneTransformTilingData* tilingData)
{
    t_ = *tilingData;
    const int64_t C = (t_.cols > 0) ? t_.cols : 1;
    const int64_t gb = 32 * C;
    int64_t B = (t_.batchGroups > 0) ? t_.batchGroups : 1;
    tileBytes_ = B * gb;
    if (tileBytes_ < 32) {
        tileBytes_ = 32;
    }
    const int64_t totalBytes = t_.totalGroups * gb;
    inGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x), totalBytes);
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), totalBytes);

    pipe.InitBuffer(inBuf_, static_cast<uint32_t>(tileBytes_));
    pipe.InitBuffer(outBuf_, static_cast<uint32_t>(tileBytes_));
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
__aicore__ inline void BitPlaneTransform<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t C = (t_.cols > 0) ? t_.cols : 1;
    const int64_t gb = 32 * C;
    const int64_t full = t_.fullGroups;
    const bool isInv = (t_.dirFlag != 0);

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

/*!
 * \file group_norm.h
 * \brief GroupNorm kernel (group-per-core, double buffered, no cross-core reduce)
 *
 *  y[n,c,*] = (x[n,c,*] - mean_{group}) * rsqrt(var_{group}+eps) * gamma[c] + beta[c]
 *
 *  A group (n,g) is a contiguous block of CpG*HW elements in row-major x.
 *  Each group is fully handled by one core (deterministic, no cross-core reduce).
 *  Internal FP32; fp16 / bf16 input is Cast to fp32 first; vector counts are
 *  multiples of 64.
 *
 *  v3 performance rewrite:
 *    - pass 1 accumulates element-wise partial sums of (x-x0) and (x-x0)^2 into
 *      two running vector accumulators across ALL tiles, then does ONE
 *      ReduceSum each. This removes the per-tile ReduceSum + V->S scalar read.
 *    - load / store use depth-2 TQue double buffers (VECIN / VECOUT): the MTE2
 *      load of tile t+1 overlaps the vector work of tile t, and the MTE3 store
 *      overlaps the next compute. Cross-pipe sync is inserted by the queue
 *      (EnQue/DeQue/FreeTensor) instead of a PipeBarrier<PIPE_ALL> per tile.
 *    - the only remaining per-tile scalar work is the tail-lane patch on the
 *      last tile of a group (pad lanes -> x0), which happens once per group.
 *
 *  The group statistics are accumulated around an offset x0 = x[first element]
 *  instead of around 0: pass 1 accumulates sum(x-x0) and sum((x-x0)^2), then
 *  mean = x0 + E[x-x0] and var = E[(x-x0)^2] - E[x-x0]^2.  The shifted form
 *  keeps full precision for large-mean / small-spread groups, where the naive
 *  E[x^2] - mean^2 form cancels catastrophically in fp32.
 */

#ifndef GROUPNORM_H
#define GROUPNORM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_tiling_data.h"
#include "group_norm_tiling_key.h"

namespace NsGroupNorm {

using namespace AscendC;

template <typename T>
class GroupNorm {
public:
    __aicore__ inline GroupNorm(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                const GroupNormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadTile(const LocalTensor<float>& dstF, int64_t elemOff, int64_t cnt, int64_t cntR);
    __aicore__ inline void StoreTile(const LocalTensor<float>& srcF, int64_t elemOff, int64_t cnt, int64_t cntR);
    __aicore__ inline void ProcessGroup(int64_t gidx);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ_;    // raw input tiles (T)
    TQue<QuePosition::VECOUT, 2> outQ_;  // output tiles (T)

    TBuf<QuePosition::VECCALC> fx_;      // fp32 working tile
    TBuf<QuePosition::VECCALC> sq_;      // fp32 square tile
    TBuf<QuePosition::VECCALC> acc_;     // running sum(x-x0)
    TBuf<QuePosition::VECCALC> acc2_;    // running sum((x-x0)^2)
    TBuf<QuePosition::VECCALC> work_;    // ReduceSum scratch
    TBuf<QuePosition::VECCALC> work2_;   // ReduceSum scratch
    TBuf<QuePosition::VECCALC> red_;     // reduce result
    TBuf<QuePosition::VECCALC> red2_;    // reduce result

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> bGm_;

    GroupNormTilingData t_;
};

template <typename T>
__aicore__ inline void GroupNorm<T>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                          const GroupNormTilingData* tilingData)
{
    t_ = *tilingData;
    const int64_t total = static_cast<int64_t>(t_.totalGroups) * t_.CpG * t_.HW;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), total);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), total);
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma), t_.G * t_.CpG);
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), t_.G * t_.CpG);

    const uint32_t ch = static_cast<uint32_t>(t_.chunk);
    pipe.InitBuffer(inQ_, 2, ch * (uint32_t)sizeof(T));
    pipe.InitBuffer(outQ_, 2, ch * (uint32_t)sizeof(T));
    pipe.InitBuffer(fx_, ch * 4);
    pipe.InitBuffer(sq_, ch * 4);
    if (static_cast<int64_t>(t_.CpG) * t_.HW > t_.chunk) {
        pipe.InitBuffer(acc_, ch * 4);
        pipe.InitBuffer(acc2_, ch * 4);
    }
    pipe.InitBuffer(work_, ch * 4);
    pipe.InitBuffer(work2_, ch * 4);
    pipe.InitBuffer(red_, 64 * 4);
    pipe.InitBuffer(red2_, 64 * 4);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::LoadTile(const LocalTensor<float>& dstF, int64_t elemOff,
                                              int64_t cnt, int64_t cntR)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * (int64_t)sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    int64_t bytes = cnt * (int64_t)sizeof(T);
    int64_t padBytes = ((bytes + 31) / 32) * 32 - bytes;
    int64_t padElems = padBytes / (int64_t)sizeof(T);
    DataCopyPadExtParams<T> pp{padElems > 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};

    LocalTensor<T> in = inQ_.AllocTensor<T>();
    DataCopyPad(in, xGm_[elemOff], p, pp);
    inQ_.EnQue(in);
    LocalTensor<T> inV = inQ_.DeQue<T>();
    if constexpr (sizeof(T) == 2) {
        Cast(dstF, inV, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        Adds(dstF, inV, 0.0f, static_cast<int32_t>(cntR));
    }
    inQ_.FreeTensor(inV);
    // DataCopyPad zero-fills only up to the next 32B boundary; lanes between that
    // boundary and the 64-element vector alignment are stale UB. Zero them (the
    // start address is 32B aligned by construction) so the pass-1 shift leaves
    // the tail at exactly 0, matching the old scalar lane patch.
    const int64_t padEnd = cnt + padElems;
    if (padEnd < cntR) {
        LocalTensor<float> tail = dstF[static_cast<uint32_t>(padEnd)];
        Duplicate(tail, 0.0f, static_cast<int32_t>(cntR - padEnd));
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::StoreTile(const LocalTensor<float>& srcF, int64_t elemOff,
                                               int64_t cnt, int64_t cntR)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * (int64_t)sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;

    LocalTensor<T> out = outQ_.AllocTensor<T>();
    if constexpr (sizeof(T) == 2) {
        Cast(out, srcF, RoundMode::CAST_RINT, static_cast<int32_t>(cntR));
    } else {
        Adds(out, srcF, 0.0f, static_cast<int32_t>(cntR));
    }
    outQ_.EnQue(out);
    LocalTensor<T> outV = outQ_.DeQue<T>();
    DataCopyPad(yGm_[elemOff], outV, p);
    outQ_.FreeTensor(outV);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ProcessGroup(int64_t gidx)
{
    const int64_t CpG = t_.CpG;
    const int64_t HW = t_.HW;
    const int64_t gsz = CpG * HW;
    const int64_t chunk = t_.chunk;
    const int64_t G = t_.G;
    const float eps = t_.epsilon;
    const int64_t base = gidx * gsz;
    const int64_t g = (G > 0) ? (gidx % G) : 0;
    const int64_t cbase = g * CpG;

    LocalTensor<float> fx = fx_.Get<float>();
    LocalTensor<float> sq = sq_.Get<float>();
    LocalTensor<float> acc;
    LocalTensor<float> acc2;
    LocalTensor<float> work = work_.Get<float>();
    LocalTensor<float> work2 = work2_.Get<float>();
    LocalTensor<float> red = red_.Get<float>();
    LocalTensor<float> red2 = red2_.Get<float>();

    // ---- pass 1: shifted sum and shifted sum of squares ----
    // A single-tile group is its own accumulator: the reduction input is the
    // (x-x0) / (x-x0)^2 tile itself, bit-identical to accumulating into a
    // zero-initialised vector accumulator.
    const bool singleTile = (gsz <= chunk);
    if (!singleTile) {
        acc = acc_.Get<float>();
        acc2 = acc2_.Get<float>();
        Duplicate(acc, 0.0f, static_cast<int32_t>(chunk));
        Duplicate(acc2, 0.0f, static_cast<int32_t>(chunk));
    }

    float x0 = 0.0f;
    bool firstTile = true;
    for (int64_t off = 0; off < gsz; off += chunk) {
        int64_t cnt = (gsz - off < chunk) ? (gsz - off) : chunk;
        int64_t cntR = ((cnt + 63) / 64) * 64;
        LoadTile(fx, base + off, cnt, cntR);
        if (firstTile) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            x0 = fx.GetValue(0);
            firstTile = false;
        }
        // subtract x0 on the real lanes only: the pad lanes are already exactly 0
        // and stay 0, so the values entering the reduction are bit-identical to
        // the old "patch the lanes to x0, then subtract" form.
        Adds(fx, fx, -x0, static_cast<int32_t>(cnt));
        Mul(sq, fx, fx, static_cast<int32_t>(cntR));
        if (!singleTile) {
            Add(acc, acc, fx, static_cast<int32_t>(cntR));
            Add(acc2, acc2, sq, static_cast<int32_t>(cntR));
        }
    }
    if (singleTile) {
        ReduceSum(red, fx, work, static_cast<int32_t>(chunk));
        ReduceSum(red2, sq, work2, static_cast<int32_t>(chunk));
    } else {
        ReduceSum(red, acc, work, static_cast<int32_t>(chunk));
        ReduceSum(red2, acc2, work2, static_cast<int32_t>(chunk));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float s1 = red.GetValue(0);
    float s2 = red2.GetValue(0);

    const float invN = 1.0f / static_cast<float>(gsz);
    const float meanShift = s1 * invN;                 // E[x - x0]
    const float mean = x0 + meanShift;
    float var = s2 * invN - meanShift * meanShift;     // E[(x-x0)^2] - E[x-x0]^2
    if (var < 0.0f) {
        var = 0.0f;
    }
    LocalTensor<float> vb = red2_.Get<float>();
    Duplicate(vb, var + eps, 64);
    Sqrt(vb, vb, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float rstd = 1.0f / vb.GetValue(0);

    // ---- pass 2: per-channel normalize + affine, store ----
    // y = (x - mean) * (rstd * gamma) + beta (reference formula, element by element)
    for (int64_t c = 0; c < CpG; ++c) {
        float gc = gGm_.GetValue(cbase + c);
        float bc = bGm_.GetValue(cbase + c);
        float a = rstd * gc;
        int64_t chBase = base + c * HW;
        for (int64_t off = 0; off < HW; off += chunk) {
            int64_t cnt = (HW - off < chunk) ? (HW - off) : chunk;
            int64_t cntR = ((cnt + 63) / 64) * 64;
            LoadTile(fx, chBase + off, cnt, cntR);
            Adds(fx, fx, -mean, static_cast<int32_t>(cntR));
            Muls(fx, fx, a, static_cast<int32_t>(cntR));
            Adds(fx, fx, bc, static_cast<int32_t>(cntR));
            StoreTile(fx, chBase + off, cnt, cntR);
        }
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t total = t_.totalGroups;

    int64_t nEach = (blockNum > 0) ? (total / blockNum) : 0;
    int64_t rem = (blockNum > 0) ? (total % blockNum) : 0;
    int64_t start = blockIdx * nEach + (blockIdx < rem ? blockIdx : rem);
    int64_t cnt = nEach + (blockIdx < rem ? 1 : 0);
    for (int64_t gi = start; gi < start + cnt; ++gi) {
        ProcessGroup(gi);
    }
}

} // namespace NsGroupNorm
#endif // GROUPNORM_H

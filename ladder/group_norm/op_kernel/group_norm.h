/*!
 * \file group_norm.h
 * \brief GroupNorm kernel (group-per-core, no cross-core reduce)
 *
 *  y[n,c,*] = (x[n,c,*] - mean_{group}) * rsqrt(var_{group}+eps) * gamma[c] + beta[c]
 *  A group (n,g) is a contiguous block of CpG*HW elements in row-major x.
 *  Each group is fully handled by one core (deterministic, no cross-core reduce).
 *  Internal FP32; fp16 / bf16 input is Cast to fp32 first; vector counts are
 *  multiples of 64.
 *
 *  The group statistics are accumulated around an offset x0 = x[first element of
 *  the group] instead of around 0: pass 1 accumulates sum(x-x0) and
 *  sum((x-x0)^2), then mean = x0 + E[x-x0] and var = E[(x-x0)^2] - E[x-x0]^2.
 *  For data with a large mean and a small spread (e.g. 1e4 + noise) the naive
 *  E[x^2] - mean^2 form cancels catastrophically in fp32 and the variance turns
 *  to garbage; the shifted form keeps full precision for that case.
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
    __aicore__ inline void LoadToFloat(const LocalTensor<float>& dstF, int64_t elemOff, int64_t cnt, int64_t cntR);
    __aicore__ inline void StoreFromFloat(const LocalTensor<float>& srcF, int64_t elemOff, int64_t cnt, int64_t cntR);
    __aicore__ inline void ProcessGroup(int64_t gidx);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> raw_;
    TBuf<QuePosition::VECCALC> fx_;
    TBuf<QuePosition::VECCALC> sq_;
    TBuf<QuePosition::VECCALC> work_;
    TBuf<QuePosition::VECCALC> work2_;
    TBuf<QuePosition::VECCALC> red_;
    TBuf<QuePosition::VECCALC> red2_;
    TBuf<QuePosition::VECCALC> oraw_;

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
    const int64_t total = t_.N * t_.C * t_.HW;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), total);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), total);
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma), t_.C);
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), t_.C);

    const int64_t ch = t_.chunk;
    pipe.InitBuffer(fx_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(sq_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(work_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(work2_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(red_, static_cast<uint32_t>(64 * 4));
    pipe.InitBuffer(red2_, static_cast<uint32_t>(64 * 4));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(raw_, static_cast<uint32_t>(ch * 2));
        pipe.InitBuffer(oraw_, static_cast<uint32_t>(ch * 2));
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::LoadToFloat(const LocalTensor<float>& dstF, int64_t elemOff,
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
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> raw = raw_.Get<T>();
        if (cnt < cntR) {
            Duplicate(raw, static_cast<T>(0), static_cast<int32_t>(cntR));
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(raw, xGm_[elemOff], p, pp);
        PipeBarrier<PIPE_ALL>();
        Cast(dstF, raw, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        if (cnt < cntR) {
            Duplicate(dstF, 0.0f, static_cast<int32_t>(cntR));
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(dstF, xGm_[elemOff], p, pp);
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::StoreFromFloat(const LocalTensor<float>& srcF, int64_t elemOff,
                                                    int64_t cnt, int64_t cntR)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * (int64_t)sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> oraw = oraw_.Get<T>();
        Cast(oraw, srcF, RoundMode::CAST_RINT, static_cast<int32_t>(cntR));
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(yGm_[elemOff], oraw, p);
    } else {
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(yGm_[elemOff], srcF, p);
    }
    PipeBarrier<PIPE_ALL>();   // MTE3 done before srcF buffer is reused (WAR)
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ProcessGroup(int64_t gidx)
{
    const int64_t CpG = t_.CpG;
    const int64_t HW = t_.HW;
    const int64_t gsz = t_.gsz;
    const int64_t chunk = t_.chunk;
    const int64_t G = t_.G;
    const float eps = t_.epsilon;
    const int64_t base = gidx * gsz;          // element offset of this group in x/y
    const int64_t g = (G > 0) ? (gidx % G) : 0;
    const int64_t cbase = g * CpG;            // global channel base of this group

    LocalTensor<float> fx = fx_.Get<float>();
    LocalTensor<float> sq = sq_.Get<float>();
    LocalTensor<float> work = work_.Get<float>();
    LocalTensor<float> work2 = work2_.Get<float>();
    LocalTensor<float> red = red_.Get<float>();
    LocalTensor<float> red2 = red2_.Get<float>();

    // ---- pass 1: shifted sum and shifted sum of squares over the whole group ----
    // x0 is the first element of the group; accumulating (x - x0) and (x - x0)^2
    // keeps the variance accurate when the group has a large mean and a small
    // spread, where E[x^2] - mean^2 would cancel in fp32. x0 is taken from the
    // first tile after the Cast, so no scalar T -> float conversion is needed
    // (bisheng rejects bf16 scalar casts) and the padded lanes are patched in
    // the float domain.
    float x0 = 0.0f;
    bool firstTile = true;
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int64_t off = 0; off < gsz; off += chunk) {
        int64_t cnt = (gsz - off < chunk) ? (gsz - off) : chunk;
        int64_t cntR = ((cnt + 63) / 64) * 64;
        LocalTensor<float> fxc = fx;
        LoadToFloat(fxc, base + off, cnt, cntR);
        PipeBarrier<PIPE_ALL>();
        if (firstTile) {
            x0 = fxc.GetValue(0);
            firstTile = false;
        }
        // lanes beyond the real element count must read as x0 so that the shift
        // makes them contribute exactly 0 to both reductions
        for (int64_t j = cnt; j < cntR; ++j) {
            fxc.SetValue(j, x0);
        }
        PipeBarrier<PIPE_ALL>(); // S -> V
        Adds(fx, fx, -x0, static_cast<int32_t>(cntR));
        Mul(sq, fx, fx, static_cast<int32_t>(cntR));
        PipeBarrier<PIPE_ALL>();
        ReduceSum(red, fx, work, static_cast<int32_t>(cntR));
        ReduceSum(red2, sq, work2, static_cast<int32_t>(cntR));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        s1 += red.GetValue(0);
        s2 += red2.GetValue(0);
        PipeBarrier<PIPE_ALL>();
    }

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
    // y = (x - mean) * (rstd * gamma) + beta, i.e. the reference formula element
    // by element. Folding it into y = x*a + b with b = beta - mean*a would cost
    // one vector op less, but b loses |mean*a| worth of precision, which ruins
    // outputs whose |mean|*rstd*|gamma| is much larger than |y| (constant or
    // large-offset groups).
    for (int64_t c = 0; c < CpG; ++c) {
        float gc = gGm_.GetValue(cbase + c);
        float bc = bGm_.GetValue(cbase + c);
        float a = rstd * gc;
        int64_t chBase = base + c * HW;
        for (int64_t off = 0; off < HW; off += chunk) {
            int64_t cnt = (HW - off < chunk) ? (HW - off) : chunk;
            int64_t cntR = ((cnt + 63) / 64) * 64;
            LoadToFloat(fx, chBase + off, cnt, cntR);
            PipeBarrier<PIPE_ALL>();
            Adds(fx, fx, -mean, static_cast<int32_t>(cntR));
            Muls(fx, fx, a, static_cast<int32_t>(cntR));
            Adds(fx, fx, bc, static_cast<int32_t>(cntR));
            StoreFromFloat(fx, chBase + off, cnt, cntR);
        }
    }
    PipeBarrier<PIPE_ALL>();
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

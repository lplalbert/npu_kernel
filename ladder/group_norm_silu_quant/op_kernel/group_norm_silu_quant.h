/*!
 * \file group_norm_silu_quant.h
 * \brief GroupNormSiluQuant kernel (group-per-core, no cross-core reduce)
 *
 *  For each group (n,g) (a contiguous block of CpG*HW elements in row-major self):
 *    mean = E[x],  var = E[(x-mean)^2],  rstd = 1/sqrt(var+eps)
 *    gn   = ((x - mean) * rstd * gamma[c]) + beta[c]     (per-channel affine)
 *    silu = gn / (1 + exp(-gn))
 *    out  = clamp(rint(silu / quantScale[c]), -128, 127) -> int8
 *
 *  The golden reference (problem spec 3.1) is the plain numpy implementation, whose
 *  `ndarray.mean/var` over the trailing axes of a C-contiguous array reduce with numpy's
 *  flat "Parker" pairwise summation (verified bit-exactly against numpy 1.26: PW_BLOCKSIZE
 *  128, 8 interleaved accumulators per block, `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` combine,
 *  `n2 = n/2 - (n/2)%8` split).  int8 output must match bit-exactly, so the group statistics
 *  are replicated with exactly that summation order (two-pass variance, mean is `sum / n`).
 *
 *  Arithmetic order per element follows the golden exactly, one rounding per operation:
 *    (x - mean) * rstd * gamma + beta,  1 + exp(-gn),  gn / (...)  ,  silu / quantScale
 *  and `rint` is round-half-even (the 1.5*2^23 add/sub trick).
 */

#ifndef GROUPNORMSILUQUANT_H
#define GROUPNORMSILUQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_silu_quant_tiling_data.h"
#include "group_norm_silu_quant_tiling_key.h"

namespace NsGroupNormSiluQuant {

using namespace AscendC;

// maximum length handled by one pairwise-sum leaf (numpy PW_BLOCKSIZE)
constexpr int64_t kPwBlock = 128;
// max recursion depth of the pairwise split (n <= 2^40 elements is far beyond device memory)
constexpr int kPwMaxDepth = 64;
// numpy's reduce internal buffer size (NPY_BUFSIZE); blocks of this many elements are
// summed independently and accumulated sequentially.
constexpr int64_t kNpyBufSize = 8192;

template <typename T>
class GroupNormSiluQuant {
public:
    __aicore__ inline GroupNormSiluQuant(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale,
                                GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
                                const GroupNormSiluQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadToFloat(const LocalTensor<float>& dstF, int64_t elemOff,
                                       int64_t cnt, int64_t cntR, T padT);
    __aicore__ inline void StoreInt8(const LocalTensor<int8_t>& srcI8, int64_t elemOff, int64_t cnt);
    __aicore__ inline void StoreStat(GlobalTensor<T>& dst, int64_t idx, float val);
    __aicore__ inline void ProcessGroup(int64_t gidx);
    __aicore__ inline float PwLeaf(int64_t off, int64_t len, bool centered, float mean);
    __aicore__ inline float PwSum(int64_t off, int64_t len, bool centered, float mean);
    __aicore__ inline float PwSumChunked(int64_t off, int64_t n, bool centered, float mean);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> raw_;   // fp16 staging (T)
    TBuf<QuePosition::VECCALC> fx_;    // value / result (float)
    TBuf<QuePosition::VECCALC> ta_;    // scratch (float, or int32 in quant stage)
    TBuf<QuePosition::VECCALC> tb_;    // scratch (float)
    TBuf<QuePosition::VECCALC> tc_;    // scratch (float)
    TBuf<QuePosition::VECCALC> oi8_;   // int8 staging
    TBuf<QuePosition::VECCALC> red_;   // reduce result (64 floats)
    TBuf<QuePosition::VECCALC> red2_;  // reduce temp / result (64 floats)
    TBuf<QuePosition::VECCALC> leafF_; // pairwise leaf staging (float)
    TBuf<QuePosition::VECCALC> leafR_; // pairwise leaf staging (T)

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gGm_;
    GlobalTensor<T> bGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> qsGm_;
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> rstdGm_;

    GroupNormSiluQuantTilingData t_;
};

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
                                                   GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut,
                                                   GM_ADDR rstdOut,
                                                   const GroupNormSiluQuantTilingData* tilingData)
{
    t_ = *tilingData;
    const int64_t total = t_.N * t_.C * t_.HW;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), total);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out), total);
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), t_.C);
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), t_.C);
    qsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale), t_.qsPerChannel ? t_.C : 1);
    meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(meanOut), t_.totalGroups);
    rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstdOut), t_.totalGroups);

    const int64_t ch = t_.chunk;
    pipe.InitBuffer(fx_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(ta_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(tb_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(tc_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(oi8_, static_cast<uint32_t>(((ch + 31) / 32) * 32));
    pipe.InitBuffer(red_, static_cast<uint32_t>(64 * 4));
    pipe.InitBuffer(red2_, static_cast<uint32_t>(64 * 4));
    pipe.InitBuffer(leafF_, static_cast<uint32_t>(kPwBlock * 4));
    pipe.InitBuffer(leafR_, static_cast<uint32_t>(kPwBlock * sizeof(T)));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(raw_, static_cast<uint32_t>(((ch + 31) / 32) * 32 * 2));
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::LoadToFloat(const LocalTensor<float>& dstF, int64_t elemOff,
                                                          int64_t cnt, int64_t cntR, T padT)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
    int64_t padBytes = ((bytes + 31) / 32) * 32 - bytes;
    int64_t padElems = padBytes / static_cast<int64_t>(sizeof(T));
    DataCopyPadExtParams<T> pp{padElems > 0, 0, static_cast<uint8_t>(padElems), padT};
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> raw = raw_.Get<T>();
        if (cnt < cntR) {
            Duplicate(raw, padT, static_cast<int32_t>(cntR));
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(raw, xGm_[elemOff], p, pp);
        PipeBarrier<PIPE_ALL>();
        Cast(dstF, raw, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        if (cnt < cntR) {
            Duplicate(dstF, static_cast<float>(padT), static_cast<int32_t>(cntR));
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(dstF, xGm_[elemOff], p, pp);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::StoreInt8(const LocalTensor<int8_t>& srcI8, int64_t elemOff,
                                                        int64_t cnt)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPad(yGm_[elemOff], srcI8, p);
    PipeBarrier<PIPE_ALL>();   // MTE3 done before the staging buffer is reused (WAR)
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::StoreStat(GlobalTensor<T>& dst, int64_t idx, float val)
{
    // GM scalar stores via GlobalTensor::SetValue are not flushed reliably on this arch;
    // stage through UB and use DataCopyPad instead.
    LocalTensor<float> sb = red_.Get<float>();
    Duplicate(sb, val, 64);
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> hb = raw_.Get<T>();
        Cast(hb, sb, RoundMode::CAST_RINT, 64);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[idx], hb, p);
    } else {
        DataCopyPad(dst[idx], sb, p);
    }
    PipeBarrier<PIPE_ALL>();
}

// ---------------------------------------------------------------------------
// numpy pairwise summation (exact replication of npy_pairwise_sum for float32)
// ---------------------------------------------------------------------------

// One leaf of the pairwise tree: 8 <= len <= kPwBlock follows numpy's unrolled
// 8-accumulator block, len < 8 follows numpy's plain sequential loop.
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::PwLeaf(int64_t off, int64_t len, bool centered, float mean)
{
    LocalTensor<float> v = leafF_.Get<float>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> raw = leafR_.Get<T>();
        DataCopyPad(raw, xGm_[off], p, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
        PipeBarrier<PIPE_ALL>();
        int64_t cntR = ((len + 63) / 64) * 64;
        Cast(v, raw, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        DataCopyPad(v, xGm_[off], p, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
    }
    PipeBarrier<PIPE_ALL>();

    if (len < 8) {
        float res = 0.0f;
        for (int64_t i = 0; i < len; ++i) {
            float x = v.GetValue(i);
            if (centered) {
                float d = x - mean;
                res += d * d;
            } else {
                res += x;
            }
        }
        return res;
    }
    float r[8];
    for (int64_t j = 0; j < 8; ++j) {
        float x = v.GetValue(j);
        if (centered) {
            float d = x - mean;
            r[j] = d * d;
        } else {
            r[j] = x;
        }
    }
    int64_t i = 8;
    const int64_t lim = len - (len % 8);
    for (; i < lim; i += 8) {
        for (int64_t j = 0; j < 8; ++j) {
            float x = v.GetValue(i + j);
            if (centered) {
                float d = x - mean;
                r[j] += d * d;
            } else {
                r[j] += x;
            }
        }
    }
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < len; ++i) {
        float x = v.GetValue(i);
        if (centered) {
            float d = x - mean;
            res += d * d;
        } else {
            res += x;
        }
    }
    return res;
}

// Iterative post-order walk of numpy's pairwise recursion:
//   n < 8        -> sequential
//   n <= 128     -> PwLeaf
//   else         -> PwSum(n2) + PwSum(n-n2),  n2 = n/2 - (n/2)%8
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::PwSum(int64_t off, int64_t len, bool centered, float mean)
{
    if (len <= kPwBlock) {
        return PwLeaf(off, len, centered, mean);
    }
    struct Frame {
        int64_t off;
        int64_t len;
        int32_t stage;   // 0: push left, 1: left pushed, 2: left done, 3: right pushed
        float left;
    };
    Frame st[kPwMaxDepth];
    int sp = 0;
    st[sp].off = off;
    st[sp].len = len;
    st[sp].stage = 0;
    st[sp].left = 0.0f;
    ++sp;

    float pending = 0.0f;
    bool have = false;
    while (sp > 0) {
        Frame& f = st[sp - 1];
        if (f.len <= kPwBlock) {
            pending = PwLeaf(f.off, f.len, centered, mean);
            have = true;
            --sp;
            continue;
        }
        if (have) {
            if (f.stage == 1) {
                f.left = pending;
                f.stage = 2;
                have = false;
                continue;
            }
            // f.stage == 3: right child returned
            pending = f.left + pending;
            have = true;
            --sp;
            continue;
        }
        int64_t n2 = f.len / 2;
        n2 -= n2 % 8;
        if (f.stage == 0) {
            f.stage = 1;
            st[sp].off = f.off;
            st[sp].len = n2;
            st[sp].stage = 0;
            st[sp].left = 0.0f;
            ++sp;
        } else if (f.stage == 2) {
            f.stage = 3;
            st[sp].off = f.off + n2;
            st[sp].len = f.len - n2;
            st[sp].stage = 0;
            st[sp].left = 0.0f;
            ++sp;
        } else {
            // unreachable (guards against a corrupted stack)
            --sp;
        }
    }
    return pending;
}

// numpy's reduce buffers the iteration in blocks of NPY_BUFSIZE (8192) elements:
// each buffered block is summed with the pairwise scheme above and the per-block
// results are accumulated sequentially into the fp32 output.  Verified bit-exact
// against numpy 1.26 for n in {105, 8191, 8192, 8193, 16385, 32768, 131072, ...}.
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::PwSumChunked(int64_t off, int64_t n, bool centered, float mean)
{
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i += kNpyBufSize) {
        int64_t len = (n - i < kNpyBufSize) ? (n - i) : kNpyBufSize;
        acc = acc + PwSum(off + i, len, centered, mean);
    }
    return acc;
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::ProcessGroup(int64_t gidx)
{
    const int64_t CpG = t_.CpG;
    const int64_t HW = t_.HW;
    const int64_t gsz = t_.gsz;
    const int64_t chunk = t_.chunk;
    const int64_t G = t_.G;
    const float eps = t_.epsilon;
    const bool doSilu = (t_.activateSilu != 0);
    const int64_t base = gidx * gsz;
    const int64_t g = (G > 0) ? (gidx % G) : 0;
    const int64_t cbase = g * CpG;

    LocalTensor<float> fx = fx_.Get<float>();
    LocalTensor<float> ta = ta_.Get<float>();
    LocalTensor<float> tb = tb_.Get<float>();
    LocalTensor<float> tc = tc_.Get<float>();
    LocalTensor<float> red = red_.Get<float>();
    LocalTensor<float> red2 = red2_.Get<float>();

    // ---- pass 1: mean = numpy_pairwise_sum(x) / n   (numpy: umr_sum / rcount, fp32 divide) ----
    const float sum = PwSumChunked(base, gsz, false, 0.0f);
    Duplicate(red, sum, 64);
    Duplicate(red2, static_cast<float>(gsz), 64);
    PipeBarrier<PIPE_ALL>();
    Div(red, red, red2, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float mean = red.GetValue(0);
    PipeBarrier<PIPE_ALL>();

    // ---- pass 2: var = numpy_pairwise_sum((x-mean)^2) / n  (numpy var: two-pass) ----
    const float sumsq = PwSumChunked(base, gsz, true, mean);
    Duplicate(red, sumsq, 64);
    Duplicate(red2, static_cast<float>(gsz), 64);
    PipeBarrier<PIPE_ALL>();
    Div(red, red, red2, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float var = red.GetValue(0);
    PipeBarrier<PIPE_ALL>();

    // rstd = 1.0 / np.sqrt(var + eps)
    Duplicate(red2, var + eps, 64);
    PipeBarrier<PIPE_ALL>();
    Sqrt(red2, red2, 64);
    Duplicate(red, 1.0f, 64);
    PipeBarrier<PIPE_ALL>();
    Div(red2, red, red2, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float rstd = red2.GetValue(0);
    PipeBarrier<PIPE_ALL>();

    // store meanOut / rstdOut (dtype T)
    StoreStat(meanGm_, gidx, mean);
    StoreStat(rstdGm_, gidx, rstd);

    // ---- pass 3: per-channel normalize + SiLU + quant, store int8 ----
    for (int64_t c = 0; c < CpG; ++c) {
        const int64_t gc = cbase + c;
        float gcv = static_cast<float>(gGm_.GetValue(gc));
        float bcv = static_cast<float>(bGm_.GetValue(gc));
        float qsv = qsGm_.GetValue(t_.qsPerChannel ? gc : 0);
        int64_t chBase = base + c * HW;
        LocalTensor<float> qsb = tb_.Get<float>();
        Duplicate(qsb, qsv, static_cast<int32_t>(chunk));
        PipeBarrier<PIPE_ALL>();
        for (int64_t off = 0; off < HW; off += chunk) {
            int64_t cnt = (HW - off < chunk) ? (HW - off) : chunk;
            int64_t cntR = ((cnt + 63) / 64) * 64;
            int32_t n = static_cast<int32_t>(cntR);
            LoadToFloat(fx, chBase + off, cnt, cntR, static_cast<T>(0));
            PipeBarrier<PIPE_ALL>();
            // gn = ((x - mean) * rstd * gamma) + beta  (golden op order)
            Adds(fx, fx, -mean, n);
            Muls(fx, fx, rstd, n);
            Muls(fx, fx, gcv, n);
            Adds(fx, fx, bcv, n);
            if (doSilu) {
                // silu = fx / (1 + exp(-fx))
                Muls(ta, fx, -1.0f, n);
                PipeBarrier<PIPE_ALL>();
                Exp(ta, ta, n);
                PipeBarrier<PIPE_ALL>();
                Adds(ta, ta, 1.0f, n);
                PipeBarrier<PIPE_ALL>();
                Div(fx, fx, ta, n);
            }
            // quant: clamp to int8 range, then round-to-nearest-even in fp32 via the
            // 1.5*2^23 add/sub trick (ulp == 1 there), matching numpy rint exactly.
            Div(fx, fx, qsb, n);
            Maxs(fx, fx, -128.0f, n);
            Mins(fx, fx, 127.0f, n);
            Adds(fx, fx, 12582912.0f, n);
            Adds(fx, fx, -12582912.0f, n);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<half> thi = tc_.Get<half>();
            Cast(thi, fx, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<int8_t> i8 = oi8_.Get<int8_t>();
            Cast(i8, thi, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_ALL>();
            StoreInt8(i8, chBase + off, cnt);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Process()
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

} // namespace NsGroupNormSiluQuant
#endif // GROUPNORMSILUQUANT_H

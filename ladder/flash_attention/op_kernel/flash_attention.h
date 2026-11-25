/*!
 * \file flash_attention.h
 * \brief FlashAttention kernel (fp32 internal, multi-core over query rows)
 *
 *  out[b,n,i,:] = softmax_j( scale * q[b,n,i,:] . k[b,n,j,:] + mask(i,j) ) @ v[b,n,:,:]
 *  Causal mask adds -1e4 to j>i (matches golden NEG_MASK).  attn_mask additive.
 *  Per query row: full scores row in UB, stable softmax (subtract row max), then
 *  tiled V accumulation.  Rows are independent -> trivially multi-core.
 */

#ifndef FLASHATTENTION_H
#define FLASHATTENTION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "flash_attention_tiling_data.h"
#include "flash_attention_tiling_key.h"

namespace NsFlashAttention {

using namespace AscendC;

constexpr float NEG_MASK = -1e4f;
constexpr int32_t MAXD = 256;

template <typename T>
class FlashAttention {
public:
    __aicore__ inline FlashAttention(){};

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR out,
                                const FlashAttentionTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeRow(int64_t r);
    __aicore__ inline void LoadCastK(LocalTensor<float>& dstF, int64_t gmOff, int64_t cnt, bool useV);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> qF_;
    TBuf<QuePosition::VECCALC> qraw_;
    TBuf<QuePosition::VECCALC> kf_;
    TBuf<QuePosition::VECCALC> kraw_;
    TBuf<QuePosition::VECCALC> sbuf_;
    TBuf<QuePosition::VECCALC> acc_;
    TBuf<QuePosition::VECCALC> tmp_;
    TBuf<QuePosition::VECCALC> work_;
    TBuf<QuePosition::VECCALC> red_;
    TBuf<QuePosition::VECCALC> mbuf_;
    TBuf<QuePosition::VECCALC> mraw_;
    TBuf<QuePosition::VECCALC> cau_;
    TBuf<QuePosition::VECCALC> oraw_;

    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> mGm_;
    GlobalTensor<T> outGm_;

    FlashAttentionTilingData t_;
};

template <typename T>
__aicore__ inline void FlashAttention<T>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR out,
                                               const FlashAttentionTilingData* tilingData)
{
    t_ = *tilingData;
    const int64_t D = t_.D;
    const int64_t S = t_.S;
    const int64_t BK = t_.keyBlock;
    const int64_t qkvNum = t_.B * t_.N * S * D;
    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q), qkvNum);
    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k), qkvNum);
    vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v), qkvNum);
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), qkvNum);
    if (t_.hasMask != 0) {
        mGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attn_mask), t_.B * S * S);
    }

    pipe.InitBuffer(qF_, static_cast<uint32_t>(D * 4));
    pipe.InitBuffer(kf_, static_cast<uint32_t>(BK * D * 4));
    pipe.InitBuffer(sbuf_, static_cast<uint32_t>(S * 4));
    pipe.InitBuffer(acc_, static_cast<uint32_t>(D * 4));
    pipe.InitBuffer(tmp_, static_cast<uint32_t>(D * 4));
    int64_t workN = (D > S) ? D : S;
    pipe.InitBuffer(work_, static_cast<uint32_t>(workN * 4));
    pipe.InitBuffer(red_, static_cast<uint32_t>(64 * 4));
    pipe.InitBuffer(oraw_, static_cast<uint32_t>(D * sizeof(T)));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(qraw_, static_cast<uint32_t>(D * 2));
        pipe.InitBuffer(kraw_, static_cast<uint32_t>(BK * D * 2));
    }
    if (t_.hasMask != 0) {
        pipe.InitBuffer(mbuf_, static_cast<uint32_t>(S * 4));
        if constexpr (sizeof(T) == 2) {
            pipe.InitBuffer(mraw_, static_cast<uint32_t>(S * 2));
        }
    }
    if (t_.isCausal != 0) {
        pipe.InitBuffer(cau_, static_cast<uint32_t>(S * 4));
    }
}

// Load cnt elements of T from GM at gmOff, produce float in dstF.
template <typename T>
__aicore__ inline void FlashAttention<T>::LoadCastK(LocalTensor<float>& dstF, int64_t gmOff, int64_t cnt, bool useV)
{
    GlobalTensor<T>& g = useV ? vGm_ : kGm_;
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * (int64_t)sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> raw = kraw_.Get<T>();
        DataCopyPad(raw, g[gmOff], p, pp);
        PipeBarrier<PIPE_ALL>();
        Cast(dstF, raw, RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
    } else {
        DataCopyPad(dstF, g[gmOff], p, pp);
    }
}

template <typename T>
__aicore__ inline void FlashAttention<T>::ComputeRow(int64_t r)
{
    const int64_t B = t_.B;
    const int64_t N = t_.N;
    const int64_t S = t_.S;
    const int64_t D = t_.D;
    const int64_t BK = t_.keyBlock;
    const bool isCausal = (t_.isCausal != 0);
    const bool hasMask = (t_.hasMask != 0);
    const float scale = t_.softmaxScale;

    const int64_t head = r / S;
    const int64_t i = r - head * S;
    const int64_t b = head / N;
    const int64_t qkvBase = head * S * D;

    LocalTensor<float> qF = qF_.Get<float>();
    LocalTensor<float> kf = kf_.Get<float>();
    LocalTensor<float> sbuf = sbuf_.Get<float>();
    LocalTensor<float> acc = acc_.Get<float>();
    LocalTensor<float> tmp = tmp_.Get<float>();
    LocalTensor<float> work = work_.Get<float>();
    LocalTensor<float> red = red_.Get<float>();

    // ---- load q row ----
    DataCopyExtParams pl;
    pl.blockCount = 1; pl.srcStride = 0; pl.dstStride = 0; pl.rsv = 0;
    DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> qraw = qraw_.Get<T>();
        pl.blockLen = static_cast<uint32_t>(D * 2);
        DataCopyPad(qraw, qGm_[qkvBase + i * D], pl, pp);
        PipeBarrier<PIPE_ALL>();
        Cast(qF, qraw, RoundMode::CAST_NONE, static_cast<int32_t>(D));
    } else {
        pl.blockLen = static_cast<uint32_t>(D * 4);
        DataCopyPad(qF, qGm_[qkvBase + i * D], pl, pp);
    }

    // ---- load mask row (optional) ----
    LocalTensor<float> mbuf;
    if (hasMask) {
        mbuf = mbuf_.Get<float>();
        int64_t mOff = b * S * S + i * S;
        DataCopyExtParams pm;
        pm.blockCount = 1; pm.srcStride = 0; pm.dstStride = 0; pm.rsv = 0;
        if constexpr (sizeof(T) == 2) {
            LocalTensor<T> mraw = mraw_.Get<T>();
            pm.blockLen = static_cast<uint32_t>(S * 2);
            DataCopyPad(mraw, mGm_[mOff], pm, pp);
            PipeBarrier<PIPE_ALL>();
            Cast(mbuf, mraw, RoundMode::CAST_NONE, static_cast<int32_t>(S));
        } else {
            pm.blockLen = static_cast<uint32_t>(S * 4);
            DataCopyPad(mbuf, mGm_[mOff], pm, pp);
        }
    }
    PipeBarrier<PIPE_ALL>();

    // ---- scores (batched: ReduceSum writes sbuf[j] directly, no scalar round-trip) ----
    for (int64_t j0 = 0; j0 < S; j0 += BK) {
        int64_t bk = (S - j0 < BK) ? (S - j0) : BK;
        LoadCastK(kf, qkvBase + j0 * D, bk * D, false);
        PipeBarrier<PIPE_ALL>();
        for (int64_t jj = 0; jj < bk; ++jj) {
            Mul(tmp, qF, kf[jj * D], static_cast<int32_t>(D));
            ReduceSum(sbuf[j0 + jj], tmp, work, static_cast<int32_t>(D));
        }
        PipeBarrier<PIPE_ALL>();
    }
    // post: scale, additive mask, causal (-1e4 on j>i)
    Muls(sbuf, sbuf, scale, static_cast<int32_t>(S));
    if (hasMask) {
        Add(sbuf, sbuf, mbuf, static_cast<int32_t>(S));
    }
    if (isCausal) {
        LocalTensor<float> cau = cau_.Get<float>();
        int64_t nz = S - 1 - i;   // #masked positions (j>i)
        if (nz <= 0) {
            Duplicate(cau, 0.0f, static_cast<int32_t>(S));
        } else {
            Duplicate(cau, NEG_MASK, static_cast<int32_t>(S));
            if (i == 0) {
                cau.SetValue(0, 0.0f);
            } else {
                Duplicate(cau, 0.0f, static_cast<int32_t>(i + 1));  // zero prefix [0..i]
            }
        }
        Add(sbuf, sbuf, cau, static_cast<int32_t>(S));
    }

    // ---- stable softmax ----
    ReduceMax(red, sbuf, work, static_cast<int32_t>(S));
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float m = red.GetValue(0);
    Adds(sbuf, sbuf, -m, static_cast<int32_t>(S));
    Exp(sbuf, sbuf, static_cast<int32_t>(S));
    ReduceSum(red, sbuf, work, static_cast<int32_t>(S));
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float l = red.GetValue(0);
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;

    // ---- value accumulation ----
    Duplicate(acc, 0.0f, static_cast<int32_t>(D));
    for (int64_t j0 = 0; j0 < S; j0 += BK) {
        int64_t bk = (S - j0 < BK) ? (S - j0) : BK;
        // load V block into kf
        LoadCastK(kf, qkvBase + j0 * D, bk * D, true);
        PipeBarrier<PIPE_ALL>();
        for (int64_t jj = 0; jj < bk; ++jj) {
            float p = sbuf.GetValue(j0 + jj);
            Muls(tmp, kf[jj * D], p, static_cast<int32_t>(D));
            Add(acc, acc, tmp, static_cast<int32_t>(D));
        }
        PipeBarrier<PIPE_ALL>();
    }
    Muls(acc, acc, inv, static_cast<int32_t>(D));
    PipeBarrier<PIPE_ALL>();

    // ---- store ----
    DataCopyExtParams ps;
    ps.blockCount = 1; ps.srcStride = 0; ps.dstStride = 0; ps.rsv = 0;
    ps.blockLen = static_cast<uint32_t>(D * sizeof(T));
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> oraw = oraw_.Get<T>();
        Cast(oraw, acc, RoundMode::CAST_RINT, static_cast<int32_t>(D));
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(outGm_[qkvBase + i * D], oraw, ps);
    } else {
        DataCopyPad(outGm_[qkvBase + i * D], acc, ps);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void FlashAttention<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t total = t_.totalRows;

    int64_t base = (blockNum > 0) ? (total / blockNum) : 0;
    int64_t rem = (blockNum > 0) ? (total % blockNum) : 0;
    int64_t start = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
    int64_t cnt = base + (blockIdx < rem ? 1 : 0);
    for (int64_t r = start; r < start + cnt; ++r) {
        ComputeRow(r);
    }
}

} // namespace NsFlashAttention
#endif // FLASHATTENTION_H


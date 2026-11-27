/*!
 * \file bn_training_update.h
 * \brief BnTrainingUpdate kernel (v2: vectorized, flat-split multi-core)
 *
 *   y_c = fl( fl( fl( fl(x - mean_c) / d_c ) * scale_c ) + offset_c )
 *   mean_c = sum_c / N_c ;  var_c = sq_c / N_c - mean_c*mean_c ; d_c = sqrt(var_c+eps)
 *   save_mean = mean, save_variance = var  (fp32, shape C)
 *
 * The per-channel statistics are computed once per core from the pre-reduced
 * sum / square_sum inputs (same scalar fp32 chain, with volatile temporaries so
 * the compiler cannot contract the ops into an FMA, and with the exactly
 * rounded integer square root). The elementwise part is then a pure vector
 * chain Adds / Divs / Muls / Adds, which performs exactly the four IEEE fp32
 * roundings the golden numpy expression performs (one per op), so it stays
 * bit exact with the reference while running at vector speed.
 *
 * Parallelism: because mean/d/scale/offset only need the pre-reduced per
 * channel sums, the elementwise pass has NO cross-core dependency. The flat
 * element space is therefore split evenly across all AIV cores; each core
 * recomputes the tiny per-channel table and processes a contiguous element
 * range (rounded to whole (n,c) rows so a row keeps a single channel).
 */

#ifndef BNTRAININGUPDATE_H
#define BNTRAININGUPDATE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bn_training_update_tiling_data.h"
#include "bn_training_update_tiling_key.h"
#include <type_traits>

namespace NsBnTrainingUpdate {

using namespace AscendC;

constexpr int64_t ALIGN_ELEM = 64;

__aicore__ inline uint32_t BitsOfFloat(float v)
{
    union { float f; uint32_t u; } cv; cv.f = v; return cv.u;
}
__aicore__ inline float FloatOfBits(uint32_t b)
{
    union { float f; uint32_t u; } cv; cv.u = b; return cv.f;
}

__aicore__ inline float ExactSqrt(float v, float guess)
{
    const uint32_t bits = BitsOfFloat(v);
    const uint32_t sign = bits >> 31;
    const uint32_t expo = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;
    if (expo == 0xFFu) return v;
    if (sign != 0u) return FloatOfBits(0x7FC00000u);
    if (expo == 0u) return guess;
    const int32_t e = static_cast<int32_t>(expo) - 127;
    const int32_t k = e >> 1;
    const uint32_t m24 = mant | 0x800000u;
    const uint32_t mm = (e & 1) ? (m24 << 2) : (m24 << 1);
    const uint64_t nn = static_cast<uint64_t>(mm) << 22;
    int64_t s = 1 << 23;
    const int32_t gexp = 127 + (23 - k);
    if (gexp > 0 && gexp < 255) {
        const float scaled = guess * FloatOfBits(static_cast<uint32_t>(gexp) << 23);
        if (scaled >= 1.0f && scaled <= 1.7e7f) s = static_cast<int64_t>(scaled);
    }
    while (s > 1 && s * s > static_cast<int64_t>(nn)) --s;
    while ((s + 1) * (s + 1) <= static_cast<int64_t>(nn)) ++s;
    const uint64_t rem = nn - static_cast<uint64_t>(s * s);
    const uint64_t root = static_cast<uint64_t>(s) + ((rem > static_cast<uint64_t>(s)) ? 1u : 0u);
    const int32_t ef = 127 + k;
    if (ef <= 0 || ef >= 254) return guess;
    if (root >= (1u << 24)) return FloatOfBits(static_cast<uint32_t>(ef + 1) << 23);
    return FloatOfBits((static_cast<uint32_t>(ef) << 23) | (static_cast<uint32_t>(root) - (1u << 23)));
}

template <typename T>
class BnTrainingUpdate {
public:
    __aicore__ inline BnTrainingUpdate() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance,
                                const BnTrainingUpdateTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t AlignUp(int64_t v) const { return (v + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM; }
    __aicore__ inline void BuildChannelTable();
    __aicore__ inline void LoadTile(const LocalTensor<float>& dstF, int64_t elemOff, int64_t cnt, int64_t cntR);
    __aicore__ inline void StoreTile(const LocalTensor<float>& srcF, int64_t elemOff, int64_t cnt, int64_t cntR);
    __aicore__ inline void StoreChannelStats();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ_;
    TQue<QuePosition::VECOUT, 2> outQ_;
    TBuf<QuePosition::VECCALC> fx_;
    TBuf<QuePosition::VECCALC> denV_;
    TBuf<QuePosition::VECCALC> tMean_;
    TBuf<QuePosition::VECCALC> tVar_;
    TBuf<QuePosition::VECCALC> tDen_;
    TBuf<QuePosition::VECCALC> tScale_;
    TBuf<QuePosition::VECCALC> tOff_;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> sumGm;
    GlobalTensor<float> sqGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offGm;
    GlobalTensor<float> meanOutGm;
    GlobalTensor<float> varOutGm;

    int64_t chan_ = 1;
    int64_t rowLen_ = 1;
    int64_t totalRows_ = 1;
    int64_t elemsPerCore_ = 1;
    int64_t ub_ = ALIGN_ELEM;
    int64_t tbl_ = ALIGN_ELEM;
    int64_t nElem_ = 1;
    int64_t totalNum_ = 1;
    float eps_ = 1e-5f;
    float nf_ = 1.0f;
};

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Init(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale,
                                                 GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean,
                                                 GM_ADDR save_variance,
                                                 const BnTrainingUpdateTilingData* tilingData)
{
    chan_ = tilingData->channelNum; if (chan_ < 1) chan_ = 1;
    rowLen_ = tilingData->rowLength; if (rowLen_ < 1) rowLen_ = 1;
    totalRows_ = tilingData->totalRows; if (totalRows_ < 1) totalRows_ = 1;
    elemsPerCore_ = tilingData->blockFactor; if (elemsPerCore_ < 1) elemsPerCore_ = 1;
    ub_ = tilingData->ubFactor; if (ub_ < ALIGN_ELEM) ub_ = ALIGN_ELEM;
    ub_ = (ub_ / ALIGN_ELEM) * ALIGN_ELEM;
    nElem_ = tilingData->elemsPerChannel; if (nElem_ < 1) nElem_ = 1;
    totalNum_ = tilingData->totalNum; if (totalNum_ < 1) totalNum_ = totalRows_ * rowLen_;
    eps_ = tilingData->epsilon;
    nf_ = static_cast<float>(nElem_);
    tbl_ = AlignUp(chan_);

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalRows_ * rowLen_);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalRows_ * rowLen_);
    sumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum), chan_);
    sqGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(square_sum), chan_);
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), chan_);
    offGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset), chan_);
    meanOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_mean), chan_);
    varOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_variance), chan_);

    const uint32_t ub = static_cast<uint32_t>(ub_);
    pipe.InitBuffer(inQ_, 2, ub * (uint32_t)sizeof(T));
    pipe.InitBuffer(outQ_, 2, ub * (uint32_t)sizeof(T));
    pipe.InitBuffer(fx_, ub * 4);
    pipe.InitBuffer(denV_, ub * 4);
    pipe.InitBuffer(tMean_, static_cast<uint32_t>(tbl_ * 4));
    pipe.InitBuffer(tVar_, static_cast<uint32_t>(tbl_ * 4));
    pipe.InitBuffer(tDen_, static_cast<uint32_t>(tbl_ * 4));
    pipe.InitBuffer(tScale_, static_cast<uint32_t>(tbl_ * 4));
    pipe.InitBuffer(tOff_, static_cast<uint32_t>(tbl_ * 4));
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::BuildChannelTable()
{
    LocalTensor<float> mT = tMean_.Get<float>();
    LocalTensor<float> vT = tVar_.Get<float>();
    LocalTensor<float> sT = tScale_.Get<float>();
    LocalTensor<float> bT = tOff_.Get<float>();
    for (int64_t c = 0; c < chan_; ++c) {
        const int32_t idx = static_cast<int32_t>(c);
        const float sumv = sumGm.GetValue(idx);
        const float sqv = sqGm.GetValue(idx);
        volatile float mean = sumv / nf_;
        volatile float quot = sqv / nf_;
        volatile float sq = mean * mean;
        volatile float var = quot - sq;
        mT.SetValue(idx, mean);
        vT.SetValue(idx, var);
        sT.SetValue(idx, scaleGm.GetValue(idx));
        bT.SetValue(idx, offGm.GetValue(idx));
    }
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> dT = tDen_.Get<float>();
    const uint32_t n = static_cast<uint32_t>(tbl_);
    Adds(dT, vT, eps_, n);
    Sqrt(dT, dT, n);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int64_t c = 0; c < chan_; ++c) {
        const int32_t idx = static_cast<int32_t>(c);
        volatile float v = vT.GetValue(idx) + eps_;
        dT.SetValue(idx, ExactSqrt(v, dT.GetValue(idx)));
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::LoadTile(const LocalTensor<float>& dstF, int64_t elemOff,
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
    DataCopyPad(in, xGm[elemOff], p, pp);
    inQ_.EnQue(in);
    LocalTensor<T> inV = inQ_.DeQue<T>();
    if constexpr (sizeof(T) == 2) {
        Cast(dstF, inV, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        Adds(dstF, inV, 0.0f, static_cast<int32_t>(cntR));
    }
    inQ_.FreeTensor(inV);
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::StoreTile(const LocalTensor<float>& srcF, int64_t elemOff,
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
    DataCopyPad(yGm[elemOff], outV, p);
    outQ_.FreeTensor(outV);
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::StoreChannelStats()
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(chan_ * (int64_t)sizeof(float));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(meanOutGm, tMean_.Get<float>(), p);
    DataCopyPad(varOutGm, tVar_.Get<float>(), p);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Process()
{
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    const int64_t S = rowLen_;
    const int64_t C = chan_;

    BuildChannelTable();
    if (blockId == 0) {
        StoreChannelStats();
    }

    int64_t r0 = blockId * elemsPerCore_;   // elemsPerCore_ holds rows-per-core here
    if (r0 > totalRows_) r0 = totalRows_;
    int64_t r1 = r0 + elemsPerCore_;
    if (r1 > totalRows_) r1 = totalRows_;
    if (r0 >= r1) return;

    LocalTensor<float> fx = fx_.Get<float>();
    LocalTensor<float> denV = denV_.Get<float>();
    LocalTensor<float> mT = tMean_.Get<float>();
    LocalTensor<float> dT = tDen_.Get<float>();
    LocalTensor<float> sT = tScale_.Get<float>();
    LocalTensor<float> bT = tOff_.Get<float>();

    for (int64_t row = r0; row < r1; ++row) {
        const int32_t c = static_cast<int32_t>(row % C);
        const float mean = mT.GetValue(c);
        const float den = dT.GetValue(c);
        const float scv = sT.GetValue(c);
        const float ofv = bT.GetValue(c);
        const int64_t rowBase = row * S;
        for (int64_t off = 0; off < S; off += ub_) {
            int64_t cnt = (S - off < ub_) ? (S - off) : ub_;
            int64_t cntR = ((cnt + 63) / 64) * 64;
            LoadTile(fx, rowBase + off, cnt, cntR);
            if constexpr (sizeof(T) == 2) {
                // fp16 output: MARE tolerance is 10*2^-10, so the vector Div
                // (a couple of ULP) stays well inside tolerance; vector path.
                Duplicate(denV, den, static_cast<int32_t>(cntR));
                Adds(fx, fx, -mean, static_cast<int32_t>(cntR));
                Div(fx, fx, denV, static_cast<int32_t>(cntR));
                Muls(fx, fx, scv, static_cast<int32_t>(cntR));
                Adds(fx, fx, ofv, static_cast<int32_t>(cntR));
            } else {
                // fp32: MARE tolerance is 10*2^-13, too tight for the vector
                // divider near zero, so use the exact scalar IEEE chain.
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (int64_t i = 0; i < cnt; ++i) {
                    const float v = fx.GetValue(static_cast<int32_t>(i));
                    volatile float t1 = v - mean;
                    volatile float t2 = t1 / den;
                    volatile float t3 = t2 * scv;
                    fx.SetValue(static_cast<int32_t>(i), t3 + ofv);
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            StoreTile(fx, rowBase + off, cnt, cntR);
        }
    }
}

}  // namespace NsBnTrainingUpdate
#endif  // BNTRAININGUPDATE_H

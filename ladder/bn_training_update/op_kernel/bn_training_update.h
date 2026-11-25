/*!
 * \file bn_training_update.h
 * \brief BnTrainingUpdate kernel class definition
 *
 *   y_c       = (x_c - mean_c) / sqrt(var_c + eps) * scale_c + offset_c
 *   mean_c    = sum_c / N_c
 *   var_c     = square_sum_c / N_c - mean_c * mean_c
 *   save_mean = mean, save_variance = var          (both fp32, shape (C,))
 *
 * Precision first: the per element arithmetic is evaluated as scalar fp32 in
 * exactly the same operation order as the reference numpy implementation,
 *   fl( fl( fl( fl(x - mean) / d ) * scale ) + offset ),
 * with IEEE scalar division. Volatile intermediates stop the compiler from
 * fusing the last multiply-add into an FMA or from reassociating the chain,
 * because a fused form loses one rounding step and would then differ from the
 * golden by one ulp (which matters for output elements that sit near zero
 * after the offset cancels the normalized value).
 *
 * Per channel mean / var / d are evaluated once per core into small UB tables
 * (C <= 4096, so at most 5 * 16KB). d = sqrt(var + eps) is first approximated
 * by the hardware vector Sqrt and then replaced by the exactly rounded root
 * computed with integer arithmetic (ExactSqrt below): the hardware root is
 * about one ulp off for roughly one value in ten, and a one ulp error in d
 * shifts every output element of that channel by one ulp. The elementwise part
 * then only needs scalar loads from those tables: no per element sqrt, and no
 * per element division other than the unavoidable (x - mean) / d that the
 * golden also performs.
 *
 * Multi core split is along rows ((n, c) planes), and the rows of one core are
 * a contiguous element range, so every core writes only its own output block
 * and no cross core reduction or barrier is required.
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

constexpr int64_t ALIGN_ELEM = 64;  // vector op count alignment (one 256B repeat)

__aicore__ inline uint32_t BitsOfFloat(float v)
{
    union {
        float f;
        uint32_t u;
    } cv;
    cv.f = v;
    return cv.u;
}

__aicore__ inline float FloatOfBits(uint32_t b)
{
    union {
        float f;
        uint32_t u;
    } cv;
    cv.u = b;
    return cv.f;
}

/*
 * Correctly rounded square root of a positive normal float, evaluated exactly
 * with integer arithmetic on the mantissa.
 *
 * The vector Sqrt instruction is only accurate to about one ulp, and d =
 * sqrt(var + eps) enters every output element of its channel through
 * (x - mean) / d, so a one ulp error in d moves a whole channel by one ulp and
 * breaks the exact match with the reference implementation (visible on output
 * elements that sit close to zero). The hardware result is therefore used only
 * as the starting point of an exact integer fix up:
 *
 *   v = M * 2^(2k-24), M in [2^24, 2^26)   ->  sqrt(v) = sqrt(M) * 2^(k-12)
 *   N = M * 2^22 in [2^46, 2^48)           ->  sqrt(N) = sqrt(v) * 2^(23-k)
 *   S = isqrt(N) has 24 bits, so S is exactly the 24 bit mantissa of sqrt(v)
 *   and it only has to be rounded: with R = N - S*S the exact root rounds up
 *   iff R > S (a tie is impossible because R is an integer and the tie point
 *   is S + 0.25).
 *
 * All arithmetic stays exact in 64 bit integers (N < 2^48), so the result is
 * the correctly rounded float, not just an approximation.
 */
__aicore__ inline float ExactSqrt(float v, float guess)
{
    const uint32_t bits = BitsOfFloat(v);
    const uint32_t sign = bits >> 31;
    const uint32_t expo = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;
    if (expo == 0xFFu) {                                  // inf / nan
        return v;
    }
    if (sign != 0u) {                                     // negative -> quiet nan
        return FloatOfBits(0x7FC00000u);
    }
    if (expo == 0u) {                                     // zero / subnormal
        return guess;
    }
    const int32_t e = static_cast<int32_t>(expo) - 127;
    const int32_t k = e >> 1;                             // floor(e / 2)
    const uint32_t m24 = mant | 0x800000u;
    const uint32_t mm = (e & 1) ? (m24 << 2) : (m24 << 1);  // M in [2^24, 2^26)
    const uint64_t nn = static_cast<uint64_t>(mm) << 22;    // N in [2^46, 2^48)

    // seed the integer search with the hardware root, rescaled to sqrt(N)
    int64_t s = 1 << 23;
    const int32_t gexp = 127 + (23 - k);
    if (gexp > 0 && gexp < 255) {
        const float scaled = guess * FloatOfBits(static_cast<uint32_t>(gexp) << 23);
        if (scaled >= 1.0f && scaled <= 1.7e7f) {
            s = static_cast<int64_t>(scaled);
        }
    }
    while (s > 1 && s * s > static_cast<int64_t>(nn)) {
        --s;
    }
    while ((s + 1) * (s + 1) <= static_cast<int64_t>(nn)) {
        ++s;
    }

    const uint64_t rem = nn - static_cast<uint64_t>(s * s);
    const uint64_t root = static_cast<uint64_t>(s) + ((rem > static_cast<uint64_t>(s)) ? 1u : 0u);
    const int32_t ef = 127 + k;
    if (ef <= 0 || ef >= 254) {                           // out of normal range
        return guess;
    }
    if (root >= (1u << 24)) {                             // mantissa carry
        return FloatOfBits(static_cast<uint32_t>(ef + 1) << 23);
    }
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
    __aicore__ inline int64_t AlignUp(int64_t v) const
    {
        return (v + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
    }
    __aicore__ inline void BuildChannelTable();
    __aicore__ inline void ComputeChunk(const LocalTensor<float>& src, const LocalTensor<float>& dst,
                                        int64_t off, int64_t cnt);
    __aicore__ inline void RunChunk(int64_t off, int64_t cnt);
    __aicore__ inline void StoreChannelStats();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tIn;     // input tile
    TBuf<QuePosition::VECCALC> tOut;    // output tile
    TBuf<QuePosition::VECCALC> fWork;   // fp16 -> fp32 working tile
    TBuf<QuePosition::VECCALC> tMean;   // per channel mean
    TBuf<QuePosition::VECCALC> tVar;    // per channel variance
    TBuf<QuePosition::VECCALC> tDen;    // per channel sqrt(var + eps)
    TBuf<QuePosition::VECCALC> tScale;  // per channel scale
    TBuf<QuePosition::VECCALC> tOff;    // per channel offset

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
    int64_t rowsPerCore_ = 1;
    int64_t ub_ = ALIGN_ELEM;
    int64_t tbl_ = ALIGN_ELEM;
    int64_t nElem_ = 1;
    float eps_ = 1e-5f;
    float nf_ = 1.0f;
};

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Init(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale,
                                                 GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean,
                                                 GM_ADDR save_variance,
                                                 const BnTrainingUpdateTilingData* tilingData)
{
    chan_ = tilingData->channelNum;
    if (chan_ < 1) {
        chan_ = 1;
    }
    rowLen_ = tilingData->rowLength;
    if (rowLen_ < 1) {
        rowLen_ = 1;
    }
    totalRows_ = tilingData->totalRows;
    if (totalRows_ < 1) {
        totalRows_ = 1;
    }
    rowsPerCore_ = tilingData->blockFactor;
    if (rowsPerCore_ < 1) {
        rowsPerCore_ = 1;
    }
    ub_ = tilingData->ubFactor;
    if (ub_ < ALIGN_ELEM) {
        ub_ = ALIGN_ELEM;
    }
    ub_ = (ub_ / ALIGN_ELEM) * ALIGN_ELEM;
    nElem_ = tilingData->elemsPerChannel;
    if (nElem_ < 1) {
        nElem_ = 1;
    }
    eps_ = tilingData->epsilon;
    nf_ = static_cast<float>(nElem_);
    tbl_ = AlignUp(chan_);

    const int64_t total = totalRows_ * rowLen_;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), total);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), total);
    sumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum), chan_);
    sqGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(square_sum), chan_);
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), chan_);
    offGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset), chan_);
    meanOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_mean), chan_);
    varOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_variance), chan_);

    pipe.InitBuffer(tIn, static_cast<uint32_t>(ub_ * sizeof(T)));
    pipe.InitBuffer(tOut, static_cast<uint32_t>(ub_ * sizeof(T)));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fWork, static_cast<uint32_t>(ub_ * sizeof(float)));
    }
    pipe.InitBuffer(tMean, static_cast<uint32_t>(tbl_ * sizeof(float)));
    pipe.InitBuffer(tVar, static_cast<uint32_t>(tbl_ * sizeof(float)));
    pipe.InitBuffer(tDen, static_cast<uint32_t>(tbl_ * sizeof(float)));
    pipe.InitBuffer(tScale, static_cast<uint32_t>(tbl_ * sizeof(float)));
    pipe.InitBuffer(tOff, static_cast<uint32_t>(tbl_ * sizeof(float)));
}

/*
 * Per channel statistics. The scalar chain mirrors the golden numpy code:
 *   mean     = sum   / N
 *   var      = sq    / N - mean * mean
 *   denom    = sqrt(var + eps)
 * Each product / quotient is kept in a volatile temporary so that the compiler
 * cannot contract "sq / N - mean * mean" into an FMA and skip one rounding.
 * The square root itself is done by the vector unit over the whole channel
 * table in one shot, which is both cheap and NaN-correct for var + eps < 0.
 */
template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::BuildChannelTable()
{
    LocalTensor<float> mT = tMean.Get<float>();
    LocalTensor<float> vT = tVar.Get<float>();
    LocalTensor<float> sT = tScale.Get<float>();
    LocalTensor<float> bT = tOff.Get<float>();
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
    PipeBarrier<PIPE_ALL>();  // scalar writes visible to the vector unit

    LocalTensor<float> dT = tDen.Get<float>();
    const uint32_t n = static_cast<uint32_t>(tbl_);
    Adds(dT, vT, eps_, n);
    Sqrt(dT, dT, n);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);  // vector sqrt result visible to scalar reads

    // Replace the hardware root (about one ulp accurate) by the exactly rounded
    // one. Only the C channel values are involved, so this costs nothing next
    // to the elementwise loop.
    for (int64_t c = 0; c < chan_; ++c) {
        const int32_t idx = static_cast<int32_t>(c);
        volatile float v = vT.GetValue(idx) + eps_;
        dT.SetValue(idx, ExactSqrt(v, dT.GetValue(idx)));
    }
    PipeBarrier<PIPE_ALL>();  // scalar writes visible to the elementwise loop
}

/*
 * Elementwise part of one UB chunk. [off, off + cnt) is a contiguous range of
 * the flat (n, c, h, w) element space, so it is walked plane by plane; inside
 * one plane all four per channel operands are constant.
 */
template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::ComputeChunk(const LocalTensor<float>& src,
                                                         const LocalTensor<float>& dst,
                                                         int64_t off, int64_t cnt)
{
    LocalTensor<float> mT = tMean.Get<float>();
    LocalTensor<float> dT = tDen.Get<float>();
    LocalTensor<float> sT = tScale.Get<float>();
    LocalTensor<float> bT = tOff.Get<float>();

    const int64_t lim = off + cnt;
    int64_t pos = off;
    int64_t row = pos / rowLen_;
    int64_t rowEnd = (row + 1) * rowLen_;
    int32_t c = static_cast<int32_t>(row % chan_);

    while (pos < lim) {
        if (pos >= rowEnd) {
            ++row;
            rowEnd = (row + 1) * rowLen_;
            c = static_cast<int32_t>(row % chan_);
        }
        const int64_t segEnd = (rowEnd < lim) ? rowEnd : lim;
        const float mean = mT.GetValue(c);
        const float den = dT.GetValue(c);
        const float scv = sT.GetValue(c);
        const float ofv = bT.GetValue(c);

        const int64_t i0 = pos - off;
        const int64_t i1 = segEnd - off;
        for (int64_t i = i0; i < i1; ++i) {
            const float v = src.GetValue(static_cast<int32_t>(i));
            volatile float t1 = v - mean;
            volatile float t2 = t1 / den;
            volatile float t3 = t2 * scv;
            const float r = t3 + ofv;
            dst.SetValue(static_cast<int32_t>(i), r);
        }
        pos = segEnd;
    }
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::RunChunk(int64_t off, int64_t cnt)
{
    const uint32_t acnt = static_cast<uint32_t>(AlignUp(cnt));

    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};

    LocalTensor<T> in = tIn.Get<T>();
    LocalTensor<T> out = tOut.Get<T>();
    DataCopyPad(in, xGm[off], p, pp);

    if constexpr (std::is_same<T, float>::value) {
        // scalar reads follow the DMA, so the copy completion has to be
        // signalled to the scalar pipe as well
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        ComputeChunk(in, out, off, cnt);
        PipeBarrier<PIPE_ALL>();  // scalar work done before the store
        DataCopyPad(yGm[off], out, p);
    } else {
        LocalTensor<float> fb = fWork.Get<float>();
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fb, in, RoundMode::CAST_NONE, acnt);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        ComputeChunk(fb, fb, off, cnt);  // fp32 result kept in place
        PipeBarrier<PIPE_ALL>();         // scalar work done before the cast
        Cast(out, fb, RoundMode::CAST_RINT, acnt);
        PipeBarrier<PIPE_ALL>();  // vector cast done before the store
        DataCopyPad(yGm[off], out, p);
    }
    PipeBarrier<PIPE_ALL>();  // store finished before the tiles are reused
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::StoreChannelStats()
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(chan_ * static_cast<int64_t>(sizeof(float)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(meanOutGm, tMean.Get<float>(), p);
    DataCopyPad(varOutGm, tVar.Get<float>(), p);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Process()
{
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t r0 = blockId * rowsPerCore_;
    if (r0 > totalRows_) {
        r0 = totalRows_;
    }
    int64_t r1 = r0 + rowsPerCore_;
    if (r1 > totalRows_) {
        r1 = totalRows_;
    }
    if (r0 >= r1) {
        return;
    }

    BuildChannelTable();

    // One core owns a contiguous element range, so the per channel statistics
    // are written once, by core 0 only, without any cross core barrier.
    if (blockId == 0) {
        StoreChannelStats();
    }

    const int64_t off = r0 * rowLen_;
    const int64_t end = r1 * rowLen_;
    for (int64_t o = off; o < end; o += ub_) {
        int64_t cnt = end - o;
        if (cnt > ub_) {
            cnt = ub_;
        }
        RunChunk(o, cnt);
    }
}

}  // namespace NsBnTrainingUpdate
#endif  // BNTRAININGUPDATE_H

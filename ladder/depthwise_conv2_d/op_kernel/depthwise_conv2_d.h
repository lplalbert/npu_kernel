/*!
 * \file depthwise_conv2_d.h
 * \brief DepthwiseConv2D kernel implementation.
 *
 * Semantics (groups = C, weight layout (C, 1, kH, kW)):
 *   y[n,c,oh,ow] = b[c] + sum_{kh,kw} xp[n,c, oh*stride + kh*dil, ow*stride + kw*dil] * w[c,0,kh,kw]
 * where xp is x zero padded by pad rows/cols on every side. Because Hout/Wout
 * follow the usual floor formula, every tap stays inside the padded tensor, so
 * the padding always contributes exact zeros.
 *
 * Vector path (tiling field useScalar == 0):
 *   Output rows are produced one at a time. The input row needed by a tap is
 *   copied into a UB row buffer that holds only the valid samples; the zero
 *   padding is modelled by a dedicated all-zero lane at index zoff, which is
 *   never written after initialisation. For every tap a single Gather with a
 *   precomputed byte offset table (one table per kw) builds the sampled vector,
 *   which is then scaled by the tap weight and added into the row accumulator.
 *   All vector operands are 32B aligned and every vector op count is a multiple
 *   of 64.
 *
 *   The reduction order reproduces the reference implementation: the
 *   accumulator starts at zero, taps are folded in ascending (kh, kw) order and
 *   the bias is added once at the very end. Each tap performs a separately
 *   rounded multiply (Muls) followed by a separately rounded add (Add), which
 *   matches the reference "y += patch * w" double rounding; no fused
 *   multiply-add is used anywhere.
 *
 * Scalar path (tiling field useScalar == 1):
 *   Element-wise GM gather, used only when the UB budget cannot hold the vector
 *   path buffers. Same reduction order.
 *
 * Output rows are partitioned into disjoint contiguous ranges across AIV cores,
 * so no cross-core synchronisation, atomics or workspace are needed and the
 * result is deterministic.
 *
 * This operator never touches the workspace argument.
 */

#ifndef DEPTHWISECONV2D_H
#define DEPTHWISECONV2D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "depthwise_conv2_d_tiling_data.h"
#include "depthwise_conv2_d_tiling_key.h"
#include <type_traits>

namespace NsDepthwiseConv2D {

using namespace AscendC;

/* vector ops are limited to 255 repeats per call, so long buffers are handled
 * in slices of 4096 elements (a multiple of the 64 element vector length) */
constexpr int64_t kZeroSlice = 4096;

__aicore__ inline void ZeroFillF(const LocalTensor<float>& dst, int64_t count)
{
    for (int64_t o = 0; o < count; o += kZeroSlice) {
        int64_t c = count - o;
        if (c > kZeroSlice) {
            c = kZeroSlice;
        }
        Duplicate(dst[static_cast<int32_t>(o)], 0.0f, static_cast<uint32_t>(c));
    }
}

template <typename T>
class DepthwiseConv2D {
public:
    __aicore__ inline DepthwiseConv2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                const DepthwiseConv2DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeights(int64_t c);
    __aicore__ inline void LoadRow(int64_t off);
    __aicore__ inline void DoRow(int64_t nn, int64_t c, int64_t oh);
    __aicore__ inline float ComputeOne(int64_t idx);
    __aicore__ inline void FlushChunk(int64_t base, int64_t cnt);
    __aicore__ inline void ScalarProcess();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fXr;    /* input row, float */
    TBuf<QuePosition::VECCALC> fXh;    /* input row staging, io dtype */
    TBuf<QuePosition::VECCALC> fAcc;   /* output row accumulator */
    TBuf<QuePosition::VECCALC> fXg;    /* gathered tap samples */
    TBuf<QuePosition::VECCALC> fIdx;   /* gather byte offsets, per kw */
    TBuf<QuePosition::VECCALC> fW;     /* tap weights, float */
    TBuf<QuePosition::VECCALC> fWh;    /* tap weights staging, io dtype */
    TBuf<QuePosition::VECCALC> fAccT;  /* output row, io dtype */
    TBuf<QuePosition::VECCALC> fStage; /* scalar path staging, float */
    TBuf<QuePosition::VECCALC> fStageT;/* scalar path staging, io dtype */

    GlobalTensor<T> xGm;
    GlobalTensor<T> wGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<T> yGm;

    DepthwiseConv2DTilingData td_;
    int64_t n_ = 1;
    int64_t chIn_ = 1;
    int64_t hin_ = 1;
    int64_t win_ = 1;
    int64_t hout_ = 1;
    int64_t wout_ = 1;
    int64_t kh_ = 1;
    int64_t kw_ = 1;
    int64_t stride_ = 1;
    int64_t pad_ = 0;
    int64_t dil_ = 1;
    int64_t totalOut_ = 0;
    int64_t totalRows_ = 0;
    int64_t accSize_ = 64;
    int64_t xSize_ = 128;
    int64_t zoff_ = 64;
    int64_t wPad_ = 64;
    int64_t nw_ = 1;
    int64_t chunk_ = 1024;
    int32_t hasBias_ = 0;
    int32_t useScalar_ = 0;
    int64_t wLoaded_ = -1; /* channel whose weights are currently staged */
};

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                                const DepthwiseConv2DTilingData* tilingData)
{
    td_ = *tilingData;
    n_ = td_.n > 0 ? td_.n : 1;
    chIn_ = td_.chIn > 0 ? td_.chIn : 1;
    hin_ = td_.hin > 0 ? td_.hin : 1;
    win_ = td_.win > 0 ? td_.win : 1;
    hout_ = td_.hout > 0 ? td_.hout : 1;
    wout_ = td_.wout > 0 ? td_.wout : 1;
    kh_ = td_.kh > 0 ? td_.kh : 1;
    kw_ = td_.kw > 0 ? td_.kw : 1;
    stride_ = td_.stride > 0 ? td_.stride : 1;
    pad_ = td_.pad;
    dil_ = td_.dil > 0 ? td_.dil : 1;
    hasBias_ = td_.hasBias;
    totalOut_ = td_.totalNum;
    totalRows_ = td_.totalRows > 0 ? td_.totalRows : (n_ * chIn_ * hout_);
    useScalar_ = td_.useScalar;
    accSize_ = td_.accSize > 0 ? td_.accSize : 64;
    xSize_ = td_.xSize > 0 ? td_.xSize : 128;
    zoff_ = td_.zoff > 0 ? td_.zoff : 64;
    wPad_ = td_.wPad > 0 ? td_.wPad : 64;
    nw_ = kh_ * kw_;

    chunk_ = td_.ubFactor;
    if (chunk_ < 64) {
        chunk_ = 64;
    }
    chunk_ = (chunk_ / 64) * 64;

    xGm.SetGlobalBuffer((__gm__ T*)x, n_ * chIn_ * hin_ * win_);
    wGm.SetGlobalBuffer((__gm__ T*)w, chIn_ * nw_);
    yGm.SetGlobalBuffer((__gm__ T*)y, totalOut_);
    if (hasBias_ != 0) {
        biasGm.SetGlobalBuffer((__gm__ float*)b, chIn_);
    }

    if (useScalar_ != 0) {
        pipe.InitBuffer(fStage, static_cast<uint32_t>((chunk_ + 64) * sizeof(float)));
        if constexpr (!std::is_same<T, float>::value) {
            pipe.InitBuffer(fStageT, static_cast<uint32_t>((chunk_ + 64) * sizeof(T)));
        }
        return;
    }

    pipe.InitBuffer(fXr, static_cast<uint32_t>(xSize_ * sizeof(float)));
    pipe.InitBuffer(fAcc, static_cast<uint32_t>(accSize_ * sizeof(float)));
    pipe.InitBuffer(fXg, static_cast<uint32_t>(accSize_ * sizeof(float)));
    pipe.InitBuffer(fIdx, static_cast<uint32_t>(kw_ * accSize_ * sizeof(int32_t)));
    pipe.InitBuffer(fW, static_cast<uint32_t>(wPad_ * sizeof(float)));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fXh, static_cast<uint32_t>(xSize_ * sizeof(T)));
        pipe.InitBuffer(fWh, static_cast<uint32_t>(wPad_ * sizeof(T)));
        pipe.InitBuffer(fAccT, static_cast<uint32_t>(accSize_ * sizeof(T)));
    }

    /* The row buffer is zeroed once. Only [0, win) is ever rewritten, so the
     * all-zero lane at zoff (and everything above it) stays exactly zero and
     * models the spatial zero padding. */
    LocalTensor<float> fx = fXr.Get<float>();
    ZeroFillF(fx, xSize_);

    /* Gather byte offsets. For every kw the sampled position of output ow is
     * iw = ow*stride + kw*dil - pad; positions outside the input row are
     * redirected to the zero lane. */
    LocalTensor<int32_t> idx = fIdx.Get<int32_t>();
    const int32_t zeroOff = static_cast<int32_t>(zoff_ * 4);
    for (int64_t kw = 0; kw < kw_; ++kw) {
        int64_t baseK = kw * accSize_;
        int64_t tapOff = kw * dil_ - pad_;
        for (int64_t ow = 0; ow < accSize_; ++ow) {
            int32_t off = zeroOff;
            int64_t iw = ow * stride_ + tapOff;
            if (iw >= 0 && iw < win_) {
                off = static_cast<int32_t>(iw * 4);
            }
            idx.SetValue(static_cast<int32_t>(baseK + ow), off);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

/* Stage the kh*kw weights of one channel plus its bias. */
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::LoadWeights(int64_t c)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(nw_ * sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    LocalTensor<float> fw = fW.Get<float>();
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(fw, wGm[c * nw_], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
    } else {
        LocalTensor<T> wh = fWh.Get<T>();
        DataCopyPad(wh, wGm[c * nw_], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(fw, wh, RoundMode::CAST_NONE, static_cast<int32_t>(wPad_));
    }
    PipeBarrier<PIPE_ALL>();
    wLoaded_ = c;
}

/* Copy input row `off` (already offset to the row start in GM) to the UB row
 * buffer. Only the first win_ lanes are rewritten. */
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::LoadRow(int64_t off)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(win_ * sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    PipeBarrier<PIPE_ALL>(); /* V(read fXr by the previous Gather) -> MTE2, WAR */
    LocalTensor<float> fx = fXr.Get<float>();
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(fx, xGm[off], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
    } else {
        LocalTensor<T> xh = fXh.Get<T>();
        DataCopyPad(xh, xGm[off], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>(); /* MTE2 -> V */
        Cast(fx, xh, RoundMode::CAST_NONE, static_cast<int32_t>(zoff_));
    }
    PipeBarrier<PIPE_ALL>(); /* MTE2/V -> V */
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::DoRow(int64_t nn, int64_t c, int64_t oh)
{
    LocalTensor<float> acc = fAcc.Get<float>();
    LocalTensor<float> xg = fXg.Get<float>();
    LocalTensor<float> fw = fW.Get<float>();
    LocalTensor<float> fx = fXr.Get<float>();
    LocalTensor<uint32_t> idx = fIdx.Get<int32_t>().template ReinterpretCast<uint32_t>();

    Duplicate(acc, 0.0f, static_cast<uint32_t>(accSize_));

    const int64_t xRowStride = hin_ * win_;
    const int64_t xChanBase = (nn * chIn_ + c) * xRowStride;

    for (int64_t kh = 0; kh < kh_; ++kh) {
        int64_t local = oh * stride_ + kh * dil_;
        int64_t ih = local - pad_;
        if (ih < 0 || ih >= hin_) {
            continue; /* pure padding row: contributes exact zeros */
        }
        LoadRow(xChanBase + ih * win_);
        for (int64_t kw = 0; kw < kw_; ++kw) {
            float wv = fw.GetValue(static_cast<int32_t>(kh * kw_ + kw));
            Gather(xg, fx, idx[static_cast<int32_t>(kw * accSize_)], 0,
                   static_cast<uint32_t>(accSize_));
            Muls(xg, xg, wv, static_cast<uint32_t>(accSize_));
            Add(acc, acc, xg, static_cast<uint32_t>(accSize_));
        }
    }

    if (hasBias_ != 0) {
        float bias = biasGm.GetValue(c);
        Adds(acc, acc, bias, static_cast<uint32_t>(accSize_));
    }

    int64_t rowBase = ((nn * chIn_ + c) * hout_ + oh) * wout_;
    PipeBarrier<PIPE_ALL>(); /* V -> MTE3 */
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(wout_ * sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(yGm[rowBase], acc, p);
    } else {
        LocalTensor<T> accT = fAccT.Get<T>();
        Cast(accT, acc, RoundMode::CAST_RINT, static_cast<int32_t>(accSize_));
        PipeBarrier<PIPE_ALL>(); /* V -> MTE3 */
        DataCopyPad(yGm[rowBase], accT, p);
    }
    PipeBarrier<PIPE_ALL>(); /* MTE3 done before the next row reuses the buffers */
}

/* Single output element, used by the scalar fallback path. */
template <typename T>
__aicore__ inline float DepthwiseConv2D<T>::ComputeOne(int64_t idx)
{
    const int64_t hw = hout_ * wout_;
    int64_t rem = idx;
    int64_t nn = rem / (chIn_ * hw);
    rem -= nn * (chIn_ * hw);
    int64_t c = rem / hw;
    rem -= c * hw;
    int64_t oh = rem / wout_;
    int64_t ow = rem - oh * wout_;

    float acc = 0.0f;
    const int64_t xRowStride = hin_ * win_;
    const int64_t xChanBase = (nn * chIn_ + c) * xRowStride;
    const int64_t wChanBase = c * nw_;

    for (int64_t kh = 0; kh < kh_; ++kh) {
        int64_t ih = oh * stride_ + kh * dil_ - pad_;
        if (ih < 0 || ih >= hin_) {
            continue;
        }
        for (int64_t kw = 0; kw < kw_; ++kw) {
            int64_t iw = ow * stride_ + kw * dil_ - pad_;
            if (iw < 0 || iw >= win_) {
                continue;
            }
            float xv = static_cast<float>(xGm.GetValue(xChanBase + ih * win_ + iw));
            float wv = static_cast<float>(wGm.GetValue(wChanBase + kh * kw_ + kw));
            /* separate rounding of the product and of the accumulation, to
             * match the reference "y += patch * w" evaluation order */
            volatile float prod = xv * wv;
            acc = acc + prod;
        }
    }
    if (hasBias_ != 0) {
        acc += biasGm.GetValue(c);
    }
    return acc;
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::FlushChunk(int64_t base, int64_t cnt)
{
    LocalTensor<float> f = fStage.Get<float>();
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(yGm[base], f, p);
    } else {
        LocalTensor<T> h = fStageT.Get<T>();
        int64_t cn = ((cnt + 63) / 64) * 64;
        Cast(h, f, RoundMode::CAST_RINT, static_cast<int32_t>(cn));
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(yGm[base], h, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::ScalarProcess()
{
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t bn = blockNum < 1 ? 1 : blockNum;
    int64_t per = (totalOut_ + bn - 1) / bn;
    if (per < 1) {
        per = 1;
    }
    int64_t s = blockId * per;
    if (s >= totalOut_) {
        return;
    }
    int64_t e = s + per;
    if (e > totalOut_) {
        e = totalOut_;
    }
    LocalTensor<float> f = fStage.Get<float>();
    int64_t base = s;
    while (base < e) {
        int64_t cnt = e - base;
        if (cnt > chunk_) {
            cnt = chunk_;
        }
        for (int64_t j = 0; j < cnt; ++j) {
            f.SetValue(static_cast<int32_t>(j), ComputeOne(base + j));
        }
        FlushChunk(base, cnt);
        base += cnt;
    }
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::Process()
{
    if (useScalar_ != 0) {
        ScalarProcess();
        return;
    }
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t bn = blockNum < 1 ? 1 : blockNum;
    int64_t per = (totalRows_ + bn - 1) / bn;
    if (per < 1) {
        per = 1;
    }
    int64_t s = blockId * per;
    if (s >= totalRows_) {
        return;
    }
    int64_t e = s + per;
    if (e > totalRows_) {
        e = totalRows_;
    }

    const int64_t rowChan = chIn_ * hout_;
    for (int64_t rid = s; rid < e; ++rid) {
        int64_t nn = rid / rowChan;
        int64_t rem = rid - nn * rowChan;
        int64_t c = rem / hout_;
        int64_t oh = rem - c * hout_;
        if (c != wLoaded_) {
            LoadWeights(c);
        }
        DoRow(nn, c, oh);
    }
}

} // namespace NsDepthwiseConv2D
#endif // DEPTHWISECONV2D_H

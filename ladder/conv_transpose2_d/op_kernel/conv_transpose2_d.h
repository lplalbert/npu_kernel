/*!
 * \file conv_transpose2_d.h
 * \brief ConvTranspose2D kernel
 *
 * Semantics (groups = 1):
 *   y[n,co,oh,ow] = sum_{kh,kw,ci} x[n,ci,ih,iw] * w[ci,co,kh,kw] + b[co]
 *   ih = (oh+pad-kh*dil)/stride, iw = (ow+pad-kw*dil)/stride
 *   only integer, in-range (ih,iw) contribute.
 *
 * Path selection by host tiling (field useScalar):
 *  0 = ow-block vectorized path (fast):
 *      For each output row (n,co,oh) the row is processed in 32B-aligned blocks
 *      of WB (=multiple of 64) output elements kept in natural order in UB. For
 *      every (ci,kh,kw) the strided input row (iw = (ow+pad-kw*dil)/stride) is
 *      collected into a contiguous UB vector with Gather using a per-kw
 *      precomputed byte-offset table (invalid positions point at a zero lane).
 *      A vector mac (Axpy) accumulates the block. All vector operands are
 *      32B aligned and all vector op counts are multiples of 64.
 *  1 = scalar GM-direct gather (fallback for extreme shapes / UB overflow).
 *
 * Reduction order (field accMode): the reference implementation accumulates
 *   over the input channels inside one tap first and then over the taps, with
 *   the bias added last.  To stay as close as possible to that rounding:
 *  2 = one accumulator per (kh,kw) tap, channels ascending inside the tap,
 *      taps folded in ascending (kh,kw) order, bias last.  Same association as
 *      the reference and no extra data movement.
 *  1 = one accumulator per (ci,kh) row of kw taps, folded with Kahan
 *      compensation (used when the kh*kw tap accumulators do not fit in UB).
 *  0 = one accumulator per (ci,kh) row of kw taps, plain folding.
 *  3 = single fp32 accumulator written directly (no extra chunk buffer; used
 *      for very long rows so they still run on the vector path).
 *   The bias is added once at the very end in every mode.
 *
 * Work is partitioned by disjoint row / element ranges across AIV cores
 * (no cross-core sync, no atomics -> deterministic output).
 */

#ifndef CONVTRANSPOSE2D_H
#define CONVTRANSPOSE2D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "conv_transpose2_d_tiling_data.h"
#include "conv_transpose2_d_tiling_key.h"
#include <type_traits>

namespace NsConvTranspose2D {

using namespace AscendC;

__aicore__ inline int64_t Ceil64(int64_t x)
{
    return ((x + 63) / 64) * 64;
}

// vector ops are limited to 255 repeats per call, so long buffers are zeroed in
// slices of 4096 elements (a multiple of the 64 element vector length)
__aicore__ inline void ZeroFill(const LocalTensor<float>& dst, int64_t count)
{
    constexpr int64_t kSlice = 4096;
    for (int64_t o = 0; o < count; o += kSlice) {
        int64_t c = count - o;
        if (c > kSlice) {
            c = kSlice;
        }
        Duplicate(dst[static_cast<int32_t>(o)], 0.0f, static_cast<uint32_t>(c));
    }
}

template <typename T>
class ConvTranspose2D {
public:
    __aicore__ inline ConvTranspose2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, const ConvTranspose2DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // scalar path
    __aicore__ inline float ComputeOne(int64_t idx);
    __aicore__ inline void ScalarProcess();
    __aicore__ inline void FlushChunk(int64_t base, int64_t cnt);
    // vector path
    __aicore__ inline void VectorProcess();
    __aicore__ inline void LoadXRow(int64_t off, int64_t win);
    __aicore__ inline void DoRow(int64_t n, int64_t co, int64_t oh);
    __aicore__ inline void FoldChunk(const LocalTensor<float>& acc, const LocalTensor<float>& tmp);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fStage;  // scalar staging
    TBuf<QuePosition::VECCALC> fStageH; // scalar fp16 staging
    TBuf<QuePosition::VECCALC> fAcc;    // output row accumulator (accSize)
    TBuf<QuePosition::VECCALC> fTmp;    // per (ci,kh) chunk accumulator (accSize)
    TBuf<QuePosition::VECCALC> fTap;    // per (kh,kw) tap accumulators (kh*kw*accSize)
    TBuf<QuePosition::VECCALC> fComp;   // Kahan compensation (accSize)
    TBuf<QuePosition::VECCALC> fYv;     // Kahan scratch (WB)
    TBuf<QuePosition::VECCALC> fTt;     // Kahan scratch (WB)
    TBuf<QuePosition::VECCALC> fXg;     // gathered x vector (WB)
    TBuf<QuePosition::VECCALC> fX;      // x row (float, p2)
    TBuf<QuePosition::VECCALC> xhBuf;   // x row raw T (fp16)
    TBuf<QuePosition::VECCALC> idxTab;  // gather byte offsets, per kw: [kw*accSize + ow]
    TBuf<QuePosition::VECCALC> accH;    // fp16 output cast

    GlobalTensor<T> xGm;
    GlobalTensor<T> wGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<T> yGm;

    ConvTranspose2DTilingData td_;
    int64_t n_ = 1;
    int64_t chIn_ = 1;
    int64_t chOut_ = 1;
    int64_t hin_ = 1;
    int64_t win_ = 1;
    int64_t hout_ = 1;
    int64_t wout_ = 1;
    int64_t kh_ = 1;
    int64_t kw_ = 1;
    int64_t stride_ = 1;
    int64_t pad_ = 0;
    int64_t outPad_ = 0;
    int64_t dil_ = 1;
    int32_t hasBias_ = 0;
    int64_t chunk_ = 1024;
    int64_t totalOut_ = 0;
    int32_t useScalar_ = 1;
    int32_t accMode_ = 0;

    // vector path derived
    int64_t wb_ = 64;
    int64_t accSize_ = 64;
    int64_t p2_ = 128;
    int64_t zoff_ = 64;
};

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                                const ConvTranspose2DTilingData* tilingData)
{
    td_ = *tilingData;
    n_ = td_.n > 0 ? td_.n : 1;
    chIn_ = td_.chIn > 0 ? td_.chIn : 1;
    chOut_ = td_.chOut > 0 ? td_.chOut : 1;
    hin_ = td_.hin > 0 ? td_.hin : 1;
    win_ = td_.win > 0 ? td_.win : 1;
    hout_ = td_.hout > 0 ? td_.hout : 1;
    wout_ = td_.wout > 0 ? td_.wout : 1;
    kh_ = td_.kh > 0 ? td_.kh : 1;
    kw_ = td_.kw > 0 ? td_.kw : 1;
    stride_ = td_.stride > 0 ? td_.stride : 1;
    pad_ = td_.pad;
    outPad_ = td_.outPad;
    dil_ = td_.dil > 0 ? td_.dil : 1;
    hasBias_ = td_.hasBias;
    totalOut_ = td_.totalNum;
    useScalar_ = td_.useScalar;
    accMode_ = (td_.accMode >= 0 && td_.accMode <= 3) ? td_.accMode : 0;

    chunk_ = td_.ubFactor;
    if (chunk_ < 64) {
        chunk_ = 64;
    }
    chunk_ = (chunk_ / 64) * 64;

    xGm.SetGlobalBuffer((__gm__ T*)x, n_ * chIn_ * hin_ * win_);
    wGm.SetGlobalBuffer((__gm__ T*)w, chIn_ * chOut_ * kh_ * kw_);
    yGm.SetGlobalBuffer((__gm__ T*)y, totalOut_);
    if (hasBias_ != 0) {
        biasGm.SetGlobalBuffer((__gm__ float*)b, chOut_);
    }

    if (useScalar_ != 0) {
        accMode_ = 0;
        pipe.InitBuffer(fStage, static_cast<uint32_t>((chunk_ + 64) * sizeof(float)));
        if constexpr (!std::is_same<T, float>::value) {
            pipe.InitBuffer(fStageH, static_cast<uint32_t>((chunk_ + 64) * sizeof(T)));
        }
        return;
    }

    int64_t woutPad = Ceil64(wout_);
    wb_ = woutPad < 256 ? woutPad : 256;
    if (wb_ < 64) {
        wb_ = 64;
    }
    accSize_ = ((woutPad + wb_ - 1) / wb_) * wb_;
    p2_ = Ceil64(win_ + 64);
    zoff_ = Ceil64(win_);

    pipe.InitBuffer(fAcc, static_cast<uint32_t>(accSize_ * sizeof(float)));
    pipe.InitBuffer(fXg, static_cast<uint32_t>(wb_ * sizeof(float)));
    pipe.InitBuffer(fX, static_cast<uint32_t>(p2_ * sizeof(float)));
    pipe.InitBuffer(idxTab, static_cast<uint32_t>(kw_ * accSize_ * sizeof(int32_t)));
    if (accMode_ == 2) {
        pipe.InitBuffer(fTap, static_cast<uint32_t>(kh_ * kw_ * accSize_ * sizeof(float)));
    } else if (accMode_ == 1) {
        pipe.InitBuffer(fTmp, static_cast<uint32_t>(accSize_ * sizeof(float)));
        pipe.InitBuffer(fComp, static_cast<uint32_t>(accSize_ * sizeof(float)));
        pipe.InitBuffer(fYv, static_cast<uint32_t>(wb_ * sizeof(float)));
        pipe.InitBuffer(fTt, static_cast<uint32_t>(wb_ * sizeof(float)));
    } else if (accMode_ == 0) {
        pipe.InitBuffer(fTmp, static_cast<uint32_t>(accSize_ * sizeof(float)));
    }
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(xhBuf, static_cast<uint32_t>(p2_ * sizeof(T)));
        pipe.InitBuffer(accH, static_cast<uint32_t>(accSize_ * sizeof(T)));
    }

    // zero the x row tail once so padding lanes (and zoff_ lane) are exactly 0
    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<float> fx = fX.Get<float>();
        Duplicate(fx, 0.0f, static_cast<uint32_t>(p2_));
    } else {
        LocalTensor<T> xh = xhBuf.Get<T>();
        Duplicate(xh, static_cast<T>(0), static_cast<uint32_t>(p2_));
    }

    // precompute per-kw gather byte offsets into fX (float)
    LocalTensor<int32_t> idx = idxTab.Get<int32_t>();
    const int32_t zeroOff = static_cast<int32_t>(zoff_ * 4);
    for (int64_t kw = 0; kw < kw_; ++kw) {
        int64_t baseK = kw * accSize_;
        for (int64_t ow = 0; ow < accSize_; ++ow) {
            int32_t off = zeroOff;
            if (ow < wout_) {
                int64_t t = ow + pad_ - kw * dil_;
                if (t >= 0 && (t % stride_) == 0) {
                    int64_t iw = t / stride_;
                    if (iw < win_) {
                        off = static_cast<int32_t>(iw * 4);
                    }
                }
            }
            idx.SetValue(static_cast<int32_t>(baseK + ow), off);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::LoadXRow(int64_t off, int64_t win)
{
    LocalTensor<float> fx = fX.Get<float>();
    PipeBarrier<PIPE_ALL>();  // V(read fX by prev Gather/Axpy) -> MTE2(write fX), WAR
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(win * sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(fx, xGm[off], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
    } else {
        LocalTensor<T> xh = xhBuf.Get<T>();
        DataCopyPad(xh, xGm[off], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();  // MTE2 -> V
        Cast(fx, xh, RoundMode::CAST_NONE, static_cast<int32_t>(p2_));
    }
    PipeBarrier<PIPE_ALL>();  // MTE2 -> V
}

// fold the (ci,kh) chunk accumulator into the row accumulator.
// mode 1: Kahan compensated (y = chunk - c; t = acc + y; c = (t - acc) - y; acc = t)
// mode 0: plain add
template <typename T>
__aicore__ inline void ConvTranspose2D<T>::FoldChunk(const LocalTensor<float>& acc,
                                                     const LocalTensor<float>& tmp)
{
    const int64_t nblk = accSize_ / wb_;
    if (accMode_ != 1) {
        for (int64_t b = 0; b < nblk; ++b) {
            const int32_t o = static_cast<int32_t>(b * wb_);
            Add(acc[o], acc[o], tmp[o], static_cast<uint32_t>(wb_));
        }
        return;
    }
    LocalTensor<float> comp = fComp.Get<float>();
    LocalTensor<float> yv = fYv.Get<float>();
    LocalTensor<float> tt = fTt.Get<float>();
    for (int64_t b = 0; b < nblk; ++b) {
        const int32_t o = static_cast<int32_t>(b * wb_);
        Sub(yv, tmp[o], comp[o], static_cast<uint32_t>(wb_));      // yv = chunk - c
        Add(tt, acc[o], yv, static_cast<uint32_t>(wb_));           // tt = acc + yv
        Sub(comp[o], tt, acc[o], static_cast<uint32_t>(wb_));      // comp = tt - acc
        Sub(comp[o], comp[o], yv, static_cast<uint32_t>(wb_));     // comp = (tt - acc) - yv
        Adds(acc[o], tt, 0.0f, static_cast<uint32_t>(wb_));        // acc = tt
    }
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::DoRow(int64_t n, int64_t co, int64_t oh)
{
    LocalTensor<float> acc = fAcc.Get<float>();
    LocalTensor<float> xg = fXg.Get<float>();
    LocalTensor<float> fx = fX.Get<float>();
    LocalTensor<uint32_t> idxT = idxTab.Get<int32_t>().template ReinterpretCast<uint32_t>();

    float bias = (hasBias_ != 0) ? biasGm.GetValue(co) : 0.0f;
    Duplicate(acc, 0.0f, static_cast<uint32_t>(accSize_));
    LocalTensor<float> tapBuf;
    if (accMode_ == 2) {
        tapBuf = fTap.Get<float>();
        ZeroFill(tapBuf, kh_ * kw_ * accSize_);
    } else if (accMode_ == 1) {
        Duplicate(fComp.Get<float>(), 0.0f, static_cast<uint32_t>(accSize_));
    }

    const int64_t xStrideC = hin_ * win_;
    const int64_t xStrideN = chIn_ * xStrideC;
    const int64_t wStrideC = chOut_ * kh_ * kw_;
    const int64_t baseXN = n * xStrideN;
    const int64_t nblk = accSize_ / wb_;

    for (int64_t ci = 0; ci < chIn_; ++ci) {
        int64_t xCiN = baseXN + ci * xStrideC;
        for (int64_t kh = 0; kh < kh_; ++kh) {
            int64_t th = oh + pad_ - kh * dil_;
            if (th < 0) {
                break;
            }
            if (th % stride_ != 0) {
                continue;
            }
            int64_t ih = th / stride_;
            if (ih >= hin_) {
                continue;
            }
            LoadXRow(xCiN + ih * win_, win_);
            if (accMode_ == 2) {
                // reference order: inside one tap the channels are accumulated
                // in ascending order, so every tap keeps its own accumulator
                for (int64_t kw = 0; kw < kw_; ++kw) {
                    float wv = static_cast<float>(wGm.GetValue(ci * wStrideC + co * kh_ * kw_ + kh * kw_ + kw));
                    int64_t baseK = kw * accSize_;
                    int32_t cbase = static_cast<int32_t>((kh * kw_ + kw) * accSize_);
                    for (int64_t b = 0; b < nblk; ++b) {
                        int64_t ow0 = b * wb_;
                        Gather(xg, fx, idxT[static_cast<int32_t>(baseK + ow0)], 0,
                               static_cast<uint32_t>(wb_));
                        Axpy(tapBuf[cbase + static_cast<int32_t>(ow0)], xg, wv,
                             static_cast<uint32_t>(wb_));
                    }
                }
                continue;
            }
            LocalTensor<float> tmp = (accMode_ == 3) ? acc : fTmp.Get<float>();
            for (int64_t kw = 0; kw < kw_; ++kw) {
                float wv = static_cast<float>(wGm.GetValue(ci * wStrideC + co * kh_ * kw_ + kh * kw_ + kw));
                int64_t baseK = kw * accSize_;
                for (int64_t b = 0; b < nblk; ++b) {
                    int64_t ow0 = b * wb_;
                    Gather(xg, fx, idxT[static_cast<int32_t>(baseK + ow0)], 0,
                           static_cast<uint32_t>(wb_));
                    if (kw == 0 && accMode_ != 3) {
                        Muls(tmp[static_cast<int32_t>(ow0)], xg, wv, static_cast<uint32_t>(wb_));
                    } else {
                        Muls(xg, xg, wv, static_cast<uint32_t>(wb_));
                        Add(tmp[static_cast<int32_t>(ow0)], tmp[static_cast<int32_t>(ow0)], xg,
                            static_cast<uint32_t>(wb_));
                    }
                }
            }
            if (accMode_ != 3) {
                FoldChunk(acc, tmp);
            }
        }
    }

    if (accMode_ == 2) {
        LocalTensor<float> tap = fTap.Get<float>();
        const int64_t nTap = kh_ * kw_;
        // fold the tap accumulators in ascending (kh,kw) order; acc is still 0 so
        // the first add reproduces the reference initialisation exactly
        for (int64_t t = 0; t < nTap; ++t) {
            Add(acc, acc, tap[static_cast<int32_t>(t * accSize_)], static_cast<uint32_t>(accSize_));
        }
    }
    if (hasBias_ != 0) {
        Adds(acc, acc, bias, static_cast<uint32_t>(accSize_));  // bias added last
    }

    int64_t rowBase = ((n * chOut_ + co) * hout_ + oh) * wout_;
    PipeBarrier<PIPE_ALL>();  // V -> MTE3
    if constexpr (std::is_same<T, float>::value) {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(wout_ * sizeof(float));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(yGm[rowBase], acc, p);
    } else {
        LocalTensor<T> ah = accH.Get<T>();
        Cast(ah, acc, RoundMode::CAST_RINT, static_cast<int32_t>(accSize_));
        PipeBarrier<PIPE_ALL>();  // V -> MTE3
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(wout_ * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(yGm[rowBase], ah, p);
    }
    PipeBarrier<PIPE_ALL>();  // MTE3 done before next row reuse
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::VectorProcess()
{
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    const int64_t rows = n_ * chOut_ * hout_;
    int64_t bn = blockNum < 1 ? 1 : blockNum;
    int64_t per = (rows + bn - 1) / bn;
    if (per < 1) {
        per = 1;
    }
    int64_t s = blockId * per;
    int64_t e = s + per;
    if (e > rows) {
        e = rows;
    }
    if (s >= rows) {
        return;
    }
    const int64_t rowsHW = chOut_ * hout_;
    for (int64_t rid = s; rid < e; ++rid) {
        int64_t n = rid / rowsHW;
        int64_t rr = rid - n * rowsHW;
        int64_t co = rr / hout_;
        int64_t oh = rr - co * hout_;
        DoRow(n, co, oh);
    }
}

template <typename T>
__aicore__ inline float ConvTranspose2D<T>::ComputeOne(int64_t idx)
{
    const int64_t hw = hout_ * wout_;
    int64_t tmp = idx;
    int64_t n = tmp / (chOut_ * hw);
    tmp -= n * (chOut_ * hw);
    int64_t co = tmp / hw;
    tmp -= co * hw;
    int64_t oh = tmp / wout_;
    int64_t ow = tmp - oh * wout_;

    // reference order: taps outer, channels ascending, bias last
    float acc = 0.0f;
    const int64_t xStrideC = hin_ * win_;
    const int64_t xStrideN = chIn_ * xStrideC;
    const int64_t wStrideC = chOut_ * kh_ * kw_;
    const int64_t baseX = n * xStrideN;

    for (int64_t kh = 0; kh < kh_; ++kh) {
        int64_t th = oh + pad_ - kh * dil_;
        if (th < 0) {
            break;
        }
        if (th % stride_ != 0) {
            continue;
        }
        int64_t ih = th / stride_;
        if (ih >= hin_) {
            continue;
        }
        for (int64_t kw = 0; kw < kw_; ++kw) {
            int64_t tw = ow + pad_ - kw * dil_;
            if (tw < 0) {
                break;
            }
            if (tw % stride_ != 0) {
                continue;
            }
            int64_t iw = tw / stride_;
            if (iw >= win_) {
                continue;
            }
            int64_t xOff0 = baseX + ih * win_ + iw;
            int64_t wOff0 = co * kh_ * kw_ + kh * kw_ + kw;
            for (int64_t ci = 0; ci < chIn_; ++ci) {
                float xv = static_cast<float>(xGm.GetValue(xOff0 + ci * xStrideC));
                float wv = static_cast<float>(wGm.GetValue(wOff0 + ci * wStrideC));
                acc = xv * wv + acc;
            }
        }
    }
    if (hasBias_ != 0) {
        acc += biasGm.GetValue(co);
    }
    return acc;
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::FlushChunk(int64_t base, int64_t cnt)
{
    LocalTensor<float> f = fStage.Get<float>();
    PipeBarrier<PIPE_ALL>();
    if constexpr (std::is_same<T, float>::value) {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(cnt * sizeof(float));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(yGm[base], f, p);
    } else {
        LocalTensor<T> h = fStageH.Get<T>();
        Cast(h, f, RoundMode::CAST_RINT, static_cast<int32_t>(Ceil64(cnt)));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(cnt * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(yGm[base], h, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::ScalarProcess()
{
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
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
            f.SetValue(j, ComputeOne(base + j));
        }
        FlushChunk(base, cnt);
        base += cnt;
    }
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::Process()
{
    if (useScalar_ != 0) {
        ScalarProcess();
    } else {
        VectorProcess();
    }
}

} // namespace NsConvTranspose2D
#endif // CONVTRANSPOSE2D_H

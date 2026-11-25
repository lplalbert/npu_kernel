/*!
 * \file multi_add_rms_norm_dynamic_quant.h
 * \brief MultiAddRmsNormDynamicQuant kernel class
 *
 * Row parallel: each AI core owns rowsPerCore rows.
 * - gamma / smooth1 / smooth2 are preloaded into UB once per core (fp32) when
 *   tiling sets preload=1, so the per-row GM traffic disappears.
 * - Three UB passes per row (rowBuf holds x -> z -> quantized), so every input is
 *   read from GM once and only the six outputs are written back.
 * - Every vector op count is a multiple of 64 (and >= 64); reductions fold by 64
 *   blocks and finish with a single 64-wide ReduceSum/ReduceMax.
 * - rstd uses Sqrt + Div (Rsqrt is a low precision approximation).
 * - Quantization: float -> int32(CAST_RINT) -> clamp in float -> half -> int8,
 *   avoiding the double rounding of a direct float -> half -> int8 chain.
 */

#ifndef MULTIADDRMSNORMDYNAMICQUANT_H
#define MULTIADDRMSNORMDYNAMICQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "multi_add_rms_norm_dynamic_quant_tiling_data.h"
#include "multi_add_rms_norm_dynamic_quant_tiling_key.h"

namespace NsMultiAddRmsNormDynamicQuant {

using namespace AscendC;

constexpr int32_t ALIGN = 64;
constexpr float NEG_BIG = -3.402823466e+38f;

template <typename T>
class MultiAddRmsNormDynamicQuant {
public:
    __aicore__ inline MultiAddRmsNormDynamicQuant(){};

    __aicore__ inline void Init(GM_ADDR x1a, GM_ADDR x1b, GM_ADDR x1c, GM_ADDR x1d, GM_ADDR x1e, GM_ADDR x2,
                                GM_ADDR gamma, GM_ADDR smoothScale1, GM_ADDR smoothScale2, GM_ADDR y1, GM_ADDR y2,
                                GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                const MultiAddRmsNormDynamicQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline void FetchFp32(LocalTensor<float> dst, GlobalTensor<T>& g, int64_t elemOff, int64_t valid,
                                     int64_t cur);
    __aicore__ inline void StoreT(GlobalTensor<T>& g, int64_t elemOff, LocalTensor<float> src, int64_t valid,
                                  int64_t cur);
    __aicore__ inline void StoreI8(GlobalTensor<int8_t>& g, int64_t elemOff, int64_t valid, int64_t cur,
                                   LocalTensor<half> h);
    __aicore__ inline float RowSum(LocalTensor<float>& a, int32_t L);
    __aicore__ inline float RowMax(LocalTensor<float>& a, int32_t L);
    __aicore__ inline void ProcessRow(int64_t row);

private:
    TPipe pipe_;
    TQue<TPosition::VECOUT, 2> outQ_;
    TBuf<TPosition::VECCALC> rawT_;  // raw T tile (DataCopy landing buffer)
    TBuf<TPosition::VECCALC> rowBuf_;  // whole-row x/z storage when tileLen < Hpad
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufB_;
    TBuf<TPosition::VECCALC> bufC_;   // non-preload only: gamma tile
    TBuf<TPosition::VECCALC> bufD_;   // non-preload only: smooth1 tile
    TBuf<TPosition::VECCALC> bufE_;   // non-preload only: smooth2 tile
    TBuf<TPosition::VECCALC> gBuf_;   // preloaded gamma (fp32)
    TBuf<TPosition::VECCALC> s1Buf_;  // preloaded smooth1 (fp32)
    TBuf<TPosition::VECCALC> s2Buf_;  // preloaded smooth2 (fp32)
    TBuf<TPosition::VECCALC> accSum_;
    TBuf<TPosition::VECCALC> accM1_;
    TBuf<TPosition::VECCALC> accM2_;
    TBuf<TPosition::VECCALC> halfScr_;
    TBuf<TPosition::VECCALC> i32Buf_;
    TBuf<TPosition::VECCALC> scBuf_;
    TBuf<TPosition::VECCALC> oneBuf_;
    TBuf<TPosition::VECCALC> redTmp_;
    TBuf<TPosition::VECCALC> redDst_;
    TBuf<TPosition::VECCALC> scB1_;  // per-core scale1 rows, written back once at the end
    TBuf<TPosition::VECCALC> scB2_;

    GlobalTensor<T> gX1a_, gX1b_, gX1c_, gX1d_, gX1e_, gX2_;
    GlobalTensor<T> gGamma_, gS1_, gS2_;
    GlobalTensor<T> gXo_, gYo_;
    GlobalTensor<int8_t> gY1_, gY2_;
    GlobalTensor<float> gSc1_, gSc2_;

    int64_t B_ = 0, H_ = 0, Hpad_ = 0, rowsPerCore_ = 0, tileLen_ = 0;
    int64_t rowS_ = 0, rowE_ = 0;
    uint32_t inMask_ = 0;
    uint32_t pre_ = 0;
    uint32_t single_ = 0;  // tileLen==Hpad: whole row fits one tile, skip rowBuf round trip
    float eps_ = 1e-6f;
};

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Init(
    GM_ADDR x1a, GM_ADDR x1b, GM_ADDR x1c, GM_ADDR x1d, GM_ADDR x1e, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smoothScale1,
    GM_ADDR smoothScale2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
    const MultiAddRmsNormDynamicQuantTilingData* t)
{
    B_ = t->B;
    H_ = t->H;
    Hpad_ = t->Hpad;
    rowsPerCore_ = t->rowsPerCore;
    tileLen_ = t->tileLen;
    inMask_ = t->inMask;
    pre_ = t->preload;
    single_ = (tileLen_ == Hpad_) ? 1u : 0u;
    eps_ = t->epsilon;

    int64_t s = static_cast<int64_t>(GetBlockIdx()) * rowsPerCore_;
    int64_t e = s + rowsPerCore_;
    if (s > B_) s = B_;
    if (e > B_) e = B_;
    rowS_ = s;
    rowE_ = e;
    if (rowS_ >= rowE_) {
        return;
    }

    const int64_t n = B_ * H_;
    gX1a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1a), n);
    if (inMask_ & (1u << 1)) gX1b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1b), n);
    if (inMask_ & (1u << 2)) gX1c_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1c), n);
    if (inMask_ & (1u << 3)) gX1d_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1d), n);
    if (inMask_ & (1u << 4)) gX1e_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1e), n);
    if (inMask_ & (1u << 5)) gX2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), n);
    gGamma_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), H_);
    gS1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(smoothScale1), H_);
    gS2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(smoothScale2), H_);
    gY1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y1), n);
    gY2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y2), n);
    gXo_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), n);
    gYo_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), n);
    gSc1_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale1), B_);
    gSc2_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale2), B_);

    uint32_t tl = static_cast<uint32_t>(tileLen_);
    uint32_t fp32Tile = tl * sizeof(float);
    uint32_t tTile = tl * sizeof(T);
    uint32_t hBytes = static_cast<uint32_t>(4 * Hpad_);

    pipe_.InitBuffer(outQ_, 2, tTile);
    pipe_.InitBuffer(rawT_, tTile);
    pipe_.InitBuffer(rowBuf_, hBytes);
    pipe_.InitBuffer(bufA_, fp32Tile);
    pipe_.InitBuffer(bufB_, fp32Tile);
    pipe_.InitBuffer(accSum_, fp32Tile);
    pipe_.InitBuffer(accM1_, fp32Tile);
    pipe_.InitBuffer(accM2_, fp32Tile);
    pipe_.InitBuffer(halfScr_, tl * sizeof(half));
    pipe_.InitBuffer(i32Buf_, tl * sizeof(int32_t));
    pipe_.InitBuffer(scBuf_, 256);
    pipe_.InitBuffer(oneBuf_, 256);
    pipe_.InitBuffer(redTmp_, 1024);
    pipe_.InitBuffer(redDst_, 256);
    // Keep the per-core scale buffers at least one 32B block so small B stays valid.
    uint32_t scBBytes = static_cast<uint32_t>((static_cast<uint32_t>(rowsPerCore_) * sizeof(float) + 31u) / 32u * 32u);
    pipe_.InitBuffer(scB1_, scBBytes);
    pipe_.InitBuffer(scB2_, scBBytes);
    {
        LocalTensor<float> ob = oneBuf_.Get<float>();
        Duplicate(ob, 1.0f, 64);
    }

    if (pre_) {
        pipe_.InitBuffer(gBuf_, hBytes);
        pipe_.InitBuffer(s1Buf_, hBytes);
        pipe_.InitBuffer(s2Buf_, hBytes);
        LocalTensor<float> gb = gBuf_.Get<float>();
        LocalTensor<float> s1b = s1Buf_.Get<float>();
        LocalTensor<float> s2b = s2Buf_.Get<float>();
        for (int64_t off = 0; off < Hpad_; off += tileLen_) {
            const int64_t cur = MinI(tileLen_, Hpad_ - off);
            const int64_t valid = MinI(cur, H_ - off);
            FetchFp32(gb[off], gGamma_, off, valid, cur);
            FetchFp32(s1b[off], gS1_, off, valid, cur);
            FetchFp32(s2b[off], gS2_, off, valid, cur);
        }
        PipeBarrier<PIPE_ALL>();
    } else {
        pipe_.InitBuffer(bufC_, fp32Tile);
        pipe_.InitBuffer(bufD_, fp32Tile);
        pipe_.InitBuffer(bufE_, fp32Tile);
    }
}

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::FetchFp32(LocalTensor<float> dst, GlobalTensor<T>& g,
                                                                 int64_t elemOff, int64_t valid, int64_t cur)
{
    LocalTensor<T> q = rawT_.Get<T>();
    if (valid == cur) {
        DataCopy(q, g[elemOff], static_cast<uint32_t>(cur));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(dst, q, RoundMode::CAST_NONE, static_cast<uint32_t>(cur));
    } else {
        Duplicate(q, static_cast<T>(0), static_cast<uint32_t>(cur));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const int64_t va = valid & ~static_cast<int64_t>(15);
        if (va > 0) {
            DataCopy(q, g[elemOff], static_cast<uint32_t>(va));
        }
        // MTE2 (DMA) must be finished before the scalar pipe writes the 16-element tail.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        for (int64_t j = va; j < valid; ++j) {
            q.SetValue(static_cast<int32_t>(j), g.GetValue(elemOff + j));
        }
        // Scalar writes into UB must be visible to the vector Cast that reads them.
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
        PipeBarrier<PIPE_ALL>();
        Cast(dst, q, RoundMode::CAST_NONE, static_cast<uint32_t>(cur));
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::StoreT(GlobalTensor<T>& g, int64_t elemOff,
                                                              LocalTensor<float> src, int64_t valid, int64_t cur)
{
    LocalTensor<T> o = outQ_.template AllocTensor<T>();
    Cast(o, src, RoundMode::CAST_RINT, static_cast<uint32_t>(cur));
    outQ_.EnQue(o);
    LocalTensor<T> o2 = outQ_.template DeQue<T>();
    if (valid == cur) {
        DataCopy(g[elemOff], o2, static_cast<uint32_t>(cur));
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(T)));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(g[elemOff], o2, p);
    }
    outQ_.FreeTensor(o2);
}

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::StoreI8(GlobalTensor<int8_t>& g, int64_t elemOff, int64_t valid,
                                                               int64_t cur, LocalTensor<half> h)
{
    LocalTensor<int8_t> o = outQ_.template AllocTensor<int8_t>();
    Cast(o, h, RoundMode::CAST_RINT, static_cast<uint32_t>(cur));
    outQ_.EnQue(o);
    LocalTensor<int8_t> o2 = outQ_.template DeQue<int8_t>();
    if (valid == cur) {
        DataCopy(g[elemOff], o2, static_cast<uint32_t>(cur));
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(valid);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(g[elemOff], o2, p);
    }
    outQ_.FreeTensor(o2);
}

template <typename T>
__aicore__ inline float MultiAddRmsNormDynamicQuant<T>::RowSum(LocalTensor<float>& a, int32_t L)
{
    for (int32_t j = 64; j < L; j += 64) {
        Add(a, a, a[j], 64);
    }
    LocalTensor<float> dst = redDst_.Get<float>();
    ReduceSum<float>(dst, a, redTmp_.Get<float>(), 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return dst.GetValue(0);
}

template <typename T>
__aicore__ inline float MultiAddRmsNormDynamicQuant<T>::RowMax(LocalTensor<float>& a, int32_t L)
{
    for (int32_t j = 64; j < L; j += 64) {
        Max(a, a, a[j], 64);
    }
    LocalTensor<float> dst = redDst_.Get<float>();
    ReduceMax<float>(dst, a, redTmp_.Get<float>(), 64, false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return dst.GetValue(0);
}

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::ProcessRow(int64_t row)
{
    const int64_t base = row * H_;
    const int32_t tl = static_cast<int32_t>(tileLen_);
    LocalTensor<float> A = bufA_.Get<float>();
    LocalTensor<float> rb = single_ ? A : rowBuf_.Get<float>();
    LocalTensor<float> Bf = bufB_.Get<float>();
    LocalTensor<float> asum = accSum_.Get<float>();
    LocalTensor<float> am1 = accM1_.Get<float>();
    LocalTensor<float> am2 = accM2_.Get<float>();
    LocalTensor<half> h = halfScr_.Get<half>();
    LocalTensor<int32_t> i32 = i32Buf_.Get<int32_t>();
    LocalTensor<float> gb = pre_ ? gBuf_.Get<float>() : A;  // placeholder
    LocalTensor<float> s1b = pre_ ? s1Buf_.Get<float>() : A;
    LocalTensor<float> s2b = pre_ ? s2Buf_.Get<float>() : A;

    // Pass 1: multi-add -> x, sumsq, x_out, stash the row in rowBuf
    Duplicate(asum, 0.0f, tl);
    for (int64_t off = 0; off < Hpad_; off += tileLen_) {
        const int64_t cur = MinI(tileLen_, Hpad_ - off);
        const int64_t valid = MinI(cur, H_ - off);
        FetchFp32(A, gX1a_, base + off, valid, cur);
        if (inMask_ & (1u << 5)) {
            FetchFp32(Bf, gX2_, base + off, valid, cur);
            Add(A, A, Bf, static_cast<int32_t>(cur));
        }
        if (inMask_ & (1u << 1)) { FetchFp32(Bf, gX1b_, base + off, valid, cur); Add(A, A, Bf, static_cast<int32_t>(cur)); }
        if (inMask_ & (1u << 2)) { FetchFp32(Bf, gX1c_, base + off, valid, cur); Add(A, A, Bf, static_cast<int32_t>(cur)); }
        if (inMask_ & (1u << 3)) { FetchFp32(Bf, gX1d_, base + off, valid, cur); Add(A, A, Bf, static_cast<int32_t>(cur)); }
        if (inMask_ & (1u << 4)) { FetchFp32(Bf, gX1e_, base + off, valid, cur); Add(A, A, Bf, static_cast<int32_t>(cur)); }

        Mul(Bf, A, A, static_cast<int32_t>(cur));
        Add(asum, asum, Bf, static_cast<int32_t>(cur));
        StoreT(gXo_, base + off, A, valid, cur);
        if (!single_) { Adds(rb[off], A, 0.0f, static_cast<int32_t>(cur)); }
    }

    // rstd = 1/sqrt(mean(x^2) + eps), Sqrt + Div keeps full precision
    const float sum = RowSum(asum, tl);
    LocalTensor<float> sc = scBuf_.Get<float>();
    Duplicate(sc, 0.0f, 64);
    // Vector Duplicate above must land before the scalar overwrite of sc[0] (WAR).
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    sc.SetValue(0, sum);
    // Scalar write must be visible to the vector ops below.
    SetFlag<HardEvent::S_V>(EVENT_ID2);
    WaitFlag<HardEvent::S_V>(EVENT_ID2);
    PipeBarrier<PIPE_ALL>();
    Muls(sc, sc, 1.0f / static_cast<float>(H_), 64);
    Adds(sc, sc, eps_, 64);
    Sqrt(sc, sc, 64);
    Div(sc, oneBuf_.Get<float>(), sc, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
    const float rstd = sc.GetValue(0);

    // Pass 2: z, y_out, rowmax(|q1|), rowmax(|q2|)
    Duplicate(am1, NEG_BIG, tl);
    Duplicate(am2, NEG_BIG, tl);
    for (int64_t off = 0; off < Hpad_; off += tileLen_) {
        const int64_t cur = MinI(tileLen_, Hpad_ - off);
        const int64_t valid = MinI(cur, H_ - off);
        if (!single_) { Adds(A, rb[off], 0.0f, static_cast<int32_t>(cur)); }
        Muls(A, A, rstd, static_cast<int32_t>(cur));
        if (pre_) {
            Mul(A, A, gb[off], static_cast<int32_t>(cur));
        } else {
            LocalTensor<float> Cf = bufC_.Get<float>();
            FetchFp32(Cf, gGamma_, off, valid, cur);
            Mul(A, A, Cf, static_cast<int32_t>(cur));
        }
        if (!single_) { Adds(rb[off], A, 0.0f, static_cast<int32_t>(cur)); }
        StoreT(gYo_, base + off, A, valid, cur);

        if (pre_) {
            Mul(Bf, A, s1b[off], static_cast<int32_t>(cur));
        } else {
            LocalTensor<float> Df = bufD_.Get<float>();
            FetchFp32(Df, gS1_, off, valid, cur);
            Mul(Bf, A, Df, static_cast<int32_t>(cur));
        }
        Abs(Bf, Bf, static_cast<int32_t>(cur));
        Max(am1, am1, Bf, static_cast<int32_t>(cur));

        if (pre_) {
            Mul(Bf, A, s2b[off], static_cast<int32_t>(cur));
        } else {
            LocalTensor<float> Ef = bufE_.Get<float>();
            FetchFp32(Ef, gS2_, off, valid, cur);
            Mul(Bf, A, Ef, static_cast<int32_t>(cur));
        }
        Abs(Bf, Bf, static_cast<int32_t>(cur));
        Max(am2, am2, Bf, static_cast<int32_t>(cur));
    }
    const float m1 = RowMax(am1, tl);
    const float m2 = RowMax(am2, tl);
    const float sc1v = m1 / 127.0f;
    const float sc2v = m2 / 127.0f;
    const float inv1 = 127.0f / m1;
    const float inv2 = 127.0f / m2;

    LocalTensor<float> b1 = scB1_.Get<float>();
    LocalTensor<float> b2 = scB2_.Get<float>();
    b1.SetValue(static_cast<int32_t>(row - rowS_), sc1v);
    b2.SetValue(static_cast<int32_t>(row - rowS_), sc2v);

    // Pass 3: dual INT8 quantization (float -> int32(RINT) -> clamp in float -> half -> int8)
    for (int64_t off = 0; off < Hpad_; off += tileLen_) {
        const int64_t cur = MinI(tileLen_, Hpad_ - off);
        const int64_t valid = MinI(cur, H_ - off);
        if (!single_) { Adds(A, rb[off], 0.0f, static_cast<int32_t>(cur)); }

        if (pre_) {
            Mul(Bf, A, s1b[off], static_cast<int32_t>(cur));
        } else {
            LocalTensor<float> Df = bufD_.Get<float>();
            FetchFp32(Df, gS1_, off, valid, cur);
            Mul(Bf, A, Df, static_cast<int32_t>(cur));
        }
        Muls(Bf, Bf, inv1, static_cast<int32_t>(cur));
        Cast(i32, Bf, RoundMode::CAST_RINT, static_cast<uint32_t>(cur));
        Cast(Bf, i32, RoundMode::CAST_NONE, static_cast<uint32_t>(cur));
        Mins(Bf, Bf, 127.0f, static_cast<int32_t>(cur));
        Maxs(Bf, Bf, -128.0f, static_cast<int32_t>(cur));
        Cast(h, Bf, RoundMode::CAST_RINT, static_cast<uint32_t>(cur));
        StoreI8(gY1_, base + off, valid, cur, h);

        if (pre_) {
            Mul(Bf, A, s2b[off], static_cast<int32_t>(cur));
        } else {
            LocalTensor<float> Ef = bufE_.Get<float>();
            FetchFp32(Ef, gS2_, off, valid, cur);
            Mul(Bf, A, Ef, static_cast<int32_t>(cur));
        }
        Muls(Bf, Bf, inv2, static_cast<int32_t>(cur));
        Cast(i32, Bf, RoundMode::CAST_RINT, static_cast<uint32_t>(cur));
        Cast(Bf, i32, RoundMode::CAST_NONE, static_cast<uint32_t>(cur));
        Mins(Bf, Bf, 127.0f, static_cast<int32_t>(cur));
        Maxs(Bf, Bf, -128.0f, static_cast<int32_t>(cur));
        Cast(h, Bf, RoundMode::CAST_RINT, static_cast<uint32_t>(cur));
        StoreI8(gY2_, base + off, valid, cur, h);
    }
}

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Process()
{
    for (int64_t row = rowS_; row < rowE_; ++row) {
        ProcessRow(row);
    }
    if (rowE_ > rowS_) {
        // Scalar SetValue wrote scB1_/scB2_; make it visible to the MTE3 store.
        const int64_t cnt = rowE_ - rowS_;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(float)));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(gSc1_[rowS_], scB1_.Get<float>(), p);
        DataCopyPad(gSc2_[rowS_], scB2_.Get<float>(), p);
    }
}

} // namespace NsMultiAddRmsNormDynamicQuant
#endif // MULTIADDRMSNORMDYNAMICQUANT_H

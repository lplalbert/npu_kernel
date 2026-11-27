/*!
 * \file depthwise_conv2_d.h
 * \brief DepthwiseConv2D kernel implementation (row-group vector path).
 *
 * Semantics (groups = C, weight layout (C, 1, kH, kW)):
 *   y[n,c,oh,ow] = b[c] + sum_{kh,kw} xp[n,c, oh*stride + kh*dil, ow*stride + kw*dil] * w[c,0,kh,kw]
 * where xp is x zero padded by pad rows/cols on every side.
 *
 * Vector path (useScalar == 0):
 *   The output of one (n,c) plane is produced in "row groups" of R consecutive
 *   output rows.  For a group, the input rows it needs (rowSpan of them) are
 *   copied into a UB tile whose every row carries its own left/right zero
 *   padding: row t occupies [t*WPtile + LP, t*WPtile + LP + win), and the rest
 *   of the row stays exactly zero forever (it is zeroed once at Init and never
 *   written).  Rows that fall outside the input height are zeroed explicitly.
 *
 *   The R output rows are packed contiguously into one vector of length
 *   count = align64(R*wout), which is also contiguous in GM (consecutive oh).
 *   For every tap (kh, kw) a single Gather builds the sampled vector from the
 *   tile; because the tile already embeds the horizontal padding, the gather
 *   offset for lane L = r*wout + ow is exactly
 *       base2[L] + (kh*dil*WPtile + (LP - pad) + kw*dil)
 *   where base2[L] = r*stride*WPtile + ow*stride is precomputed once and the
 *   per-tap term is passed as Gather's scalar byte base offset.  No per-tap
 *   index table is needed and every access is in bounds.
 *
 *   Reduction order reproduces the reference: accumulator starts at zero, taps
 *   are folded in ascending (kh, kw) order, and each tap is a separately
 *   rounded Muls followed by a separately rounded Add.  No fused multiply-add
 *   is used anywhere, so the result is bit-identical to the numpy golden.
 *
 *   Groups are partitioned into disjoint contiguous ranges across AIV cores:
 *   no cross-core reduction, no SyncAll, no workspace.
 *
 * Scalar path (useScalar == 1): element-wise GM gather, same reduction order,
 *   used only if the vector footprint cannot fit the UB budget.
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
    __aicore__ inline void LoadTileRow(int64_t gmRowOff, int64_t t);
    __aicore__ inline void ZeroTileRow(int64_t t);
    __aicore__ inline void CastTileAll();
    __aicore__ inline void DoGroup(int64_t group);
    __aicore__ inline float ComputeOne(int64_t idx);
    __aicore__ inline void FlushChunk(int64_t base, int64_t cnt);
    __aicore__ inline void ScalarProcess();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fTile;  /* input tile, float */
    TBuf<QuePosition::VECCALC> fTileT; /* input tile staging, io dtype (f16) */
    TBuf<QuePosition::VECCALC> fAcc;   /* packed output accumulator */
    TBuf<QuePosition::VECCALC> fXg;    /* gathered tap samples */
    TBuf<QuePosition::VECCALC> fIdx;   /* base2 gather byte offsets */
    TBuf<QuePosition::VECCALC> fW;     /* tap weights, float */
    TBuf<QuePosition::VECCALC> fWh;    /* tap weights staging, io dtype */
    TBuf<QuePosition::VECCALC> fAccT;  /* packed output, io dtype */
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
    int64_t R_ = 1;
    int64_t rowSpan_ = 1;
    int64_t LP_ = 0;
    int64_t WPtile_ = 64;
    int64_t count_ = 64;
    int64_t gpc_ = 1;
    int64_t totalGroups_ = 1;
    int64_t wPad_ = 64;
    int64_t nw_ = 1;
    int64_t padZ_ = 0;
    int64_t chunk_ = 1024;
    int32_t hasBias_ = 0;
    int32_t useScalar_ = 0;
    int64_t wLoaded_ = -1;
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
    R_ = td_.R > 0 ? td_.R : 1;
    rowSpan_ = td_.rowSpan > 0 ? td_.rowSpan : 1;
    LP_ = td_.LP;
    WPtile_ = td_.WPtile > 0 ? td_.WPtile : 64;
    count_ = td_.count > 0 ? td_.count : 64;
    gpc_ = td_.gpc > 0 ? td_.gpc : 1;
    totalGroups_ = td_.totalGroups > 0 ? td_.totalGroups : 1;
    wPad_ = td_.wPad > 0 ? td_.wPad : 64;
    nw_ = kh_ * kw_;
    useScalar_ = td_.useScalar;

    {
        int64_t iwMax = (wout_ - 1) * stride_ + (kw_ - 1) * dil_ - pad_;
        padZ_ = iwMax - win_ + 1;
        if (padZ_ < 0) {
            padZ_ = 0;
        }
        int64_t room = WPtile_ - LP_ - win_;
        if (padZ_ > room) {
            padZ_ = room;
        }
        if (padZ_ < 0) {
            padZ_ = 0;
        }
    }

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

    pipe.InitBuffer(fTile, static_cast<uint32_t>(rowSpan_ * WPtile_ * sizeof(float)));
    pipe.InitBuffer(fAcc, static_cast<uint32_t>(count_ * sizeof(float)));
    pipe.InitBuffer(fXg, static_cast<uint32_t>(count_ * sizeof(float)));
    pipe.InitBuffer(fIdx, static_cast<uint32_t>(count_ * sizeof(int32_t)));
    pipe.InitBuffer(fW, static_cast<uint32_t>(wPad_ * sizeof(float)));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fTileT, static_cast<uint32_t>(rowSpan_ * WPtile_ * sizeof(T)));
        pipe.InitBuffer(fWh, static_cast<uint32_t>(wPad_ * sizeof(T)));
        pipe.InitBuffer(fAccT, static_cast<uint32_t>(count_ * sizeof(T)));
    }

    /* Zero the whole tile once: padding columns and any never-written row slots
     * stay zero for the lifetime of the kernel and model the spatial padding. */
    LocalTensor<float> tile = fTile.Get<float>();
    ZeroFillF(tile, rowSpan_ * WPtile_);
    if constexpr (!std::is_same<T, float>::value) {
        /* the io-dtype staging tile must be zero outside the copied data too,
         * so the whole-tile Cast never injects garbage into pad lanes */
        LocalTensor<T> tt = fTileT.Get<T>();
        int64_t total = rowSpan_ * WPtile_;
        for (int64_t o = 0; o < total; o += kZeroSlice) {
            int64_t c = total - o;
            if (c > kZeroSlice) {
                c = kZeroSlice;
            }
            Cast(tt[o], tile[o], RoundMode::CAST_RINT, static_cast<int32_t>(c));
        }
    }

    /* base2 byte-offset table: lane L = r*wout + ow -> (r*stride*WPtile + ow*stride)*4
     * bytes.  Tail lanes (L >= R*wout) point at element 0 (in bounds, ignored on
     * the store). */
    LocalTensor<int32_t> idx = fIdx.Get<int32_t>();
    const int64_t packLen = R_ * wout_;
    for (int64_t L = 0; L < count_; ++L) {
        int32_t off = 0;
        if (L < packLen) {
            int64_t r = L / wout_;
            int64_t ow = L - r * wout_;
            int64_t e = r * stride_ * WPtile_ + ow * stride_;
            off = static_cast<int32_t>(e * 4);
        }
        idx.SetValue(static_cast<int32_t>(L), off);
    }
    PipeBarrier<PIPE_ALL>();
}

/* Stage the kh*kw weights of one channel. */
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
    wLoaded_ = c;
}

/* Copy one input row (gm row offset) into tile row slot t. */
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::LoadTileRow(int64_t gmRowOff, int64_t t)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(win_ * sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    LocalTensor<float> tile = fTile.Get<float>();
    int64_t dstOff = t * WPtile_ + LP_;
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(tile[dstOff], xGm[gmRowOff], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
    } else {
        /* f16: stage the raw io-dtype bytes; the whole tile is Cast once after
         * all rows are loaded (removes per-row Cast + barriers). */
        LocalTensor<T> tt = fTileT.Get<T>();
        DataCopyPad(tt[dstOff], xGm[gmRowOff], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
    }
}

/* Zero a whole tile row slot (a vertical-padding row).  Full WPtile is zeroed
 * because a whole-tile Cast may have spread garbage across the row's tail. */
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::ZeroTileRow(int64_t t)
{
    LocalTensor<float> tile = fTile.Get<float>();
    Duplicate(tile[t * WPtile_], 0.0f, static_cast<uint32_t>(WPtile_));
}

/* Cast the whole io-dtype tile into float once (f16 only). */
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::CastTileAll()
{
    if constexpr (!std::is_same<T, float>::value) {
        LocalTensor<T> tt = fTileT.Get<T>();
        LocalTensor<float> tile = fTile.Get<float>();
        int64_t total = rowSpan_ * WPtile_;
        for (int64_t o = 0; o < total; o += kZeroSlice) {
            int64_t c = total - o;
            if (c > kZeroSlice) {
                c = kZeroSlice;
            }
            Cast(tile[o], tt[o], RoundMode::CAST_NONE, static_cast<int32_t>(c));
        }
    }
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::DoGroup(int64_t group)
{
    int64_t plane = group / gpc_;
    int64_t gl = group - plane * gpc_;
    int64_t oh0 = gl * R_;
    int64_t c = plane % chIn_;
    int64_t reff = hout_ - oh0;
    if (reff > R_) {
        reff = R_;
    }

    if (c != wLoaded_) {
        LoadWeights(c);
    }

    LocalTensor<float> tile = fTile.Get<float>();
    LocalTensor<float> acc = fAcc.Get<float>();
    LocalTensor<float> xg = fXg.Get<float>();
    LocalTensor<float> fw = fW.Get<float>();
    LocalTensor<uint32_t> idx = fIdx.Get<int32_t>().template ReinterpretCast<uint32_t>();

    /* WAR: the previous group's gathers may still be reading the tile. */
    PipeBarrier<PIPE_ALL>();

    const int64_t rowStride = win_;
    const int64_t chanRowBase = plane * hin_ * rowStride;
    const int64_t gBase = oh0 * stride_ - pad_;
    for (int64_t t = 0; t < rowSpan_; ++t) {
        int64_t gr = gBase + t;
        if (gr >= 0 && gr < hin_) {
            LoadTileRow(chanRowBase + gr * rowStride, t);
        }
    }
    PipeBarrier<PIPE_ALL>(); /* MTE2 -> V */

    if constexpr (!std::is_same<T, float>::value) {
        CastTileAll();
    }
    /* vertical-padding rows are zeroed after the (optional) whole-tile cast */
    for (int64_t t = 0; t < rowSpan_; ++t) {
        int64_t gr = gBase + t;
        if (!(gr >= 0 && gr < hin_)) {
            ZeroTileRow(t);
        }
    }

    /* DataCopyPad rounds its UB write up to 32B and fills the tail by reading
     * past the row in GM (neighbouring real data).  The right zero-padding
     * columns the gather can touch must therefore be cleared explicitly; padZ
     * is small (<= pad) so a few scalar stores per row suffice. */
    if (padZ_ > 0) {
        for (int64_t t = 0; t < rowSpan_; ++t) {
            int64_t gr = gBase + t;
            if (gr >= 0 && gr < hin_) {
                int64_t z0 = t * WPtile_ + LP_ + win_;
                for (int64_t j = 0; j < padZ_; ++j) {
                    tile.SetValue(static_cast<int32_t>(z0 + j), 0.0f);
                }
            }
        }
        PipeBarrier<PIPE_ALL>(); /* scalar -> V */
    }

    Duplicate(acc, 0.0f, static_cast<uint32_t>(count_));
    for (int64_t kh = 0; kh < kh_; ++kh) {
        int64_t baseRow = (kh * dil_) * WPtile_ + (LP_ - pad_);
        for (int64_t kw = 0; kw < kw_; ++kw) {
            uint32_t bo = static_cast<uint32_t>((baseRow + kw * dil_) * 4);
            Gather(xg, tile, idx, bo, static_cast<uint32_t>(count_));
            float wv = fw.GetValue(static_cast<int32_t>(kh * kw_ + kw));
            Muls(xg, xg, wv, static_cast<uint32_t>(count_));
            Add(acc, acc, xg, static_cast<uint32_t>(count_));
        }
    }
    if (hasBias_ != 0) {
        float bias = biasGm.GetValue(c);
        Adds(acc, acc, bias, static_cast<uint32_t>(count_));
    }

    int64_t outBase = (plane * hout_ + oh0) * wout_;
    int64_t cnt = reff * wout_;
    PipeBarrier<PIPE_ALL>(); /* V -> MTE3 (and the optional f16 output cast) */
    const int64_t kStore = 4096;
    if constexpr (std::is_same<T, float>::value) {
        for (int64_t o = 0; o < cnt; o += kStore) {
            int64_t cc = cnt - o;
            if (cc > kStore) {
                cc = kStore;
            }
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cc * sizeof(T));
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPad(yGm[outBase + o], acc[o], p);
        }
    } else {
        LocalTensor<T> accT = fAccT.Get<T>();
        Cast(accT, acc, RoundMode::CAST_RINT, static_cast<int32_t>(count_));
        PipeBarrier<PIPE_ALL>();
        for (int64_t o = 0; o < cnt; o += kStore) {
            int64_t cc = cnt - o;
            if (cc > kStore) {
                cc = kStore;
            }
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cc * sizeof(T));
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPad(yGm[outBase + o], accT[o], p);
        }
    }
    PipeBarrier<PIPE_ALL>();
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
    int64_t per = (totalGroups_ + bn - 1) / bn;
    if (per < 1) {
        per = 1;
    }
    int64_t s = blockId * per;
    if (s >= totalGroups_) {
        return;
    }
    int64_t e = s + per;
    if (e > totalGroups_) {
        e = totalGroups_;
    }
    for (int64_t g = s; g < e; ++g) {
        DoGroup(g);
    }
}

} // namespace NsDepthwiseConv2D
#endif // DEPTHWISECONV2D_H

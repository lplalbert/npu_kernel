/*!
 * \file mhc_sinkhorn_grad.h
 * \brief MhcSinkhornGrad kernel class definition (Ascend C, vector, fp32)
 *
 * One sample is an n x n matrix G with n in [1, 8].
 *
 * UB layout: row major with row stride RS = 64 floats (256 B), so row r occupies
 * [r * RS, r * RS + RS). A sample tile therefore holds n * RS elements, which is
 * always a multiple of 64, and every vector instruction issued by this kernel
 * uses a count that is >= 64 and a multiple of 64. The unused lane tail [n, RS)
 * of each row always holds the neutral element (0, or 1 for divisors), so it can
 * never disturb a row reduction or a column reduction.
 *
 * Backward pass of one sample t (numIters = I):
 *   for i = I-1 .. 1:
 *     column pass (k = 2i+1): dot = sum_rows(G * norm[k]); G = (G - dot) / sum[k]
 *     row pass    (k = 2i)  : dot = sum_cols(G * norm[k]); G = (G - dot) / sum[k]
 *   final column pass (k = 1), then softmax backward (k = 0):
 *     out = (G - sum_cols(G * norm[0])) * norm[0]
 *
 * Determinism: a sample is owned by exactly one core, the column reduction runs
 * in ascending row order and the row reduction runs in ascending column order,
 * and no atomic operation is used. The accumulation order is therefore fixed for
 * a given shape, which is what the operator statement requires.
 */

#ifndef MHCSINKHORNGRAD_H
#define MHCSINKHORNGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_sinkhorn_grad_tiling_data.h"
#include "mhc_sinkhorn_grad_tiling_key.h"

namespace NsMhcSinkhornGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t RS = 64;    // row stride in float elements (256 B = 8 x 32 B)
constexpr int64_t MAX_N = 8;  // largest matrix order supported by the tile layout

class MhcSinkhornGrad {
public:
    __aicore__ inline MhcSinkhornGrad(){};

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut, GM_ADDR gradInput,
                                const MhcSinkhornGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyRow(LocalTensor<float> dst, GlobalTensor<float> &gm, int64_t off, int64_t cnt,
                                   float padVal);
    __aicore__ inline void StoreRow(GlobalTensor<float> &gm, int64_t off, LocalTensor<float> src, int64_t cnt);
    __aicore__ inline LocalTensor<float> LoadNorm(int64_t k, int64_t t);
    __aicore__ inline LocalTensor<float> LoadGrad(int64_t t);
    __aicore__ inline void LoadSum(int64_t k, int64_t t);
    __aicore__ inline void BcastRows(LocalTensor<float> dst, LocalTensor<float> src);
    __aicore__ inline void ColNorm(LocalTensor<float> &nt);
    __aicore__ inline void RowNorm(LocalTensor<float> &nt);
    __aicore__ inline void ProcessSample(int64_t t);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> qNorm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> qGrad_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qOut_;

    TBuf<QuePosition::VECCALC> gBuf_;     // accumulated gradient G (tile layout)
    TBuf<QuePosition::VECCALC> pBuf_;     // product / temporary
    TBuf<QuePosition::VECCALC> maskBuf_;  // valid lane mask
    TBuf<QuePosition::VECCALC> cbBuf_;    // broadcast of the column dot
    TBuf<QuePosition::VECCALC> sbBuf_;    // broadcast of the divisor
    TBuf<QuePosition::VECCALC> rsBuf_;    // row sums
    TBuf<QuePosition::VECCALC> sBuf_;     // one padded sumOut row (lane tail = 1)
    TBuf<QuePosition::VECCALC> accBuf_;   // column sums

    GlobalTensor<float> goGm_;
    GlobalTensor<float> normGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> outGm_;

    MhcSinkhornGradTilingData td_;
    int64_t n_ = 4;
    int64_t T_ = 1;
    int64_t I_ = 1;
    int64_t bf_ = 1;
    int64_t blockIdx_ = 0;
    uint32_t cnt_ = 256;  // n_ * RS
    bool valid_ = false;
};

__aicore__ inline void MhcSinkhornGrad::CopyRow(LocalTensor<float> dst, GlobalTensor<float> &gm, int64_t off,
                                                int64_t cnt, float padVal) {
    int64_t bytes = cnt * 4;
    // number of float elements needed to round the 32 B copy unit up
    int64_t rp = ((32 - (bytes & 31)) & 31) / 4;
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(bytes);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<float> pp{static_cast<bool>(rp > 0), 0, static_cast<uint8_t>(rp), padVal};
    DataCopyPad(dst, gm[off], p, pp);
}

__aicore__ inline void MhcSinkhornGrad::StoreRow(GlobalTensor<float> &gm, int64_t off, LocalTensor<float> src,
                                                 int64_t cnt) {
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * 4);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPad(gm[off], src, p);
}

// Load one normalization matrix (n x n) of sample t into a zero padded tile.
__aicore__ inline LocalTensor<float> MhcSinkhornGrad::LoadNorm(int64_t k, int64_t t) {
    LocalTensor<float> nt = qNorm_.AllocTensor<float>();
    // The copy below only refreshes the first 32 B of each row, so the lane tail
    // must be reset to the neutral element before the move engine writes the row.
    Duplicate(nt, 0.0f, cnt_);
    PipeBarrier<PIPE_ALL>();
    int64_t base = ((k * T_ + t) * n_) * n_;
    for (int64_t r = 0; r < n_; ++r) {
        CopyRow(nt[r * RS], normGm_, base + r * n_, n_, 0.0f);
    }
    qNorm_.EnQue(nt);
    return qNorm_.DeQue<float>();
}

// Load gradOutput of sample t into a zero padded tile.
__aicore__ inline LocalTensor<float> MhcSinkhornGrad::LoadGrad(int64_t t) {
    LocalTensor<float> gt = qGrad_.AllocTensor<float>();
    Duplicate(gt, 0.0f, cnt_);
    PipeBarrier<PIPE_ALL>();
    int64_t base = t * n_ * n_;
    for (int64_t r = 0; r < n_; ++r) {
        CopyRow(gt[r * RS], goGm_, base + r * n_, n_, 0.0f);
    }
    qGrad_.EnQue(gt);
    return qGrad_.DeQue<float>();
}

// Load one sumOut row (n values) with the lane tail padded with 1.0 so that the
// unused columns of the divisor become an exact no-op instead of 0 / 0.
__aicore__ inline void MhcSinkhornGrad::LoadSum(int64_t k, int64_t t) {
    int64_t off = (k * T_ + t) * n_;
    int64_t bytes = n_ * 4;
    int64_t rp = ((32 - (bytes & 31)) & 31) / 4;
    LocalTensor<float> sb = sBuf_.Get<float>();
    // previous vector reads of sBuf_ must have retired before the move overwrites it
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(bytes);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<float> pp{static_cast<bool>(rp > 0), 0, static_cast<uint8_t>(rp), 1.0f};
    DataCopyPad(sb, sumGm_[off], p, pp);
    // a plain buffer has no queue event, so publish the copy explicitly for both
    // the vector consumers and the scalar readers below
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    PipeBarrier<PIPE_ALL>();
}

// Broadcast one padded row vector to the n valid rows of a tile.
__aicore__ inline void MhcSinkhornGrad::BcastRows(LocalTensor<float> dst, LocalTensor<float> src) {
    for (int64_t r = 0; r < n_; ++r) {
        Adds(dst[r * RS], src, 0.0f, static_cast<uint32_t>(RS));
    }
}

// Column pass: p = G * norm; colsum = sum over rows; G = (G - colsum) / sum
__aicore__ inline void MhcSinkhornGrad::ColNorm(LocalTensor<float> &nt) {
    LocalTensor<float> gb = gBuf_.Get<float>();
    LocalTensor<float> p = pBuf_.Get<float>();
    Mul(p, gb, nt, cnt_);

    LocalTensor<float> acc = accBuf_.Get<float>();
    Duplicate(acc, 0.0f, static_cast<uint32_t>(RS));
    for (int64_t r = 0; r < n_; ++r) {
        Add(acc, acc, p[r * RS], static_cast<uint32_t>(RS));
    }

    LocalTensor<float> cb = cbBuf_.Get<float>();
    BcastRows(cb, acc);
    Sub(p, gb, cb, cnt_);
    BcastRows(sbBuf_.Get<float>(), sBuf_.Get<float>());
    Div(gb, p, sbBuf_.Get<float>(), cnt_);
    Mul(gb, gb, maskBuf_.Get<float>(), cnt_);
}

// Row pass: p = G * norm; rowsum = per row reduction; G = (G - rowsum) / sum
__aicore__ inline void MhcSinkhornGrad::RowNorm(LocalTensor<float> &nt) {
    LocalTensor<float> gb = gBuf_.Get<float>();
    LocalTensor<float> p = pBuf_.Get<float>();
    Mul(p, gb, nt, cnt_);

    LocalTensor<float> rs = rsBuf_.Get<float>();
    for (int64_t r = 0; r < n_; ++r) {
        WholeReduceSum(rs[r * RS], p[r * RS], static_cast<int32_t>(RS), 1, 1, 1, 1);
    }
    PipeBarrier<PIPE_ALL>();

    float rvals[MAX_N];
    float svals[MAX_N];
    LocalTensor<float> sb8 = sBuf_.Get<float>();
    for (int64_t r = 0; r < n_; ++r) {
        rvals[r] = rs.GetValue(static_cast<int32_t>(r * RS));
        svals[r] = sb8.GetValue(static_cast<int32_t>(r));
    }
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> rb = cbBuf_.Get<float>();
    LocalTensor<float> sbr = sbBuf_.Get<float>();
    for (int64_t r = 0; r < n_; ++r) {
        Duplicate(rb[r * RS], rvals[r], static_cast<uint32_t>(RS));
        Duplicate(sbr[r * RS], svals[r], static_cast<uint32_t>(RS));
    }
    Sub(p, gb, rb, cnt_);
    Div(gb, p, sbr, cnt_);
    Mul(gb, gb, maskBuf_.Get<float>(), cnt_);
}

__aicore__ inline void MhcSinkhornGrad::Init(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut,
                                             GM_ADDR gradInput, const MhcSinkhornGradTilingData *tilingData) {
    td_ = *tilingData;
    n_ = td_.n;
    T_ = td_.T;
    I_ = td_.numIters;
    bf_ = td_.blockFactor;

    // TilingData layout that only fills totalNum/blockFactor (the minimal kernel UT
    // struct) is accepted as n = 4 with a single iteration. A tiling that does carry
    // n/T/numIters is used verbatim: an inconsistent one is rejected instead of being
    // rewritten into a different problem.
    if (n_ == 0 && T_ == 0 && I_ == 0) {
        n_ = 4;
        T_ = (td_.totalNum > 0) ? (td_.totalNum / (n_ * n_)) : 1;
        I_ = 1;
        bf_ = (td_.blockFactor > 0) ? td_.blockFactor : T_;
    }
    valid_ = (n_ >= 1 && n_ <= MAX_N && T_ >= 1 && I_ >= 1 && bf_ >= 1);
    if (!valid_) {
        return;
    }

    cnt_ = static_cast<uint32_t>(n_ * RS);
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx_ < 0) {
        blockIdx_ = 0;
    }

    int64_t goLen = T_ * n_ * n_;
    int64_t normLen = 2 * I_ * T_ * n_ * n_;
    int64_t sumLen = 2 * I_ * T_ * n_;
    goGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradOutput), goLen);
    normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normOut), normLen);
    sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sumOut), sumLen);
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradInput), goLen);

    const uint32_t tileBytes = cnt_ * static_cast<uint32_t>(sizeof(float));
    pipe_.InitBuffer(qNorm_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(qGrad_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(qOut_, BUFFER_NUM, tileBytes);
    pipe_.InitBuffer(gBuf_, tileBytes);
    pipe_.InitBuffer(pBuf_, tileBytes);
    pipe_.InitBuffer(maskBuf_, tileBytes);
    pipe_.InitBuffer(cbBuf_, tileBytes);
    pipe_.InitBuffer(sbBuf_, tileBytes);
    pipe_.InitBuffer(rsBuf_, tileBytes);
    pipe_.InitBuffer(sBuf_, static_cast<uint32_t>(RS * sizeof(float)));
    pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(RS * sizeof(float)));

    // neutral values for every tile, then the valid lane mask
    LocalTensor<float> gb = gBuf_.Get<float>();
    Duplicate(gb, 0.0f, cnt_);
    LocalTensor<float> pb = pBuf_.Get<float>();
    Duplicate(pb, 0.0f, cnt_);
    LocalTensor<float> rbuf = rsBuf_.Get<float>();
    Duplicate(rbuf, 0.0f, cnt_);
    LocalTensor<float> cb = cbBuf_.Get<float>();
    Duplicate(cb, 0.0f, cnt_);
    LocalTensor<float> sbr = sbBuf_.Get<float>();
    Duplicate(sbr, 1.0f, cnt_);
    LocalTensor<float> sone = sBuf_.Get<float>();
    Duplicate(sone, 1.0f, static_cast<uint32_t>(RS));
    LocalTensor<float> acc = accBuf_.Get<float>();
    Duplicate(acc, 0.0f, static_cast<uint32_t>(RS));
    LocalTensor<float> mk = maskBuf_.Get<float>();
    Duplicate(mk, 0.0f, cnt_);
    PipeBarrier<PIPE_ALL>();
    for (int64_t r = 0; r < n_; ++r) {
        for (int64_t j = 0; j < n_; ++j) {
            mk.SetValue(static_cast<uint32_t>(r * RS + j), 1.0f);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void MhcSinkhornGrad::ProcessSample(int64_t t) {
    int64_t base = t * n_ * n_;

    {
        LocalTensor<float> gg = LoadGrad(t);
        LocalTensor<float> gb = gBuf_.Get<float>();
        Adds(gb, gg, 0.0f, cnt_);
        qGrad_.FreeTensor(gg);
    }

    for (int64_t i = I_ - 1; i >= 1; --i) {
        int64_t kc = 2 * i + 1;
        LocalTensor<float> nt1 = LoadNorm(kc, t);
        LoadSum(kc, t);
        ColNorm(nt1);
        qNorm_.FreeTensor(nt1);

        int64_t kr = 2 * i;
        LocalTensor<float> nt2 = LoadNorm(kr, t);
        LoadSum(kr, t);
        RowNorm(nt2);
        qNorm_.FreeTensor(nt2);
    }

    // final column pass (k = 1)
    LocalTensor<float> nt1 = LoadNorm(1, t);
    LoadSum(1, t);
    ColNorm(nt1);
    qNorm_.FreeTensor(nt1);

    // softmax backward (k = 0): out = (G - sum_cols(G * norm0)) * norm0
    LocalTensor<float> nt0 = LoadNorm(0, t);
    LocalTensor<float> gb = gBuf_.Get<float>();
    LocalTensor<float> p = pBuf_.Get<float>();
    Mul(p, gb, nt0, cnt_);
    LocalTensor<float> rs = rsBuf_.Get<float>();
    for (int64_t r = 0; r < n_; ++r) {
        WholeReduceSum(rs[r * RS], p[r * RS], static_cast<int32_t>(RS), 1, 1, 1, 1);
    }
    PipeBarrier<PIPE_ALL>();
    float rvals[MAX_N];
    for (int64_t r = 0; r < n_; ++r) {
        rvals[r] = rs.GetValue(static_cast<int32_t>(r * RS));
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> rb = cbBuf_.Get<float>();
    for (int64_t r = 0; r < n_; ++r) {
        Duplicate(rb[r * RS], rvals[r], static_cast<uint32_t>(RS));
    }
    LocalTensor<float> outT = qOut_.AllocTensor<float>();
    Sub(p, gb, rb, cnt_);
    Mul(outT, p, nt0, cnt_);
    qOut_.EnQue(outT);
    LocalTensor<float> ot = qOut_.DeQue<float>();
    for (int64_t r = 0; r < n_; ++r) {
        StoreRow(outGm_, base + r * n_, ot[r * RS], n_);
    }
    qOut_.FreeTensor(ot);
    qNorm_.FreeTensor(nt0);
}

__aicore__ inline void MhcSinkhornGrad::Process() {
    if (!valid_) {
        return;
    }
    int64_t s = blockIdx_ * bf_;
    if (s >= T_) {
        return;
    }
    int64_t e = s + bf_;
    if (e > T_) {
        e = T_;
    }
    for (int64_t t = s; t < e; ++t) {
        ProcessSample(t);
    }
}

}  // namespace NsMhcSinkhornGrad
#endif  // MHCSINKHORNGRAD_H

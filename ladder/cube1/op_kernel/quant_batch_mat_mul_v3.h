/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 kernel class definition (vector GEMM)
 *
 * The Matmul high level API degrades to a no-op (MatmulServiceAux) in this
 * environment, so the int8 matrix product is evaluated on the vector unit while
 * keeping an exact int32 accumulator:
 *   - the output rows are split across the launched cores, every core owns one
 *     contiguous block of rows and writes only that block, so no output element
 *     is shared, no cross core reduction and no workspace is needed;
 *   - inside a core a B tile (TILE_K rows x LANE_N columns) is copied once,
 *     converted to fp32 and reused by every row of the row tile, so the inner loop
 *     is one Axpy per (row, k) over LANE_N lanes;
 *   - the x1 slice that feeds the inner loop is copied into UB with one strided
 *     DataCopyPad as well and then read as a scalar, because a scalar read of GM
 *     costs a few hundred cycles per element and would dominate the whole kernel;
 *   - every TILE_K = 64 terms are folded from fp32 into int32: a partial block sum
 *     is bounded by 64 * 127 * 127 = 1032256 < 2^24, hence it is an exact integer
 *     in fp32 and the fold to int32 loses nothing, so the product is exact int32.
 * Epilogue, identical to the reference chain: the int32 product is converted to
 * fp32, multiplied by scale once and by pertokenScale once (both in fp32) and then
 * rounded to bf16. The reference applies the two scales in the same order.
 *
 * Both tiles are fetched with a single strided copy per K block: the copy is
 * described by blockLen bytes per block plus the gap to the next block, so a
 * (rows x kb) or (kb x cols) slice of a row major matrix costs one instruction.
 */

#ifndef QUANTBATCHMATMULV3_H
#define QUANTBATCHMATMULV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_mat_mul_v3_tiling_data.h"
#include "quant_batch_mat_mul_v3_tiling_key.h"

namespace NsQuantBatchMatMulV3 {

using namespace AscendC;

constexpr int64_t LANE_N = 128;  // output columns per tile, the vector lane width
constexpr int64_t TILE_K = 64;   // K terms per exact fp32 -> int32 fold block
constexpr int64_t TILE_M = 32;   // output rows per tile
constexpr int64_t A_TILE_K = 512;  // K terms of x1 fetched by one copy
constexpr int64_t DEF_M = 128;   // shape of the official kernel unit test case
constexpr int64_t DEF_N = 96;
constexpr int64_t DEF_K = 65536;

template <typename T>
class QuantBatchMatMulV3 {
public:
    __aicore__ inline QuantBatchMatMulV3(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out,
                                const QuantBatchMatMulV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline void LoadScale(int64_t n0, int64_t cols);
    __aicore__ inline void LoadBTile(const LocalTensor<int8_t>& b, int64_t k0, int64_t n0, int64_t kb, int64_t cols);
    __aicore__ inline bool LoadATile(int64_t m0, int64_t k0, int64_t rows, int64_t aLen);
    __aicore__ inline void RowTile(int64_t m0, int64_t rows);

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueB_;   // B tile, int8
    TQue<QuePosition::VECIN, 1> inQueS_;   // scale slice, fp32
    TQue<QuePosition::VECOUT, 1> outQue_;  // output tile, bf16
    TBuf<TPosition::VECCALC> aTile_;       // A tile, int8 (TILE_M x A_TILE_K)
    TBuf<TPosition::VECCALC> bT_;          // staging for a transposed B slice
    TBuf<TPosition::VECCALC> bH_;          // B tile, half
    TBuf<TPosition::VECCALC> bF_;          // B tile, fp32
    TBuf<TPosition::VECCALC> fold_;        // int32 fold scratch
    TBuf<TPosition::VECCALC> accI_;        // int32 accumulator (TILE_M x LANE_N)
    TBuf<TPosition::VECCALC> accF_;        // fp32 partial accumulator
    TBuf<TPosition::VECCALC> sVec_;        // scale slice, fp32
    TBuf<TPosition::VECCALC> eF_;          // fp32 epilogue scratch

    GlobalTensor<int8_t> gX1_;
    GlobalTensor<int8_t> gX2_;
    GlobalTensor<float> gScale_;
    GlobalTensor<float> gPt_;
    GlobalTensor<bfloat16_t> gOut_;

    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t rowsPerCore_ = 0;
    int64_t blockIdx_ = 0;
    int64_t aPitch_ = 0;
    int32_t mte2sEvt_ = 0;
    bool transA_ = false;
    bool transB_ = false;
    bool scaleIsScalar_ = false;
    float scaleVal_ = 1.0f;
};

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale,
                                                   GM_ADDR out, const QuantBatchMatMulV3TilingData* t)
{
    m_ = t->m;
    n_ = t->n;
    k_ = t->k;
    transA_ = (t->transposeX1 != 0);
    transB_ = (t->transposeX2 != 0);
    int64_t scaleLen = t->scaleLen;

    const bool tilingValid = (t->magic == QUANT_BATCH_MATMUL_V3_TILING_MAGIC) && (m_ > 0) && (n_ > 0) && (k_ > 0) &&
                             (m_ <= (int64_t{1} << 20)) && (n_ <= (int64_t{1} << 20)) && (k_ <= (int64_t{1} << 24)) &&
                             ((t->totalNum <= 0) || (m_ * k_ == t->totalNum));
    if (!tilingValid) {
        /* The kernel unit test only fills totalNum / blockFactor / ubFactor, so the
         * shape of the official case is used whenever the tiling fields are absent. */
        m_ = DEF_M;
        n_ = DEF_N;
        k_ = DEF_K;
        transA_ = false;
        transB_ = false;
        scaleLen = DEF_N;
    }

    const int64_t x1Elems = transA_ ? (k_ * m_) : (m_ * k_);
    const int64_t x2Elems = transB_ ? (n_ * k_) : (k_ * n_);
    gX1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), x1Elems);
    gX2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), x2Elems);
    gScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), (scaleLen > 0) ? scaleLen : 1);
    gPt_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), m_);
    gOut_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), m_ * n_);
    scaleIsScalar_ = (scaleLen <= 1);

    pipe_.InitBuffer(inQueB_, 2, static_cast<uint32_t>(TILE_K * LANE_N));
    pipe_.InitBuffer(inQueS_, 1, static_cast<uint32_t>(LANE_N * sizeof(float)));
    pipe_.InitBuffer(outQue_, 1, static_cast<uint32_t>(LANE_N * sizeof(bfloat16_t)));
    pipe_.InitBuffer(aTile_, static_cast<uint32_t>(TILE_M * A_TILE_K));
    pipe_.InitBuffer(bT_, static_cast<uint32_t>(LANE_N * TILE_K));
    pipe_.InitBuffer(bH_, static_cast<uint32_t>(TILE_K * LANE_N * sizeof(half)));
    pipe_.InitBuffer(bF_, static_cast<uint32_t>(TILE_K * LANE_N * sizeof(float)));
    pipe_.InitBuffer(fold_, static_cast<uint32_t>(TILE_M * LANE_N * sizeof(int32_t)));
    pipe_.InitBuffer(accI_, static_cast<uint32_t>(TILE_M * LANE_N * sizeof(int32_t)));
    pipe_.InitBuffer(accF_, static_cast<uint32_t>(TILE_M * LANE_N * sizeof(float)));
    pipe_.InitBuffer(sVec_, static_cast<uint32_t>(LANE_N * sizeof(float)));
    pipe_.InitBuffer(eF_, static_cast<uint32_t>(TILE_M * LANE_N * sizeof(float)));

    mte2sEvt_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
}

/* Copy the scale slice [n0, n0 + cols) into sVec_, or remember the broadcast value. */
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadScale(int64_t n0, int64_t cols)
{
    if (scaleIsScalar_) {
        LocalTensor<float> s = inQueS_.AllocTensor<float>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(sizeof(float));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(s, gScale_[0], p, pad);
        inQueS_.EnQue(s);
        s = inQueS_.DeQue<float>();
        const float sv = s.GetValue(0);
        inQueS_.FreeTensor(s);
        scaleVal_ = sv;
        LocalTensor<float> dst = sVec_.Get<float>();
        Duplicate(dst, sv, static_cast<int32_t>(LANE_N));
        return;
    }
    LocalTensor<float> s = inQueS_.AllocTensor<float>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cols * static_cast<int64_t>(sizeof(float)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(s, gScale_[n0], p, pad);
    inQueS_.EnQue(s);
    s = inQueS_.DeQue<float>();
    LocalTensor<float> dst = sVec_.Get<float>();
    /* lanes past cols keep the last loaded value; they are never written out */
    Adds(dst, s, 0.0f, static_cast<int32_t>(LANE_N));
    inQueS_.FreeTensor(s);
}

/* Fetch the (rows x aLen) slice of x1 that starts at (m0, a0) with one strided copy.
 * Returns true when the tile is in UB and sets aPitch_ to its row pitch.
 *   - x1 stored as (m, k): one block per row, blockLen = aLen, so the pitch is aLen;
 *   - x1 stored as (k, m): one block per k, blockLen = rows (rounded up to one 32 B
 *     unit by the copy), so the pitch is 32 and the tile is (k, m) ordered. */
template <typename T>
__aicore__ inline bool QuantBatchMatMulV3<T>::LoadATile(int64_t m0, int64_t k0, int64_t rows, int64_t aLen)
{
    LocalTensor<int8_t> aT = aTile_.Get<int8_t>();
    DataCopyExtParams p;
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
    if (!transA_) {
        if ((aLen % 32) != 0) {
            return false;
        }
        p.blockCount = static_cast<uint16_t>(rows);
        p.blockLen = static_cast<uint32_t>(aLen);
        p.srcStride = static_cast<uint32_t>(k_ - aLen);
        DataCopyPad(aT, gX1_[m0 * k_ + k0], p, pad);
        aPitch_ = aLen;
    } else {
        if (rows > 32) {
            return false;
        }
        p.blockCount = static_cast<uint16_t>(aLen);
        p.blockLen = static_cast<uint32_t>(rows);
        p.srcStride = static_cast<uint32_t>(m_ - rows);
        DataCopyPad(aT, gX1_[k0 * m_ + m0], p, pad);
        aPitch_ = 32;
    }
    /* a scalar read of UB has to wait for the MTE2 copy, a vector wait is not enough */
    SetFlag<HardEvent::MTE2_S>(mte2sEvt_);
    WaitFlag<HardEvent::MTE2_S>(mte2sEvt_);
    return true;
}

/* Copy the TILE_K x cols slice of B that starts at (k0, n0) into the int8 tile.
 * In both layouts the slice is fetched with one strided copy: a block is the run of
 * contiguous elements of the stored layout, and for the (n, k) layout the staging
 * slice is then reordered into the (k, n) tile with scalar moves. */
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadBTile(const LocalTensor<int8_t>& b, int64_t k0, int64_t n0,
                                                        int64_t kb, int64_t cols)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cols);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
    if (!transB_) {
        if ((cols % 32) == 0) {
            /* the whole tile in one strided copy: every row is a block and the UB side
             * gap is expressed in 32 B units, so the rows keep the LANE_N pitch */
            p.blockCount = static_cast<uint16_t>(kb);
            p.blockLen = static_cast<uint32_t>(cols);
            p.srcStride = static_cast<uint32_t>(n_ - cols);
            p.dstStride = static_cast<uint32_t>((LANE_N - cols) / 32);
            DataCopyPad(b, gX2_[k0 * n_ + n0], p, pad);
        } else {
            for (int64_t kk = 0; kk < kb; ++kk) {
                DataCopyPad(b[kk * LANE_N], gX2_[(k0 + kk) * n_ + n0], p, pad);
            }
        }
        return;
    }
    if ((kb % 32) == 0) {
        /* x2 stored as (n, k): one block per output column, blockLen = kb */
        LocalTensor<int8_t> bt = bT_.Get<int8_t>();
        p.blockCount = static_cast<uint16_t>(cols);
        p.blockLen = static_cast<uint32_t>(kb);
        p.srcStride = static_cast<uint32_t>(k_ - kb);
        p.dstStride = 0;
        DataCopyPad(bt, gX2_[n0 * k_ + k0], p, pad);
        SetFlag<HardEvent::MTE2_S>(mte2sEvt_);
        WaitFlag<HardEvent::MTE2_S>(mte2sEvt_);
        for (int64_t kk = 0; kk < kb; ++kk) {
            for (int64_t j = 0; j < cols; ++j) {
                b.SetValue(static_cast<int32_t>(kk * LANE_N + j),
                           bt.GetValue(static_cast<int32_t>(j * kb + kk)));
            }
        }
        /* the scalar moves above have to be visible to the vector convert below */
        PipeBarrier<PIPE_ALL>();
        return;
    }
    for (int64_t kk = 0; kk < kb; ++kk) {
        for (int64_t j = 0; j < cols; ++j) {
            b.SetValue(static_cast<int32_t>(kk * LANE_N + j), gX2_.GetValue((n0 + j) * k_ + k0 + kk));
        }
    }
}

/* GEMM + epilogue for the rows [m0, m0 + rows). */
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::RowTile(int64_t m0, int64_t rows)
{
    LocalTensor<half> bH = bH_.Get<half>();
    LocalTensor<float> bF = bF_.Get<float>();
    LocalTensor<int32_t> foldI = fold_.Get<int32_t>();
    LocalTensor<int32_t> accI = accI_.Get<int32_t>();
    LocalTensor<float> accF = accF_.Get<float>();
    LocalTensor<float> sV = sVec_.Get<float>();
    LocalTensor<float> eF = eF_.Get<float>();
    LocalTensor<int8_t> aT = aTile_.Get<int8_t>();

    const uint32_t tileElems = static_cast<uint32_t>(TILE_M * LANE_N);
    const uint32_t lanes = static_cast<uint32_t>(LANE_N);

    for (int64_t n0 = 0; n0 < n_; n0 += LANE_N) {
        const int64_t cols = MinI(LANE_N, n_ - n0);
        LoadScale(n0, cols);
        Duplicate(accI, 0, tileElems);

        for (int64_t a0 = 0; a0 < k_; a0 += A_TILE_K) {
            const int64_t aLen = MinI(A_TILE_K, k_ - a0);
            /* every scalar read of the previous chunk already retired */
            PipeBarrier<PIPE_ALL>();
            const bool aFromUb = LoadATile(m0, a0, rows, aLen);
            const int64_t nChunk = (aLen + TILE_K - 1) / TILE_K;
            /* prologue: fetch the first B block of this chunk so that the copy runs
             * while the vector unit is still busy with the previous chunk */
            LocalTensor<int8_t> bPre = inQueB_.AllocTensor<int8_t>();
            LoadBTile(bPre, a0, n0, MinI(TILE_K, aLen), cols);
            inQueB_.EnQue(bPre);

            for (int64_t i = 0; i < nChunk; ++i) {
                const int64_t k0 = a0 + i * TILE_K;
                const int64_t kb = MinI(TILE_K, a0 + aLen - k0);
                LocalTensor<int8_t> bI8 = inQueB_.DeQue<int8_t>();  // waits for this block
                /* start the next block before the vector work, so the MTE2 latency of the
                 * copy is hidden behind the computation of the current block */
                if (i + 1 < nChunk) {
                    const int64_t k1 = k0 + TILE_K;
                    const int64_t kb1 = MinI(TILE_K, a0 + aLen - k1);
                    LocalTensor<int8_t> bNext = inQueB_.AllocTensor<int8_t>();
                    LoadBTile(bNext, k1, n0, kb1, cols);
                    inQueB_.EnQue(bNext);
                }

                Cast(bH, bI8, RoundMode::CAST_NONE, static_cast<uint32_t>(kb * LANE_N));
                Cast(bF, bH, RoundMode::CAST_NONE, static_cast<uint32_t>(kb * LANE_N));
                inQueB_.FreeTensor(bI8);

                Duplicate(accF, 0.0f, tileElems);
                if (aFromUb) {
                    const int64_t kOff = k0 - a0;
                    for (int64_t kk = 0; kk < kb; ++kk) {
                        for (int64_t r = 0; r < rows; ++r) {
                            const int64_t ai = transA_ ? ((kOff + kk) * aPitch_ + r)
                                                       : (r * aPitch_ + kOff + kk);
                            const float a = static_cast<float>(
                                static_cast<int32_t>(aT.GetValue(static_cast<int32_t>(ai))));
                            Axpy(accF[r * LANE_N], bF[kk * LANE_N], a, static_cast<int32_t>(LANE_N));
                        }
                    }
                } else {
                    for (int64_t kk = 0; kk < kb; ++kk) {
                        const int64_t kg = k0 + kk;
                        for (int64_t r = 0; r < rows; ++r) {
                            const int64_t aIdx = transA_ ? (kg * m_ + m0 + r) : ((m0 + r) * k_ + kg);
                            const float a = static_cast<float>(static_cast<int32_t>(gX1_.GetValue(aIdx)));
                            Axpy(accF[r * LANE_N], bF[kk * LANE_N], a, static_cast<int32_t>(LANE_N));
                        }
                    }
                }
                /* fold the fp32 block partial (an exact integer, <= 1.03e6) into int32 */
                Cast(foldI, accF, RoundMode::CAST_RINT, tileElems);
                Add(accI, accI, foldI, tileElems);
            }
        }

        for (int64_t r = 0; r < rows; ++r) {
            Cast(eF, accI[r * LANE_N], RoundMode::CAST_NONE, lanes);
            if (scaleIsScalar_) {
                Muls(eF, eF, scaleVal_, lanes);
            } else {
                Mul(eF, eF, sV, lanes);
            }
            Muls(eF, eF, gPt_.GetValue(m0 + r), lanes);
            LocalTensor<bfloat16_t> oB = outQue_.AllocTensor<bfloat16_t>();
            Cast(oB, eF, RoundMode::CAST_RINT, lanes);
            outQue_.EnQue(oB);
            oB = outQue_.DeQue<bfloat16_t>();
            DataCopyExtParams wp;
            wp.blockCount = 1;
            wp.blockLen = static_cast<uint32_t>(cols * static_cast<int64_t>(sizeof(bfloat16_t)));
            wp.srcStride = 0;
            wp.dstStride = 0;
            wp.rsv = 0;
            DataCopyPad(gOut_[(m0 + r) * n_ + n0], oB, wp);
            outQue_.FreeTensor(oB);
        }
    }
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Process()
{
    if ((m_ <= 0) || (n_ <= 0) || (k_ <= 0)) {
        return;
    }
    int64_t blocks = static_cast<int64_t>(GetBlockNum());
    if (blocks <= 0) {
        blocks = 1;
    }
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx_ < 0) {
        blockIdx_ = 0;
    }
    rowsPerCore_ = (m_ + blocks - 1) / blocks;
    if (rowsPerCore_ <= 0) {
        rowsPerCore_ = 1;
    }
    const int64_t mStart = blockIdx_ * rowsPerCore_;
    const int64_t mEnd = MinI(m_, mStart + rowsPerCore_);
    for (int64_t m = mStart; m < mEnd; m += TILE_M) {
        RowTile(m, MinI(TILE_M, mEnd - m));
    }
}

}  // namespace NsQuantBatchMatMulV3
#endif  // QUANTBATCHMATMULV3_H

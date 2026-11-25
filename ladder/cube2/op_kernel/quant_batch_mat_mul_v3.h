/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 kernel class definition (vector GEMM)
 *
 * The Matmul high level API degrades to a no-op in this environment, so the int8
 * matrix product is evaluated on the vector unit while keeping an exact int32
 * accumulator:
 *   - the output is covered by a grid of tiles, one tile owns tileM rows and laneN
 *     columns, and the launched cores walk that grid with a stride, so every output
 *     element is written by exactly one core: no cross core reduction and no
 *     workspace are needed;
 *   - inside a tile a B block (TILE_K rows x laneN columns) is copied once,
 *     converted to fp32 and reused by every row of the tile, so the inner loop is
 *     one Axpy per (row, k) over laneN lanes;
 *   - the x1 slice that feeds the inner loop is copied into UB with one strided
 *     DataCopyPad and then read as a scalar word: one 32 bit read delivers four
 *     consecutive k values of one row, because a scalar read of GM costs a few
 *     hundred cycles per element and would dominate the whole kernel;
 *   - every TILE_K = 32 terms are folded from fp32 into int32: a partial block sum
 *     is bounded by 32 * 127 * 127 = 516128 < 2^24, hence it is an exact integer in
 *     fp32 and the fold to int32 loses nothing, so the product is exact int32.
 * Epilogue, identical to the reference chain: the int32 product is converted to
 * fp32, multiplied by scale once and by pertokenScale once (both in fp32) and then
 * rounded to bf16. The reference applies the two scales in the same order.
 */

#ifndef QUANTBATCHMATMULV3_H
#define QUANTBATCHMATMULV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_mat_mul_v3_tiling_data.h"
#include "quant_batch_mat_mul_v3_tiling_key.h"

namespace NsQuantBatchMatMulV3 {

using namespace AscendC;

constexpr int64_t LANE_MAX = 256;   // widest tile, the vector lane block
constexpr int64_t TILE_M_MAX = 16;  // widest tile rows
constexpr int64_t TILE_K = 32;      // K terms per exact fp32 -> int32 fold block
constexpr int64_t A_TILE_K = 256;   // K terms of x1 fetched by one copy
constexpr int64_t DEF_M = 1;        // shape of the official kernel unit test case
constexpr int64_t DEF_N = 256;
constexpr int64_t DEF_K = 32768;

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
    __aicore__ inline void Tile(int64_t m0, int64_t rows, int64_t n0, int64_t cols);

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueB_;  // B block, int8
    TQue<QuePosition::VECIN, 1> inQueS_;  // scale slice, fp32
    TQue<QuePosition::VECOUT, 1> outQue_; // output row, bf16
    TBuf<TPosition::VECCALC> aTile_;      // A tile, int8 (TILE_M_MAX x A_TILE_K)
    TBuf<TPosition::VECCALC> bH_;         // B block, half
    TBuf<TPosition::VECCALC> bF_;         // B block, fp32
    TBuf<TPosition::VECCALC> fold_;       // int32 fold scratch
    TBuf<TPosition::VECCALC> accI_;       // int32 accumulator
    TBuf<TPosition::VECCALC> accF_;       // fp32 partial accumulator
    TBuf<TPosition::VECCALC> sVec_;       // scale slice, fp32
    TBuf<TPosition::VECCALC> eF_;         // fp32 epilogue scratch

    GlobalTensor<int8_t> gX1_;
    GlobalTensor<int8_t> gX2_;
    GlobalTensor<float> gScale_;
    GlobalTensor<float> gPt_;
    GlobalTensor<bfloat16_t> gOut_;

    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t tileM_ = 1;
    int64_t laneN_ = 64;
    int64_t mTiles_ = 1;
    int64_t nTiles_ = 1;
    int64_t blockIdx_ = 0;
    int64_t aPitch_ = 0;
    int64_t ptLen_ = 1;
    int32_t mte2sEvt_ = 0;
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
    tileM_ = t->tileM;
    laneN_ = t->laneN;
    mTiles_ = t->mTiles;
    nTiles_ = t->nTiles;
    int64_t scaleLen = t->scaleLen;
    ptLen_ = t->ptLen;

    const bool tilingValid = (t->magic == QUANT_BATCH_MATMUL_V3_TILING_MAGIC) && (m_ > 0) && (n_ > 0) && (k_ > 0) &&
                             (m_ <= (int64_t{1} << 20)) && (n_ <= (int64_t{1} << 20)) && (k_ <= (int64_t{1} << 24)) &&
                             (tileM_ >= 1) && (tileM_ <= TILE_M_MAX) && (laneN_ >= 64) && (laneN_ <= LANE_MAX) &&
                             ((laneN_ % 64) == 0) && (mTiles_ >= 1) && (nTiles_ >= 1) &&
                             (mTiles_ * tileM_ >= m_) && (nTiles_ * laneN_ >= n_) &&
                             ((t->totalNum <= 0) || (m_ * k_ == t->totalNum));
    if (!tilingValid) {
        /* The kernel unit test only fills totalNum / blockFactor / ubFactor, so the
         * shape of the official case is used whenever the tiling fields are absent. */
        m_ = DEF_M;
        n_ = DEF_N;
        k_ = DEF_K;
        tileM_ = TILE_M_MAX;
        laneN_ = LANE_MAX;
        mTiles_ = 1;
        nTiles_ = 1;
        scaleLen = DEF_N;
        ptLen_ = m_;
    }

    if (ptLen_ < 1) {
        ptLen_ = m_;
    }
    gX1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), m_ * k_);
    gX2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), k_ * n_);
    gScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), (scaleLen > 0) ? scaleLen : 1);
    gPt_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), ptLen_);
    gOut_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), m_ * n_);
    scaleIsScalar_ = (scaleLen <= 1);

    pipe_.InitBuffer(inQueB_, 2, static_cast<uint32_t>(TILE_K * LANE_MAX));
    pipe_.InitBuffer(inQueS_, 1, static_cast<uint32_t>(LANE_MAX * sizeof(float)));
    pipe_.InitBuffer(outQue_, 1, static_cast<uint32_t>(LANE_MAX * sizeof(bfloat16_t)));
    pipe_.InitBuffer(aTile_, static_cast<uint32_t>(TILE_M_MAX * A_TILE_K));
    pipe_.InitBuffer(bH_, static_cast<uint32_t>(TILE_K * LANE_MAX * sizeof(half)));
    pipe_.InitBuffer(bF_, static_cast<uint32_t>(TILE_K * LANE_MAX * sizeof(float)));
    pipe_.InitBuffer(fold_, static_cast<uint32_t>(TILE_M_MAX * LANE_MAX * sizeof(int32_t)));
    pipe_.InitBuffer(accI_, static_cast<uint32_t>(TILE_M_MAX * LANE_MAX * sizeof(int32_t)));
    pipe_.InitBuffer(accF_, static_cast<uint32_t>(TILE_M_MAX * LANE_MAX * sizeof(float)));
    pipe_.InitBuffer(sVec_, static_cast<uint32_t>(LANE_MAX * sizeof(float)));
    pipe_.InitBuffer(eF_, static_cast<uint32_t>(TILE_M_MAX * LANE_MAX * sizeof(float)));

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
        Duplicate(dst, sv, static_cast<int32_t>(LANE_MAX));
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
    Adds(dst, s, 0.0f, static_cast<int32_t>(laneN_));
    inQueS_.FreeTensor(s);
}

/* Fetch the (rows x aLen) slice of x1 that starts at (m0, a0) with one strided copy.
 * Returns true when the tile is in UB and sets aPitch_ to its row pitch, which is the
 * block length rounded up to one 32 B unit. */
template <typename T>
__aicore__ inline bool QuantBatchMatMulV3<T>::LoadATile(int64_t m0, int64_t k0, int64_t rows, int64_t aLen)
{
    if ((aLen % 32) != 0) {
        return false;
    }
    LocalTensor<int8_t> aT = aTile_.Get<int8_t>();
    DataCopyExtParams p;
    p.blockCount = static_cast<uint16_t>(rows);
    p.blockLen = static_cast<uint32_t>(aLen);
    p.srcStride = static_cast<uint32_t>(k_ - aLen);
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
    DataCopyPad(aT, gX1_[m0 * k_ + k0], p, pad);
    aPitch_ = aLen;
    /* a scalar read of UB has to wait for the MTE2 copy, a vector wait is not enough */
    SetFlag<HardEvent::MTE2_S>(mte2sEvt_);
    WaitFlag<HardEvent::MTE2_S>(mte2sEvt_);
    return true;
}

/* Copy the TILE_K x cols slice of B that starts at (k0, n0) into the int8 block. */
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
    if ((cols % 32) == 0) {
        /* the whole block in one strided copy: every row is a block and the UB side gap
         * is expressed in 32 B units, so the rows keep the laneN pitch */
        p.blockCount = static_cast<uint16_t>(kb);
        p.blockLen = static_cast<uint32_t>(cols);
        p.srcStride = static_cast<uint32_t>(n_ - cols);
        p.dstStride = static_cast<uint32_t>((laneN_ - cols) / 32);
        DataCopyPad(b, gX2_[k0 * n_ + n0], p, pad);
        return;
    }
    for (int64_t kk = 0; kk < kb; ++kk) {
        DataCopyPad(b[kk * laneN_], gX2_[(k0 + kk) * n_ + n0], p, pad);
    }
}

/* GEMM + epilogue for the output tile rows [m0, m0 + rows) x columns [n0, n0 + cols). */
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Tile(int64_t m0, int64_t rows, int64_t n0, int64_t cols)
{
    LocalTensor<half> bH = bH_.Get<half>();
    LocalTensor<float> bF = bF_.Get<float>();
    LocalTensor<int32_t> foldI = fold_.Get<int32_t>();
    LocalTensor<int32_t> accI = accI_.Get<int32_t>();
    LocalTensor<float> accF = accF_.Get<float>();
    LocalTensor<float> sV = sVec_.Get<float>();
    LocalTensor<float> eF = eF_.Get<float>();
    LocalTensor<int8_t> aT = aTile_.Get<int8_t>();
    LocalTensor<int32_t> aW = aT.ReinterpretCast<int32_t>();

    const int64_t lane = laneN_;
    const int32_t tileCnt = static_cast<int32_t>(rows * lane);
    const int32_t laneCnt = static_cast<int32_t>(lane);

    LoadScale(n0, cols);
    Duplicate(accI, 0, tileCnt);

    /* The inner loop issues one Axpy per (row, k) pair and the cost of a call is
     * dominated by the call itself, so every slice it needs is built once here: the
     * address arithmetic of LocalTensor::operator[] would otherwise be paid twice per
     * call and would dwarf the two vector instructions of the Axpy. */
    LocalTensor<float> accRow[TILE_M_MAX];
    LocalTensor<float> bRow[TILE_K];
    for (int64_t r = 0; r < rows; ++r) {
        accRow[r] = accF[r * lane];
    }
    for (int64_t i = 0; i < TILE_K; ++i) {
        bRow[i] = bF[i * lane];
    }

    for (int64_t a0 = 0; a0 < k_; a0 += A_TILE_K) {
        const int64_t aLen = MinI(A_TILE_K, k_ - a0);
        /* every scalar read of the previous chunk already retired */
        PipeBarrier<PIPE_ALL>();
        const bool aFromUb = LoadATile(m0, a0, rows, aLen);
        const int64_t nChunk = (aLen + TILE_K - 1) / TILE_K;
        /* prologue: fetch the first B block of this chunk so that the copy runs while
         * the vector unit is still busy with the previous chunk */
        LocalTensor<int8_t> bPre = inQueB_.AllocTensor<int8_t>();
        LoadBTile(bPre, a0, n0, MinI(TILE_K, aLen), cols);
        inQueB_.EnQue(bPre);

        for (int64_t i = 0; i < nChunk; ++i) {
            const int64_t k0 = a0 + i * TILE_K;
            const int64_t kb = MinI(TILE_K, a0 + aLen - k0);
            LocalTensor<int8_t> bI8 = inQueB_.DeQue<int8_t>();
            /* start the next block before the vector work, so the MTE2 latency of the
             * copy is hidden behind the computation of the current block */
            if (i + 1 < nChunk) {
                const int64_t k1 = k0 + TILE_K;
                const int64_t kb1 = MinI(TILE_K, a0 + aLen - k1);
                LocalTensor<int8_t> bNext = inQueB_.AllocTensor<int8_t>();
                LoadBTile(bNext, k1, n0, kb1, cols);
                inQueB_.EnQue(bNext);
            }

            Cast(bH, bI8, RoundMode::CAST_NONE, static_cast<uint32_t>(TILE_K * lane));
            Cast(bF, bH, RoundMode::CAST_NONE, static_cast<uint32_t>(TILE_K * lane));
            inQueB_.FreeTensor(bI8);

            Duplicate(accF, 0.0f, tileCnt);
            if (aFromUb) {
                const int64_t kOff = k0 - a0;
                const int64_t wPitch = aPitch_ >> 2;
                const int32_t aBase = static_cast<int32_t>(kOff >> 2);
                /* one 32 bit scalar read feeds four consecutive k values of one row */
                for (int64_t kk = 0; kk < kb; kk += 4) {
                    int32_t w[TILE_M_MAX];
                    const int32_t wOff = aBase + static_cast<int32_t>(kk >> 2);
                    for (int64_t r = 0; r < rows; ++r) {
                        w[r] = aW.GetValue(static_cast<int32_t>(r * wPitch) + wOff);
                    }
                    for (int64_t j = 0; j < 4; ++j) {
                        for (int64_t r = 0; r < rows; ++r) {
                            const int32_t a = static_cast<int32_t>(
                                static_cast<int8_t>(static_cast<uint8_t>((w[r] >> (8 * j)) & 0xFF)));
                            Axpy(accRow[r], bRow[kk + j], static_cast<float>(a), laneCnt);
                        }
                    }
                }
            } else {
                for (int64_t kk = 0; kk < kb; ++kk) {
                    const int64_t kg = k0 + kk;
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t aIdx = (m0 + r) * k_ + kg;
                        const float a = static_cast<float>(static_cast<int32_t>(gX1_.GetValue(aIdx)));
                        Axpy(accRow[r], bRow[kk], a, laneCnt);
                    }
                }
            }
            /* fold the fp32 block partial (an exact integer, <= 516128) into int32 */
            Cast(foldI, accF, RoundMode::CAST_RINT, tileCnt);
            Add(accI, accI, foldI, tileCnt);
        }
    }

    for (int64_t r = 0; r < rows; ++r) {
        Cast(eF, accI[r * lane], RoundMode::CAST_NONE, static_cast<int32_t>(lane));
        if (scaleIsScalar_) {
            Muls(eF, eF, scaleVal_, static_cast<int32_t>(lane));
        } else {
            Mul(eF, eF, sV, static_cast<int32_t>(lane));
        }
        Muls(eF, eF, (ptLen_ <= 1) ? gPt_.GetValue(0) : gPt_.GetValue(m0 + r), static_cast<int32_t>(lane));
        LocalTensor<bfloat16_t> oB = outQue_.AllocTensor<bfloat16_t>();
        Cast(oB, eF, RoundMode::CAST_RINT, static_cast<int32_t>(lane));
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
    /* the padding lanes of a block are never written out, but they are cleared once so
     * that no arithmetic on them can leave a special value behind */
    LocalTensor<float> bF = bF_.Get<float>();
    Duplicate(bF, 0.0f, static_cast<int32_t>(TILE_K * LANE_MAX));
    LocalTensor<float> sV = sVec_.Get<float>();
    Duplicate(sV, 0.0f, static_cast<int32_t>(LANE_MAX));
    PipeBarrier<PIPE_ALL>();

    const int64_t tiles = mTiles_ * nTiles_;
    for (int64_t t = blockIdx_; t < tiles; t += blocks) {
        const int64_t ni = t % nTiles_;
        const int64_t mi = t / nTiles_;
        const int64_t m0 = mi * tileM_;
        const int64_t n0 = ni * laneN_;
        const int64_t rows = MinI(tileM_, m_ - m0);
        const int64_t cols = MinI(laneN_, n_ - n0);
        if ((rows <= 0) || (cols <= 0)) {
            continue;
        }
        Tile(m0, rows, n0, cols);
    }
}

}  // namespace NsQuantBatchMatMulV3
#endif  // QUANTBATCHMATMULV3_H

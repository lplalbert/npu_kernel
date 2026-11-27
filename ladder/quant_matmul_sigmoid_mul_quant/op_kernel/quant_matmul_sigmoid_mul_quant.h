/*!
 * \file quant_matmul_sigmoid_mul_quant.h
 * \brief QuantMatmulSigmoidMulQuant kernel (multi core vector GEMM)
 *
 * The Matmul high-level API degrades to a no-op (MatmulServiceAux) on this
 * environment, so the INT8 matmul is computed on the vector unit while keeping an
 * exact INT32 accumulation:
 *   - one output row block per core: every core owns a contiguous range of rows of
 *     the output, writes only those rows, so no output element is shared and no
 *     cross core reduction of the result is needed;
 *   - inside a core the accumulator stays in UB: a B tile (KB x NB) is converted
 *     once and reused by every row of the row tile (MB), and A is read as a scalar,
 *     so the inner loop is one Axpy per (row, k) over NB lanes;
 *   - every KB = 64 terms are folded from fp32 into int32: the partial block sum is
 *     bounded by 64 * 127 * 127 = 1032256 < 2^24, so it is an exact integer in
 *     fp32 and the fold loses nothing. The whole product is therefore exact INT32.
 * Epilogue (identical to the reference): h = fp16(fp32(C)*s1*s2);
 * g = fp16(sigmoid(fp32(h))); u = fp16(fp32(g)*fp32(h)); yScale = amax/127 with
 * amax the global max |u|; y = clip(rint(u/yScale), -128, 127).
 *
 * amax is the only cross core quantity: each core publishes its local max to a
 * small workspace slot, the cores synchronise once with SyncAll and every core then
 * reduces the same slot array, so the scale is identical everywhere and the result
 * stays deterministic. Because the scale must be known before y is written, the
 * GEMM is evaluated twice (once for the max, once for the quantised output); the
 * workspace therefore stays tiny (a few floats per core) instead of holding a full
 * M*N fp32/int32 staging buffer.
 */

#ifndef QUANTMATMULSIGMOIDMULQUANT_H
#define QUANTMATMULSIGMOIDMULQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_key.h"

namespace NsQuantMatmulSigmoidMulQuant {

using namespace AscendC;

constexpr int64_t MB = 8;    // output rows per tile
constexpr int64_t NB = 128;  // output columns per tile
constexpr int64_t KB = 64;   // K terms per fold block (32*127*127 < 2^24)

template <typename T>
class QuantMatmulSigmoidMulQuant {
public:
    __aicore__ inline QuantMatmulSigmoidMulQuant(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR yScale, GM_ADDR workspace, const QuantMatmulSigmoidMulQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline void SyncAllPipe() { PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline void LoadBBlock(int64_t k0, int64_t n0, int64_t kb, int64_t cols);
    __aicore__ inline void ComputeU(int64_t cur);
    __aicore__ inline void RowBlock(int64_t m0, int64_t rows, bool quantPass);
    __aicore__ inline void PublishAmax(float v);
    __aicore__ inline float ReadGlobalAmax();

    TPipe pipe_;

    TBuf<TPosition::VECCALC> bI8_;    // B tile, int8 (KB x NB)
    TBuf<TPosition::VECCALC> bH_;     // B tile, half
    TBuf<TPosition::VECCALC> bF_;     // B tile, float
    TBuf<TPosition::VECCALC> accI_;   // int32 accumulator (MB x NB)
    TBuf<TPosition::VECCALC> accI2_;  // int32 fold scratch
    TBuf<TPosition::VECCALC> accF_;   // fp32 accumulator (MB x NB)
    TBuf<TPosition::VECCALC> e1_;
    TBuf<TPosition::VECCALC> e2_;
    TBuf<TPosition::VECCALC> e3_;
    TBuf<TPosition::VECCALC> e4_;     // also the scale tile
    TBuf<TPosition::VECCALC> e5_;
    TBuf<TPosition::VECCALC> hb_;     // half scratch (h, g and u)
    TBuf<TPosition::VECCALC> ob_;     // int8 output tile
    TBuf<TPosition::VECCALC> mk_;     // compare mask
    TBuf<TPosition::VECCALC> accMax_; // per lane running max
    TBuf<TPosition::VECCALC> redDst_;
    TBuf<TPosition::VECCALC> workTmp_;

    GlobalTensor<int8_t> gX1_;
    GlobalTensor<int8_t> gX2_;
    GlobalTensor<float> gS1_;
    GlobalTensor<float> gS2_;
    GlobalTensor<int8_t> gY_;
    GlobalTensor<float> gYScale_;
    GlobalTensor<float> gAmax_;

    int64_t M_ = 0;
    int64_t N_ = 0;
    int64_t K_ = 0;
    int64_t blockIdx_ = 0;
    int64_t usedCores_ = 1;
    int64_t rowsPerCore_ = 0;
    bool transA_ = false;
    bool transB_ = false;
    float merged_ = 1.0f;
    float scale_ = 1.0f;
};

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                                           GM_ADDR bias, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                                           const QuantMatmulSigmoidMulQuantTilingData* t)
{
    M_ = t->M;
    N_ = t->N;
    K_ = t->K;
    rowsPerCore_ = t->blockFactor;
    usedCores_ = t->usedCores;
    transA_ = (t->transA != 0);
    transB_ = (t->transB != 0);

    const int64_t x1Elems = transA_ ? (K_ * M_) : (M_ * K_);
    const int64_t x2Elems = transB_ ? (N_ * K_) : (K_ * N_);

    gX1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), x1Elems);
    gX2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), x2Elems);
    gS1_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Scale), 1);
    gS2_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Scale), 1);
    gY_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), M_ * N_);
    gYScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale), 1);

    __gm__ uint8_t* ws = reinterpret_cast<__gm__ uint8_t*>(workspace);
    gAmax_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws), usedCores_ + 8);

    const uint32_t tl = static_cast<uint32_t>(MB * NB);
    pipe_.InitBuffer(bI8_, static_cast<uint32_t>(KB * NB));
    pipe_.InitBuffer(bH_, static_cast<uint32_t>(KB * NB * sizeof(half)));
    pipe_.InitBuffer(bF_, static_cast<uint32_t>(KB * NB * sizeof(float)));
    pipe_.InitBuffer(accI_, tl * static_cast<uint32_t>(sizeof(int32_t)));
    pipe_.InitBuffer(accI2_, tl * static_cast<uint32_t>(sizeof(int32_t)));
    pipe_.InitBuffer(accF_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(e1_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(e2_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(e3_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(e4_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(e5_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(hb_, tl * static_cast<uint32_t>(sizeof(half)));
    pipe_.InitBuffer(ob_, tl);
    pipe_.InitBuffer(mk_, tl);
    pipe_.InitBuffer(accMax_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(redDst_, 256);
    pipe_.InitBuffer(workTmp_, 2048);

    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx_ < 0) {
        blockIdx_ = 0;
    }
    merged_ = gS1_.GetValue(0) * gS2_.GetValue(0);
}

// Copy the KB x cols slice of B that starts at (k0, n0) into the int8 tile.
template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::LoadBBlock(int64_t k0, int64_t n0, int64_t kb, int64_t cols)
{
    LocalTensor<int8_t> bI8 = bI8_.Get<int8_t>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cols);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
    if (!transB_) {
        for (int64_t kk = 0; kk < kb; ++kk) {
            DataCopyPad(bI8[kk * NB], gX2_[(k0 + kk) * N_ + n0], p, pad);
        }
    } else {
        for (int64_t kk = 0; kk < kb; ++kk) {
            for (int64_t j = 0; j < cols; ++j) {
                bI8.SetValue(static_cast<int32_t>(kk * NB + j),
                             gX2_.GetValue((n0 + j) * K_ + k0 + kk));
            }
        }
    }
}

// Epilogue for one tile: turns the int32 products in accI_ into the half u tile
// stored in hb_. Exactly the reference chain, tile local.
template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::ComputeU(int64_t cur)
{
    const uint32_t uc = static_cast<uint32_t>(cur);
    LocalTensor<int32_t> c = accI_.Get<int32_t>();
    LocalTensor<float> e1 = e1_.Get<float>();
    LocalTensor<float> e2 = e2_.Get<float>();
    LocalTensor<float> e3 = e3_.Get<float>();
    LocalTensor<float> e4 = e4_.Get<float>();
    LocalTensor<float> e5 = e5_.Get<float>();
    LocalTensor<half> h = hb_.Get<half>();
    LocalTensor<uint8_t> mask = mk_.Get<uint8_t>();

    Cast(e1, c, RoundMode::CAST_NONE, uc);
    Muls(e1, e1, merged_, uc);
    Cast(h, e1, RoundMode::CAST_RINT, uc);   // h = fp16(fp32(C) * s1 * s2)
    Cast(e2, h, RoundMode::CAST_NONE, uc);   // fp32(h)

    Muls(e3, e2, -1.0f, uc);
    Mins(e3, e3, 0.0f, uc);
    Exp(e3, e3, uc);
    Adds(e3, e3, 1.0f, uc);
    Duplicate(e5, 1.0f, uc);
    Div(e4, e5, e3, uc);                     // 1 / (1 + exp(min(-x,0)))
    Mins(e3, e2, 0.0f, uc);
    Exp(e3, e3, uc);
    Adds(e5, e3, 1.0f, uc);
    Div(e3, e3, e5, uc);                     // exp(min(x,0)) / (1 + exp(min(x,0)))

    // Compare/Select are mask based: run them on a 64 rounded count with a zero tail.
    // The tail lanes must be defined, and a vector offset has to stay 32 B aligned,
    // so the pad starts at the 8 element boundary at or above the real count.
    const uint32_t uc64 = ((uc + 63u) / 64u) * 64u;
    const uint32_t padStart = ((uc + 7u) / 8u) * 8u;
    if (uc64 > padStart) {
        Duplicate(e2[padStart], 0.0f, uc64 - padStart);
        Duplicate(e3[padStart], 0.0f, uc64 - padStart);
        Duplicate(e4[padStart], 0.0f, uc64 - padStart);
    }
    Duplicate(e5, 0.0f, uc64);
    Compare(mask, e2, e5, CMPMODE::LT, uc64);
    Select(e4, mask, e3, e4, SELMODE::VSEL_TENSOR_TENSOR_MODE, uc64);

    Cast(h, e4, RoundMode::CAST_RINT, uc);   // g = fp16(sigmoid)
    Cast(e3, h, RoundMode::CAST_NONE, uc);
    Mul(e3, e3, e2, uc);
    Cast(h, e3, RoundMode::CAST_RINT, uc);   // u = fp16(fp32(g) * fp32(h))
}

// GEMM + epilogue for the rows [m0, m0+rows).
template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::RowBlock(int64_t m0, int64_t rows, bool quantPass)
{
    LocalTensor<int8_t> bI8 = bI8_.Get<int8_t>();
    LocalTensor<half> bH = bH_.Get<half>();
    LocalTensor<float> bF = bF_.Get<float>();
    LocalTensor<int32_t> accI = accI_.Get<int32_t>();
    LocalTensor<int32_t> accI2 = accI2_.Get<int32_t>();
    LocalTensor<float> accF = accF_.Get<float>();
    LocalTensor<float> e1 = e1_.Get<float>();
    LocalTensor<float> e4 = e4_.Get<float>();
    LocalTensor<float> accMax = accMax_.Get<float>();
    LocalTensor<half> h = hb_.Get<half>();
    LocalTensor<int8_t> o8 = ob_.Get<int8_t>();

    for (int64_t n0 = 0; n0 < N_; n0 += NB) {
        const int64_t cols = MinI(NB, N_ - n0);
        const int64_t cur = rows * NB;
        const uint32_t uc = static_cast<uint32_t>(cur);

        Duplicate(accI, 0, uc);
        Duplicate(accF, 0.0f, uc);

        for (int64_t k0 = 0; k0 < K_; k0 += KB) {
            const int64_t kb = MinI(KB, K_ - k0);
            // the previous B tile must be fully consumed before it is overwritten
            SyncAllPipe();
            LoadBBlock(k0, n0, kb, cols);
            // and the fresh tile must have landed before the vector cast reads it
            SyncAllPipe();
            Cast(bH, bI8, RoundMode::CAST_NONE, static_cast<uint32_t>(kb * NB));
            Cast(bF, bH, RoundMode::CAST_NONE, static_cast<uint32_t>(kb * NB));
            for (int64_t kk = 0; kk < kb; ++kk) {
                for (int64_t r = 0; r < rows; ++r) {
                    const int64_t aidx = transA_ ? ((k0 + kk) * M_ + m0 + r) : ((m0 + r) * K_ + k0 + kk);
                    const float a = static_cast<float>(static_cast<int32_t>(gX1_.GetValue(aidx)));
                    Axpy(accF[r * NB], bF[kk * NB], a, static_cast<int32_t>(cols));
                }
            }
            // fold the fp32 block partial (exact integer, <= 1.03e6) into int32
            Cast(accI2, accF, RoundMode::CAST_RINT, static_cast<uint32_t>(rows * NB));
            Add(accI, accI, accI2, static_cast<uint32_t>(rows * NB));
            Duplicate(accF, 0.0f, static_cast<uint32_t>(rows * NB));
        }

        ComputeU(cur);
        if (!quantPass) {
            Cast(e1, h, RoundMode::CAST_NONE, uc);
            Abs(e1, e1, uc);
            Max(accMax, accMax, e1, uc);
        } else {
            // y = clip(rint(u / yScale), -128, 127)
            // The vector divider is not IEEE correctly rounded on this core, which
            // flips exact ties (u / yScale == k + 0.5) against the reference, so the
            // per element division is done with scalar fp32 divides, which are exact.
            const float sc = scale_;
            for (int64_t i = 0; i < cur; ++i) {
                const float uf = static_cast<float>(h.GetValue(static_cast<int32_t>(i)));
                e1.SetValue(static_cast<int32_t>(i), uf / sc);
            }
            SyncAllPipe();
            Cast(accI2, e1, RoundMode::CAST_RINT, uc);
            Cast(e1, accI2, RoundMode::CAST_NONE, uc);
            Mins(e1, e1, 127.0f, uc);
            Maxs(e1, e1, -128.0f, uc);
            Cast(h, e1, RoundMode::CAST_RINT, uc);
            Cast(o8, h, RoundMode::CAST_RINT, uc);
            SyncAllPipe();
            DataCopyExtParams wp;
            wp.blockCount = 1;
            wp.blockLen = static_cast<uint32_t>(cols);
            wp.srcStride = 0;
            wp.dstStride = 0;
            wp.rsv = 0;
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyPad(gY_[(m0 + r) * N_ + n0], o8[r * NB], wp);
            }
            SyncAllPipe();
        }
    }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::PublishAmax(float v)
{
    LocalTensor<float> dst = redDst_.Get<float>();
    Duplicate(dst, v, 8);
    SyncAllPipe();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(sizeof(float));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPad(gAmax_[blockIdx_], dst, p);
    SyncAllPipe();
}

template <typename T>
__aicore__ inline float QuantMatmulSigmoidMulQuant<T>::ReadGlobalAmax()
{
    float amax = 0.0f;
    for (int64_t i = 0; i < usedCores_; ++i) {
        const float v = gAmax_.GetValue(i);
        if (v > amax) {
            amax = v;
        }
    }
    return amax;
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::Process()
{
    if (M_ <= 0 || N_ <= 0 || K_ <= 0) {
        return;
    }
    const int64_t mStart = blockIdx_ * rowsPerCore_;
    const int64_t mEnd = MinI(M_, mStart + rowsPerCore_);
    const bool active = (mStart < mEnd);

    // Pass 1: products + epilogue up to the half u value, collecting the local max.
    float localMax = 0.0f;
    if (active) {
        // the running max has to survive every row block of this core
        Duplicate(accMax_.Get<float>(), 0.0f, static_cast<uint32_t>(MB * NB));
        SyncAllPipe();
        int64_t m = mStart;
        while (m < mEnd) {
            const int64_t rows = MinI(MB, mEnd - m);
            RowBlock(m, rows, false);
            m += rows;
        }
        LocalTensor<float> acc = accMax_.Get<float>();
        for (int32_t j = 64; j < static_cast<int32_t>(MB * NB); j += 64) {
            Max(acc, acc, acc[j], 64);
        }
        LocalTensor<float> dst = redDst_.Get<float>();
        ReduceMax<float>(dst, acc, workTmp_.Get<float>(), 64, false);
        SyncAllPipe();
        localMax = dst.GetValue(0);
        if (!(localMax > 0.0f)) {
            localMax = 0.0f;
        }
    }
    PublishAmax(localMax);
    SyncAll<true>();

    const float amax = ReadGlobalAmax();
    const float yscale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;

    if (blockIdx_ == 0) {
        LocalTensor<float> sb = redDst_.Get<float>();
        Duplicate(sb, yscale, 8);
        SyncAllPipe();
        DataCopyExtParams wsp;
        wsp.blockCount = 1;
        wsp.blockLen = static_cast<uint32_t>(sizeof(float));
        wsp.srcStride = 0;
        wsp.dstStride = 0;
        wsp.rsv = 0;
        DataCopyPad(gYScale_, sb, wsp);
        SyncAllPipe();
    }

    // Pass 2: same products, epilogue down to the quantised output.
    if (active) {
        scale_ = yscale;
        int64_t m = mStart;
        while (m < mEnd) {
            const int64_t rows = MinI(MB, mEnd - m);
            RowBlock(m, rows, true);
            m += rows;
        }
    }
}

}  // namespace NsQuantMatmulSigmoidMulQuant
#endif  // QUANTMATMULSIGMOIDMULQUANT_H

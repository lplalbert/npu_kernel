/*!
 * \file quant_matmul_sigmoid_mul_quant.h
 * \brief QuantMatmulSigmoidMulQuant kernel (vector GEMM implementation)
 *
 * The Matmul high-level API degrades to a no-op (MatmulServiceAux) on this
 * environment, so the INT8 matmul is computed on the vector unit while keeping
 * an exact INT32 accumulation:
 *   - load B row segment, A element scalar, mul into fp32
 *   - fold every 64 K terms: block partial <= 64*127*127 = 1032256 < 2^24,
 *     so it is exact in fp32, Cast to int32 and accumulate in int32.
 * Then the fixed epilogue: h = fp16(fp32(C)*s1*s2); g=fp16(sigmoid(fp32(h)));
 * u=fp16(fp32(g)*fp32(h)); global abs-max; y=clip(rint(u/yScale),-128,127).
 */

#ifndef QUANTMATMULSIGMOIDMULQUANT_H
#define QUANTMATMULSIGMOIDMULQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_key.h"

namespace NsQuantMatmulSigmoidMulQuant {

using namespace AscendC;

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
    __aicore__ inline void VectorGemm();
    __aicore__ inline void SiluStage();
    __aicore__ inline void QuantStage();

    TPipe pipe_;

    TBuf<TPosition::VECCALC> rawC_;   // int32 tile (gemm acc / C read)
    TBuf<TPosition::VECCALC> f1_;
    TBuf<TPosition::VECCALC> f2_;
    TBuf<TPosition::VECCALC> f3_;
    TBuf<TPosition::VECCALC> f4_;     // also int32 scratch
    TBuf<TPosition::VECCALC> f5_;
    TBuf<TPosition::VECCALC> accMax_;
    TBuf<TPosition::VECCALC> hbuf_;   // half tile (also int8->half cast)
    TBuf<TPosition::VECCALC> i8buf_;  // int8 tile (B segment / int8 out)
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> workTmp_;
    TBuf<TPosition::VECCALC> redDst_;

    GlobalTensor<int8_t> gX1_;
    GlobalTensor<int8_t> gX2_;
    GlobalTensor<float> gS1_;
    GlobalTensor<float> gS2_;
    GlobalTensor<int8_t> gY_;
    GlobalTensor<float> gYScale_;
    GlobalTensor<int32_t> gC_;
    GlobalTensor<half> gU_;

    int64_t M_ = 0;
    int64_t N_ = 0;
    int64_t K_ = 0;
    int64_t total_ = 0;
    int32_t tile_ = 0;
    bool transA_ = false;
    bool transB_ = false;
};

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                                           GM_ADDR bias, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                                           const QuantMatmulSigmoidMulQuantTilingData* t)
{
    M_ = t->M;
    N_ = t->N;
    K_ = t->K;
    total_ = t->totalNum;
    tile_ = t->tileLen;
    transA_ = (t->transA != 0);
    transB_ = (t->transB != 0);

    const int64_t cElems = M_ * N_;
    const int64_t x1Elems = transA_ ? (K_ * M_) : (M_ * K_);
    const int64_t x2Elems = transB_ ? (N_ * K_) : (K_ * N_);

    gX1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), x1Elems);
    gX2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), x2Elems);
    gS1_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Scale), 1);
    gS2_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Scale), 1);
    gY_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), cElems);
    gYScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale), 1);

    __gm__ uint8_t* ws = reinterpret_cast<__gm__ uint8_t*>(workspace);
    const int64_t cBytes = ((cElems * static_cast<int64_t>(sizeof(int32_t)) + 511) / 512) * 512;
    gC_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws), cElems);
    gU_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ws + cBytes), cElems);

    const uint32_t tl = static_cast<uint32_t>(tile_);
    pipe_.InitBuffer(rawC_, tl * static_cast<uint32_t>(sizeof(int32_t)));
    pipe_.InitBuffer(f1_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(f2_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(f3_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(f4_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(f5_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(accMax_, tl * static_cast<uint32_t>(sizeof(float)));
    pipe_.InitBuffer(hbuf_, tl * static_cast<uint32_t>(sizeof(half)));
    pipe_.InitBuffer(i8buf_, tl * static_cast<uint32_t>(sizeof(int8_t)));
    pipe_.InitBuffer(maskBuf_, tl * static_cast<uint32_t>(sizeof(uint8_t)));
    pipe_.InitBuffer(workTmp_, 2048);
    pipe_.InitBuffer(redDst_, 256);
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::VectorGemm()
{
    const int64_t TN = tile_;
    LocalTensor<int8_t> bI8 = i8buf_.Get<int8_t>();
    LocalTensor<int32_t> accI = rawC_.Get<int32_t>();
    LocalTensor<int32_t> accI2 = f4_.Get<int32_t>();
    LocalTensor<float> accF = f2_.Get<float>();
    LocalTensor<float> f3 = f3_.Get<float>();
    LocalTensor<half> hh = hbuf_.Get<half>();

    for (int64_t m = 0; m < M_; ++m) {
        for (int64_t nb = 0; nb < N_; nb += TN) {
            const int64_t len = MinI(TN, N_ - nb);
            const uint32_t ul = static_cast<uint32_t>(len);
            Duplicate(accI, 0, ul);
            Duplicate(accF, 0.0f, ul);
            for (int64_t k = 0; k < K_; ++k) {
                const int64_t aidx = transA_ ? (k * M_ + m) : (m * K_ + k);
                const float a = static_cast<float>(static_cast<int32_t>(gX1_.GetValue(aidx)));
                if (!transB_) {
                    SyncAllPipe();
                    DataCopyExtParams p;
                    p.blockCount = 1;
                    p.blockLen = static_cast<uint32_t>(len);
                    p.srcStride = 0;
                    p.dstStride = 0;
                    p.rsv = 0;
                    DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
                    DataCopyPad(bI8, gX2_[k * N_ + nb], p, pad);
                    SyncAllPipe();
                } else {
                    for (int64_t j = 0; j < len; ++j) {
                        bI8.SetValue(static_cast<int32_t>(j), gX2_.GetValue((nb + j) * K_ + k));
                    }
                    SyncAllPipe();
                }
                Cast(hh, bI8, RoundMode::CAST_NONE, ul);   // int8 -> half (exact)
                Cast(f3, hh, RoundMode::CAST_NONE, ul);    // half -> float (exact)
                Muls(f3, f3, a, ul);
                Add(accF, accF, f3, ul);
                if ((k & 63) == 63) {
                    Cast(accI2, accF, RoundMode::CAST_RINT, ul);
                    Add(accI, accI, accI2, ul);
                    Duplicate(accF, 0.0f, ul);
                }
            }
            Cast(accI2, accF, RoundMode::CAST_RINT, ul);
            Add(accI, accI, accI2, ul);
            SyncAllPipe();
            DataCopyExtParams w;
            w.blockCount = 1;
            w.blockLen = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(int32_t)));
            w.srcStride = 0;
            w.dstStride = 0;
            w.rsv = 0;
            DataCopyPad(gC_[m * N_ + nb], accI, w);
            SyncAllPipe();
        }
    }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::SiluStage()
{
    const float merged = gS1_.GetValue(0) * gS2_.GetValue(0);

    LocalTensor<int32_t> c = rawC_.Get<int32_t>();
    LocalTensor<float> f1 = f1_.Get<float>();
    LocalTensor<float> f2 = f2_.Get<float>();
    LocalTensor<float> f3 = f3_.Get<float>();
    LocalTensor<float> f4 = f4_.Get<float>();
    LocalTensor<float> f5 = f5_.Get<float>();
    LocalTensor<float> acc = accMax_.Get<float>();
    LocalTensor<half> h = hbuf_.Get<half>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    Duplicate(acc, 0.0f, tile_);

    for (int64_t off = 0; off < total_; off += tile_) {
        const int64_t cur = MinI(tile_, total_ - off);
        const uint32_t uc = static_cast<uint32_t>(cur);

        SyncAllPipe();
        DataCopyExtParams rp;
        rp.blockCount = 1;
        rp.blockLen = static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(int32_t)));
        rp.srcStride = 0;
        rp.dstStride = 0;
        rp.rsv = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(c, gC_[off], rp, pad);
        SyncAllPipe();

        Cast(f1, c, RoundMode::CAST_NONE, uc);
        Muls(f1, f1, merged, uc);
        Cast(h, f1, RoundMode::CAST_RINT, uc);
        Cast(f2, h, RoundMode::CAST_NONE, uc);

        Muls(f3, f2, -1.0f, uc);
        Mins(f3, f3, 0.0f, uc);
        Exp(f3, f3, uc);
        Adds(f3, f3, 1.0f, uc);
        Duplicate(f5, 1.0f, uc);
        Div(f4, f5, f3, uc);
        Mins(f3, f2, 0.0f, uc);
        Exp(f3, f3, uc);
        Adds(f5, f3, 1.0f, uc);
        Div(f3, f3, f5, uc);
        // Compare/Select are mask-based and skip the count%64 tail; run them on a
        // 64-rounded count with a zero-padded tail (pad from a 32B-aligned offset;
        // lanes in [uc,padStart) are beyond the real count and discarded).
        const uint32_t uc64 = ((uc + 63u) / 64u) * 64u;
        const uint32_t padStart = ((uc + 7u) / 8u) * 8u;
        if (uc64 > padStart) {
            Duplicate(f2[padStart], 0.0f, uc64 - padStart);
            Duplicate(f3[padStart], 0.0f, uc64 - padStart);
            Duplicate(f4[padStart], 0.0f, uc64 - padStart);
        }
        Duplicate(f5, 0.0f, uc64);
        Compare(mask, f2, f5, CMPMODE::LT, uc64);
        Select(f4, mask, f3, f4, SELMODE::VSEL_TENSOR_TENSOR_MODE, uc64);

        Cast(h, f4, RoundMode::CAST_RINT, uc);
        Cast(f3, h, RoundMode::CAST_NONE, uc);
        Mul(f3, f3, f2, uc);
        Cast(h, f3, RoundMode::CAST_RINT, uc);

        SyncAllPipe();
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(half)));
        wp.srcStride = 0;
        wp.dstStride = 0;
        wp.rsv = 0;
        DataCopyPad(gU_[off], h, wp);
        SyncAllPipe();

        Cast(f3, h, RoundMode::CAST_NONE, uc);
        Abs(f3, f3, uc);
        Max(acc, acc, f3, uc);
    }

    for (int32_t j = 64; j < tile_; j += 64) {
        Max(acc, acc, acc[j], 64);
    }
    LocalTensor<float> dst = redDst_.Get<float>();
    ReduceMax<float>(dst, acc, workTmp_.Get<float>(), 64, false);
    SyncAllPipe();
    float amax = dst.GetValue(0);
    if (!(amax > 0.0f)) {
        amax = 0.0f;
    }
    float yscale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;

    LocalTensor<float> sb = redDst_.Get<float>();
    Duplicate(sb, yscale, 8);
    SyncAllPipe();
    gYScale_.SetValue(0, yscale);
    SyncAllPipe();

    Duplicate(f5, yscale, tile_);
    SyncAllPipe();
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::QuantStage()
{
    LocalTensor<int32_t> ic = rawC_.Get<int32_t>();
    LocalTensor<float> f1 = f1_.Get<float>();
    LocalTensor<float> f5 = f5_.Get<float>();
    LocalTensor<half> h = hbuf_.Get<half>();
    LocalTensor<int8_t> o8 = i8buf_.Get<int8_t>();

    for (int64_t off = 0; off < total_; off += tile_) {
        const int64_t cur = MinI(tile_, total_ - off);
        const uint32_t uc = static_cast<uint32_t>(cur);

        SyncAllPipe();
        DataCopyExtParams rp;
        rp.blockCount = 1;
        rp.blockLen = static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(half)));
        rp.srcStride = 0;
        rp.dstStride = 0;
        rp.rsv = 0;
        DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        DataCopyPad(h, gU_[off], rp, pad);
        SyncAllPipe();

        Cast(f1, h, RoundMode::CAST_NONE, uc);
        Div(f1, f1, f5, uc);
        Cast(ic, f1, RoundMode::CAST_RINT, uc);
        Cast(f1, ic, RoundMode::CAST_NONE, uc);
        Mins(f1, f1, 127.0f, uc);
        Maxs(f1, f1, -128.0f, uc);
        Cast(h, f1, RoundMode::CAST_RINT, uc);
        Cast(o8, h, RoundMode::CAST_RINT, uc);

        SyncAllPipe();
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = static_cast<uint32_t>(cur);
        wp.srcStride = 0;
        wp.dstStride = 0;
        wp.rsv = 0;
        DataCopyPad(gY_[off], o8, wp);
        SyncAllPipe();
    }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::Process()
{
    if (total_ <= 0) {
        return;
    }
    VectorGemm();
    SiluStage();
    QuantStage();
}

} // namespace NsQuantMatmulSigmoidMulQuant
#endif // QUANTMATMULSIGMOIDMULQUANT_H

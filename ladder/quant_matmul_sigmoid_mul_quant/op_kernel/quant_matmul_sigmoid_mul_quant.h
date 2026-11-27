/*!
 * \file quant_matmul_sigmoid_mul_quant.h
 * \brief QuantMatmulSigmoidMulQuant on the cube unit via the AscendC Matmul (KFC)
 *        MIX AIC/AIV high level API.
 *
 * Pipeline (identical rounding points to the reference / golden):
 *   P = int32(x1 @ x2)                        (cube int8 Mmad -> exact int32)
 *   H = fp16(fp32(P) * s1 * s2)               (merged = s1*s2, ONE fp32 multiply)
 *   G = fp16(sigmoid(fp32(H)))
 *   U = fp16(fp32(G) * fp32(H))
 *   yScale = amax(|fp32(U)|) / 127            (== 1.0 when amax == 0)
 *   y = clip(rint(U / yScale), -128, 127)     (RNE round first, clip after)
 *
 * The int32 matmul result is written to the user workspace. The vector epilogue
 * runs in two passes over that buffer (pass 1: build U + global amax, pass 2:
 * quantise), exactly reproducing the reference rounding order.
 *
 * The transposeX1 / transposeX2 attrs are compile-time template parameters of the
 * Matmul types (the ND offsets then describe the physical layout), so all four
 * combinations are correctly served by the cube unit with NO scalar B gather.
 */
#ifndef QUANTMATMULSIGMOIDMULQUANT_H
#define QUANTMATMULSIGMOIDMULQUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_key.h"

namespace NsQuantMatmulSigmoidMulQuant {

using namespace AscendC;
using namespace matmul;

constexpr MatmulConfig QMSQ_CFG = GetMDLConfig();

template <bool AT, bool BT>
class QMSQ {
public:
    TPipe pipe;

    using AType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, AT>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, BT>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<AType, BType, CType, BiasType, QMSQ_CFG> mm;

    __aicore__ inline QMSQ(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR workspace, const QuantMatmulSigmoidMulQuantTilingData* t)
    {
        t_ = t;
        M_ = t->M;
        N_ = t->N;
        K_ = t->K;
        total_ = M_ * N_;
        tile_ = (t->tileLen > 0) ? t->tileLen : 1024;

        const int64_t cElems = total_;
        const int64_t x1Elems = AT ? (K_ * M_) : (M_ * K_);
        const int64_t x2Elems = BT ? (N_ * K_) : (K_ * N_);

        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), x1Elems);
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), x2Elems);
        s1_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Scale), 1);
        s2_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Scale), 1);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), cElems);
        ys_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale), 1);

        const int64_t cBytes = ((cElems * static_cast<int64_t>(sizeof(int32_t)) + 511) / 512) * 512;
        const int64_t uBytes = ((cElems * static_cast<int64_t>(sizeof(half)) + 511) / 512) * 512;
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), cElems);
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace + cBytes), cElems);
        /* one 32B slot per core for the per-core partial amax (cross core reduce) */
        part_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + cBytes + uBytes), 256);

        const uint32_t tl = static_cast<uint32_t>(tile_);
        pipe.InitBuffer(rawC_, tl * static_cast<uint32_t>(sizeof(int32_t)));
        pipe.InitBuffer(f1_, tl * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(f2_, tl * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(f3_, tl * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(f4_, tl * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(f5_, tl * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(accMax_, tl * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(hbuf_, tl * static_cast<uint32_t>(sizeof(half)));
        pipe.InitBuffer(i8buf_, tl * static_cast<uint32_t>(sizeof(int8_t)));
        pipe.InitBuffer(maskBuf_, tl * static_cast<uint32_t>(sizeof(uint8_t)));
        pipe.InitBuffer(workTmp_, 2048);
        pipe.InitBuffer(redDst_, 256);
        pipe.InitBuffer(partUb_, 2048);

        merged_ = s1_.GetValue(0) * s2_.GetValue(0);
    }

    __aicore__ inline void Process()
    {
        if (total_ <= 0 || M_ <= 0 || N_ <= 0 || K_ <= 0) {
            mm.End();
            return;
        }
        const int64_t gridM = t_->gridM > 0 ? t_->gridM : 1;
        const int64_t gridN = t_->gridN > 0 ? t_->gridN : 1;
        const int64_t core = static_cast<int64_t>(GetBlockIdx());
        const int64_t mPer = (M_ + gridM - 1) / gridM;
        const int64_t nPer = (N_ + gridN - 1) / gridN;
        const int64_t mi = (gridN > 0) ? (core / gridN) : 0;
        const int64_t ni = (gridN > 0) ? (core % gridN) : 0;
        const int64_t m0 = mi * mPer;
        const int64_t n0 = ni * nPer;
        const int64_t m1 = (m0 < M_) ? MinI(m0 + mPer, M_) : 0;
        const int64_t n1 = (n0 < N_) ? MinI(n0 + nPer, N_) : 0;
        const bool hasWork = (m0 < m1) && (n0 < n1);

        if (gridM * gridN == 1) {
            /* ---- single core: the whole result is visible to this core ---- */
            MatmulAll(0, M_, 0, N_);
            AmaxRect(0, M_, 0, N_);
            const float yscale = FinishAmax(ReduceAcc());
            PublishYScale(yscale);
            Duplicate(f5_.Get<float>(), yscale, static_cast<uint32_t>(tile_));
            QuantRect(0, M_, 0, N_);
            mm.End();
            return;
        }
        /* ---- multi core: each core owns one (rows x cols) rectangle ---- */
        float localAmax = 0.0f;
        if (hasWork) {
            MatmulAll(m0, m1, n0, n1);
            AmaxRect(m0, m1, n0, n1);
            localAmax = ReduceAcc();
        }
        /* publish this core's partial amax, then an AIV barrier: each block's IterateAll
         * returns only once its own AIC matmul is complete, so after the barrier every
         * int32 tile is in GM and every core can read all partials. */
        PartWrite(core, localAmax);
        AscendC::SyncAll<true>();
        const float amax = PartReadMax(gridM * gridN);
        const float yscale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        if (core == 0) {
            PublishYScale(yscale);
        }
        Duplicate(f5_.Get<float>(), yscale, static_cast<uint32_t>(tile_));
        if (hasWork) {
            QuantRect(m0, m1, n0, n1);
        }
        mm.End();
    }

private:
    __aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline void SyncAllPipe() { PipeBarrier<PIPE_ALL>(); }

    /* int8 matmul of the (rows x cols) tile; the int32 result goes to c_ */
    __aicore__ inline void MatmulAll(int64_t m0, int64_t m1, int64_t n0, int64_t n1)
    {
        const int64_t curM = m1 - m0;
        const int64_t curN = n1 - n0;
        /* orgKa / orgKb / orgKc are the *row strides* of A / B / C and must be the real
         * leading dims; the 3-arg SetOrgShape would set orgKb = K and overflow its 16-bit
         * field once K >= 65536. */
        mm.SetOrgShape(M_, N_, AT ? M_ : K_, BT ? K_ : N_, N_);
        mm.SetSingleShape(curM, curN, K_);
        mm.SetTensorA(x1_[AT ? m0 : (m0 * K_)], AT);
        mm.SetTensorB(x2_[BT ? (n0 * K_) : n0], BT);
        mm.IterateAll(c_[m0 * N_ + n0], false);
    }

    /* per-core partial amax: one 32B slot per core, written with a 4 byte payload */
    __aicore__ inline void PartWrite(int64_t core, float v)
    {
        LocalTensor<float> t = partUb_.Get<float>();
        SyncAllPipe();
        t.SetValue(0, v);
        SyncAllPipe();
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = static_cast<uint32_t>(sizeof(float));
        wp.srcStride = 0;
        wp.dstStride = 0;
        wp.rsv = 0;
        DataCopyPad(part_[core * 8], t, wp);
        SyncAllPipe();
    }

    __aicore__ inline float PartReadMax(int64_t cores)
    {
        LocalTensor<float> t = partUb_.Get<float>();
        SyncAllPipe();
        DataCopyExtParams rp;
        rp.blockCount = 1;
        rp.blockLen = static_cast<uint32_t>(cores * 32);
        rp.srcStride = 0;
        rp.dstStride = 0;
        rp.rsv = 0;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(t, part_[0], rp, pad);
        SyncAllPipe();
        float m = 0.0f;
        for (int64_t i = 0; i < cores; ++i) {
            const float v = t.GetValue(static_cast<int32_t>(i * 8));
            if (v > m) {
                m = v;
            }
        }
        SyncAllPipe();
        return m;
    }

    __aicore__ inline void PublishYScale(float yscale)
    {
        SyncAllPipe();
        ys_.SetValue(0, yscale);
        SyncAllPipe();
    }


    /* one chunk of pass 1: U = fp16(g*h) into the workspace, absorb |U| into acc */
    __aicore__ inline void AmaxChunk(int64_t off, int64_t cur)
    {
        LocalTensor<int32_t> c = rawC_.Get<int32_t>();
        LocalTensor<float> f1 = f1_.Get<float>();
        LocalTensor<float> f2 = f2_.Get<float>();
        LocalTensor<float> f3 = f3_.Get<float>();
        LocalTensor<float> f4 = f4_.Get<float>();
        LocalTensor<float> f5 = f5_.Get<float>();
        LocalTensor<float> acc = accMax_.Get<float>();
        LocalTensor<half> h = hbuf_.Get<half>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        const uint32_t uc = static_cast<uint32_t>(cur);

        SyncAllPipe();
        DataCopyExtParams rp;
        rp.blockCount = 1;
        rp.blockLen = static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(int32_t)));
        rp.srcStride = 0;
        rp.dstStride = 0;
        rp.rsv = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(c, c_[off], rp, pad);
        SyncAllPipe();

        Cast(f1, c, RoundMode::CAST_NONE, uc);
        Muls(f1, f1, merged_, uc);
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
        /* Compare/Select are mask ops that skip the count%64 tail: run them on a
         * 64-rounded count with the tail zero filled from a 32B aligned offset. */
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
        DataCopyPad(u_[off], h, wp);
        SyncAllPipe();

        Cast(f3, h, RoundMode::CAST_NONE, uc);
        Abs(f3, f3, uc);
        Max(acc, acc, f3, uc);
    }

    /* pass 1 over the (rows x cols) rectangle */
    __aicore__ inline void AmaxRect(int64_t m0, int64_t m1, int64_t n0, int64_t n1)
    {
        Duplicate(accMax_.Get<float>(), 0.0f, static_cast<uint32_t>(tile_));
        if (n0 == 0 && n1 == N_) {
            /* gridN == 1: the row block is contiguous, walk it as one flat run */
            const int64_t f0 = m0 * N_;
            const int64_t f1 = m1 * N_;
            for (int64_t off = f0; off < f1; off += tile_) {
                AmaxChunk(off, MinI(tile_, f1 - off));
            }
            return;
        }
        for (int64_t row = m0; row < m1; ++row) {
            for (int64_t col = n0; col < n1; col += tile_) {
                AmaxChunk(row * N_ + col, MinI(tile_, n1 - col));
            }
        }
    }

    /* reduce the per-lane |U| accumulator of this core to one scalar */
    __aicore__ inline float ReduceAcc()
    {
        LocalTensor<float> acc = accMax_.Get<float>();
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
        return amax;
    }

    __aicore__ inline float FinishAmax(float amax)
    {
        const float yscale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        return yscale;
    }

    /* one chunk of pass 2: y = clip(rint(U / yScale)) */
    __aicore__ inline void QuantChunk(int64_t off, int64_t cur)
    {
        LocalTensor<int32_t> ic = rawC_.Get<int32_t>();
        LocalTensor<float> f1 = f1_.Get<float>();
        LocalTensor<float> f5 = f5_.Get<float>();
        LocalTensor<half> h = hbuf_.Get<half>();
        LocalTensor<int8_t> o8 = i8buf_.Get<int8_t>();
        const uint32_t uc = static_cast<uint32_t>(cur);

        SyncAllPipe();
        DataCopyExtParams rp;
        rp.blockCount = 1;
        rp.blockLen = static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(half)));
        rp.srcStride = 0;
        rp.dstStride = 0;
        rp.rsv = 0;
        DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        DataCopyPad(h, u_[off], rp, pad);
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
        DataCopyPad(y_[off], o8, wp);
        SyncAllPipe();
    }

    /* pass 2 over the (rows x cols) rectangle */
    __aicore__ inline void QuantRect(int64_t m0, int64_t m1, int64_t n0, int64_t n1)
    {
        if (n0 == 0 && n1 == N_) {
            const int64_t f0 = m0 * N_;
            const int64_t f1 = m1 * N_;
            for (int64_t off = f0; off < f1; off += tile_) {
                QuantChunk(off, MinI(tile_, f1 - off));
            }
            return;
        }
        for (int64_t row = m0; row < m1; ++row) {
            for (int64_t col = n0; col < n1; col += tile_) {
                QuantChunk(row * N_ + col, MinI(tile_, n1 - col));
            }
        }
    }

    GlobalTensor<int8_t> x1_;
    GlobalTensor<int8_t> x2_;
    GlobalTensor<float> s1_;
    GlobalTensor<float> s2_;
    GlobalTensor<int8_t> y_;
    GlobalTensor<float> ys_;
    GlobalTensor<int32_t> c_;
    GlobalTensor<half> u_;
    GlobalTensor<float> part_;

    TBuf<TPosition::VECCALC> rawC_;
    TBuf<TPosition::VECCALC> f1_;
    TBuf<TPosition::VECCALC> f2_;
    TBuf<TPosition::VECCALC> f3_;
    TBuf<TPosition::VECCALC> f4_;
    TBuf<TPosition::VECCALC> f5_;
    TBuf<TPosition::VECCALC> accMax_;
    TBuf<TPosition::VECCALC> hbuf_;
    TBuf<TPosition::VECCALC> i8buf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> workTmp_;
    TBuf<TPosition::VECCALC> redDst_;
    TBuf<TPosition::VECCALC> partUb_;

    const QuantMatmulSigmoidMulQuantTilingData* t_ = nullptr;
    int64_t M_ = 0;
    int64_t N_ = 0;
    int64_t K_ = 0;
    int64_t total_ = 0;
    int32_t tile_ = 1024;
    float merged_ = 1.0f;
};

template <bool AT, bool BT>
__aicore__ inline void RunQmsq(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale,
                               GM_ADDR userWS, const QuantMatmulSigmoidMulQuantTilingData* td)
{
    QMSQ<AT, BT> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &(td->mm));
    op.Init(x1, x2, x1Scale, x2Scale, y, yScale, userWS, td);
    op.Process();
}

} // namespace NsQuantMatmulSigmoidMulQuant
#endif // QUANTMATMULSIGMOIDMULQUANT_H

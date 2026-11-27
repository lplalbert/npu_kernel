/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 on the cube unit via the AscendC Matmul (KFC) high level API.
 *        out[m,n] = bf16( fp32(int32(x1 @ x2)) * scale[n] * pertokenScale[m] )
 *
 * Multi-core output grid (one disjoint (rows x cols) rectangle per core) and a
 * K split into chunks <= KC so that k >= 65536 never overflows the 16 bit cube K
 * fields and never falls into the slow loop-based GM->L1 load path. The int32
 * partial products of the K chunks are accumulated exactly with the atomic-C mode
 * of IterateAll. The vector epilogue processes whole (rows x cols) sub-tiles so the
 * number of pipe synchronisations is O(tiles) instead of O(rows).
 */
#ifndef QBMM_MM_H
#define QBMM_MM_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_mat_mul_v3_tiling_data.h"

namespace NsQBMM {

using namespace AscendC;
using namespace matmul;

/* Two compile-time Matmul configs:
 *  - FAST (intrinsicsCheck=false, the default) is used whenever every physical leading
 *    dim fits 16 bits (k<=65535, n<=65535). It has no loop-based load, so k==65535 does
 *    NOT fall into the slow path.
 *  - SLOW (intrinsicsCheck=true) is the correct fallback when a leading dim exceeds
 *    65535 (only k>65535 or n>65535), where the loop-based GM->L1 load is required. */
template <bool SLOW>
struct QbmmCfgPick;
template <>
struct QbmmCfgPick<false> {
    static constexpr MatmulConfig v = GetMDLConfig();
};
template <>
struct QbmmCfgPick<true> {
    static constexpr MatmulConfig v = GetMDLConfig(true);
};
constexpr int64_t EP_N = 256;     // epilogue column chunk (elements)
constexpr int64_t EP_R = 32;      // epilogue row chunk (rows)
constexpr int64_t STAGE_UB = 98304;  // UB staging buffer for the compact K-slice copy

template <bool AT, bool BT, bool SLOW>
class QBMM {
public:
    TPipe pipe;

    using AType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, AT>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, BT>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<AType, BType, CType, BiasType, QbmmCfgPick<SLOW>::v> mm;

    __aicore__ inline QBMM(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out,
                                GM_ADDR workspace, const QuantBatchMatMulV3TilingData* t)
    {
        t_ = t;
        m_ = t->m; n_ = t->n; k_ = t->k;
        scaleLen_ = t->scaleLen; ptLen_ = t->ptLen;
        x1_.SetGlobalBuffer((__gm__ int8_t*)x1);
        x2_.SetGlobalBuffer((__gm__ int8_t*)x2);
        scale_.SetGlobalBuffer((__gm__ float*)scale);
        pt_.SetGlobalBuffer((__gm__ float*)pertokenScale);
        y_.SetGlobalBuffer((__gm__ bfloat16_t*)out);
        c_.SetGlobalBuffer((__gm__ int32_t*)workspace, static_cast<uint64_t>(m_) * n_ + 256);
        pipe.InitBuffer(bC_, static_cast<uint32_t>(EP_R * EP_N * sizeof(int32_t)));
        pipe.InitBuffer(bF_, static_cast<uint32_t>(EP_R * EP_N * sizeof(float)));
        pipe.InitBuffer(bS_, static_cast<uint32_t>(EP_N * sizeof(float)));
        pipe.InitBuffer(bQ_, static_cast<uint32_t>(EP_R * EP_N * sizeof(bfloat16_t)));
        pipe.InitBuffer(bPt_, static_cast<uint32_t>(EP_R * sizeof(float)));
        pipe.InitBuffer(stageBuf_, static_cast<uint32_t>(STAGE_UB));
        wsBase_ = workspace;
        x1Base_ = x1;
        x2Base_ = x2;
        cEpi_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), static_cast<uint64_t>(m_) * n_ + 256);
        cPitch_ = n_;
        yPitch_ = n_;
        cBytes_ = (((m_ * n_ * static_cast<int64_t>(sizeof(int32_t))) + 511) / 512) * 512;
        const int64_t kp = (t->kc > 0) ? static_cast<int64_t>(t->kc) : 1;
        aReg_ = (((m_ * kp) + 511) / 512) * 512;
        bReg_ = (((n_ * kp) + 511) / 512) * 512;
        if (t->np > 0) {
            cStgReg_ = (((m_ * static_cast<int64_t>(t->np) * sizeof(int32_t)) + 511) / 512) * 512;
            bStgReg_ = (((k_ * static_cast<int64_t>(t->np)) + 511) / 512) * 512;
            nStgOff_ = static_cast<int64_t>(t->gridN) * cStgReg_;
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t core = static_cast<int64_t>(GetBlockIdx());
        const int64_t gridK = t_->gridK > 0 ? static_cast<int64_t>(t_->gridK) : 1;
        if (gridK > 1) {
            /* K split across cores: A/B read once, int32 partials atomically summed. */
            ZeroCSlice(core, gridK);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<true>();
            const int64_t kPer = (t_->kc > 0) ? static_cast<int64_t>(t_->kc) : ((k_ + gridK - 1) / gridK);
            const int64_t kc0 = core * kPer;
            if (kc0 < k_) {
                const int64_t kc1 = MinI(k_, kc0 + kPer);
                const int64_t kc = kc1 - kc0;
                const bool nA = (t_->needA != 0);
                const bool nB = (t_->needB != 0);
                bool aT = AT;
                bool bT = BT;
                GlobalTensor<int8_t> gA;
                GlobalTensor<int8_t> gB;
                if (nA) {
                    /* A is (m,k) with k > 65535: stage this K slice to a compact (m x kc)
                     * workspace (row stride = kc) so orgKa = kc fits 16 bits. */
                    GM_ADDR pa = wsBase_ + cBytes_ + core * aReg_;
                    StageSlice(x1_, pa, m_, k_, kc0, kc);
                    gA.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(pa), m_ * kc);
                    aT = false;
                } else {
                    gA.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1Base_ + (AT ? (kc0 * m_) : kc0)),
                                       m_ * k_ + 1);
                }
                if (nB) {
                    /* B is (n,k) with k > 65535: stage this K slice to a compact (n x kc)
                     * workspace so orgKb = kc fits 16 bits (still read as transposed B). */
                    GM_ADDR pb = wsBase_ + cBytes_ + (nA ? (gridK * aReg_) : 0) + core * bReg_;
                    StageSlice(x2_, pb, n_, k_, kc0, kc);
                    gB.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(pb), n_ * kc);
                    bT = true;
                } else {
                    gB.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2Base_ + (BT ? kc0 : (kc0 * n_))),
                                       k_ * n_ + 1);
                }
                mm.SetOrgShape(m_, n_, nA ? kc : (AT ? m_ : k_), nB ? kc : (BT ? k_ : n_), n_);
                mm.SetSingleShape(m_, n_, kc);
                mm.SetTensorA(gA, aT);
                mm.SetTensorB(gB, bT);
                mm.IterateAll(c_[0], true);  // enAtomic add into the zeroed C
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<true>();
            const int64_t rPer = (m_ + gridK - 1) / gridK;
            const int64_t r0 = core * rPer;
            const int64_t r1 = MinI(m_, r0 + rPer);
            if (r0 < r1) { Epilogue(r0, 0, r1 - r0, n_, 0); }
            mm.End();
            return;
        }
        if (t_->np > 0) {
            /* N-split (n > 65535): each core owns nPer columns of the output; B's N-tile is
             * staged so orgKb fits and C is a compact (m x nPer) per-core buffer. */
            const int64_t gridN = t_->gridN > 0 ? static_cast<int64_t>(t_->gridN) : 1;
            const int64_t nPer = static_cast<int64_t>(t_->np);
            const int64_t n0 = core * nPer;
            const int64_t ncol = MinI(nPer, n_ - n0);
            if (ncol > 0) {
                const bool nB = (t_->needB != 0);
                GlobalTensor<int8_t> gA;
                GlobalTensor<int8_t> gB;
                gA.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1Base_), m_ * k_ + 1);
                if (nB) {
                    GM_ADDR pb = wsBase_ + nStgOff_ + core * bStgReg_;
                    StageSlice(x2_, pb, k_, n_, n0, ncol);
                    gB.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(pb), k_ * ncol);
                } else {
                    gB.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2Base_ + n0 * k_), n_ * k_ + 1);
                }
                const int64_t orgKa = AT ? m_ : k_;
                const int64_t orgKb = nB ? ncol : (BT ? k_ : n_);
                GM_ADDR pc = wsBase_ + core * cStgReg_;
                GlobalTensor<int32_t> gC;
                gC.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pc), m_ * ncol);
                mm.SetOrgShape(m_, ncol, orgKa, orgKb, ncol);
                mm.SetSingleShape(m_, ncol, k_);
                mm.SetTensorA(gA, AT);
                mm.SetTensorB(gB, BT);
                mm.IterateAll(gC, false);
                cEpi_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pc), m_ * ncol);
                cPitch_ = ncol;
                yPitch_ = n_;
                Epilogue(0, 0, m_, ncol, n0);
            }
            mm.End();
            return;
        }
        int64_t gridM = t_->gridM > 0 ? t_->gridM : 1;
        int64_t gridN = t_->gridN > 0 ? t_->gridN : 1;
        int64_t mPer = (m_ + gridM - 1) / gridM;
        int64_t nPer = (n_ + gridN - 1) / gridN;
        int64_t tiles = gridM * gridN;
        for (int64_t ti = core; ti < tiles; ti += static_cast<int64_t>(GetBlockNum())) {
            int64_t mi = ti / gridN;
            int64_t ni = ti % gridN;
            int64_t m0 = mi * mPer;
            int64_t n0 = ni * nPer;
            int64_t curM = MinI(mPer, m_ - m0);
            int64_t curN = MinI(nPer, n_ - n0);
            if (curM <= 0 || curN <= 0) { continue; }
            /* K split into chunks < 65536; first chunk overwrites C, later chunks add.
             * t->kc is 0 when a physical leading dim exceeds 65535 (only possible for
             * k >= 65536); then a single chunk + the loop-based fallback is used. */
            const int64_t kstep = (t_->kc > 0) ? static_cast<int64_t>(t_->kc) : k_;
            for (int64_t k0 = 0; k0 < k_; k0 += kstep) {
                const int64_t kc = MinI(kstep, k_ - k0);
                const int64_t offA = AT ? (m0 + k0 * m_) : (m0 * k_ + k0);
                const int64_t offB = BT ? (n0 * k_ + k0) : (k0 * n_ + n0);
                /* orgKa/orgKb are the row strides of A/B; use the real leading dims. */
                mm.SetOrgShape(m_, n_, AT ? m_ : k_, BT ? k_ : n_, n_);
                mm.SetSingleShape(curM, curN, kc);
                mm.SetTensorA(x1_[offA], AT);
                mm.SetTensorB(x2_[offB], BT);
                mm.IterateAll(c_[m0 * n_ + n0], k0 > 0);
            }
            Epilogue(m0, n0, curM, curN, n0);
        }
        mm.End();
    }

private:
    __aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }

    /* Copy the (rows x kc) column block starting at column kc0 of a row-major (rows x kfull)
     * int8 matrix into a compact destination with row stride kc (GM -> UB -> GM). */
    __aicore__ inline void StageSlice(GlobalTensor<int8_t>& src, GM_ADDR dst, int64_t rows,
                                      int64_t kfull, int64_t kc0, int64_t kc)
    {
        if (rows <= 0 || kc <= 0) { return; }
        GlobalTensor<int8_t> d;
        d.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(dst), rows * kc);
        LocalTensor<int8_t> ub = stageBuf_.Get<int8_t>();
        int64_t rmax = STAGE_UB / kc;
        if (rmax < 1) { rmax = 1; }
        for (int64_t r = 0; r < rows; r += rmax) {
            const int64_t rb = MinI(rmax, rows - r);
            DataCopyExtParams gp;
            gp.blockCount = static_cast<uint16_t>(rb);
            gp.blockLen = static_cast<uint32_t>(kc);
            gp.srcStride = static_cast<uint32_t>(kfull - kc);
            gp.dstStride = 0;
            gp.rsv = 0;
            DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
            DataCopyPad(ub, src[r * kfull + kc0], gp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyExtParams wp;
            wp.blockCount = static_cast<uint16_t>(rb);
            wp.blockLen = static_cast<uint32_t>(kc);
            wp.srcStride = 0;
            wp.dstStride = 0;
            wp.rsv = 0;
            DataCopyPad(d[r * kc], ub, wp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void ZeroCSlice(int64_t core, int64_t gridK)
    {
        LocalTensor<int32_t> z = bC_.Get<int32_t>();
        const int64_t tile = EP_R * EP_N;
        Duplicate(z, 0, static_cast<int32_t>(tile));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        const int64_t total = m_ * n_;
        const int64_t per = ((total + gridK - 1) / gridK + tile - 1) / tile * tile;
        const int64_t s = core * per;
        const int64_t e = MinI(total, s + per);
        for (int64_t off = s; off < e; off += tile) {
            const int64_t cnt = MinI(tile, e - off);
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(int32_t)));
            p.srcStride = 0; p.dstStride = 0; p.rsv = 0;
            DataCopyPad(c_[off], z, p);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    }

    /* Whole (rows x cols) sub-tile epilogue: one strided copy in, vector scaling per
     * row (scale broadcast over the column chunk, pertoken per row), one strided copy
     * out. Few pipe synchronisations per row chunk instead of per single row. */
    __aicore__ inline void Epilogue(int64_t m0, int64_t cCol0, int64_t rows, int64_t cols, int64_t yCol0)
    {
        const bool scalarScale = (scaleLen_ <= 1);
        const float sv = scalarScale ? scale_.GetValue(0) : 0.0f;
        const float ptScalar = (ptLen_ <= 1) ? pt_.GetValue(0) : 0.0f;

        for (int64_t c0 = 0; c0 < cols; c0 += EP_N) {
            const int64_t ncol = MinI(EP_N, cols - c0);
            const int64_t pitchF = ((ncol + 7) / 8) * 8;   /* 32B aligned fp32 count */
            const int64_t pitchB = ((ncol + 15) / 16) * 16; /* 32B aligned bf16 count */
            LocalTensor<float> sV = bS_.Get<float>();
            if (!scalarScale) {
                DataCopyExtParams sp;
                sp.blockCount = 1;
                sp.blockLen = static_cast<uint32_t>(ncol * static_cast<int64_t>(sizeof(float)));
                sp.srcStride = 0; sp.dstStride = 0; sp.rsv = 0;
                DataCopyPadExtParams<float> spad{false, 0, 0, 0};
                DataCopyPad(sV, scale_[yCol0 + c0], sp, spad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
            for (int64_t r0 = 0; r0 < rows; r0 += EP_R) {
                const int64_t rb = MinI(EP_R, rows - r0);
                LocalTensor<int32_t> qc = bC_.Get<int32_t>();
                DataCopyExtParams cp;
                cp.blockCount = static_cast<uint16_t>(rb);
                cp.blockLen = static_cast<uint32_t>(ncol * static_cast<int64_t>(sizeof(int32_t)));
                cp.srcStride = static_cast<uint32_t>((cPitch_ - ncol) * static_cast<int64_t>(sizeof(int32_t)));
                cp.dstStride = static_cast<uint32_t>((pitchF - ncol) * static_cast<int64_t>(sizeof(int32_t)) / 32);
                cp.rsv = 0;
                DataCopyPadExtParams<int32_t> cpad{false, 0, 0, 0};
                DataCopyPad(qc, cEpi_[((m0 + r0) * cPitch_ + cCol0) + c0], cp, cpad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                LocalTensor<float> ptB = bPt_.Get<float>();
                if (ptLen_ > 1) {
                    DataCopyExtParams pp;
                    pp.blockCount = 1;
                    pp.blockLen = static_cast<uint32_t>(rb * static_cast<int64_t>(sizeof(float)));
                    pp.srcStride = 0; pp.dstStride = 0; pp.rsv = 0;
                    DataCopyPadExtParams<float> ppad{false, 0, 0, 0};
                    DataCopyPad(ptB, pt_[m0 + r0], pp, ppad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                }
                LocalTensor<float> f = bF_.Get<float>();
                for (int64_t r = 0; r < rb; ++r) {
                    const float pv = (ptLen_ <= 1) ? ptScalar : ptB.GetValue(static_cast<int32_t>(r));
                    Cast(f[r * pitchF], qc[r * pitchF], RoundMode::CAST_NONE, static_cast<int32_t>(ncol));
                    if (scalarScale) {
                        Muls(f[r * pitchF], f[r * pitchF], sv, static_cast<int32_t>(ncol));
                    } else {
                        Mul(f[r * pitchF], f[r * pitchF], sV, static_cast<int32_t>(ncol));
                    }
                    Muls(f[r * pitchF], f[r * pitchF], pv, static_cast<int32_t>(ncol));
                }
                LocalTensor<bfloat16_t> qy = bQ_.Get<bfloat16_t>();
                for (int64_t r = 0; r < rb; ++r) {
                    Cast(qy[r * pitchB], f[r * pitchF], RoundMode::CAST_RINT, static_cast<int32_t>(ncol));
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams wp;
                wp.blockCount = static_cast<uint16_t>(rb);
                wp.blockLen = static_cast<uint32_t>(ncol * static_cast<int64_t>(sizeof(bfloat16_t)));
                wp.srcStride = static_cast<uint32_t>((pitchB - ncol) * static_cast<int64_t>(sizeof(bfloat16_t)) / 32);
                wp.dstStride = static_cast<uint32_t>((yPitch_ - ncol) * static_cast<int64_t>(sizeof(bfloat16_t)));
                wp.rsv = 0;
                DataCopyPad(y_[((m0 + r0) * yPitch_ + yCol0) + c0], qy, wp);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                /* bC_/bF_/bQ_ reused by the next row chunk: WAR on the vector reads */
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
    }

    GlobalTensor<int8_t> x1_;
    GlobalTensor<int8_t> x2_;
    GlobalTensor<float> scale_;
    GlobalTensor<float> pt_;
    GlobalTensor<bfloat16_t> y_;
    GlobalTensor<int32_t> c_;
    GlobalTensor<int32_t> cEpi_;
    int64_t cPitch_ = 0, yPitch_ = 0;
    int64_t cStgReg_ = 0, bStgReg_ = 0, nStgOff_ = 0;
    TBuf<TPosition::VECCALC> bC_;
    TBuf<TPosition::VECCALC> bF_;
    TBuf<TPosition::VECCALC> bS_;
    TBuf<TPosition::VECCALC> bQ_;
    TBuf<TPosition::VECCALC> bPt_;
    TBuf<TPosition::VECCALC> stageBuf_;
    GM_ADDR wsBase_ = nullptr;
    GM_ADDR x1Base_ = nullptr;
    GM_ADDR x2Base_ = nullptr;
    int64_t cBytes_ = 0, aReg_ = 0, bReg_ = 0;
    const QuantBatchMatMulV3TilingData* t_ = nullptr;
    int64_t m_ = 0, n_ = 0, k_ = 0;
    int32_t scaleLen_ = 1, ptLen_ = 1;
};

}  // namespace NsQBMM
#endif

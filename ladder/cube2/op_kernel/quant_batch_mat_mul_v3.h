/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 on the cube unit via the AscendC Matmul (KFC) high level API.
 *        out[m,n] = bf16( fp32(int32(x1 @ x2)) * scale[n] * pertokenScale[m] )
 *        K is split into chunks <= KC so that k >= 65536 does not overflow the
 *        uint16 K fields of the cube tiling; the int32 partial products are
 *        accumulated exactly with the atomic-C mode of IterateAll.
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

constexpr MatmulConfig QBMM_CFG = GetMDLConfig(true);  // intrinsicsCheck: loop-based GM->L1 load for inner axis >= 65535
constexpr int64_t EP_N = 64;      // epilogue vector block (one 64-lane chunk)
constexpr int64_t KC = (int64_t)1 << 40;  // single K chunk (intrinsicsCheck handles k >= 65535)

template <bool AT, bool BT>
class QBMM {
public:
    TPipe pipe;

    using AType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, AT>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, BT>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<AType, BType, CType, BiasType, QBMM_CFG> mm;

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
        pipe.InitBuffer(bC_, static_cast<uint32_t>(EP_N * sizeof(int32_t)));
        pipe.InitBuffer(bF_, static_cast<uint32_t>(EP_N * sizeof(float)));
        pipe.InitBuffer(bS_, static_cast<uint32_t>(EP_N * sizeof(float)));
        pipe.InitBuffer(bQ_, static_cast<uint32_t>(EP_N * sizeof(bfloat16_t)));
    }

    __aicore__ inline void Process()
    {
        /* Output grid chosen by the host: gridM x gridN tiles, one per core.
         * Each tile owns a disjoint m0/n0 rectangle so no output is shared. */
        int64_t gridM = t_->gridM > 0 ? t_->gridM : 1;
        int64_t gridN = t_->gridN > 0 ? t_->gridN : 1;
        int64_t mPer = (m_ + gridM - 1) / gridM;
        int64_t nPer = (n_ + gridN - 1) / gridN;
        int64_t tiles = gridM * gridN;
        int64_t core = static_cast<int64_t>(GetBlockIdx());
        for (int64_t ti = core; ti < tiles; ti += static_cast<int64_t>(GetBlockNum())) {
            int64_t mi = ti / gridN;
            int64_t ni = ti % gridN;
            int64_t m0 = mi * mPer;
            int64_t n0 = ni * nPer;
            int64_t curM = MinI(mPer, m_ - m0);
            int64_t curN = MinI(nPer, n_ - n0);
            if (curM <= 0 || curN <= 0) { continue; }
            /* K is split so that every chunk keeps the cube K fields < 65536.
             * The first chunk overwrites C, later chunks atomically add into it. */
            for (int64_t k0 = 0; k0 < k_; k0 += KC) {
                const int64_t kc = MinI(KC, k_ - k0);
                const int64_t offA = AT ? (m0 + k0 * m_) : (m0 * k_ + k0);
                const int64_t offB = BT ? (n0 * k_ + k0) : (k0 * n_ + n0);
                /* orgKa / orgKb are the *row strides* of A and B, so they must be the
                 * real leading dims (k_ for a (m,k) A, n_ for a (k,n) B) and not the
                 * reduce length. The 3-arg SetOrgShape(M,N,K) sets orgKb = orgK, which
                 * overflows the 16-bit K field once k >= 65536 and corrupts B. */
                mm.SetOrgShape(m_, n_, AT ? m_ : k_, BT ? k_ : n_, n_);
                mm.SetSingleShape(curM, curN, kc);
                mm.SetTensorA(x1_[offA], AT);
                mm.SetTensorB(x2_[offB], BT);
                mm.IterateAll(c_[m0 * n_ + n0], k0 > 0);
            }
            Epilogue(m0, n0, curM, curN);
        }
        mm.End();
    }

private:
    __aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }

    __aicore__ inline void Epilogue(int64_t m0, int64_t n0, int64_t rows, int64_t cols)
    {
        const bool scalarScale = (scaleLen_ <= 1);
        const float sv = scalarScale ? scale_.GetValue(0) : 0.0f;
        for (int64_t r = 0; r < rows; ++r) {
            const float pv = (ptLen_ <= 1) ? pt_.GetValue(0) : pt_.GetValue(m0 + r);
            int64_t c0 = 0;
            while (c0 < cols) {
                const int64_t ncol = MinI(EP_N, cols - c0);
                const int32_t cnt = static_cast<int32_t>(EP_N);
                LocalTensor<int32_t> qc = bC_.Get<int32_t>();
                DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = static_cast<uint32_t>(ncol * static_cast<int64_t>(sizeof(int32_t)));
                cp.srcStride = 0; cp.dstStride = 0; cp.rsv = 0;
                DataCopyPadExtParams<int32_t> cpad{false, 0, 0, 0};
                DataCopyPad(qc, c_[((m0 + r) * n_ + n0) + c0], cp, cpad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                LocalTensor<float> f = bF_.Get<float>();
                Cast(f, qc, RoundMode::CAST_NONE, cnt);
                if (scalarScale) {
                    Muls(f, f, sv, cnt);
                } else {
                    LocalTensor<float> s = bS_.Get<float>();
                    DataCopyExtParams sp;
                    sp.blockCount = 1;
                    sp.blockLen = static_cast<uint32_t>(ncol * static_cast<int64_t>(sizeof(float)));
                    sp.srcStride = 0; sp.dstStride = 0; sp.rsv = 0;
                    DataCopyPadExtParams<float> spad{false, 0, 0, 0};
                    DataCopyPad(s, scale_[n0 + c0], sp, spad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    Mul(f, f, s, cnt);
                }
                Muls(f, f, pv, cnt);

                LocalTensor<bfloat16_t> qy = bQ_.Get<bfloat16_t>();
                Cast(qy, f, RoundMode::CAST_RINT, cnt);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams wp;
                wp.blockCount = 1;
                wp.blockLen = static_cast<uint32_t>(ncol * static_cast<int64_t>(sizeof(bfloat16_t)));
                wp.srcStride = 0; wp.dstStride = 0; wp.rsv = 0;
                DataCopyPad(y_[((m0 + r) * n_ + n0) + c0], qy, wp);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                /* bC_ / bQ_ are reused by the next chunk: WAR on the vector reads */
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                c0 += EP_N;
            }
        }
    }

    GlobalTensor<int8_t> x1_;
    GlobalTensor<int8_t> x2_;
    GlobalTensor<float> scale_;
    GlobalTensor<float> pt_;
    GlobalTensor<bfloat16_t> y_;
    GlobalTensor<int32_t> c_;
    TBuf<TPosition::VECCALC> bC_;
    TBuf<TPosition::VECCALC> bF_;
    TBuf<TPosition::VECCALC> bS_;
    TBuf<TPosition::VECCALC> bQ_;
    const QuantBatchMatMulV3TilingData* t_ = nullptr;
    int64_t m_ = 0, n_ = 0, k_ = 0;
    int32_t scaleLen_ = 1, ptLen_ = 1;
};

}  // namespace NsQBMM
#endif

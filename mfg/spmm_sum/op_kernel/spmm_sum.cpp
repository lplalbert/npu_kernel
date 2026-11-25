// Kernel side of SpmmSum (CSR sparse-dense matmul with row-sum reduction)
//   out[r, :] = sum_{e in [rowptr[r], rowptr[r+1])} value[e] * mat[col[e], :]
//   empty row -> out slot = 0; accumulate in float32, write back in input dtype.
//
// Throughput-oriented design:
//   * interleaved row assignment (core c takes rows c, c+coreStride, ...) so that
//     power-law hub rows spread over cores;
//   * per row, column chunks of ck columns;
//   * edges are processed in batches of BE: BE DataCopyPad calls are queued first and
//     only ONE MTE2->V sync covers the whole batch (sync count drops from O(E) to O(E/BE));
//     each batch row lands at a 32B-aligned UB offset (dstStride = 0 packing);
//   * all vector op counts are alignUp(cn,64) (CANN 9.0.0 small-count crash), only the
//     real cn columns are written back.
#include "kernel_operator.h"

#include "spmm_sum_tiling.h"
#include "tiling_key_spmm_sum.h"

using namespace AscendC;

namespace {
constexpr int32_t LA = 64;   // minimum vector count
constexpr int32_t UB_PAD = 8;
constexpr int32_t BE = 8;    // edges batched per sync round
}  // namespace

template <typename T>
class KernelSpmmSum {
public:
    __aicore__ inline KernelSpmmSum() {}

    __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR mat, GM_ADDR out,
                                const SpmmSumTilingData &t) {
        m_ = static_cast<int64_t>(t.m);
        n_ = static_cast<int64_t>(t.n);
        k_ = static_cast<int64_t>(t.k);
        e_ = static_cast<int64_t>(t.e);
        rpc_ = static_cast<int64_t>(t.rowsPerCore);
        stride_ = static_cast<int64_t>(t.coreStride);
        ck_ = static_cast<int64_t>(t.ck);
        block_ = static_cast<int64_t>(GetBlockIdx());
        mode_ = t.dtypeMode;

        rowGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rowptr), m_ + 1);
        colGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(col), e_ > 0 ? e_ : 1);
        valGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value), e_ > 0 ? e_ : 1);
        matGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(mat), n_ * k_ > 0 ? n_ * k_ : 1);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), m_ * k_ > 0 ? m_ * k_ : 1);

        const int32_t ck = static_cast<int32_t>(ck_);
        rowStride_ = ((ck * static_cast<int32_t>(sizeof(T)) + 31) / 32) * 32;   // bytes per batch row
        const uint32_t rawBytes = static_cast<uint32_t>(BE * rowStride_ + 32);
        pipe_.InitBuffer(rawBuf_, rawBytes);
        pipe_.InitBuffer(accBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(outBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(T)));
        if constexpr (sizeof(T) == 2) {
            fRowStride_ = ((ck * 4 + 31) / 32) * 32;
            pipe_.InitBuffer(fBuf_, static_cast<uint32_t>(BE * fRowStride_ + 32));
        }
    }

    __aicore__ inline void Process() {
        if (m_ <= 0 || k_ <= 0) {
            return;
        }
        LocalTensor<T> raw = rawBuf_.Get<T>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<T> outH = outBuf_.Get<T>();

        const int64_t stride = (stride_ > 0) ? stride_ : 1;
        const int32_t ck = static_cast<int32_t>(ck_);

        // interleaved rows: balances power-law hub rows across cores
        for (int64_t r = block_; r < m_; r += stride) {
            const int64_t e0 = rowGm_.GetValue(r);
            const int64_t e1 = rowGm_.GetValue(r + 1);
            for (int64_t kc = 0; kc < k_; kc += ck) {
                const int32_t cn = static_cast<int32_t>((kc + ck < k_) ? ck : (k_ - kc));
                const int32_t cnOp = ((cn + LA - 1) / LA) * LA;
                Duplicate(acc, 0.0f, cnOp);
                int64_t eb0 = e0;
                int64_t eb1 = e1;
                if (eb0 < 0) {
                    eb0 = 0;
                }
                if (eb1 > e_) {
                    eb1 = e_;
                }
                for (int64_t eb = eb0; eb < eb1; eb += BE) {
                    const int32_t nb = static_cast<int32_t>((eb + BE < eb1) ? BE : (eb1 - eb));
                    // queue nb copies (no sync between them)
                    for (int32_t i = 0; i < nb; ++i) {
                        const int32_t c = colGm_.GetValue(eb + i);
                        const int64_t moff = static_cast<int64_t>(c) * k_ + kc;
                        DataCopyExtParams p;
                        p.blockCount = 1;
                        p.blockLen = static_cast<uint32_t>(cn * sizeof(T));
                        p.srcStride = 0;
                        p.dstStride = 0;
                        p.rsv = 0;
                        DataCopyPadExtParams<T> pp;
                        LocalTensor<T> dst = raw[i * rowStride_ / static_cast<int32_t>(sizeof(T))];
                        DataCopyPad(dst, matGm_[static_cast<int32_t>(moff)], p, pp);
                    }
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    for (int32_t i = 0; i < nb; ++i) {
                        const float v = static_cast<float>(valGm_.GetValue(eb + i));
                        LocalTensor<T> srcT = raw[i * rowStride_ / static_cast<int32_t>(sizeof(T))];
                        if constexpr (sizeof(T) == 2) {
                            LocalTensor<float> mf = fBuf_.Get<float>();
                            LocalTensor<float> fd = mf[i * fRowStride_ / 4];
                            Cast(fd, srcT, RoundMode::CAST_NONE, static_cast<uint32_t>(cnOp));
                            Muls(tmp, fd, v, cnOp);
                        } else {
                            Muls(tmp, srcT.template ReinterpretCast<float>(), v, cnOp);
                        }
                        Add(acc, acc, tmp, cnOp);
                    }
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
                // write back real cn columns
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                if constexpr (sizeof(T) == 2) {
                    Cast(outH, acc, RoundMode::CAST_NONE, static_cast<uint32_t>(cnOp));
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                }
                DataCopyExtParams q;
                q.blockCount = 1;
                q.blockLen = static_cast<uint32_t>(cn * sizeof(T));
                q.srcStride = 0;
                q.dstStride = 0;
                q.rsv = 0;
                if constexpr (sizeof(T) == 2) {
                    DataCopyPad(outGm_[static_cast<int32_t>(r * k_ + kc)], outH, q);
                } else {
                    DataCopyPad(outGm_[static_cast<int32_t>(r * k_ + kc)], acc, q);
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> fBuf_;
    GlobalTensor<int64_t> rowGm_;
    GlobalTensor<int32_t> colGm_;
    GlobalTensor<T> valGm_;
    GlobalTensor<T> matGm_;
    GlobalTensor<T> outGm_;
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t e_ = 0;
    int64_t rpc_ = 0;
    int64_t stride_ = 1;
    int64_t ck_ = 0;
    int64_t block_ = 0;
    int32_t rowStride_ = 0;
    int32_t fRowStride_ = 0;
    uint32_t mode_ = 1;
};

template <typename DT_VALUE>
__global__ __aicore__ void spmm_sum(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR mat, GM_ADDR out,
                                    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SpmmSumTilingData);
    GET_TILING_DATA_WITH_STRUCT(SpmmSumTilingData, tiling_data, tiling);
    KernelSpmmSum<DT_VALUE> op;
    op.Init(rowptr, col, value, mat, out, tiling_data);
    op.Process();
}

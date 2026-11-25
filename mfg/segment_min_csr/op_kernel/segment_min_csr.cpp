// Kernel side of SegmentMinCsr
//   out[r, c]     = min over src[indptr[r]:indptr[r+1], c]
//   arg_out[r, c] = first e (global src row) achieving that min (strict < updates)
//   empty segment: out = 0, arg_out = E (sentinel)
//   f16 input compared as f32; value written back unchanged.
//
// Implementation notes:
//   * rows of out split across cores (rowsPerCore); column chunks of ck columns;
//   * per chunk keep minV (f32) and argV (f32, holds global row index e);
//     per edge: load src row chunk -> f32, Compare(cur < minV), Select minV/argV
//     (strict < keeps the first occurrence -> required tie-break);
//   * int64 arg output: Cast f32 -> int32 into ai[0..cnOp), zero ai[ck..2ck) with a
//     float view Duplicate (0.0f => int32 0), then Gather interleaves [v,0] pairs
//     (int64 little endian = [low, high]); only proven float ops plus Cast are used.
//   * every vector op count is alignUp(ck,64) (CANN 9.0.0 small-count crash);
//     only the real ck columns are written back.
#include "kernel_operator.h"

#include "segment_min_csr_tiling.h"
#include "tiling_key_segment_min_csr.h"

using namespace AscendC;

namespace {
constexpr int32_t LA = 64;
constexpr int32_t UB_PAD = 8;
}  // namespace

template <typename T>
class KernelSegmentMinCsr {
public:
    __aicore__ inline KernelSegmentMinCsr() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indptr, GM_ADDR out, GM_ADDR argOut,
                                const SegmentMinCsrTilingData &t) {
        n_ = static_cast<int64_t>(t.n);
        e_ = static_cast<int64_t>(t.e);
        k_ = static_cast<int64_t>(t.k);
        rpc_ = static_cast<int64_t>(t.rowsPerCore);
        ck_ = static_cast<int64_t>(t.ck);
        block_ = static_cast<int64_t>(GetBlockIdx());
        mode_ = t.dtypeMode;

        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), e_ * k_ > 0 ? e_ * k_ : 1);
        ptrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(indptr), n_ + 1);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), n_ * k_ > 0 ? n_ * k_ : 1);
        argGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(argOut), (n_ * k_ > 0 ? n_ * k_ : 1) * 2);

        const int32_t ck = static_cast<int32_t>(ck_);
        pipe_.InitBuffer(rawBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(curBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(minBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(argBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(eBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(maskBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(uint8_t)));
        pipe_.InitBuffer(outBuf_, static_cast<uint32_t>((ck + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(aiBuf_, static_cast<uint32_t>((2 * ck + 2 * LA + UB_PAD) * sizeof(int32_t)));
        pipe_.InitBuffer(wOutBuf_, static_cast<uint32_t>((2 * ck + 2 * LA + UB_PAD) * sizeof(int32_t)));
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>((2 * ck + 2 * LA + UB_PAD) * sizeof(uint32_t)));

        // interleave offsets (bytes into ai): j even -> (j/2)*4 ; j odd -> (ck + j/2)*4
        LocalTensor<int32_t> offs = offBuf_.Get<int32_t>();
        for (int32_t j = 0; j < 2 * ck + 2 * LA; ++j) {
            const int32_t half = j / 2;
            const int32_t idx = (j % 2 == 0) ? half : (ck + half);
            offs.SetValue(j, idx * 4);
        }
    }

    __aicore__ inline void Process() {
        if (n_ <= 0 || k_ <= 0) {
            return;
        }
        LocalTensor<T> raw = rawBuf_.Get<T>();
        LocalTensor<float> cur = curBuf_.Get<float>();
        LocalTensor<float> minV = minBuf_.Get<float>();
        LocalTensor<float> argV = argBuf_.Get<float>();
        LocalTensor<float> eVec = eBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        LocalTensor<int32_t> ai = aiBuf_.Get<int32_t>();
        LocalTensor<int32_t> wOut = wOutBuf_.Get<int32_t>();
        LocalTensor<int32_t> offs = offBuf_.Get<int32_t>();

        const int64_t r0 = block_ * rpc_;
        if (r0 >= n_) {
            return;
        }
        const int64_t r1 = (r0 + rpc_ < n_) ? (r0 + rpc_) : n_;
        const int32_t ck = static_cast<int32_t>(ck_);
        LocalTensor<float> aiF = ai.template ReinterpretCast<float>();

        for (int64_t r = r0; r < r1; ++r) {
            int64_t p0 = ptrGm_.GetValue(r);
            int64_t p1 = ptrGm_.GetValue(r + 1);
            if (p0 < 0) {
                p0 = 0;
            }
            if (p1 > e_) {
                p1 = e_;
            }
            const bool empty = (p1 <= p0);
            for (int64_t kc = 0; kc < k_; kc += ck) {
                const int32_t cn = static_cast<int32_t>((kc + ck < k_) ? ck : (k_ - kc));
                const int32_t cnOp = ((cn + LA - 1) / LA) * LA;
                if (!empty) {
                    Duplicate(minV, 1e38f, cnOp);
                    Duplicate(argV, static_cast<float>(e_), cnOp);
                    for (int64_t e = p0; e < p1; ++e) {
                        const int64_t off = e * k_ + kc;
                        DataCopyExtParams p;
                        p.blockCount = 1;
                        p.blockLen = static_cast<uint32_t>(cn * sizeof(T));
                        p.srcStride = 0;
                        p.dstStride = 0;
                        p.rsv = 0;
                        DataCopyPadExtParams<T> pp;
                        DataCopyPad(raw, srcGm_[static_cast<int32_t>(off)], p, pp);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        if constexpr (sizeof(T) == 2) {
                            Cast(cur, raw, RoundMode::CAST_NONE, static_cast<uint32_t>(cnOp));
                        } else {
                            Adds(cur, raw.template ReinterpretCast<float>(), 0.0f,
                                 static_cast<uint32_t>(cnOp));
                        }
                        Duplicate(eVec, static_cast<float>(e), cnOp);
                        Compare(mask, cur, minV, CMPMODE::LT, static_cast<uint32_t>(cnOp));
                        SetFlag<HardEvent::V_S>(EVENT_ID0);
                        WaitFlag<HardEvent::V_S>(EVENT_ID0);
                        Select(minV, mask, cur, minV, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                               static_cast<uint32_t>(cnOp));
                        Select(argV, mask, eVec, argV, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                               static_cast<uint32_t>(cnOp));
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    }
                }
                // ---- write out chunk ----
                LocalTensor<T> ot = outBuf_.Get<T>();
                if (!empty) {
                    if constexpr (sizeof(T) == 2) {
                        Cast(ot, minV, RoundMode::CAST_NONE, static_cast<uint32_t>(cnOp));
                    } else {
                        Adds(ot.template ReinterpretCast<float>(), minV, 0.0f,
                             static_cast<uint32_t>(cnOp));
                    }
                } else {
                    Duplicate(ot, static_cast<T>(0), cnOp);
                }
                // ---- write arg chunk (int64) ----
                if (!empty) {
                    Cast(ai, argV, RoundMode::CAST_RINT, static_cast<uint32_t>(cnOp));
                } else {
                    Duplicate(eVec, static_cast<float>(e_), cnOp);
                    Cast(ai, eVec, RoundMode::CAST_RINT, static_cast<uint32_t>(cnOp));
                }
                // zero the upper half region [ck, 2ck) using the float view (0.0f => bits 0)
                Duplicate(aiF[ck], 0.0f, ck);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                Gather(wOut, ai, offs.template ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(2 * cnOp));
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams q;
                q.blockCount = 1;
                q.blockLen = static_cast<uint32_t>(cn * sizeof(T));
                q.srcStride = 0;
                q.dstStride = 0;
                q.rsv = 0;
                DataCopyPad(outGm_[static_cast<int32_t>(r * k_ + kc)], ot, q);
                DataCopyExtParams qa;
                qa.blockCount = 1;
                qa.blockLen = static_cast<uint32_t>(2 * cn * sizeof(int32_t));
                qa.srcStride = 0;
                qa.dstStride = 0;
                qa.rsv = 0;
                DataCopyPad(argGm_[static_cast<int32_t>((r * k_ + kc) * 2)], wOut, qa);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> curBuf_;
    TBuf<TPosition::VECCALC> minBuf_;
    TBuf<TPosition::VECCALC> argBuf_;
    TBuf<TPosition::VECCALC> eBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> aiBuf_;
    TBuf<TPosition::VECCALC> wOutBuf_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<T> srcGm_;
    GlobalTensor<int64_t> ptrGm_;
    GlobalTensor<T> outGm_;
    GlobalTensor<int32_t> argGm_;
    int64_t n_ = 0;
    int64_t e_ = 0;
    int64_t k_ = 0;
    int64_t rpc_ = 0;
    int64_t ck_ = 0;
    int64_t block_ = 0;
    uint32_t mode_ = 1;
};

template <typename DT_SRC>
__global__ __aicore__ void segment_min_csr(GM_ADDR src, GM_ADDR indptr, GM_ADDR out, GM_ADDR argOut,
                                           GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SegmentMinCsrTilingData);
    GET_TILING_DATA_WITH_STRUCT(SegmentMinCsrTilingData, tiling_data, tiling);
    KernelSegmentMinCsr<DT_SRC> op;
    op.Init(src, indptr, out, argOut, tiling_data);
    op.Process();
}

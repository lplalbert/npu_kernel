/*!
 * \file mse_loss.h
 * \brief MseLoss kernel class definition
 *
 *   reduction = 0 (none): out[i] = (self[i] - target[i])^2, elementwise output
 *   reduction = 1 (mean): out    = mean((self - target)^2)
 *   reduction = 2 (sum) : out    = sum((self - target)^2)
 *
 * All arithmetic is done in fp32. Vector op counts are always a multiple of 64
 * elements (one 256B repeat for fp32), which is the fast path on this SoC; a
 * partial tail chunk is zero padded up to ubFactor before being computed.
 *
 * Multi-core split is along the flat element dim. For fp32 reduction the per
 * core partial sum is merged into out with a 32B atomic add using the
 * (P0 - g) offset trick, which cancels the dirty initial out value g without
 * any cross-core barrier (SyncAll faults 507057 on this board). fp16 reduction
 * runs single core because half atomic add is unreliable. fp16 elementwise is
 * fully multi-core.
 */

#ifndef MSELOSS_H
#define MSELOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mse_loss_tiling_data.h"
#include "mse_loss_tiling_key.h"
#include <type_traits>

namespace NsMseLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 3;
constexpr int64_t A64 = 64;         // vector op count alignment (one repeat)
constexpr int64_t kSlotFloats = 8;  // 32B atomic add slot

template <typename T>
class MseLoss {
public:
    __aicore__ inline MseLoss() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR target, GM_ADDR out, const MseLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline LocalTensor<T> IssueLoad(TQue<QuePosition::VECIN, BUFFER_NUM>& q, GlobalTensor<T>& gm,
                                               int64_t off, int64_t cnt);
    __aicore__ inline LocalTensor<T> Load1D(TQue<QuePosition::VECIN, BUFFER_NUM>& q, GlobalTensor<T>& gm,
                                            int64_t off, int64_t cnt);
    __aicore__ inline void Store1D(LocalTensor<T>& src, int64_t off, int64_t cnt);
    __aicore__ inline void DoNone(int64_t s, int64_t e);
    __aicore__ inline void DoReduce(int64_t s, int64_t e, float& part);
    __aicore__ inline float FoldAcc(int64_t n);
    __aicore__ inline void StoreScalar(float v);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> qX; // self
    TQue<QuePosition::VECIN, BUFFER_NUM> qY; // target
    TQue<QuePosition::VECOUT, BUFFER_NUM> qO;
    TBuf<QuePosition::VECCALC> fX;      // fp16 -> fp32 cast buffer for self
    TBuf<QuePosition::VECCALC> fY;      // fp16 -> fp32 cast buffer for target
    TBuf<QuePosition::VECCALC> fD;      // squared diff buffer (float)
    TBuf<QuePosition::VECCALC> accBuf;  // fp32 lane accumulator for reduction
    TBuf<QuePosition::VECCALC> slotBuf; // 32B atomic add staging
    TBuf<QuePosition::VECCALC> scF;     // scalar store staging (float)
    TBuf<QuePosition::VECCALC> scT;     // scalar store staging (T)

    GlobalTensor<T> xGm; // self
    GlobalTensor<T> tGm; // target
    GlobalTensor<T> yGm; // out

    MseLossTilingData td_;
    int64_t total_ = 1;
    int64_t per_ = 1;
    int64_t ub_ = A64;
    int64_t red_ = 1;
};

template <typename T>
__aicore__ inline void MseLoss<T>::Init(GM_ADDR self, GM_ADDR target, GM_ADDR out,
                                        const MseLossTilingData* tilingData)
{
    td_ = *tilingData;
    total_ = td_.totalNum;
    if (total_ < 1) {
        total_ = 1;
    }
    per_ = td_.blockFactor;
    if (per_ < 1) {
        per_ = 1;
    }
    ub_ = td_.ubFactor;
    if (ub_ < A64) {
        ub_ = A64;
    }
    ub_ = (ub_ / A64) * A64;
    red_ = td_.reduction;
    if (red_ < 0 || red_ > 2) {
        red_ = 1; // defensive clamp (e.g. raw/uninitialized tiling buffer)
    }

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), total_);
    tGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(target), total_);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), total_);

    pipe.InitBuffer(qX, BUFFER_NUM, static_cast<uint32_t>(ub_ * sizeof(T)));
    pipe.InitBuffer(qY, BUFFER_NUM, static_cast<uint32_t>(ub_ * sizeof(T)));
    if (red_ == 0) {
        pipe.InitBuffer(qO, BUFFER_NUM, static_cast<uint32_t>(ub_ * sizeof(T)));
    }
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fX, static_cast<uint32_t>(ub_ * sizeof(float)));
        pipe.InitBuffer(fY, static_cast<uint32_t>(ub_ * sizeof(float)));
    }
    if (red_ != 0 || !std::is_same<T, float>::value) {
        pipe.InitBuffer(fD, static_cast<uint32_t>(ub_ * sizeof(float)));
    }
    if (red_ != 0) {
        pipe.InitBuffer(accBuf, static_cast<uint32_t>(ub_ * sizeof(float)));
    }
    pipe.InitBuffer(slotBuf, static_cast<uint32_t>(kSlotFloats * sizeof(float)));
    pipe.InitBuffer(scF, static_cast<uint32_t>(A64 * sizeof(float)));
    pipe.InitBuffer(scT, static_cast<uint32_t>(A64 * sizeof(T)));
}

template <typename T>
__aicore__ inline LocalTensor<T> MseLoss<T>::IssueLoad(TQue<QuePosition::VECIN, BUFFER_NUM>& q,
                                                       GlobalTensor<T>& gm, int64_t off, int64_t cnt)
{
    LocalTensor<T> t = q.template AllocTensor<T>();
    if (cnt == ub_) {
        DataCopy(t, gm[off], static_cast<uint32_t>(ub_));
    } else {
        // zero the whole chunk so the padded tail never carries stale data
        Duplicate(t, static_cast<T>(0), static_cast<uint32_t>(ub_));
        PipeBarrier<PIPE_ALL>();
        const int64_t padElems = (cnt * static_cast<int64_t>(sizeof(T)) + 31) / 32 * 32 / static_cast<int64_t>(sizeof(T)) - cnt;
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T)));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<T> pp{padElems > 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        DataCopyPad(t, gm[off], p, pp);
    }
    q.EnQue(t);
    return t;
}

template <typename T>
__aicore__ inline LocalTensor<T> MseLoss<T>::Load1D(TQue<QuePosition::VECIN, BUFFER_NUM>& q,
                                                    GlobalTensor<T>& gm, int64_t off, int64_t cnt)
{
    IssueLoad(q, gm, off, cnt);
    return q.template DeQue<T>();
}

template <typename T>
__aicore__ inline void MseLoss<T>::Store1D(LocalTensor<T>& src, int64_t off, int64_t cnt)
{
    if (cnt == ub_) {
        DataCopy(yGm[off], src, static_cast<uint32_t>(ub_));
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T)));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(yGm[off], src, p);
    }
}

template <typename T>
__aicore__ inline void MseLoss<T>::DoNone(int64_t s, int64_t e)
{
    if (s >= e) {
        return;
    }
    const uint32_t pc = static_cast<uint32_t>(ub_);
    const int64_t nch = (e - s + ub_ - 1) / ub_;
    // software 2-ahead prefetch (BUFFER_NUM == 3): chunk k+2 is issued while
    // chunk k is computed, so MTE2 reads stay ahead of the vector compute and
    // the MTE3 store of the current chunk.
    constexpr int64_t PREFETCH = 2;
    int64_t issued = 0;
    for (; issued < PREFETCH && issued < nch; ++issued) {
        int64_t o = s + issued * ub_;
        int64_t c = e - o;
        if (c > ub_) {
            c = ub_;
        }
        IssueLoad(qX, xGm, o, c);
        IssueLoad(qY, tGm, o, c);
    }

    for (int64_t k = 0; k < nch; ++k) {
        LocalTensor<T> xt = qX.template DeQue<T>();
        LocalTensor<T> yt = qY.template DeQue<T>();

        int64_t off = s + k * ub_;
        int64_t cnt = e - off;
        if (cnt > ub_) {
            cnt = ub_;
        }

        if (issued < nch) {
            int64_t o = s + issued * ub_;
            int64_t c = e - o;
            if (c > ub_) {
                c = ub_;
            }
            IssueLoad(qX, xGm, o, c);
            IssueLoad(qY, tGm, o, c);
            ++issued;
        }

        LocalTensor<T> ot = qO.template AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Sub(ot, xt, yt, pc);
            Mul(ot, ot, ot, pc);
        } else {
            LocalTensor<float> fx = fX.Get<float>();
            LocalTensor<float> fy = fY.Get<float>();
            LocalTensor<float> fd = fD.Get<float>();
            Cast(fx, xt, RoundMode::CAST_NONE, pc);
            Cast(fy, yt, RoundMode::CAST_NONE, pc);
            Sub(fd, fx, fy, pc);
            Mul(fd, fd, fd, pc);
            Cast(ot, fd, RoundMode::CAST_RINT, pc);
        }
        qO.EnQue(ot);
        ot = qO.template DeQue<T>();
        Store1D(ot, off, cnt);
        qO.FreeTensor(ot);
        qX.FreeTensor(xt);
        qY.FreeTensor(yt);
    }
}

template <typename T>
__aicore__ inline float MseLoss<T>::FoldAcc(int64_t n)
{
    LocalTensor<float> acc = accBuf.Get<float>();
    int64_t m = n;
    while (m > 128) {
        int64_t h = (m / 2 / A64) * A64; // keep every vector count a multiple of 64
        if (h < A64) {
            break;
        }
        Add(acc, acc, acc[static_cast<int32_t>(h)], static_cast<uint32_t>(h));
        m = h;
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float s = 0.0f;
    for (int64_t j = 0; j < m; ++j) {
        s += acc.GetValue(j);
    }
    return s;
}

template <typename T>
__aicore__ inline void MseLoss<T>::DoReduce(int64_t s, int64_t e, float& part)
{
    part = 0.0f;
    if (s >= e) {
        return;
    }
    const uint32_t pc = static_cast<uint32_t>(ub_);
    LocalTensor<float> acc = accBuf.Get<float>();
    Duplicate(acc, 0.0f, pc);

    int64_t off = s;
    int64_t cnt = e - off;
    if (cnt > ub_) {
        cnt = ub_;
    }
    IssueLoad(qX, xGm, off, cnt);
    IssueLoad(qY, tGm, off, cnt);

    while (true) {
        LocalTensor<T> xt = qX.template DeQue<T>();
        LocalTensor<T> yt = qY.template DeQue<T>();

        int64_t noff = off + cnt;
        bool hasNext = (noff < e);
        int64_t ncnt = 0;
        if (hasNext) {
            ncnt = e - noff;
            if (ncnt > ub_) {
                ncnt = ub_;
            }
            IssueLoad(qX, xGm, noff, ncnt);
            IssueLoad(qY, tGm, noff, ncnt);
        }

        LocalTensor<float> fd = fD.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            Sub(fd, xt, yt, pc);
        } else {
            LocalTensor<float> fx = fX.Get<float>();
            LocalTensor<float> fy = fY.Get<float>();
            Cast(fx, xt, RoundMode::CAST_NONE, pc);
            Cast(fy, yt, RoundMode::CAST_NONE, pc);
            Sub(fd, fx, fy, pc);
        }
        Mul(fd, fd, fd, pc);
        Add(acc, acc, fd, pc);

        qX.FreeTensor(xt);
        qY.FreeTensor(yt);

        if (!hasNext) {
            break;
        }
        off = noff;
        cnt = ncnt;
    }
    part = FoldAcc(ub_);
}

template <typename T>
__aicore__ inline void MseLoss<T>::StoreScalar(float v)
{
    if constexpr (std::is_same<T, float>::value) {
        yGm.SetValue(0, v);
    } else {
        LocalTensor<float> f = scF.Get<float>();
        LocalTensor<T> t = scT.Get<T>();
        f.SetValue(0, v);
        for (int64_t j = 1; j < A64; ++j) {
            f.SetValue(j, 0.0f);
        }
        PipeBarrier<PIPE_ALL>(); // S -> V
        Cast(t, f, RoundMode::CAST_RINT, static_cast<uint32_t>(A64));
        PipeBarrier<PIPE_ALL>(); // V -> S
        yGm.SetValue(0, t.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void MseLoss<T>::Process()
{
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum < 1) {
        blockNum = 1;
    }
    int64_t s = blockId * per_;
    if (s > total_) {
        s = total_;
    }
    int64_t e = s + per_;
    if (e > total_) {
        e = total_;
    }

    if (red_ == 0) {
        if (s < e) {
            DoNone(s, e);
        }
        return;
    }

    // Reduction path. Read the dirty initial value of out before any atomic write
    // so every core observes the same g (fp32 multi-core only).
    float g0 = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        if (blockNum > 1) {
            g0 = yGm.GetValue(0);
        }
    }

    float part = 0.0f;
    if (s < e) {
        DoReduce(s, e, part);
    }

    float val = part;
    if (red_ == 1) {
        val = part / static_cast<float>(total_); // sum of per-core (part/N) == mean
    }

    if (blockNum <= 1) {
        StoreScalar(val);
        return;
    }

    if constexpr (std::is_same<T, float>::value) {
        // g finite : final = g + (val0 - g) + sum(val_k) = sum(val_k)
        // g bad    : fall back to plain accumulation (keeps inf/nan propagation)
        const bool gBad = (g0 != g0) || (g0 > 3.402823466e38f) || (g0 < -3.402823466e38f);
        const float delta = (gBad || blockId != 0) ? val : (val - g0);
        LocalTensor<float> tmp = slotBuf.Get<float>();
        tmp.SetValue(0, delta);
        for (int64_t j = 1; j < kSlotFloats; ++j) {
            tmp.SetValue(j, 0.0f);
        }
        PipeBarrier<PIPE_ALL>(); // S -> MTE3
        SetAtomicAdd<float>();
        DataCopy(yGm[0], tmp, static_cast<uint32_t>(kSlotFloats)); // 32B atomic add
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    } else {
        StoreScalar(val); // fp16 reduction runs single core, should not reach here
    }
}

} // namespace NsMseLoss
#endif // MSELOSS_H

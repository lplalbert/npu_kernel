/*!
 * \file reduce_log_sum_exp.h
 * \brief ReduceLogSumExp 算子 kernel
 *
 * y = m + ln( sum( exp(x - m) ) )，m = max(x|axes)   （max-shift，防上溢）
 * 中间量一律 float32；16bit 存储侧收发（仅用向量 Cast，避免 bf16 标量转换）。
 * axes 在设备侧由 kernel 自读自解（opapi 流程中 host 侧 tiling 不可见其取值）。
 *
 * 五条路径（快路径全部向量化 + TQue 双缓冲预取，热循环内无 PipeBarrier）：
 *   MODE_COPY : reduceNum==1 —— 输出即输入，纯拷贝
 *   MODE_P2B  : 末尾维是唯一 reduce 维且能整块装进 UB
 *               —— 一次 DataCopy 装 G 个输出，逐个在 UB 内折叠（每 tile 只 1 遍 GM 读）
 *   MODE_P2A  : 末尾维是 reduce（其余情况，含 all-reduce 超大 reduceNum）
 *               —— 每输出 2 遍扫描 GM（先 max 后 sum），lane 累加 + 树折叠
 *   MODE_P3   : 末尾维是 kept（含多轴非连续）—— 对 inner 个输出列式累加，所有 lane
 *               都是不同输出，无需折叠，全程无标量热循环
 *   MODE_SC   : 兜底标量实现（tailRun<8 / inner<8 等退化几何）
 *
 * 同步：TQue EnQue/DeQue 负责 MTE2->V；累加器在 VECCALC（V 内部有序）；
 *       向量写后读标量、以及输出批 DataCopy 前后各一次 PipeBarrier<PIPE_ALL>。
 */

#ifndef REDUCE_LOG_SUM_EXP_H
#define REDUCE_LOG_SUM_EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_log_sum_exp_tiling_data.h"
#include "reduce_log_sum_exp_tiling_key.h"
#include <type_traits>

namespace NsReduceLogSumExp {

using namespace AscendC;

constexpr int64_t kTile = 8192;    // inQueue 单块元素数（双缓冲 => 两份）
constexpr int64_t kOutB = 1024;    // 输出批量（m/s/y 暂存）
constexpr int64_t kCacheAlign = 128;
constexpr float kNegBig = -3.402823466e+38f;

constexpr int32_t MODE_SC = 0;
constexpr int32_t MODE_P2A = 1;
constexpr int32_t MODE_P2B = 2;
constexpr int32_t MODE_P3 = 3;
constexpr int32_t MODE_COPY = 4;

template <typename T>
class ReduceLogSumExp {
public:
    __aicore__ inline ReduceLogSumExp(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR axes, GM_ADDR y, const ReduceLogSumExpTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // ---------------- 几何解析 ----------------
    __aicore__ inline void ResolveGeometry();
    __aicore__ inline int64_t KeptBase(int64_t idx, int64_t nk) const;
    __aicore__ inline int64_t ROff(int64_t r, int64_t nuse) const;
    __aicore__ inline void CoreRange(int64_t total, int64_t align, int64_t& s, int64_t& e) const;
    __aicore__ inline void OutRange(int64_t total, int64_t align, int64_t& s, int64_t& e) const;

    // ---------------- 搬运 / 折叠 ----------------
    __aicore__ inline void CopyIn(int64_t off, int64_t cnt);
    __aicore__ inline LocalTensor<float> Take(int64_t cnt);
    __aicore__ inline void Drop();
    __aicore__ inline void SyncVS();      // 向量 -> 标量读（不动 MTE2/MTE3 流水）
    __aicore__ inline void FullSync();    // 全 pipe 栅障（只用在输出批）
    __aicore__ inline float FoldMax(LocalTensor<float> a, int64_t n);
    __aicore__ inline float FoldSum(LocalTensor<float> a, int64_t n);
    __aicore__ inline void FlushOut(int64_t yStart, int64_t n);
    __aicore__ inline void StoreOut(int64_t yStart, LocalTensor<float> yv, int64_t cnt);

    // ---------------- 各路径 ----------------
    __aicore__ inline void ProcessCopy();
    __aicore__ inline void ProcessP2A();
    __aicore__ inline void P2AScan(int64_t base, int32_t mode, float shift);
    __aicore__ inline void P2AAcc(int64_t k, int32_t mode, float shift);
    __aicore__ inline int64_t P2Off(int64_t base, int64_t k) const;
    __aicore__ inline int64_t P2Cnt(int64_t k) const;
    __aicore__ inline void ProcessP2B();
    __aicore__ inline void ProcessP3();
    __aicore__ inline void P3Acc(int32_t mode, int64_t cnt);
    // 兜底
    __aicore__ inline void ProcessScalar();
    __aicore__ inline void ScalarOne(int64_t outIdx);
    __aicore__ inline void LoadChunk(int64_t base, int64_t r0, int64_t cur);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQ_;
    TBuf<QuePosition::VECCALC> fBuf_;    // 16bit Cast 目标 / 拷贝临时
    TBuf<QuePosition::VECCALC> aBuf_;    // 累加器 1（float）
    TBuf<QuePosition::VECCALC> bBuf_;    // 累加器 2（float，仅 P3）
    TBuf<QuePosition::VECCALC> mBuf_;    // 输出批量：m
    TBuf<QuePosition::VECCALC> sBuf_;    // 输出批量：s
    TBuf<QuePosition::VECCALC> yBuf_;    // 输出批量：y(float)
    TBuf<QuePosition::VECCALC> tBuf_;    // 输出批量：y(T)
    TBuf<QuePosition::VECCALC> scF_;     // 兜底：float 收发
    TBuf<QuePosition::VECCALC> scT_;     // 兜底：T 收发

    LocalTensor<T> curT_;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int32_t> axesGm;

    ReduceLogSumExpTilingData td_;

    // 合并后的维（x 行主序顺序）
    int64_t nd_ = 0;
    int64_t dsz_[8] = {0};
    int64_t dst_[8] = {0};
    uint8_t dred_[8] = {0};
    // kept / reduce 维
    int64_t kd_[8] = {0};
    int64_t kds_[8] = {0};
    int64_t kc_ = 0;
    int64_t rd_[8] = {0};
    int64_t rds_[8] = {0};
    int64_t rc_ = 0;

    int64_t outNum_ = 1;
    int64_t reduceNum_ = 1;
    int64_t totalNum_ = 1;
    int64_t tailRun_ = 1;    // 末尾连续 reduce 长度（末尾维为 reduce 时 = 该维大小）
    int64_t inner_ = 1;      // 末尾连续 kept 长度（末尾维为 kept 时 = 该维大小）
    int64_t nRuns_ = 1;      // P2A：每输出的“段”数 = reduceNum_ / tailRun_
    int64_t w_ = 0;          // P2A：lane 累加宽度（64 的倍数）
    int64_t ncp_ = 0;        // P2A：每段的子块数
    int32_t mode_ = MODE_SC;
};

// ============================== 几何 ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ResolveGeometry()
{
    const int64_t rank = td_.rank;
    totalNum_ = td_.totalNum;

    bool red[8] = {false, false, false, false, false, false, false, false};
    for (int64_t i = 0; i < td_.axesNum; ++i) {
        int64_t a = static_cast<int64_t>(axesGm.GetValue(i));
        if (a < 0) {
            a += rank;
        }
        if (a >= 0 && a < rank) {
            red[a] = true;
        }
    }

    // 丢弃 size-1 维 + 合并同类相邻维（dense 行主序下成立）
    nd_ = 0;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t sz = td_.xDims[i];
        if (sz == 1) {
            continue;
        }
        const bool r = red[i];
        if (nd_ > 0 && dred_[nd_ - 1] == static_cast<uint8_t>(r)) {
            dsz_[nd_ - 1] *= sz;
            dst_[nd_ - 1] = td_.xStrides[i];
        } else {
            dsz_[nd_] = sz;
            dst_[nd_] = td_.xStrides[i];
            dred_[nd_] = static_cast<uint8_t>(r);
            ++nd_;
        }
    }
    if (nd_ == 0) {   // 0 维 / 全 1 维：单元素直通
        nd_ = 1;
        dsz_[0] = 1;
        dst_[0] = 1;
        dred_[0] = 0;
    }

    kc_ = 0;
    rc_ = 0;
    for (int64_t i = 0; i < nd_; ++i) {
        if (dred_[i] != 0) {
            rd_[rc_] = dsz_[i];
            rds_[rc_] = dst_[i];
            ++rc_;
        } else {
            kd_[kc_] = dsz_[i];
            kds_[kc_] = dst_[i];
            ++kc_;
        }
    }
    outNum_ = 1;
    for (int64_t i = 0; i < kc_; ++i) {
        outNum_ *= kd_[i];
    }
    reduceNum_ = 1;
    for (int64_t i = 0; i < rc_; ++i) {
        reduceNum_ *= rd_[i];
    }
    tailRun_ = (dred_[nd_ - 1] != 0) ? dsz_[nd_ - 1] : 1;
    inner_ = (dred_[nd_ - 1] == 0) ? dsz_[nd_ - 1] : 1;
    nRuns_ = reduceNum_ / tailRun_;

    // ---- 选路 ----
    if (reduceNum_ == 1) {
        mode_ = MODE_COPY;
    } else if (dred_[nd_ - 1] != 0) {
        if (tailRun_ < 8) {
            mode_ = MODE_SC;
        } else if (nRuns_ == 1 && tailRun_ <= kTile) {
            mode_ = MODE_P2B;
        } else {
            mode_ = MODE_P2A;
        }
    } else {
        mode_ = (inner_ < 8) ? MODE_SC : MODE_P3;
    }

    if (mode_ == MODE_P2A) {
        w_ = (tailRun_ < kTile) ? tailRun_ : kTile;
        w_ = ((w_ + 63) / 64) * 64;             // 64 的倍数 => 折叠向量步 count >= 32
        if (w_ > kTile) {
            w_ = kTile;
        }
        ncp_ = (tailRun_ + w_ - 1) / w_;
    }
}

template <typename T>
__aicore__ inline int64_t ReduceLogSumExp<T>::KeptBase(int64_t idx, int64_t nk) const
{
    int64_t off = 0;
    for (int64_t i = nk - 1; i >= 0; --i) {
        const int64_t d = kd_[i];
        const int64_t q = idx % d;
        idx /= d;
        off += q * kds_[i];
    }
    return off;
}

template <typename T>
__aicore__ inline int64_t ReduceLogSumExp<T>::ROff(int64_t r, int64_t nuse) const
{
    int64_t off = 0;
    for (int64_t i = nuse - 1; i >= 0; --i) {
        const int64_t d = rd_[i];
        const int64_t q = r % d;
        r /= d;
        off += q * rds_[i];
    }
    return off;
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::CoreRange(int64_t total, int64_t align, int64_t& s, int64_t& e) const
{
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    if (blockNum < 1) {
        blockNum = 1;
    }
    if (align < 1) {
        align = 1;
    }
    int64_t per = (total + blockNum - 1) / blockNum;
    per = ((per + align - 1) / align) * align;
    s = per * blockId;
    e = s + per;
    if (s > total) {
        s = total;
    }
    if (e > total) {
        e = total;
    }
}

// 输出下标区间切分，保证 per*unit 是 128 的整数倍（避免跨核同 cache line 写）
template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::OutRange(int64_t total, int64_t unit, int64_t& s, int64_t& e) const
{
    int64_t g = unit;
    while (g > 1 && (kCacheAlign % g) != 0) {
        --g;
    }
    if (g < 1) {
        g = 1;
    }
    int64_t align = kCacheAlign / g;
    if (align < 1) {
        align = 1;
    }
    CoreRange(total, align, s, e);
}

// ============================== 基础 ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::Init(GM_ADDR x, GM_ADDR axes, GM_ADDR y,
                                                const ReduceLogSumExpTilingData* tilingData)
{
    td_ = *tilingData;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    axesGm.SetGlobalBuffer((__gm__ int32_t*)axes);
    ResolveGeometry();

    if (mode_ == MODE_SC) {
        pipe.InitBuffer(scF_, td_.ubLen * sizeof(float));
        pipe.InitBuffer(scT_, td_.ubLen * sizeof(T));
        return;
    }
    pipe.InitBuffer(inQ_, 2, kTile * sizeof(T));
    pipe.InitBuffer(aBuf_, kTile * sizeof(float));
    if (mode_ == MODE_P3) {
        pipe.InitBuffer(bBuf_, kTile * sizeof(float));
    }
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fBuf_, kTile * sizeof(float));
    }
    if (mode_ == MODE_P2A || mode_ == MODE_P2B) {
        pipe.InitBuffer(mBuf_, kOutB * sizeof(float));
        pipe.InitBuffer(sBuf_, kOutB * sizeof(float));
    }
    pipe.InitBuffer(yBuf_, kOutB * sizeof(float));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(tBuf_, kOutB * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::CopyIn(int64_t off, int64_t cnt)
{
    LocalTensor<T> t = inQ_.template AllocTensor<T>();
    const int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
    if (((bytes & 31) == 0) && (((off * static_cast<int64_t>(sizeof(T))) & 31) == 0)) {
        DataCopy(t, xGm[off], static_cast<uint32_t>(cnt));
    } else {
        DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
        DataCopyPad(t, xGm[off], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
    }
    inQ_.EnQue(t);
}

template <typename T>
__aicore__ inline LocalTensor<float> ReduceLogSumExp<T>::Take(int64_t cnt)
{
    curT_ = inQ_.template DeQue<T>();
    if constexpr (std::is_same<T, float>::value) {
        return curT_;
    } else {
        LocalTensor<float> f = fBuf_.Get<float>();
        Cast(f, curT_, RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
        return f;
    }
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::Drop()
{
    inQ_.FreeTensor(curT_);
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::SyncVS()
{
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::FullSync()
{
    PipeBarrier<PIPE_ALL>();
}

// 树折叠（向量步 count>=8，剩 <=15 lane 标量收尾）；会破坏 a 的前 n lane
template <typename T>
__aicore__ inline float ReduceLogSumExp<T>::FoldMax(LocalTensor<float> a, int64_t n)
{
    while (n > 8) {
        const int64_t h = (n + 1) >> 1;
        const int64_t c = n >> 1;
        if (c < 8) {
            break;
        }
        Max(a, a, a[static_cast<int32_t>(h)], static_cast<int32_t>(c));
        n = h;
    }
    SyncVS();   // 向量 -> 标量读
    float m = a.GetValue(0);
    for (int64_t j = 1; j < n; ++j) {
        const float v = a.GetValue(j);
        if (v > m) {
            m = v;
        }
    }
    return m;
}

template <typename T>
__aicore__ inline float ReduceLogSumExp<T>::FoldSum(LocalTensor<float> a, int64_t n)
{
    while (n > 8) {
        const int64_t h = (n + 1) >> 1;
        const int64_t c = n >> 1;
        if (c < 8) {
            break;
        }
        Add(a, a, a[static_cast<int32_t>(h)], static_cast<int32_t>(c));
        n = h;
    }
    SyncVS();   // 向量 -> 标量读
    float s = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
        s += a.GetValue(j);
    }
    return s;
}

// 把 yv 前 cnt 个 float（fp32）或 Cast 后（16bit）写到 yGm[yStart]
template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::StoreOut(int64_t yStart, LocalTensor<float> yv, int64_t cnt)
{
    const int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
    const bool aligned = ((bytes & 31) == 0) && (((yStart * static_cast<int64_t>(sizeof(T))) & 31) == 0);
    if constexpr (std::is_same<T, float>::value) {
        FullSync();   // V -> MTE3
        if (aligned) {
            DataCopy(yGm[yStart], yv, static_cast<uint32_t>(cnt));
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(yGm[yStart], yv, p);
        }
    } else {
        LocalTensor<T> tv = tBuf_.Get<T>();
        const int64_t pad = ((cnt + 7) / 8) * 8;
        Cast(tv, yv, RoundMode::CAST_RINT, static_cast<int32_t>(pad));
        FullSync();   // V -> MTE3
        if (aligned) {
            DataCopy(yGm[yStart], tv, static_cast<uint32_t>(cnt));
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(yGm[yStart], tv, p);
        }
    }
    FullSync();       // MTE3 -> V（下一批复用 yBuf/tBuf 前）
}

// 批量写回：mBuf_/sBuf_ 前 n 个有效，不足 8 的倍数用 (0,1) 填充（ln(1)=0 保证填充位无害）
template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::FlushOut(int64_t yStart, int64_t n)
{
    if (n <= 0) {
        return;
    }
    const int64_t pad = ((n + 7) / 8) * 8;
    LocalTensor<float> mv = mBuf_.Get<float>();
    LocalTensor<float> sv = sBuf_.Get<float>();
    for (int64_t j = n; j < pad; ++j) {
        mv.SetValue(j, 0.0f);
        sv.SetValue(j, 1.0f);
    }
    FullSync();   // 标量写 -> 向量读
    Ln(sv, sv, static_cast<int32_t>(pad));
    LocalTensor<float> yv = yBuf_.Get<float>();
    Add(yv, mv, sv, static_cast<int32_t>(pad));
    StoreOut(yStart, yv, n);
}

// ============================== MODE_COPY ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ProcessCopy()
{
    int64_t s = 0;
    int64_t e = 0;
    CoreRange(totalNum_, kCacheAlign, s, e);
    LocalTensor<float> yv = yBuf_.Get<float>();
    int64_t off = s;
    if (off < e) {
        int64_t c0 = e - off;
        if (c0 > kOutB) {
            c0 = kOutB;
        }
        CopyIn(off, c0);
    }
    while (off < e) {
        int64_t cnt = e - off;
        if (cnt > kOutB) {
            cnt = kOutB;
        }
        const int64_t nOff = off + cnt;
        if (nOff < e) {
            int64_t nc = e - nOff;
            if (nc > kOutB) {
                nc = kOutB;
            }
            CopyIn(nOff, nc);
        }
        LocalTensor<float> f = Take(cnt);
        Adds(yv, f, 0.0f, static_cast<int32_t>(cnt));   // 拷到输出缓冲（V）
        Drop();
        StoreOut(off, yv, cnt);
        off = nOff;
    }
}

// ============================== MODE_P2A ==============================

template <typename T>
__aicore__ inline int64_t ReduceLogSumExp<T>::P2Off(int64_t base, int64_t k) const
{
    const int64_t run = k / ncp_;
    const int64_t c = k % ncp_;
    return base + ROff(run, rc_ - 1) + c * w_;
}

template <typename T>
__aicore__ inline int64_t ReduceLogSumExp<T>::P2Cnt(int64_t k) const
{
    const int64_t c = k % ncp_;
    const int64_t rem = tailRun_ - c * w_;
    return (rem > w_) ? w_ : rem;
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::P2AAcc(int64_t k, int32_t mode, float shift)
{
    const int64_t cur = P2Cnt(k);
    LocalTensor<float> f = Take(cur);
    LocalTensor<float> acc = aBuf_.Get<float>();
    if (mode == 0) {
        Max(acc, acc, f, static_cast<int32_t>(cur));
    } else {
        Adds(f, f, -shift, static_cast<int32_t>(cur));
        Exp(f, f, static_cast<int32_t>(cur));
        Add(acc, acc, f, static_cast<int32_t>(cur));
    }
    Drop();
}

// 一遍扫描：mode 0 = max，mode 1 = sum(exp(x-shift))
template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::P2AScan(int64_t base, int32_t mode, float shift)
{
    const int64_t total = nRuns_ * ncp_;
    LocalTensor<float> acc = aBuf_.Get<float>();
    Duplicate(acc, (mode == 0) ? kNegBig : 0.0f, static_cast<int32_t>(w_));
    CopyIn(P2Off(base, 0), P2Cnt(0));
    for (int64_t k = 0; k + 1 < total; ++k) {
        CopyIn(P2Off(base, k + 1), P2Cnt(k + 1));
        P2AAcc(k, mode, shift);
    }
    P2AAcc(total - 1, mode, shift);
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ProcessP2A()
{
    int64_t s = 0;
    int64_t e = 0;
    CoreRange(outNum_, kCacheAlign, s, e);
    LocalTensor<float> mv = mBuf_.Get<float>();
    LocalTensor<float> sv = sBuf_.Get<float>();
    int64_t bi = 0;
    for (int64_t o = s; o < e; ++o) {
        const int64_t base = KeptBase(o, kc_);
        P2AScan(base, 0, 0.0f);
        const float m = FoldMax(aBuf_.Get<float>(), w_);
        P2AScan(base, 1, m);
        const float sum = FoldSum(aBuf_.Get<float>(), w_);
        mv.SetValue(bi, m);
        sv.SetValue(bi, sum);
        ++bi;
        if (bi == kOutB) {
            FlushOut(o + 1 - bi, bi);
            bi = 0;
        }
    }
    if (bi > 0) {
        FlushOut(e - bi, bi);
    }
}

// ============================== MODE_P2B ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ProcessP2B()
{
    int64_t s = 0;
    int64_t e = 0;
    CoreRange(outNum_, kCacheAlign, s, e);
    const int64_t R = tailRun_;
    int64_t G = kTile / R;
    if (G < 1) {
        G = 1;
    }
    if (G > kOutB) {
        G = kOutB;
    }
    LocalTensor<float> acc = aBuf_.Get<float>();
    LocalTensor<float> mv = mBuf_.Get<float>();
    LocalTensor<float> sv = sBuf_.Get<float>();
    int64_t o0 = s;
    if (o0 >= e) {
        return;
    }
    {
        int64_t g0 = e - o0;
        if (g0 > G) {
            g0 = G;
        }
        CopyIn(o0 * R, g0 * R);
    }
    while (o0 < e) {
        int64_t g = e - o0;
        if (g > G) {
            g = G;
        }
        const int64_t cnt = g * R;
        const int64_t nO = o0 + g;
        if (nO < e) {
            int64_t g2 = e - nO;
            if (g2 > G) {
                g2 = G;
            }
            CopyIn(nO * R, g2 * R);
        }
        LocalTensor<float> f = Take(cnt);
        Adds(acc, f, 0.0f, static_cast<int32_t>(cnt));   // max 折叠在副本上做
        // ---- max ----
        for (int64_t i = 0; i < g; ++i) {
            mv.SetValue(i, FoldMax(acc[static_cast<int32_t>(R * i)], R));
        }
        const int64_t pad = ((g + 7) / 8) * 8;
        for (int64_t i = g; i < pad; ++i) {
            mv.SetValue(i, 0.0f);
            sv.SetValue(i, 1.0f);
        }
        // ---- sum(exp(x-m)) ----
        for (int64_t i = 0; i < g; ++i) {
            LocalTensor<float> fg = f[static_cast<int32_t>(R * i)];
            Adds(fg, fg, -mv.GetValue(i), static_cast<int32_t>(R));
            Exp(fg, fg, static_cast<int32_t>(R));
            sv.SetValue(i, FoldSum(fg, R));
        }
        Drop();
        FlushOut(o0, g);
        o0 = nO;
    }
}

// ============================== MODE_P3 ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::P3Acc(int32_t mode, int64_t cnt)
{
    LocalTensor<float> f = Take(cnt);
    LocalTensor<float> am = aBuf_.Get<float>();
    if (mode == 0) {
        Max(am, am, f, static_cast<int32_t>(cnt));
    } else {
        Sub(f, f, am, static_cast<int32_t>(cnt));
        Exp(f, f, static_cast<int32_t>(cnt));
        LocalTensor<float> b2 = bBuf_.Get<float>();
        Add(b2, b2, f, static_cast<int32_t>(cnt));
    }
    Drop();
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ProcessP3()
{
    const int64_t nBlocks = outNum_ / inner_;
    int64_t bs = 0;
    int64_t be = 0;
    OutRange(nBlocks, inner_, bs, be);

    int64_t ic = inner_;
    if (ic > kTile) {
        ic = kTile;
    }
    const int64_t nIc = (inner_ + ic - 1) / ic;

    LocalTensor<float> am = aBuf_.Get<float>();
    LocalTensor<float> b2 = bBuf_.Get<float>();
    LocalTensor<float> yv = yBuf_.Get<float>();

    for (int64_t b = bs; b < be; ++b) {
        const int64_t base = KeptBase(b, kc_ - 1);
        for (int64_t c = 0; c < nIc; ++c) {
            const int64_t coff = base + c * ic;
            int64_t cnt = inner_ - c * ic;
            if (cnt > ic) {
                cnt = ic;
            }
            // ---- pass 1: max over runs ----
            Duplicate(am, kNegBig, static_cast<int32_t>(ic));
            CopyIn(coff, cnt);
            for (int64_t r = 0; r + 1 < reduceNum_; ++r) {
                CopyIn(coff + ROff(r + 1, rc_), cnt);
                P3Acc(0, cnt);
            }
            P3Acc(0, cnt);
            // ---- pass 2: sum(exp(x - m)) ----
            Duplicate(b2, 0.0f, static_cast<int32_t>(ic));
            CopyIn(coff, cnt);
            for (int64_t r = 0; r + 1 < reduceNum_; ++r) {
                CopyIn(coff + ROff(r + 1, rc_), cnt);
                P3Acc(1, cnt);
            }
            P3Acc(1, cnt);
            // ---- y = m + ln(s) ----
            const int64_t pad = ((cnt + 7) / 8) * 8;
            for (int64_t j = cnt; j < pad; ++j) {
                am.SetValue(j, 0.0f);
                b2.SetValue(j, 1.0f);
            }
            FullSync();   // 标量写 -> 向量读
            Ln(b2, b2, static_cast<int32_t>(pad));
            Add(yv, am, b2, static_cast<int32_t>(pad));
            StoreOut(b * inner_ + c * ic, yv, cnt);
        }
    }
}

// ============================== 兜底标量（原实现，已验证） ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::LoadChunk(int64_t base, int64_t r0, int64_t cur)
{
    LocalTensor<float> f = scF_.Get<float>();
    LocalTensor<T> t = scT_.Get<T>();
    const bool contig = (nRuns_ == 1);
    if (contig) {
        const int64_t bytes = cur * static_cast<int64_t>(sizeof(T));
        if constexpr (std::is_same<T, float>::value) {
            if ((bytes & 31) == 0) {
                DataCopy(f, xGm[base + r0], static_cast<uint32_t>(cur));
            } else {
                DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
                DataCopyPad(f, xGm[base + r0], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        } else {
            if ((bytes & 31) == 0) {
                DataCopy(t, xGm[base + r0], static_cast<uint32_t>(cur));
            } else {
                DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
                DataCopyPad(t, xGm[base + r0], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
            Cast(f, t, RoundMode::CAST_NONE, static_cast<int32_t>(cur));
        }
    } else {
        // 非连续归约维：逐元素收集（仅退化几何走这里）
        if constexpr (std::is_same<T, float>::value) {
            for (int64_t j = 0; j < cur; ++j) {
                f.SetValue(j, xGm.GetValue(base + ROff(r0 + j, rc_)));
            }
        } else {
            for (int64_t j = 0; j < cur; ++j) {
                t.SetValue(j, xGm.GetValue(base + ROff(r0 + j, rc_)));
            }
            PipeBarrier<PIPE_ALL>();
            Cast(f, t, RoundMode::CAST_NONE, static_cast<int32_t>(cur));
        }
    }
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ScalarOne(int64_t outIdx)
{
    const int64_t base = KeptBase(outIdx, kc_);
    LocalTensor<float> f = scF_.Get<float>();
    const int64_t ubLen = td_.ubLen;

    float m = kNegBig;
    for (int64_t r0 = 0; r0 < reduceNum_; r0 += ubLen) {
        int64_t cur = reduceNum_ - r0;
        if (cur > ubLen) {
            cur = ubLen;
        }
        LoadChunk(base, r0, cur);
        PipeBarrier<PIPE_ALL>();
        for (int64_t j = 0; j < cur; ++j) {
            const float v = f.GetValue(j);
            if (v > m) {
                m = v;
            }
        }
    }

    float s = 0.0f;
    for (int64_t r0 = 0; r0 < reduceNum_; r0 += ubLen) {
        int64_t cur = reduceNum_ - r0;
        if (cur > ubLen) {
            cur = ubLen;
        }
        LoadChunk(base, r0, cur);
        PipeBarrier<PIPE_ALL>();
        Adds(f, f, -m, static_cast<int32_t>(cur));
        Exp(f, f, static_cast<int32_t>(cur));
        PipeBarrier<PIPE_ALL>();
        for (int64_t j = 0; j < cur; ++j) {
            s += f.GetValue(j);
        }
    }

    f.SetValue(0, s);
    Ln(f, f, 1);
    f.SetValue(0, m + f.GetValue(0));
    if constexpr (std::is_same<T, float>::value) {
        yGm.SetValue(outIdx, f.GetValue(0));
    } else {
        LocalTensor<T> t = scT_.Get<T>();
        PipeBarrier<PIPE_ALL>();
        Cast(t, f, RoundMode::CAST_RINT, 1);
        yGm.SetValue(outIdx, t.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::ProcessScalar()
{
    int64_t s = 0;
    int64_t e = 0;
    CoreRange(outNum_, kCacheAlign, s, e);
    for (int64_t o = s; o < e; ++o) {
        ScalarOne(o);
    }
}

// ============================== 入口 ==============================

template <typename T>
__aicore__ inline void ReduceLogSumExp<T>::Process()
{
    if (mode_ == MODE_COPY) {
        ProcessCopy();
    } else if (mode_ == MODE_P2B) {
        ProcessP2B();
    } else if (mode_ == MODE_P2A) {
        ProcessP2A();
    } else if (mode_ == MODE_P3) {
        ProcessP3();
    } else {
        ProcessScalar();
    }
}

} // namespace NsReduceLogSumExp
#endif // REDUCE_LOG_SUM_EXP_H

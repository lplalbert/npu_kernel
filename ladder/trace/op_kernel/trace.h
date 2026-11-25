/*!
 * \file trace.h
 * \brief Trace 算子 kernel 类定义
 *
 * out = sum_{i<min(M,N)} x[i][i]，内部 FP32 累加，输出与输入同 dtype。
 *
 * 访存：主对角线跨步 stride = N+1 个元素。用 DataCopyPad 以「每行 32B」为一块做跨步收集
 *   (blockCount=行数, blockLen=32, srcStride=(N+1)*sizeof(T)-32, dstStride=0)，
 *   每行独占一个 32B 槽（天然 UB 对齐），再用向量 Gather 按字节偏移取每槽首元素，Cast 成
 *   FP32 后在 UB 累加器上向量累加。读 32B/行相对整矩阵可忽略，但把访存指令数从 K 次降到 K/块。
 *
 * 尾块：最后一个对角行（全局下标 K-1）用 blockLen=sizeof(T) 的单块 DataCopyPad 收集，避免 32B 越界；
 *   块内不足 64 行时走 ScalarRows（SetValue 收集 + Cast 64 + 标量求和），保证向量 op/Gather count>=64。
 *
 * 多核（仅 fp32）：各核按对角行分段求部分和 -> 原子累加进 out。
 *   起始 out 含脏值 g（各核在计算前先读到同一 g），core0 加 (P0-g)、其余核加 Pk => sum。
 *   若 g 非有限（如重复调用时上一次结果为 inf），则退化为直接累加（inf 传播正确）。
 *   fp16 对角线 <= 2048，用单核纯写。
 *
 * 同步：输入搬运走 TQue(VECIN)（MTE2->V / V->MTE2）；向量写->标量读用 V_S；标量写 UB->向量读、
 *   MTE3 前后用 PipeBarrier<PIPE_ALL>()。
 */

#ifndef TRACE_H
#define TRACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trace_tiling_data.h"
#include "trace_tiling_key.h"
#include <type_traits>

namespace NsTrace {

using namespace AscendC;

constexpr int64_t kSlotFloats = 8;   // 原子累加 32B = 8 float
constexpr int64_t kScalarLane = 64;  // 标量收集/Cast 的最小向量宽度

template <typename T>
class Trace {
public:
    __aicore__ inline Trace() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const TraceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoBlockChunk(int64_t off, int64_t cnt);
    __aicore__ inline float ScalarRows(int64_t start, int64_t cnt);
    __aicore__ inline float FoldAcc(int64_t n);
    __aicore__ inline float Accumulate(int64_t s, int64_t e);
    __aicore__ inline void StoreOut(float v);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQ;
    TBuf<QuePosition::VECCALC> f32Buf;  // fp16 -> fp32 Cast 目标
    TBuf<QuePosition::VECCALC> gBuf;    // 单块 Gather 结果
    TBuf<QuePosition::VECCALC> accBuf;  // lane 累加器
    TBuf<QuePosition::VECCALC> offBuf;  // Gather 字节偏移
    TBuf<QuePosition::VECCALC> scT;     // 标量收集（T）
    TBuf<QuePosition::VECCALC> scF;     // 标量收集（float）
    TBuf<QuePosition::VECCALC> outT;    // 输出 Cast（fp16）
    TBuf<QuePosition::VECCALC> slotBuf; // 原子累加 32B 暂存

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    TraceTilingData td_;
    int64_t n_ = 0;
    int64_t stride_ = 0;
    int64_t k_ = 0;
    int64_t chunk_ = 0;
    int64_t blkElems_ = 0;
};

template <typename T>
__aicore__ inline void Trace<T>::Init(GM_ADDR self, GM_ADDR out, const TraceTilingData* tilingData)
{
    td_ = *tilingData;
    n_ = td_.n;
    if (n_ < 1) {
        n_ = 1;
    }
    k_ = td_.k;
    if (k_ < 1) {
        k_ = 1; // 退化保护（正常输入 k>=1）
    }
    stride_ = n_ + 1;
    chunk_ = td_.chunk;
    if (chunk_ < kScalarLane) {
        chunk_ = kScalarLane;
    }
    blkElems_ = static_cast<int64_t>(32 / sizeof(T));

    int64_t mrows = td_.m;
    if (mrows < 1) {
        mrows = 1;
    }
    xGm.SetGlobalBuffer((__gm__ T*)self, mrows * n_);
    yGm.SetGlobalBuffer((__gm__ T*)out, 1);

    pipe.InitBuffer(inQ, 1, static_cast<uint32_t>(chunk_ * 32)); // 每行一槽 32B
    pipe.InitBuffer(gBuf, static_cast<uint32_t>(chunk_ * sizeof(float)));
    pipe.InitBuffer(accBuf, static_cast<uint32_t>(chunk_ * sizeof(float)));
    pipe.InitBuffer(offBuf, static_cast<uint32_t>(chunk_ * sizeof(int32_t)));
    pipe.InitBuffer(scT, static_cast<uint32_t>(kScalarLane * sizeof(T)));
    pipe.InitBuffer(outT, static_cast<uint32_t>(kScalarLane * sizeof(T)));
    pipe.InitBuffer(scF, static_cast<uint32_t>(kScalarLane * sizeof(float)));
    pipe.InitBuffer(slotBuf, static_cast<uint32_t>(kSlotFloats * sizeof(float)));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(f32Buf, static_cast<uint32_t>(chunk_ * blkElems_ * sizeof(float)));
    }

    // Gather 字节偏移：第 j 槽首元素在源 float 张量的 j*blkElems_ 处
    LocalTensor<int32_t> off = offBuf.Get<int32_t>();
    CreateVecIndex(off, static_cast<int32_t>(0), static_cast<uint32_t>(chunk_));
    Muls(off, off, static_cast<int32_t>(blkElems_ * 4), static_cast<uint32_t>(chunk_));
}

template <typename T>
__aicore__ inline void Trace<T>::DoBlockChunk(int64_t off, int64_t cnt)
{
    LocalTensor<T> blk = inQ.template AllocTensor<T>();

    DataCopyExtParams p;
    p.blockCount = static_cast<uint16_t>(cnt);
    p.blockLen = 32;
    p.srcStride = static_cast<uint32_t>(stride_ * static_cast<int64_t>(sizeof(T)) - 32);
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPad(blk, xGm[off * stride_], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
    inQ.EnQue(blk);

    LocalTensor<T> src = inQ.template DeQue<T>();
    LocalTensor<float> g = gBuf.Get<float>();
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<uint32_t> offv = offBuf.Get<int32_t>().template ReinterpretCast<uint32_t>();

    const uint32_t pc = static_cast<uint32_t>(((cnt + 63) / 64) * 64);
    Duplicate(g, 0.0f, pc);

    if constexpr (std::is_same<T, float>::value) {
        Gather(g, src, offv, 0, static_cast<uint32_t>(cnt));
    } else {
        LocalTensor<float> fb = f32Buf.Get<float>();
        Cast(fb, src, RoundMode::CAST_NONE, static_cast<int32_t>(cnt * blkElems_));
        Gather(g, fb, offv, 0, static_cast<uint32_t>(cnt));
    }
    Add(acc, acc, g, pc);
    inQ.FreeTensor(src);
}

template <typename T>
__aicore__ inline float Trace<T>::ScalarRows(int64_t start, int64_t cnt)
{
    LocalTensor<T> t = scT.Get<T>();
    for (int64_t j = 0; j < cnt; ++j) {
        t.SetValue(j, xGm.GetValue((start + j) * stride_));
    }
    float s = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        for (int64_t j = 0; j < cnt; ++j) {
            s += t.GetValue(j);
        }
    } else {
        LocalTensor<float> f = scF.Get<float>();
        PipeBarrier<PIPE_ALL>(); // S -> V
        Cast(f, t, RoundMode::CAST_NONE, static_cast<int32_t>(kScalarLane));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int64_t j = 0; j < cnt; ++j) {
            s += f.GetValue(j);
        }
    }
    return s;
}

template <typename T>
__aicore__ inline float Trace<T>::FoldAcc(int64_t n)
{
    LocalTensor<float> acc = accBuf.Get<float>();
    while (n > 128) {
        const int64_t h = n >> 1;
        Add(acc, acc, acc[static_cast<int32_t>(h)], static_cast<uint32_t>(h));
        n = h;
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float s = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
        s += acc.GetValue(j);
    }
    return s;
}

template <typename T>
__aicore__ inline float Trace<T>::Accumulate(int64_t s, int64_t e)
{
    LocalTensor<float> acc = accBuf.Get<float>();
    Duplicate(acc, 0.0f, static_cast<uint32_t>(chunk_));

    int64_t bEnd = e;
    bool hasLast = (k_ - 1 >= s) && (k_ - 1 < e);
    if (hasLast) {
        bEnd = k_ - 1;
    }

    int64_t off = s;
    float tail = 0.0f;
    while (off < bEnd) {
        int64_t cnt = bEnd - off;
        if (cnt > chunk_) {
            cnt = chunk_;
        }
        if (cnt >= kScalarLane) {
            DoBlockChunk(off, cnt);
        } else {
            tail += ScalarRows(off, cnt);
        }
        off += cnt;
    }

    float sum = FoldAcc(chunk_) + tail;
    if (hasLast) {
        sum += ScalarRows(k_ - 1, 1);
    }
    return sum;
}

template <typename T>
__aicore__ inline void Trace<T>::StoreOut(float v)
{
    if constexpr (std::is_same<T, float>::value) {
        yGm.SetValue(0, v);
    } else {
        LocalTensor<float> f = scF.Get<float>();
        LocalTensor<T> t = outT.Get<T>();
        f.SetValue(0, v);
        for (int64_t j = 1; j < kScalarLane; ++j) {
            f.SetValue(j, 0.0f);
        }
        PipeBarrier<PIPE_ALL>(); // S -> V
        Cast(t, f, RoundMode::CAST_RINT, static_cast<int32_t>(kScalarLane));
        PipeBarrier<PIPE_ALL>(); // V -> S
        yGm.SetValue(0, t.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void Trace<T>::Process()
{
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum < 1) {
        blockNum = 1;
    }
    const int64_t per = td_.blockFactor;
    int64_t s = blockId * per;
    if (s > k_) {
        s = k_;
    }
    int64_t e = s + per;
    if (e > k_) {
        e = k_;
    }

    // 多核原子累加前，先读 y 的脏初值 g（先于本地计算与任何原子加，保证各核读到同一初值）
    float g0 = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        if (blockNum > 1) {
            g0 = yGm.GetValue(0);
        }
    }

    float part = 0.0f;
    if (s < e) {
        part = Accumulate(s, e);
    }

    if (blockNum <= 1) {
        StoreOut(part);
        return;
    }

    if constexpr (std::is_same<T, float>::value) {
        // g 非有限（含 Inf/NaN，或重复调用时上一次结果为 Inf）时退化为直接累加，保证 Inf 传播；
        // 否则 core0 减去 g、其余核直接加，最终 y = sum(Pk)。
        const bool gBad = (g0 != g0) || (g0 > 3.402823466e38f) || (g0 < -3.402823466e38f);
        const float delta = (gBad || blockId != 0) ? part : (part - g0);
        LocalTensor<float> tmp = slotBuf.Get<float>();
        tmp.SetValue(0, delta);
        for (int64_t j = 1; j < kSlotFloats; ++j) {
            tmp.SetValue(j, 0.0f);
        }
        PipeBarrier<PIPE_ALL>(); // S -> MTE3
        SetAtomicAdd<float>();
        DataCopy(yGm[0], tmp, static_cast<uint32_t>(kSlotFloats)); // 32B 原子加
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    } else {
        StoreOut(part); // fp16 走单核，不会到这
    }
}

} // namespace NsTrace
#endif // TRACE_H

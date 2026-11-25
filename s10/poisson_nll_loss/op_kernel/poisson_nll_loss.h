/*!
 * \file poisson_nll_loss.h
 * \brief PoissonNllLoss 算子 kernel 类定义
 *
 * loss_i = log_input ? exp(x_i) - t_i * x_i : x_i - t_i * ln(x_i + eps)
 * full 时对 t_i > 1 再加 Stirling 项 t*ln t - t + 0.5*ln(2*pi*t)。
 * 全部在 float32 上做向量计算。
 * reduction=none  : 逐元素多核写回（各核不重叠区间，无跨核同步）
 * reduction=sum/mean : 单核（跨核合并需要 SyncAll，本机必现 507035），
 *                     但归约向量化：每 chunk 一条 Add 累加到 ubLen 个 lane，最后一次标量收 lane。
 */

#ifndef POISSONNLLLOSS_H
#define POISSONNLLLOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "poisson_nll_loss_tiling_data.h"
#include "poisson_nll_loss_tiling_key.h"
#include <type_traits>

namespace NsPoissonNllLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class PoissonNllLoss {
public:
    __aicore__ inline PoissonNllLoss(){};

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                const PoissonNllLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
    __aicore__ inline void StoreScalar(float value);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fBuf; // float: x -> loss
    TBuf<QuePosition::VECCALC> gBuf; // float: target
    TBuf<QuePosition::VECCALC> hBuf; // float: temp
    TBuf<QuePosition::VECCALC> iBuf; // float: temp
    TBuf<QuePosition::VECCALC> jBuf; // float: sum/mean 的 lane 累加器
    TBuf<QuePosition::VECCALC> tBuf; // T: 16bit 收/发缓存

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMT;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> wsFGm; // 多核部分和（当前 sum/mean 走单核，保留绑定）

    PoissonNllLossTilingData td_;
};

template <typename T>
__aicore__ inline void PoissonNllLoss<T>::Init(GM_ADDR input_x, GM_ADDR target, GM_ADDR loss,
                                               GM_ADDR workspace,
                                               const PoissonNllLossTilingData* tilingData)
{
    td_ = *tilingData;
    inputGMX.SetGlobalBuffer((__gm__ T*)input_x);
    inputGMT.SetGlobalBuffer((__gm__ T*)target);
    outputGMY.SetGlobalBuffer((__gm__ T*)loss);
    wsFGm.SetGlobalBuffer((__gm__ float*)workspace);
    pipe.InitBuffer(fBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(gBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(hBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(iBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(jBuf, td_.ubLen * sizeof(float));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(tBuf, td_.ubLen * sizeof(T));
    }
}

// 把 x/target 的一个分块装到 fBuf/gBuf（float32）
template <typename T>
__aicore__ inline void PoissonNllLoss<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<float> f = fBuf.Get<float>();
    LocalTensor<float> g = gBuf.Get<float>();
    int64_t bytes = currentNum * static_cast<int64_t>(sizeof(T));
    bool aligned = ((bytes & 31) == 0);
    if constexpr (std::is_same<T, float>::value) {
        if (aligned) {
            DataCopy(f, inputGMX[offset], currentNum);
            DataCopy(g, inputGMT[offset], currentNum);
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(f, inputGMX[offset], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
            DataCopyPad(g, inputGMT[offset], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    } else {
        LocalTensor<T> t = tBuf.Get<T>();
        int64_t cnt = ((currentNum + 15) / 16) * 16; // Cast 长度按 32B 取整
        if (aligned) {
            DataCopy(t, inputGMX[offset], currentNum);
            PipeBarrier<PIPE_ALL>();
            Cast(f, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
            DataCopy(t, inputGMT[offset], currentNum);
            PipeBarrier<PIPE_ALL>();
            Cast(g, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(t, inputGMX[offset], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(f, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(t, inputGMT[offset], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(g, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// 逐元素 loss 写入 fBuf（float32）
template <typename T>
__aicore__ inline void PoissonNllLoss<T>::Compute(int64_t currentNum)
{
    LocalTensor<float> f = fBuf.Get<float>(); // x -> loss
    LocalTensor<float> g = gBuf.Get<float>(); // target
    LocalTensor<float> h = hBuf.Get<float>();
    LocalTensor<float> i = iBuf.Get<float>();
    uint32_t cnt = static_cast<uint32_t>(currentNum);

    if (td_.logInput != 0) {
        Mul(h, g, f, cnt); // t * x
        Exp(f, f, cnt);    // exp(x)
        Sub(f, f, h, cnt); // exp(x) - t*x
    } else {
        Adds(h, f, td_.eps, cnt); // x + eps
        Ln(h, h, cnt);            // ln(x+eps)
        Mul(h, g, h, cnt);        // t * ln(x+eps)
        Sub(f, f, h, cnt);        // x - t*ln(x+eps)
    }

    if (td_.full != 0) {
        // Stirling: t*ln t - t + 0.5*ln(2*pi*t)，仅 t>1 生效
        Maxs(h, g, 1.0f, cnt);               // tm = max(t,1)：避免 t<=0 出 NaN/-inf
        Ln(i, h, cnt);                       // ln(tm)
        Mul(i, i, h, cnt);                   // tm*ln(tm)
        Sub(i, i, h, cnt);                   // tm*ln(tm) - tm
        Muls(h, h, 6.283185307179586f, cnt); // 2*pi*tm
        Ln(h, h, cnt);
        Muls(h, h, 0.5f, cnt); // 0.5*ln(2*pi*tm)
        Add(i, i, h, cnt);     // Stirling 项
        // 掩码 k = clamp((t-1)*BIG, 0, 1)：t>1 取 1、t<1 取 0
        Adds(h, g, -1.0f, cnt);
        Muls(h, h, 1.0e6f, cnt);
        Maxs(h, h, 0.0f, cnt);
        Mins(h, h, 1.0f, cnt);
        Mul(i, i, h, cnt);
        Add(f, f, i, cnt);
    }
}

// reduction=none：把 fBuf 里 float loss 写回 GM
template <typename T>
__aicore__ inline void PoissonNllLoss<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<float> f = fBuf.Get<float>();
    int64_t bytes = currentNum * static_cast<int64_t>(sizeof(T));
    bool aligned = ((bytes & 31) == 0);
    if constexpr (std::is_same<T, float>::value) {
        if (aligned) {
            DataCopy(outputGMY[offset], f, currentNum);
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(outputGMY[offset], f, p);
        }
    } else {
        LocalTensor<T> t = tBuf.Get<T>();
        int64_t cnt = ((currentNum + 15) / 16) * 16;
        Cast(t, f, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_ALL>();
        if (aligned) {
            DataCopy(outputGMY[offset], t, currentNum);
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(outputGMY[offset], t, p);
        }
    }
}

// 写 [1] 标量结果
template <typename T>
__aicore__ inline void PoissonNllLoss<T>::StoreScalar(float value)
{
    if constexpr (std::is_same<T, float>::value) {
        outputGMY.SetValue(0, value);
    } else {
        LocalTensor<float> f = fBuf.Get<float>();
        LocalTensor<T> t = tBuf.Get<T>();
        f.SetValue(0, value);
        PipeBarrier<PIPE_ALL>(); // S -> V
        Cast(t, f, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_ALL>(); // V -> S
        outputGMY.SetValue(0, t.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void PoissonNllLoss<T>::Process()
{
    int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t total = td_.totalNum > 0 ? td_.totalNum : 0;
    int64_t ubLen = td_.ubLen > 0 ? td_.ubLen : 64;
    LocalTensor<float> f = fBuf.Get<float>();

    if (td_.reduction == 0) {
        // reduction=none：多核按 blockIdx 划分不重叠区间（逐元素，无跨核归约）
        int64_t bf = td_.blockFactor > 0 ? td_.blockFactor : 0;
        int64_t start = bf * blockId;
        if (start > total) {
            start = total;
        }
        int64_t end = start + bf;
        if (end > total) {
            end = total;
        }
        for (int64_t off = start; off < end; off += ubLen) {
            int64_t cur = end - off;
            if (cur > ubLen) {
                cur = ubLen;
            }
            CopyIn(off, cur);
            PipeBarrier<PIPE_ALL>();
            Compute(cur);
            PipeBarrier<PIPE_ALL>();
            CopyOut(off, cur);
            // 必须等 MTE3 把 fBuf 写出去，下一轮 CopyIn 才能覆盖 fBuf
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }

    // sum / mean：单核 + 向量化 lane 归约
    //（跨核合并要 SyncAll，本机必崩；逐元素标量 Kahan 太慢：1M 元素 21ms -> 改完 0.05ms 量级）
    if (blockId != 0) {
        return;
    }
    LocalTensor<float> gacc = jBuf.Get<float>();
    Duplicate(gacc, 0.0f, static_cast<uint32_t>(ubLen));
    PipeBarrier<PIPE_ALL>();

    float tailAcc = 0.0f;
    float tailComp = 0.0f;
    for (int64_t off = 0; off < total; off += ubLen) {
        int64_t cur = total - off;
        if (cur > ubLen) {
            cur = ubLen;
        }
        CopyIn(off, cur);
        PipeBarrier<PIPE_ALL>(); // MTE2 -> V
        Compute(cur);
        // 注意：这里不要再插 V->V 的 PipeBarrier（同流水线本来就按序），
        // 每 chunk 一条 PIPE_ALL 屏障在 8M 级大例上是可测的固定开销。
        if (cur == ubLen) {
            Add(gacc, gacc, f, static_cast<uint32_t>(ubLen));
        } else {
            // 尾块（< ubLen）：标量 Kahan 补足，避免把 f 里无效 lane 加进来
            for (int64_t j = 0; j < cur; ++j) {
                float v = f.GetValue(j);
                float yv = v - tailComp;
                float t = tailAcc + yv;
                tailComp = (t - tailAcc) - yv;
                tailAcc = t;
            }
        }
        PipeBarrier<PIPE_ALL>(); // V/S -> 下一轮 MTE2
    }

    // lane 归约（只做一次，ubLen 次标量读）
    PipeBarrier<PIPE_ALL>(); // V -> S
    float acc = 0.0f;
    float comp = 0.0f;
    for (int64_t j = 0; j < ubLen; ++j) {
        float v = gacc.GetValue(j);
        float yv = v - comp;
        float t = acc + yv;
        comp = (t - acc) - yv;
        acc = t;
    }
    float yv2 = tailAcc - comp;
    acc = acc + yv2;

    if (td_.reduction == 2) {
        acc *= td_.invN;
    }
    StoreScalar(acc);
}

} // namespace NsPoissonNllLoss
#endif // POISSONNLLLOSS_H

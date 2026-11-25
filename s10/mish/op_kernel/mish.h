/*!
 * \file mish.h
 * \brief Mish 算子 kernel 类定义
 *
 * out = x * tanh(softplus(x)), softplus(x) = ln(1 + exp(x))
 * 数值稳定实现：softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
 * 中间量一律 float32 计算，16bit dtype 存储侧收发，避免 exp/ln 溢出与精度损失。
 */

#ifndef MISH_H
#define MISH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mish_tiling_data.h"
#include "mish_tiling_key.h"
#include <type_traits>

namespace NsMish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Mish {
public:
    __aicore__ inline Mish(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> xBufF;   // 16bit dtype 的 float 升配缓存
    TBuf<QuePosition::VECCALC> sBufF;   // softplus / tanh 中间量
    TBuf<QuePosition::VECCALC> tBufF;   // ln(1+exp(-|x|)) 中间量

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t offset_ = 0;       // 本核起始元素偏移
    int64_t blockLength_ = 0;  // 本核处理元素数
    int64_t ubLength_ = 0;     // 单次 UB 循环元素数
};

template <typename T>
__aicore__ inline void Mish<T>::Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData)
{
    int64_t blockStart = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    if (blockStart >= tilingData->totalNum) {
        blockLength_ = 0;
        return;
    }
    offset_ = blockStart;
    blockLength_ = tilingData->blockFactor;
    if (offset_ + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - offset_;
    }
    ubLength_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : blockLength_;

    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(sBufF, ubLength_ * sizeof(float));
    pipe.InitBuffer(tBufF, ubLength_ * sizeof(float));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(xBufF, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Mish<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xIn = inputQueueX.AllocTensor<T>();
    int64_t bytes = currentNum * static_cast<int64_t>(sizeof(T));
    if ((bytes & 31) == 0) {
        // 32B 对齐快路径（tiling 保证块起点对齐；仅尾块走变长拷贝）
        DataCopy(xIn, inputGMX[offset_ + progress * ubLength_], currentNum);
    } else {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(bytes), 0, 0, 0};
        DataCopyPad(xIn, inputGMX[offset_ + progress * ubLength_], copyInParams,
                    DataCopyPadExtParams<T>{false, 0, 0, 0});
    }
    inputQueueX.EnQue(xIn);
}

template <typename T>
__aicore__ inline void Mish<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xIn = inputQueueX.DeQue<T>();
    LocalTensor<T> yOut = outputQueueY.AllocTensor<T>();

    LocalTensor<float> sF = sBufF.Get<float>();
    LocalTensor<float> tF = tBufF.Get<float>();
    LocalTensor<float> xF;

    if constexpr (std::is_same<T, float>::value) {
        xF = xIn;  // float 存储直接零拷贝复用
    } else {
        xF = xBufF.Get<float>();
        Cast(xF, xIn, RoundMode::CAST_NONE, currentNum);
    }

    // tF = ln(1 + exp(-|x|))，恒在 [ln1, ln2]，无溢出
    Abs(tF, xF, currentNum);
    Muls(tF, tF, static_cast<float>(-1.0), currentNum);
    Exp(tF, tF, currentNum);
    Adds(tF, tF, static_cast<float>(1.0), currentNum);
    Ln(tF, tF, currentNum);

    // sF = max(x, 0) + ln(1 + exp(-|x|)) = softplus(x)
    Maxs(sF, xF, static_cast<float>(0.0), currentNum);
    Add(sF, sF, tF, currentNum);

    // sF = tanh(softplus(x))
    Tanh(sF, sF, currentNum);

    if constexpr (std::is_same<T, float>::value) {
        Mul(yOut, xIn, sF, currentNum);
    } else {
        Mul(xF, xF, sF, currentNum);
        Cast(yOut, xF, RoundMode::CAST_RINT, currentNum);
    }

    inputQueueX.FreeTensor(xIn);
    outputQueueY.EnQue(yOut);
}

template <typename T>
__aicore__ inline void Mish<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yOut = outputQueueY.DeQue<T>();
    int64_t bytes = currentNum * static_cast<int64_t>(sizeof(T));
    if ((bytes & 31) == 0) {
        DataCopy(outputGMY[offset_ + progress * ubLength_], yOut, currentNum);
    } else {
        // UB->GM 方向 DataCopyPad 为 3 参重载（字节级变长，支持非 32B 对齐）
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(bytes), 0, 0, 0};
        DataCopyPad(outputGMY[offset_ + progress * ubLength_], yOut, copyOutParams);
    }
    outputQueueY.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void Mish<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loops = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loops; ++i) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsMish
#endif // MISH_H

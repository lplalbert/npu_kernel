/*!
 * \file bessel_i0e.h
 * \brief BesselI0e 算子 kernel 类定义
 *
 * y = exp(-|x|) * I0(x)，禁止直接算 I0（大 |x| 会溢出），故按 Cephes 分段：
 *   |x| <= 3.75 : I0 = P1((fs/3.75)^2)，fs = min(|x|,3.75)，再乘 exp(-fs)
 *   |x| >  3.75 : i0e = P2(3.75/mx) / sqrt(mx)，mx = max(|x|,3.75)（渐近展开，自带指数衰减）
 * 两段用窄带 mask 线性混合，避免 inf*0 = NaN 与除零。
 *
 * 特殊值（题面 §3.6 要求）：
 *   - ±inf -> 0：mx 先 Min 到 3.0e38，避免 inf*rsqrt(inf)=inf*0=NaN；
 *   - NaN    -> NaN：额外算 poison = exp(-|x|)*0（有限值->0，inf->0，NaN->NaN），最后加到结果上，
 *                    保证即使 Max/Min 吞掉 NaN，输出仍是 NaN。
 * 多核：逐元素按 blockIdx 划分不重叠区间（无跨核同步）。
 */

#ifndef BESSELI0E_H
#define BESSELI0E_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bessel_i0e_tiling_data.h"
#include "bessel_i0e_tiling_key.h"
#include <type_traits>

namespace NsBesselI0e {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class BesselI0e {
public:
    __aicore__ inline BesselI0e(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BesselI0eTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fBuf; // float: |x| -> y
    TBuf<QuePosition::VECCALC> gBuf; // float: temp
    TBuf<QuePosition::VECCALC> hBuf; // float: temp
    TBuf<QuePosition::VECCALC> iBuf; // float: temp
    TBuf<QuePosition::VECCALC> jBuf; // float: temp
    TBuf<QuePosition::VECCALC> kBuf; // float: NaN poison
    TBuf<QuePosition::VECCALC> tBuf; // T: 16bit 收/发缓存

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    BesselI0eTilingData td_;
};

template <typename T>
__aicore__ inline void BesselI0e<T>::Init(GM_ADDR x, GM_ADDR y, const BesselI0eTilingData* tilingData)
{
    td_ = *tilingData;
    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);
    pipe.InitBuffer(fBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(gBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(hBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(iBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(jBuf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(kBuf, td_.ubLen * sizeof(float));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(tBuf, td_.ubLen * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void BesselI0e<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<float> f = fBuf.Get<float>();
    int64_t bytes = currentNum * static_cast<int64_t>(sizeof(T));
    bool aligned = ((bytes & 31) == 0);
    if constexpr (std::is_same<T, float>::value) {
        if (aligned) {
            DataCopy(f, inputGMX[offset], currentNum);
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(f, inputGMX[offset], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    } else {
        LocalTensor<T> t = tBuf.Get<T>();
        int64_t cnt = ((currentNum + 15) / 16) * 16; // Cast 长度按 32B 取整
        if (aligned) {
            DataCopy(t, inputGMX[offset], currentNum);
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(t, inputGMX[offset], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        Cast(f, t, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void BesselI0e<T>::Compute(int64_t currentNum)
{
    LocalTensor<float> f = fBuf.Get<float>(); // |x| -> y
    LocalTensor<float> g = gBuf.Get<float>();
    LocalTensor<float> h = hBuf.Get<float>();
    LocalTensor<float> i = iBuf.Get<float>();
    LocalTensor<float> j = jBuf.Get<float>();
    LocalTensor<float> k = kBuf.Get<float>(); // NaN poison
    uint32_t cnt = static_cast<uint32_t>(currentNum);

    Abs(f, f, cnt); // f = |x|

    // NaN poison：exp(-|x|) 有限->(0,1]、+inf->0、NaN->NaN；再乘 0 只剩 0 或 NaN
    Muls(k, f, -1.0f, cnt);
    Exp(k, k, cnt);
    Muls(k, k, 0.0f, cnt);

    // ---- 小 |x| 段：y1 = exp(-fs)*I0(fs)，fs = min(|x|,3.75) ----
    // 用 fs 限幅：大 |x| 时 t<=1、exp(-fs) 不下溢、P1 不溢出（该段结果随后被 mask 丢弃）
    Mins(g, f, 3.75f, cnt);                // fs
    Muls(h, g, 0.26666666666666666f, cnt); // fs/3.75
    Mul(h, h, h, cnt);                     // t
    Muls(i, h, 0.0045813f, cnt);
    Adds(i, i, 0.0360768f, cnt);
    Mul(i, i, h, cnt);
    Adds(i, i, 0.2659732f, cnt);
    Mul(i, i, h, cnt);
    Adds(i, i, 1.2067492f, cnt);
    Mul(i, i, h, cnt);
    Adds(i, i, 3.0899424f, cnt);
    Mul(i, i, h, cnt);
    Adds(i, i, 3.5156229f, cnt);
    Mul(i, i, h, cnt);
    Adds(i, i, 1.0f, cnt); // P1 = I0(t)
    Muls(g, g, -1.0f, cnt);
    Exp(g, g, cnt);       // exp(-fs)
    Mul(i, i, g, cnt);    // y1

    // ---- 大 |x| 段：y2 = P2(3.75/mx)/sqrt(mx)，mx = clamp(max(|x|,3.75), 3e38) ----
    // 限幅到 3.0e38：避免 |x|=inf 时 rsqrt(inf)=0 与 inf 相乘产生 NaN（题面要求 i0e(±inf)=0）
    Maxs(g, f, 3.75f, cnt);
    Mins(g, g, 3.0e38f, cnt);
    Rsqrt(j, g, cnt); // r = 1/sqrt(mx)（硬件近似）
    // 一次 Newton 迭代把 rsqrt 精度提到 ~1e-6：r = r*(1.5 - 0.5*mx*r*r)
    Mul(h, g, j, cnt);
    Mul(h, h, j, cnt);
    Muls(h, h, -0.5f, cnt);
    Adds(h, h, 1.5f, cnt);
    Mul(j, j, h, cnt);
    Mul(g, j, j, cnt);      // 1/mx
    Muls(g, g, 3.75f, cnt); // t2 = 3.75/mx
    Muls(h, g, 0.00392377f, cnt);
    Adds(h, h, -0.01647633f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, 0.02635537f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, -0.02057706f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, 0.00916281f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, -0.00157565f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, 0.00225319f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, 0.01328592f, cnt);
    Mul(h, h, g, cnt);
    Adds(h, h, 0.39894228f, cnt); // P2
    Mul(j, j, h, cnt);            // y2 = P2/sqrt(mx)

    // ---- 混合：y = y1 + kk*(y2-y1)，kk = clamp((|x|-3.75)*1e6+0.5, 0, 1) ----
    Adds(g, f, -3.75f, cnt);
    Muls(g, g, 1.0e6f, cnt);
    Adds(g, g, 0.5f, cnt);
    Maxs(g, g, 0.0f, cnt);
    Mins(g, g, 1.0f, cnt); // kk
    Sub(f, j, i, cnt);     // y2-y1
    Mul(f, f, g, cnt);
    Add(f, f, i, cnt); // y

    // NaN 透传（±inf 时 k 段为 0，不影响结果）
    Add(f, f, k, cnt);
}

template <typename T>
__aicore__ inline void BesselI0e<T>::CopyOut(int64_t offset, int64_t currentNum)
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

template <typename T>
__aicore__ inline void BesselI0e<T>::Process()
{
    // 多核：按 blockIdx 划分不重叠元素区间（逐元素算子，无跨核归约）
    int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t total = td_.totalNum > 0 ? td_.totalNum : 0;
    int64_t bf = td_.blockFactor > 0 ? td_.blockFactor : 0;
    int64_t start = bf * blockId;
    if (start > total) {
        start = total;
    }
    int64_t end = start + bf;
    if (end > total) {
        end = total;
    }
    int64_t ubLen = td_.ubLen > 0 ? td_.ubLen : 64;
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
        // 等 MTE3 写出 fBuf/tBuf 后再进入下一轮，避免覆盖
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace NsBesselI0e
#endif // BESSELI0E_H

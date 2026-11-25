/*!
 * \file dot.h
 * \brief Dot 算子 kernel 类定义
 *
 * y = sum(x1[i] * x2[i])，输出 shape [1]。
 *
 * 多核归约：各核部分和写用户 workspace -> SyncAll -> core0 汇总 -> 写 y。
 *
 * 关键坑（已实测定位）：kernel 收到的 `workspace` 是系统(KFC)工作区基址，用户区在
 * `GetUserWorkspace(workspace) = kfc_workspace + RESERVED_WORKSPACE`，910B 上
 * RESERVED_WORKSPACE = 16MB。**tiling 必须把 GetLibApiWorkSpaceSize()（≥16MB）算进
 * currentWorkspace[0]**，否则用户区指针落在分配区外，一写就 ACL 507035（向量核异常）。
 * 槽位按 128 元素间隔（float 512B / int64 1KB），避免多核写同一 cache line。
 *
 * 浮点：向量 Mul + lane 向量累加 + 标量收 lane；int32：int64 精确累加（与 torch 一致）。
 * 16bit 存储侧收发仅用向量 Cast（bf16/half 标量转换不被后端支持）。
 */

#ifndef DOT_H
#define DOT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dot_tiling_data.h"
#include "dot_tiling_key.h"
#include <type_traits>

#define DOT_EMPTY 0
#define DOT_DIAG 0

namespace NsDot {

using namespace AscendC;

constexpr int64_t WS_STRIDE = 128;
constexpr int64_t DOT_WS_OFFSET = 8192; // skip framework header bytes

template <typename T>
class Dot {
public:
    __aicore__ inline Dot(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                const DotTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadChunk(int64_t base, int64_t cur);
    __aicore__ inline float SumRangeF(int64_t start, int64_t len);
    __aicore__ inline void StoreF(float v);
    __aicore__ inline void StoreI(int64_t v);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> f1Buf; // ubLen * float
    TBuf<QuePosition::VECCALC> f2Buf; // ubLen * float
    TBuf<QuePosition::VECCALC> gBuf;  // ubLen * float（lane 累加器）
    TBuf<QuePosition::VECCALC> tBuf;  // ubLen * T（16bit 收发缓存）

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> wsFGm;     // 用户 workspace：浮点部分和槽
    GlobalTensor<int64_t> wsIGm;   // 用户 workspace：整型部分和槽
    GlobalTensor<int32_t> wsI32Gm; // 同上，int32 视图（DataCopy 不支持 int64）
    GlobalTensor<int32_t> yI32Gm;
    GlobalTensor<float> yF32Gm;

    DotTilingData td_;
};

template <typename T>
__aicore__ inline void Dot<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                    const DotTilingData* tilingData)
{
    td_ = *tilingData;
    x1Gm.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    // 用户区基址 = GetUserWorkspace(workspace)（= 系统 KFC 区 + 16MB 保留区）
    GM_ADDR userWS = GetUserWorkspace(workspace);
    wsFGm.SetGlobalBuffer((__gm__ float*)userWS);
    wsIGm.SetGlobalBuffer((__gm__ int64_t*)userWS);
    wsI32Gm.SetGlobalBuffer((__gm__ int32_t*)userWS);
    yI32Gm.SetGlobalBuffer((__gm__ int32_t*)y);
    yF32Gm.SetGlobalBuffer((__gm__ float*)y);
    pipe.InitBuffer(f1Buf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(f2Buf, td_.ubLen * sizeof(float));
    pipe.InitBuffer(gBuf, td_.ubLen * sizeof(float));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(tBuf, td_.ubLen * sizeof(T));
    }
}

// 装载 x1/x2 分块到 f1Buf/f2Buf(float32)
template <typename T>
__aicore__ inline void Dot<T>::LoadChunk(int64_t base, int64_t cur)
{
    LocalTensor<float> f1 = f1Buf.Get<float>();
    LocalTensor<float> f2 = f2Buf.Get<float>();
    int64_t bytes = cur * static_cast<int64_t>(sizeof(T));
    bool aligned = ((bytes & 31) == 0);
    if constexpr (std::is_same<T, float>::value) {
        if (aligned) {
            DataCopy(f1, x1Gm[base], cur);
            DataCopy(f2, x2Gm[base], cur);
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(f1, x1Gm[base], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
            DataCopyPad(f2, x2Gm[base], p, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    } else {
        LocalTensor<T> t = tBuf.Get<T>();
        int64_t cnt = ((cur + 15) / 16) * 16; // Cast 长度按 32B 取整
        if (aligned) {
            DataCopy(t, x1Gm[base], cur);
            PipeBarrier<PIPE_ALL>();
            Cast(f1, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
            DataCopy(t, x2Gm[base], cur);
            PipeBarrier<PIPE_ALL>();
            Cast(f2, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyExtParams p{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(t, x1Gm[base], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(f1, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(t, x2Gm[base], p, DataCopyPadExtParams<T>{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(f2, t, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// [start, start+len) 的 float 点积（向量 lane 累加 + 标量收 lane）
template <typename T>
__aicore__ inline float Dot<T>::SumRangeF(int64_t start, int64_t len)
{
    int64_t ubLen = td_.ubLen > 0 ? td_.ubLen : 64;
    LocalTensor<float> f1 = f1Buf.Get<float>();
    LocalTensor<float> f2 = f2Buf.Get<float>();
    LocalTensor<float> g = gBuf.Get<float>();

    Duplicate(g, 0.0f, static_cast<uint32_t>(ubLen));
    PipeBarrier<PIPE_ALL>();

    int64_t r = 0;
    for (; r + ubLen <= len; r += ubLen) {
        LoadChunk(start + r, ubLen);
        PipeBarrier<PIPE_ALL>(); // MTE2 -> V
        Mul(f1, f1, f2, static_cast<uint32_t>(ubLen));
        Add(g, g, f1, static_cast<uint32_t>(ubLen));
        PipeBarrier<PIPE_ALL>(); // V -> 下一轮 MTE2
    }

    float tail = 0.0f;
    int64_t rem = len - r;
    if (rem > 0) {
        LoadChunk(start + r, rem);
        PipeBarrier<PIPE_ALL>();
        float comp = 0.0f;
        float s = 0.0f;
        for (int64_t j = 0; j < rem; ++j) {
            float prod = f1.GetValue(j) * f2.GetValue(j);
            float yv = prod - comp;
            float t = s + yv;
            comp = (t - s) - yv;
            s = t;
        }
        tail = s;
    }

    PipeBarrier<PIPE_ALL>(); // V -> S
    float acc = 0.0f;
    float comp2 = 0.0f;
    for (int64_t j = 0; j < ubLen; ++j) {
        float v = g.GetValue(j);
        float yv = v - comp2;
        float t = acc + yv;
        comp2 = (t - acc) - yv;
        acc = t;
    }
    return acc + tail;
}

template <typename T>
__aicore__ inline void Dot<T>::StoreF(float v)
{
    if constexpr (std::is_same<T, float>::value) {
        yGm.SetValue(0, v);
    } else {
        LocalTensor<float> f1 = f1Buf.Get<float>();
        LocalTensor<T> t = tBuf.Get<T>();
        constexpr int64_t CNT = 64; // 32B 粒度
        f1.SetValue(0, v);
        for (int64_t j = 1; j < CNT; ++j) {
            f1.SetValue(j, 0.0f);
        }
        PipeBarrier<PIPE_ALL>();
        Cast(t, f1, RoundMode::CAST_RINT, CNT);
        PipeBarrier<PIPE_ALL>();
        yGm.SetValue(0, t.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void Dot<T>::StoreI(int64_t v)
{
    if constexpr (std::is_same<T, int32_t>::value) {
        yGm.SetValue(0, static_cast<int32_t>(v));
    } else {
        StoreF(static_cast<float>(v));
    }
}

template <typename T>
__aicore__ inline void Dot<T>::Process()
{
    int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum < 1) {
        blockNum = 1;
    }
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
    int64_t len = end - start;

    // ---- read y once BEFORE any core writes (needed to cancel initial garbage) ----
    float gF = 0.0f;
    int32_t gI = 0;
    if (blockNum > 1) {
        if constexpr (std::is_same<T, float>::value) {
            gF = yGm.GetValue(0);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            gI = yGm.GetValue(0);
        }
    }

    // ---- per-core partial sum ----
    int64_t accI = 0;
    float accF = 0.0f;
    if constexpr (std::is_same<T, int32_t>::value) {
        for (int64_t i = 0; i < len; ++i) {
            int64_t a = static_cast<int64_t>(x1Gm.GetValue(start + i));
            int64_t b = static_cast<int64_t>(x2Gm.GetValue(start + i));
            accI += a * b;
        }
    } else {
        if (len > 0) {
            accF = SumRangeF(start, len);
        }
    }

    // ---- single core: plain store ----
    if (blockNum <= 1) {
        if constexpr (std::is_same<T, int32_t>::value) {
            StoreI(accI);
        } else {
            StoreF(accF);
        }
        return;
    }

    // ==== multi-core WITHOUT any GM workspace: atomic accumulate straight into y ====
    // y initially holds garbage g (read by every core before any add -> all see same g).
    // core0 adds (P0 - g), others add Pk  =>  final = sum(Pk).  No SyncAll / no workspace.
    if constexpr (std::is_same<T, float>::value) {
        float delta = (blockId == 0) ? (accF - gF) : accF;
        LocalTensor<float> tmp = f1Buf.Get<float>();
        tmp.SetValue(0, delta);
        for (int64_t j = 1; j < 8; ++j) {
            tmp.SetValue(j, 0.0f);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<float>();
        DataCopy(yGm[0], tmp, 8); // 32B atomic add
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
        return;
    }
    if constexpr (std::is_same<T, int32_t>::value) {
        // exact mod 2^32 arithmetic: true sum fits in int32 per problem spec
        uint32_t u = static_cast<uint32_t>(static_cast<uint64_t>(accI) & 0xFFFFFFFFULL);
        uint32_t delta = (blockId == 0) ? (u - static_cast<uint32_t>(gI)) : u;
        LocalTensor<int32_t> w = f1Buf.Get<int32_t>();
        w.SetValue(0, static_cast<int32_t>(delta));
        for (int64_t j = 1; j < 8; ++j) {
            w.SetValue(j, 0);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        DataCopy(yGm[0], w, 8); // 32B atomic add
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
        return;
    }
    // 16-bit dtypes keep single core (tiling sets blockDim=1), never reached
    // ==== 16-bit dtypes: multi-core flag-slot reduce inside y's own page ====
    // Slot k at byte (32 + 32*k) from y: [float partial | pad | MAGIC1 | MAGIC2 | 0 ...]
    // core0 polls each slot's magic (MTE2 reads are L2-coherent with peer MTE3 writes),
    // then sums. No SyncAll / no dcci / no GM workspace / no init-value dependency.
    {
        constexpr uint32_t MG1 = 0x5A5A5A5AU;
        constexpr uint32_t MG2 = 0xA5A5A5A5U;
        LocalTensor<int32_t> w = f1Buf.Get<int32_t>();
        LocalTensor<float> wf = f1Buf.Get<float>();
        wf.SetValue(0, accF);
        w.SetValue(1, 0);
        w.SetValue(2, static_cast<int32_t>(MG1));
        w.SetValue(3, static_cast<int32_t>(MG2));
        for (int64_t j = 4; j < 8; ++j) {
            w.SetValue(j, 0);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(yI32Gm[8 + blockId * 8], w, 8); // 32B slot
        PipeBarrier<PIPE_ALL>();
        if (blockId != 0) {
            return;
        }
        LocalTensor<int32_t> rd = f2Buf.Get<int32_t>();
        LocalTensor<float> rdf = f2Buf.Get<float>();
        float totalF = 0.0f;
        float comp = 0.0f;
        for (int64_t i = 0; i < blockNum; ++i) {
            const int64_t off = 8 + i * 8;
            int64_t spin = 0;
            while (true) {
                DataCopy(rd, yI32Gm[off], 8);
                PipeBarrier<PIPE_ALL>();
                bool ok = (static_cast<uint32_t>(rd.GetValue(2)) == MG1) &&
                          (static_cast<uint32_t>(rd.GetValue(3)) == MG2);
                if (ok) {
                    break;
                }
                if (++spin > (1 << 22)) {
                    break;
                }
            }
            float v = rdf.GetValue(0);
            float yv = v - comp;
            float t = totalF + yv;
            comp = (t - totalF) - yv;
            totalF = t;
        }
        StoreF(totalF);
    }
}

} // namespace NsDot
#endif // DOT_H

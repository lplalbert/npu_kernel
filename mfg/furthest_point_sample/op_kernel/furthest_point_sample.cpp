// Kernel侧核函数实现
// FurthestPointSample: 每 batch 独立迭代 npoint 轮:
//   out[0] = 0; minD 初始 +inf;
//   每轮: 用当前选点更新 minD[i] = min(minD[i], d2(p_i, p_sel));
//         下一个选点 = minD 的最大值位置, 并列取最小索引, 不排除已选点。
//   f16 全程 half 计算与规约(不升 fp32, 否则改变并列比较)。
//
// 实现要点:
//   - 每 batch 一个核(B 通常 1~4), 点集常驻 UB: 用 Gather 一次去交错成 x/y/z;
//     f16 时先把整块 Cast 成 f32 再 Gather(f16 Gather 实机不稳), 再 Cast 回 half;
//   - 向量 op 的 count 一律用 nPad(= alignUp(n,64)), 规避 CANN 9.0.0 小 count 崩;
//   - argmax 用 64-lane 压缩: 逐 chunk Max(acc,acc,chunk) 只回读 64 个 lane,
//     再按 stride 64 找回第一个等于最大值的下标 → 并列自然取最小索引;
//   - 未命中/边界处理: nPad 之后的 lane 用 -inf 保证不会被选中;
//   - S/V 依赖: 标量读 UB(Gather 结果 / minD) 前要 V_S, 标量读完之后 V 要写同一
//     buffer 前要 S_V, 否则 WAR/RAW 竞争(uber 无自动同步)。
#include "kernel_operator.h"

#include "furthest_point_sample_tiling.h"
#include "tiling_key_furthest_point_sample.h"

using namespace AscendC;

namespace {
constexpr int32_t LA = 64;  // lane 压缩宽度, 同时是向量 op 的最小 count
}  // namespace

template <typename T>
class KernelFps {
public:
    __aicore__ inline KernelFps() {}

    __aicore__ inline void Init(GM_ADDR pts, GM_ADDR out, const FpsTilingData &t) {
        n_ = static_cast<int64_t>(t.n);
        npoint_ = static_cast<int64_t>(t.npoint);
        nPad_ = static_cast<int64_t>(t.nPad);
        const int64_t bid = static_cast<int64_t>(GetBlockIdx());
        pGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(pts), static_cast<int64_t>(t.b) * n_ * 3);
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(out), static_cast<int64_t>(t.b) * npoint_);
        base_ = bid * n_ * 3;
        oBase_ = bid * npoint_;

        const uint32_t rawBytes = static_cast<uint32_t>((n_ * 3 + LA) * sizeof(T));
        pipe_.InitBuffer(rawBuf_, rawBytes);
        const uint32_t nBytes = static_cast<uint32_t>((nPad_ + LA) * sizeof(T));
        pipe_.InitBuffer(xBuf_, nBytes);
        pipe_.InitBuffer(yBuf_, nBytes);
        pipe_.InitBuffer(zBuf_, nBytes);
        pipe_.InitBuffer(mBuf_, nBytes);  // minD
        pipe_.InitBuffer(tBuf_, nBytes);  // d2 暂存
        pipe_.InitBuffer(t2Buf_, nBytes);
        pipe_.InitBuffer(aBuf_, static_cast<uint32_t>(LA * sizeof(T)));  // lane 压缩
        pipe_.InitBuffer(oBuf_, static_cast<uint32_t>((npoint_ + LA) * sizeof(int32_t)));
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>(3 * (nPad_ + LA) * sizeof(uint32_t)));
        if constexpr (sizeof(T) == 2) {
            // f16: 整块点的 f32 视图 + 单维 Gather 的 f32 暂存
            pipe_.InitBuffer(f32RawBuf_, static_cast<uint32_t>((n_ * 3 + LA) * sizeof(float)));
            pipe_.InitBuffer(f32DimBuf_, static_cast<uint32_t>((nPad_ + LA) * sizeof(float)));
        }

        // 去交错偏移 (对 f32 源, 单位字节): o_d[j] = j*12 + d*4
        const int32_t sz = static_cast<int32_t>(nPad_ + LA);
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        for (int32_t d = 0; d < 3; ++d) {
            LocalTensor<int32_t> o = oi[d * sz];
            CreateVecIndex(o, static_cast<int32_t>(0), static_cast<uint32_t>(nPad_));
            Muls(o, o, static_cast<int32_t>(sizeof(float) * 3), nPad_);
            Adds(o, o, static_cast<int32_t>(d * sizeof(float)), nPad_);
        }
    }

    __aicore__ inline void Process() {
        if (n_ <= 0 || npoint_ <= 0) {
            return;
        }
        LocalTensor<T> xv = xBuf_.Get<T>();
        LocalTensor<T> yv = yBuf_.Get<T>();
        LocalTensor<T> zv = zBuf_.Get<T>();
        LocalTensor<T> minD = mBuf_.Get<T>();
        LocalTensor<T> tm = tBuf_.Get<T>();
        const int32_t sz = static_cast<int32_t>(nPad_ + LA);
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();

        // 1) 点集搬入 + 去交错 (用普通 TBuf 而非 TQue, 与已验证的 BallQuery/ComplexAbs 一致)
        LocalTensor<T> rd = rawBuf_.Get<T>();
        {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(n_ * 3 * sizeof(T));
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPadExtParams<T> pp;
            DataCopyPad(rd, pGm_[static_cast<int32_t>(base_)], p, pp);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        if constexpr (sizeof(T) == 2) {
            LocalTensor<float> rf = f32RawBuf_.Get<float>();
            LocalTensor<float> fd = f32DimBuf_.Get<float>();
            const uint32_t cnt3 = static_cast<uint32_t>(((n_ * 3 + LA - 1) / LA) * LA);
            Cast(rf, rd, RoundMode::CAST_NONE, cnt3);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Gather(fd, rf, oi.template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(nPad_));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Cast(xv, fd, RoundMode::CAST_NONE, static_cast<uint32_t>(nPad_));
            Gather(fd, rf, oi[sz].template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(nPad_));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Cast(yv, fd, RoundMode::CAST_NONE, static_cast<uint32_t>(nPad_));
            Gather(fd, rf, oi[2 * sz].template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(nPad_));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Cast(zv, fd, RoundMode::CAST_NONE, static_cast<uint32_t>(nPad_));
        } else {
            Gather(xv, rd, oi.template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(nPad_));
            Gather(yv, rd, oi[sz].template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(nPad_));
            Gather(zv, rd, oi[2 * sz].template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(nPad_));
        }
        // 尾部 padding 归零, 避免 Gather 到脏数据产生 NaN 污染规约
        Duplicate(xv[static_cast<int32_t>(n_)], static_cast<T>(0), LA);
        Duplicate(yv[static_cast<int32_t>(n_)], static_cast<T>(0), LA);
        Duplicate(zv[static_cast<int32_t>(n_)], static_cast<T>(0), LA);

        // 2) minD = +inf; 尾部 -inf
        Duplicate(minD, static_cast<T>(1e38f), static_cast<int32_t>(n_));
        if (nPad_ > n_) {
            Duplicate(minD[static_cast<int32_t>(n_)], static_cast<T>(-1e38f), LA);
        }
        // 标量读 x/y/z 之前必须等 V 完成
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        LocalTensor<int32_t> ob = oBuf_.Get<int32_t>();
        int32_t sel = 0;
        for (int64_t t = 0; t < npoint_; ++t) {
            if (t == 0) {
                sel = 0;
            } else {
                sel = ArgMax(minD, tm);
            }
            ob.SetValue(static_cast<int32_t>(t), sel);
            if (t + 1 < npoint_) {
                // ArgMax 的标量读 minD 完成前, V 不能覆盖 minD (WAR)
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                UpdateMin(minD, xv, yv, zv, tm, sel);
            }
        }

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams po;
        po.blockCount = 1;
        po.blockLen = static_cast<uint32_t>(npoint_ * sizeof(int32_t));
        po.srcStride = 0;
        po.dstStride = 0;
        po.rsv = 0;
        DataCopyPad(oGm_[static_cast<int32_t>(oBase_)], ob, po);
    }

private:
    __aicore__ inline float toF(T v) const {
        return static_cast<float>(v);
    }

    // minD[i] = min(minD[i], d2(p_i, p_sel))
    __aicore__ inline void UpdateMin(const LocalTensor<T> &minD, const LocalTensor<T> &xv, const LocalTensor<T> &yv,
                                     const LocalTensor<T> &zv, const LocalTensor<T> &tm, int32_t sel) {
        LocalTensor<T> t2 = t2Buf_.Get<T>();
        const int32_t np = static_cast<int32_t>(nPad_);
        const __ubuf__ T *xp = (const __ubuf__ T *)xv.GetPhyAddr();
        const float sx = toF(xp[sel]);
        const float sy = toF(((const __ubuf__ T *)yv.GetPhyAddr())[sel]);
        const float sz = toF(((const __ubuf__ T *)zv.GetPhyAddr())[sel]);
        Adds(tm, xv, static_cast<T>(0.0f - sx), np);
        Mul(tm, tm, tm, np);
        Adds(t2, yv, static_cast<T>(0.0f - sy), np);
        Mul(t2, t2, t2, np);
        Add(tm, tm, t2, np);
        Adds(t2, zv, static_cast<T>(0.0f - sz), np);
        Mul(t2, t2, t2, np);
        Add(tm, tm, t2, np);
        Min(minD, minD, tm, np);
    }

    // 64-lane 压缩求 argmax, 并列取最小索引
    __aicore__ inline int32_t ArgMax(const LocalTensor<T> &minD, const LocalTensor<T> &tm) {
        (void)tm;
        LocalTensor<T> acc = aBuf_.Get<T>();
        const int32_t np = static_cast<int32_t>(nPad_);
        const int32_t nChunk = np / LA;
        Duplicate(acc, static_cast<T>(-1e38f), LA);
        for (int32_t c = 0; c < nChunk; ++c) {
            Max(acc, acc, minD[c * LA], LA);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const __ubuf__ T *ap = (const __ubuf__ T *)acc.GetPhyAddr();
        const __ubuf__ T *mp = (const __ubuf__ T *)minD.GetPhyAddr();
        float best = -1e38f;
        for (int32_t l = 0; l < LA; ++l) {
            const float v = toF(ap[l]);
            if (v > best) {
                best = v;
            }
        }
        int32_t sel = -1;
        for (int32_t l = 0; l < LA; ++l) {
            if (toF(ap[l]) != best) {
                continue;
            }
            for (int32_t c = 0; c < nChunk; ++c) {
                const int32_t idx = c * LA + l;
                if (toF(mp[idx]) == best) {
                    if (sel < 0 || idx < sel) {
                        sel = idx;
                    }
                    break;
                }
            }
        }
        if (sel < 0) {
            sel = 0;
        }
        return sel;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> mBuf_;
    TBuf<TPosition::VECCALC> tBuf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> aBuf_;
    TBuf<TPosition::VECCALC> oBuf_;
    TBuf<TPosition::VECCALC> offBuf_;
    TBuf<TPosition::VECCALC> f32RawBuf_;
    TBuf<TPosition::VECCALC> f32DimBuf_;
    GlobalTensor<T> pGm_;
    GlobalTensor<int32_t> oGm_;
    int64_t n_ = 0;
    int64_t npoint_ = 0;
    int64_t nPad_ = 0;
    int64_t base_ = 0;
    int64_t oBase_ = 0;
};

template <typename DT_PTS>
__global__ __aicore__ void furthest_point_sample(GM_ADDR points, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FpsTilingData);
    GET_TILING_DATA_WITH_STRUCT(FpsTilingData, tiling_data, tiling);
    KernelFps<DT_PTS> op;
    op.Init(points, idx, tiling_data);
    op.Process();
}

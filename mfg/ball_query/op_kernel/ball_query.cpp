// Kernel侧核函数实现
// BallQuery: 对每个查询中心 new_xyz[b,m], 在 xyz[b] 中按参考点索引从小到大扫描,
//   d^2 = (x_k-x_q)^2+(y_k-y_q)^2+(z_k-z_q)^2, 当且仅当 d^2 < radius^2 命中;
//   取扫描顺序中最先出现的 nsample 个命中索引, 不按距离排序;
//   命中数 >= nsample 取前 nsample; 1..nsample-1 时尾部重复第一个命中索引; 0 命中整行输出 0。
//
// 实现:
//   1) 多核按查询(每个 batch 内)切分; 每个核把自己负责的查询中心复制到 UB(升级为 float32);
//   2) 参考点按 chunk 分块, 每块用 vector Gather 把 [N,3] 去交错成 x/y/z 三个数组;
//   3) 每个查询对整块用向量指令算 d^2, 然后标量顺序扫描 d^2 < radius^2 收集索引(达到 nsample 提前退出);
//   4) 结果 int32 先写 UB, 再一次性 DataCopyPad 回 GM。
#include "kernel_operator.h"

#include "ball_query_tiling.h"
#include "tiling_key_ball_query.h"

using namespace AscendC;

namespace {
constexpr int32_t UB_PAD = 8;
}  // namespace

template <typename T>
class KernelBallQuery {
public:
    __aicore__ inline KernelBallQuery() {}

    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR newXyz, GM_ADDR idx, const BallQueryTilingData &t) {
        b_ = static_cast<int64_t>(t.batch);
        n_ = static_cast<int64_t>(t.n);
        m_ = static_cast<int64_t>(t.m);
        ns_ = static_cast<int64_t>(t.nsample);
        r2_ = t.radius2;
        chunk_ = static_cast<int64_t>(t.chunk);
        qpc_ = static_cast<int32_t>(t.qPerCore);
        block_ = static_cast<int64_t>(GetBlockIdx());
        mode_ = t.dtypeMode;

        xyzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xyz), b_ * n_ * 3);
        newGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(newXyz), b_ * m_ * 3);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idx), b_ * m_ * ns_);

        const int32_t ch = static_cast<int32_t>(chunk_);
        const int32_t qAll = static_cast<int32_t>(b_ * m_ * 3);
        pipe_.InitBuffer(rawBuf_, static_cast<uint32_t>((3 * ch + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(fBuf_, static_cast<uint32_t>((3 * ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(xBuf_, static_cast<uint32_t>((ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(yBuf_, static_cast<uint32_t>((ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(zBuf_, static_cast<uint32_t>((ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(d2Buf_, static_cast<uint32_t>((ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>((ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(qRawBuf_, static_cast<uint32_t>((qAll + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(qBuf_, static_cast<uint32_t>((qAll + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(oBuf_, static_cast<uint32_t>(BQ_MAX_Q_PER_CORE * 64 * sizeof(int32_t)));
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>(3 * (ch + UB_PAD) * sizeof(uint32_t)));

        // 查询中心整块搬到 UB 并升到 float32
        LocalTensor<T> qRaw = qRawBuf_.Get<T>();
        CopyInGM(qRaw, newGm_, 0, qAll);
        if constexpr (sizeof(T) == 2) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            qf_ = qBuf_.Get<float>();
            Cast(qf_, qRaw, RoundMode::CAST_NONE, qAll);
        } else {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            qf_ = qRaw.template ReinterpretCast<float>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        // 参考点去交错偏移 (vgather offset 单位为字节): 12k, 12k+4, 12k+8
        const int32_t stride = ch + UB_PAD;
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<int32_t> ox = oi;
        LocalTensor<int32_t> oy = oi[stride];
        LocalTensor<int32_t> oz = oi[2 * stride];
        CreateVecIndex(ox, static_cast<int32_t>(0), static_cast<uint32_t>(ch));
        Muls(ox, ox, static_cast<int32_t>(3 * sizeof(float)), ch);
        Adds(oy, ox, static_cast<int32_t>(sizeof(float)), ch);
        Adds(oz, ox, static_cast<int32_t>(2 * sizeof(float)), ch);
    }

    __aicore__ inline void Process() {
        if (n_ <= 0 || m_ <= 0 || b_ <= 0 || ns_ <= 0) {
            return;
        }
        LocalTensor<float> qf = qf_;
        LocalTensor<float> xl = xBuf_.Get<float>();
        LocalTensor<float> yl = yBuf_.Get<float>();
        LocalTensor<float> zl = zBuf_.Get<float>();
        LocalTensor<float> d2 = d2Buf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<int32_t> ob = oBuf_.Get<int32_t>();

        for (int64_t b = 0; b < b_; ++b) {
            const int64_t q0 = block_ * qpc_;
            if (q0 >= m_) {
                continue;
            }
            const int64_t q1 = (q0 + qpc_ < m_) ? (q0 + qpc_) : m_;
            const int32_t nq = static_cast<int32_t>(q1 - q0);

            int32_t cnt[BQ_MAX_Q_PER_CORE];
            int32_t first[BQ_MAX_Q_PER_CORE];
            for (int32_t i = 0; i < nq; ++i) {
                cnt[i] = 0;
                first[i] = -1;
            }

            for (int64_t cs = 0; cs < n_; cs += chunk_) {
                const int64_t cn64 = (cs + chunk_ < n_) ? chunk_ : (n_ - cs);
                const int32_t cn = static_cast<int32_t>(cn64);
                LoadChunk(b, cs, cn, xl, yl, zl);

                for (int32_t i = 0; i < nq; ++i) {
                    if (cnt[i] >= static_cast<int32_t>(ns_)) {
                        continue;
                    }
                    const int64_t qIdx = 3 * (b * m_ + q0 + i);
                    const float qx = qf.GetValue(static_cast<int32_t>(qIdx));
                    const float qy = qf.GetValue(static_cast<int32_t>(qIdx + 1));
                    const float qz = qf.GetValue(static_cast<int32_t>(qIdx + 2));

                    Adds(d2, xl, -qx, cn);
                    Mul(d2, d2, d2, cn);
                    Adds(tmp, yl, -qy, cn);
                    Mul(tmp, tmp, tmp, cn);
                    Add(d2, d2, tmp, cn);
                    Adds(tmp, zl, -qz, cn);
                    Mul(tmp, tmp, tmp, cn);
                    Add(d2, d2, tmp, cn);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);

                    const __ubuf__ float *pp = (const __ubuf__ float *)d2.GetPhyAddr();
                    int32_t c = cnt[i];
                    int32_t f = first[i];
                    const int32_t cs32 = static_cast<int32_t>(cs);
                    const int32_t ns32 = static_cast<int32_t>(ns_);
                    int32_t k = 0;
                    while (k < cn && c < ns32) {
                        if (pp[k] < r2_) {
                            if (c == 0) {
                                f = cs32 + k;
                            }
                            ob.SetValue(static_cast<int32_t>(i) * ns32 + c, cs32 + k);
                            ++c;
                        }
                        ++k;
                    }
                    cnt[i] = c;
                    first[i] = f;
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
            }

            // 填充不足部分 (尾部重复第一个命中; 0 命中全 0)
            for (int32_t i = 0; i < nq; ++i) {
                const int32_t c = cnt[i];
                if (c == 0) {
                    for (int64_t k = 0; k < ns_; ++k) {
                        ob.SetValue(static_cast<int32_t>(i) * static_cast<int32_t>(ns_) + static_cast<int32_t>(k), 0);
                    }
                } else {
                    for (int64_t k = c; k < ns_; ++k) {
                        ob.SetValue(static_cast<int32_t>(i) * static_cast<int32_t>(ns_) + static_cast<int32_t>(k),
                                    first[i]);
                    }
                }
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

            const int64_t outOff = (b * m_ + q0) * ns_;
            const int64_t outCnt = static_cast<int64_t>(nq) * ns_;
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(outCnt * sizeof(int32_t));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPad(idxGm_[static_cast<int32_t>(outOff)], ob, params);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

private:
    // 把 GM 上一段 T 数据搬到 UB (按 32B 对齐情况选择 DataCopy / DataCopyPad)
    __aicore__ inline void CopyInGM(const LocalTensor<T> &dst, const GlobalTensor<T> &src, int64_t off, int64_t cnt) {
        const int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
        if ((bytes % 32) == 0 && ((off * static_cast<int64_t>(sizeof(T))) % 32) == 0) {
            DataCopy(dst, src[static_cast<int32_t>(off)], static_cast<uint32_t>(cnt));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(bytes);
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<T> padParams;
            DataCopyPad(dst, src[static_cast<int32_t>(off)], params, padParams);
        }
    }

    __aicore__ inline void LoadChunk(int64_t b, int64_t cs, int32_t cn, const LocalTensor<float> &xl,
                                     const LocalTensor<float> &yl, const LocalTensor<float> &zl) {
        LocalTensor<T> raw = rawBuf_.Get<T>();
        const int64_t off = (b * n_ + cs) * 3;
        CopyInGM(raw, xyzGm_, off, static_cast<int64_t>(cn) * 3);

        const int32_t stride = static_cast<int32_t>(chunk_) + UB_PAD;
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<uint32_t> ox = oi.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oy = oi[stride].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oz = oi[2 * stride].template ReinterpretCast<uint32_t>();

        LocalTensor<float> fsrc;
        if constexpr (sizeof(T) == 2) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            fsrc = fBuf_.Get<float>();
            Cast(fsrc, raw, RoundMode::CAST_NONE, cn * 3);
        } else {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            fsrc = raw.template ReinterpretCast<float>();
        }
        Gather(xl, fsrc, ox, 0, static_cast<uint32_t>(cn));
        Gather(yl, fsrc, oy, 0, static_cast<uint32_t>(cn));
        Gather(zl, fsrc, oz, 0, static_cast<uint32_t>(cn));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> fBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> d2Buf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> qRawBuf_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> oBuf_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<T> xyzGm_;
    GlobalTensor<T> newGm_;
    GlobalTensor<int32_t> idxGm_;
    LocalTensor<float> qf_;
    int64_t b_ = 0;
    int64_t n_ = 0;
    int64_t m_ = 0;
    int64_t ns_ = 0;
    int64_t chunk_ = 0;
    int32_t qpc_ = 0;
    int64_t block_ = 0;
    float r2_ = 0.0f;
    uint32_t mode_ = 1;
};

template <typename DT_XYZ>
__global__ __aicore__ void ball_query(GM_ADDR xyz, GM_ADDR new_xyz, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BallQueryTilingData);
    GET_TILING_DATA_WITH_STRUCT(BallQueryTilingData, tiling_data, tiling);
    KernelBallQuery<DT_XYZ> op;
    op.Init(xyz, new_xyz, idx, tiling_data);
    op.Process();
}

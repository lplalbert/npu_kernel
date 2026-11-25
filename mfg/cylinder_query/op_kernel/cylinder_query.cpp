// Kernel侧核函数实现
// CylinderQuery: 对每个查询中心 new_xyz[b,j] 和 rot[b,j](3x3 row-major):
//   delta = xyz[b,k] - new_xyz[b,j]
//   [xl,yl,zl]^T = rot[b,j]^T @ delta  (即 xl = r00*dx + r10*dy + r20*dz)
//   命中当且仅当 yl^2+zl^2 < radius^2 且 hmin < xl < hmax (严格比较)
//   按 xyz 输入索引从小到大扫描, 取最先出现的 nsample 个命中; 不足尾部重复第一个命中; 0 命中全 0。
//   f16 输入先无损转 f32, 变换/比较统一按 f32 语义。
//
// 实现:
//   1) 多核按查询切分; 每个核把整批的 new_xyz / rot 升到 f32 放 UB;
//   2) 参考点按 chunk 分块, Gather 去交错成 x/y/z (f32);
//   3) 每个查询对整块用向量指令算 lx/ly/lz, 再把三个判据合成一个 margin
//        margin = min(radius^2-(yl^2+zl^2), xl-hmin, hmax-xl)
//      margin > 0 即命中 → 标量只做一次比较, 顺序扫描收集索引 (达 nsample 提前退出);
//   4) int32 结果先写 UB, 再一次性 DataCopyPad 回 GM。
//   向量 op 的 count 用 alignUp(cn,64) 规避 CANN 9.0.0 小 count 崩, 标量扫描只用真实 cn。
#include "kernel_operator.h"

#include "cylinder_query_tiling.h"
#include "tiling_key_cylinder_query.h"

using namespace AscendC;

namespace {
constexpr int32_t UB_PAD = 8;
constexpr int32_t LA = 64;  // 向量 op 最小 count
}  // namespace

template <typename T>
class KernelCylinderQuery {
public:
    __aicore__ inline KernelCylinderQuery() {}

    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR newXyz, GM_ADDR rot, GM_ADDR idx,
                                const CylinderQueryTilingData &t) {
        b_ = static_cast<int64_t>(t.batch);
        n_ = static_cast<int64_t>(t.n);
        m_ = static_cast<int64_t>(t.m);
        ns_ = static_cast<int64_t>(t.nsample);
        r2_ = t.radius2;
        hmin_ = t.hmin;
        hmax_ = t.hmax;
        chunk_ = static_cast<int64_t>(t.chunk);
        qpc_ = static_cast<int32_t>(t.qPerCore);
        block_ = static_cast<int64_t>(GetBlockIdx());
        mode_ = t.dtypeMode;

        xyzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xyz), b_ * n_ * 3);
        newGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(newXyz), b_ * m_ * 3);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rot), b_ * m_ * 9);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idx), b_ * m_ * ns_);

        const int32_t ch = static_cast<int32_t>(chunk_);
        const int32_t q3 = static_cast<int32_t>(b_ * m_ * 3);
        const int32_t q9 = static_cast<int32_t>(b_ * m_ * 9);
        q3Pad_ = ((q3 + 7) / 8) * 8;  // DataCopyPad 的 UB 侧地址需 32B 对齐
        const int32_t qTot = q3Pad_ + q9;
        const uint32_t chPad = static_cast<uint32_t>((ch + UB_PAD) * sizeof(float));
        pipe_.InitBuffer(rawBuf_, static_cast<uint32_t>((3 * ch + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(fBuf_, static_cast<uint32_t>((3 * ch + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(xBuf_, chPad);
        pipe_.InitBuffer(yBuf_, chPad);
        pipe_.InitBuffer(zBuf_, chPad);
        pipe_.InitBuffer(lxBuf_, chPad);
        pipe_.InitBuffer(lyBuf_, chPad);
        pipe_.InitBuffer(lzBuf_, chPad);
        pipe_.InitBuffer(t1Buf_, chPad);
        pipe_.InitBuffer(t2Buf_, chPad);
        pipe_.InitBuffer(qRawBuf_, static_cast<uint32_t>((qTot + LA + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(qBuf_, static_cast<uint32_t>((qTot + LA + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(oBuf_, static_cast<uint32_t>(CQ_MAX_Q_PER_CORE * CQ_MAX_NSAMPLE * sizeof(int32_t)));
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>(3 * (ch + UB_PAD) * sizeof(uint32_t)));

        // 查询参数 (new_xyz 在前 q3 个, rot 从 q3Pad 开始) 整块搬到 UB 并升 float32
        LocalTensor<T> qRaw = qRawBuf_.Get<T>();
        CopyInGM(qRaw, newGm_, 0, q3);
        CopyInGM(qRaw[q3Pad_], rotGm_, 0, q9);
        qf_ = qBuf_.Get<float>();
        if constexpr (sizeof(T) == 2) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(qf_, qRaw, RoundMode::CAST_NONE, static_cast<uint32_t>(((qTot + LA - 1) / LA) * LA));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            qf_ = qRaw.template ReinterpretCast<float>();
        }

        // 参考点去交错偏移 (Gather offset 单位为字节): 12k, 12k+4, 12k+8
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
        LocalTensor<float> xl = xBuf_.Get<float>();
        LocalTensor<float> yl = yBuf_.Get<float>();
        LocalTensor<float> zl = zBuf_.Get<float>();
        LocalTensor<float> lx = lxBuf_.Get<float>();
        LocalTensor<float> ly = lyBuf_.Get<float>();
        LocalTensor<float> lz = lzBuf_.Get<float>();
        LocalTensor<float> t1 = t1Buf_.Get<float>();
        LocalTensor<float> t2 = t2Buf_.Get<float>();
        LocalTensor<int32_t> ob = oBuf_.Get<int32_t>();

        for (int64_t b = 0; b < b_; ++b) {
            const int64_t q0 = block_ * qpc_;
            if (q0 >= m_) {
                continue;
            }
            const int64_t q1 = (q0 + qpc_ < m_) ? (q0 + qpc_) : m_;
            const int32_t nq = static_cast<int32_t>(q1 - q0);

            int32_t cnt[CQ_MAX_Q_PER_CORE];
            int32_t first[CQ_MAX_Q_PER_CORE];
            for (int32_t i = 0; i < nq; ++i) {
                cnt[i] = 0;
                first[i] = -1;
            }

            for (int64_t cs = 0; cs < n_; cs += chunk_) {
                const int64_t cn64 = (cs + chunk_ < n_) ? chunk_ : (n_ - cs);
                const int32_t cn = static_cast<int32_t>(cn64);
                if (cn <= 0) {
                    break;
                }
                const int32_t cnOp = ((cn + LA - 1) / LA) * LA;
                if (cnOp > static_cast<int32_t>(chunk_)) {
                    break;  // chunk 应已按 64 对齐, 保险
                }
                LoadChunk(b, cs, cn, cnOp, xl, yl, zl);

                for (int32_t i = 0; i < nq; ++i) {
                    if (cnt[i] >= static_cast<int32_t>(ns_)) {
                        continue;
                    }
                    const int64_t qq = b * m_ + q0 + i;
                    const int32_t cOff = static_cast<int32_t>(qq) * 3;
                    const int32_t rOff = q3Pad_ + static_cast<int32_t>(qq) * 9;
                    const float cx = qf_.GetValue(cOff + 0);
                    const float cy = qf_.GetValue(cOff + 1);
                    const float cz = qf_.GetValue(cOff + 2);
                    const float r00 = qf_.GetValue(rOff + 0);
                    const float r01 = qf_.GetValue(rOff + 1);
                    const float r02 = qf_.GetValue(rOff + 2);
                    const float r10 = qf_.GetValue(rOff + 3);
                    const float r11 = qf_.GetValue(rOff + 4);
                    const float r12 = qf_.GetValue(rOff + 5);
                    const float r20 = qf_.GetValue(rOff + 6);
                    const float r21 = qf_.GetValue(rOff + 7);
                    const float r22 = qf_.GetValue(rOff + 8);

                    // lx = r00*x + r10*y + r20*z - (r00*cx + r10*cy + r20*cz)
                    const float cxl = r00 * cx + r10 * cy + r20 * cz;
                    Muls(lx, xl, r00, cnOp);
                    Muls(t1, yl, r10, cnOp);
                    Add(lx, lx, t1, cnOp);
                    Muls(t1, zl, r20, cnOp);
                    Add(lx, lx, t1, cnOp);
                    Adds(lx, lx, 0.0f - cxl, cnOp);
                    // ly
                    const float cyl = r01 * cx + r11 * cy + r21 * cz;
                    Muls(ly, xl, r01, cnOp);
                    Muls(t1, yl, r11, cnOp);
                    Add(ly, ly, t1, cnOp);
                    Muls(t1, zl, r21, cnOp);
                    Add(ly, ly, t1, cnOp);
                    Adds(ly, ly, 0.0f - cyl, cnOp);
                    // lz
                    const float czl = r02 * cx + r12 * cy + r22 * cz;
                    Muls(lz, xl, r02, cnOp);
                    Muls(t1, yl, r12, cnOp);
                    Add(lz, lz, t1, cnOp);
                    Muls(t1, zl, r22, cnOp);
                    Add(lz, lz, t1, cnOp);
                    Adds(lz, lz, 0.0f - czl, cnOp);

                    // t1 = radius^2 - (ly^2 + lz^2)
                    Mul(t1, ly, ly, cnOp);
                    Mul(t2, lz, lz, cnOp);
                    Add(t1, t1, t2, cnOp);
                    Muls(t1, t1, -1.0f, cnOp);
                    Adds(t1, t1, r2_, cnOp);
                    // min(t1, lx - hmin, hmax - lx)
                    Adds(t2, lx, 0.0f - hmin_, cnOp);
                    Min(t1, t1, t2, cnOp);
                    Muls(t2, lx, -1.0f, cnOp);
                    Adds(t2, t2, hmax_, cnOp);
                    Min(t1, t1, t2, cnOp);

                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const __ubuf__ float *pp = (const __ubuf__ float *)t1.GetPhyAddr();
                    int32_t c = cnt[i];
                    int32_t f = first[i];
                    const int32_t cs32 = static_cast<int32_t>(cs);
                    const int32_t ns32 = static_cast<int32_t>(ns_);
                    int32_t k = 0;
                    while (k < cn && c < ns32) {
                        if (pp[k] > 0.0f) {
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
                const int32_t rowBase = static_cast<int32_t>(i) * static_cast<int32_t>(ns_);
                if (c == 0) {
                    for (int64_t k = 0; k < ns_; ++k) {
                        ob.SetValue(rowBase + static_cast<int32_t>(k), 0);
                    }
                } else {
                    for (int64_t k = c; k < ns_; ++k) {
                        ob.SetValue(rowBase + static_cast<int32_t>(k), first[i]);
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
        if (cnt <= 0) {
            return;
        }
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

    __aicore__ inline void LoadChunk(int64_t b, int64_t cs, int32_t cn, int32_t cnOp,
                                     const LocalTensor<float> &xl, const LocalTensor<float> &yl,
                                     const LocalTensor<float> &zl) {
        LocalTensor<T> raw = rawBuf_.Get<T>();
        const int64_t off = (b * n_ + cs) * 3;
        const int64_t cnt = static_cast<int64_t>(cn) * 3;
        CopyInGM(raw, xyzGm_, off, cnt);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const int32_t stride = static_cast<int32_t>(chunk_) + UB_PAD;
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<uint32_t> ox = oi.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oy = oi[stride].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> oz = oi[2 * stride].template ReinterpretCast<uint32_t>();

        LocalTensor<float> fsrc;
        if constexpr (sizeof(T) == 2) {
            fsrc = fBuf_.Get<float>();
            const uint32_t cnt3 = static_cast<uint32_t>(((cnt + LA - 1) / LA) * LA);
            Cast(fsrc, raw, RoundMode::CAST_NONE, cnt3);
        } else {
            fsrc = raw.template ReinterpretCast<float>();
        }
        Gather(xl, fsrc, ox, 0, static_cast<uint32_t>(cnOp));
        Gather(yl, fsrc, oy, 0, static_cast<uint32_t>(cnOp));
        Gather(zl, fsrc, oz, 0, static_cast<uint32_t>(cnOp));
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> fBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> lxBuf_;
    TBuf<TPosition::VECCALC> lyBuf_;
    TBuf<TPosition::VECCALC> lzBuf_;
    TBuf<TPosition::VECCALC> t1Buf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> qRawBuf_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> oBuf_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<T> xyzGm_;
    GlobalTensor<T> newGm_;
    GlobalTensor<T> rotGm_;
    GlobalTensor<int32_t> idxGm_;
    LocalTensor<float> qf_;
    int64_t b_ = 0;
    int64_t n_ = 0;
    int64_t m_ = 0;
    int64_t ns_ = 0;
    int64_t chunk_ = 0;
    int32_t qpc_ = 0;
    int64_t block_ = 0;
    int32_t q3Pad_ = 0;
    float r2_ = 0.0f;
    float hmin_ = 0.0f;
    float hmax_ = 0.0f;
    uint32_t mode_ = 1;
};

template <typename DT_XYZ>
__global__ __aicore__ void cylinder_query(GM_ADDR xyz, GM_ADDR new_xyz, GM_ADDR rot, GM_ADDR idx, GM_ADDR workspace,
                                          GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(CylinderQueryTilingData);
    GET_TILING_DATA_WITH_STRUCT(CylinderQueryTilingData, tiling_data, tiling);
    KernelCylinderQuery<DT_XYZ> op;
    op.Init(xyz, new_xyz, rot, idx, tiling_data);
    op.Process();
}

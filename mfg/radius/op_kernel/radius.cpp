// Kernel侧核函数实现
// Radius: 对每个查询点 i（升序）:
//   在其所属 batch 的 x 里按 j 升序扫描, d^2 = ||x[j]-y[i]||^2 (float32);
//   d^2 <= r^2 且未被 ignore_same_index 剔除(剔除条件: x 的全局下标 == i) 则收集 (i,j);
//   收满 max_num_neighbors(K) 后停止(first-K 截断, 不按距离排序)。
//   写入槽位 [i*K, (i+1)*K), 未命中槽位填 (-1,-1)。
//   out 为 (2, M*K) int64: row0 = 查询下标 i, row1 = 参考下标 j。
//
// 实现要点:
//   - 多核按查询均分; 每核按 batch 段推进(ptr_x/ptr_y 用标量 GM 读);
//   - 参考点按 chunk 分块搬入 UB(f16 先 Cast 成 f32), 逐点标量扫描 + 命中计数早退;
//   - 输出: 先在 UB 里把 [row0(K 个) | row1(K 个)] 整体填 -1(用 int32 视图 Duplicate,
//     count 按 64 下限, 规避 CANN 9.0.0 小 count 崩), 再标量写命中槽位, 两次 DataCopyPad 写回。
#include "kernel_operator.h"

#include "radius_tiling.h"
#include "tiling_key_radius.h"

using namespace AscendC;

namespace {
constexpr int32_t UB_PAD = 64;
}  // namespace

template <typename T>
class KernelRadius {
public:
    __aicore__ inline KernelRadius() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY, GM_ADDR out,
                                const RadiusTilingData &t) {
        n_ = static_cast<int64_t>(t.n);
        m_ = static_cast<int64_t>(t.m);
        d_ = static_cast<int32_t>(t.dim);
        k_ = static_cast<int64_t>(t.k);
        chunk_ = static_cast<int64_t>(t.chunk);
        r2_ = t.r2;
        ignoreSame_ = (t.ignoreSame != 0u);
        const int64_t per = static_cast<int64_t>(t.qPerCore);
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        int64_t s = blk * per;
        int64_t en = s + per;
        if (s > m_) {
            s = m_;
        }
        if (en > m_) {
            en = m_;
        }
        q0_ = s;
        q1_ = en;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), n_ * d_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), m_ * d_);
        pxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptrX), 0x7ffffff);
        pyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptrY), 0x7ffffff);
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(out), 2 * m_ * k_);

        const uint32_t rawBytes = static_cast<uint32_t>((chunk_ * d_ + UB_PAD) * sizeof(T));
        pipe_.InitBuffer(rawQue_, 2, rawBytes);
        pipe_.InitBuffer(fBuf_, static_cast<uint32_t>((chunk_ * d_ + UB_PAD) * sizeof(float)));
        // 输出暂存: row0(K) + row1(K) 个 int64, 按 64 个 int32 下限补齐
        int64_t obElems = 2 * k_ + 2 * UB_PAD;
        uint32_t obBytes = static_cast<uint32_t>(obElems * sizeof(int64_t));
        if (obBytes % 32u != 0u) {
            obBytes = (obBytes / 32u + 1u) * 32u;
        }
        pipe_.InitBuffer(oBuf_, obBytes);
    }

    __aicore__ inline void Process() {
        if (q0_ >= q1_ || m_ <= 0 || k_ <= 0) {
            return;
        }
        int64_t by = 0;
        int64_t yS = pyGm_.GetValue(0);
        int64_t yE = pyGm_.GetValue(1);
        while (yE <= q0_ && by < 0x7fffffff) {
            ++by;
            yS = yE;
            yE = pyGm_.GetValue(static_cast<int32_t>(by + 1));
        }
        int64_t xS = pxGm_.GetValue(static_cast<int32_t>(by));
        int64_t xE = pxGm_.GetValue(static_cast<int32_t>(by + 1));

        for (int64_t i = q0_; i < q1_; ++i) {
            while (i >= yE) {  // 换批次
                ++by;
                yS = yE;
                yE = pyGm_.GetValue(static_cast<int32_t>(by + 1));
                xS = pxGm_.GetValue(static_cast<int32_t>(by));
                xE = pxGm_.GetValue(static_cast<int32_t>(by + 1));
            }
            SolveQuery(i, xS, xE, yS);
        }
    }

private:
    __aicore__ inline void SolveQuery(int64_t qi, int64_t xS, int64_t xE, int64_t yS) {
        LocalTensor<int64_t> ob = oBuf_.Get<int64_t>();
        LocalTensor<int32_t> ob32 = ob.template ReinterpretCast<int32_t>();
        // 整块填 -1 (int64 的 -1 = 两个 0xFFFFFFFF 字); count 至少 64
        int32_t fillCnt = static_cast<int32_t>(4 * k_ + 2 * UB_PAD);
        if (fillCnt < 64) {
            fillCnt = 64;
        }
        Duplicate(ob32, static_cast<int32_t>(-1), fillCnt);

        // y[qi]
        const int64_t yo = qi * d_;
        float yv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int32_t d = 0; d < d_; ++d) {
            yv[d] = static_cast<float>(yGm_.GetValue(static_cast<int32_t>(yo + d)));
        }

        int32_t cnt = 0;
        const int32_t k32 = static_cast<int32_t>(k_);
        for (int64_t cs = xS; cs < xE && cnt < k32; cs += chunk_) {
            const int64_t rawCn = (cs + chunk_ < xE) ? chunk_ : (xE - cs);
            const int32_t cn = static_cast<int32_t>(rawCn);
            LocalTensor<T> rd = LoadChunk(cs, cn);
            const __ubuf__ float *xp = fPtr_;
            for (int32_t kk = 0; kk < cn && cnt < k32; ++kk) {
                const int64_t gj = cs + kk;
                if (ignoreSame_ && gj == qi) {
                    continue;
                }
                const int32_t base = kk * d_;
                float dx = xp[base] - yv[0];
                float d2 = dx * dx;
                for (int32_t d = 1; d < d_; ++d) {
                    float t = xp[base + d] - yv[d];
                    d2 += t * t;
                }
                if (d2 <= r2_) {
                    ob.SetValue(static_cast<int32_t>(k_) + cnt, gj);  // row1: 全局下标(实测正确)
                    ++cnt;
                }
            }
            rawQue_.FreeTensor(rd);  // 扫描完才能归还缓冲
        }

        // row0: 命中槽位填查询下标, 其余保持 -1 (只在 t < cnt 内写)
        for (int32_t t = 0; t < cnt; ++t) {
            ob.SetValue(t, qi);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

        const int64_t baseOff = qi * k_;
        WriteOut(ob, baseOff, 0);
        WriteOut(ob[static_cast<int32_t>(k_)], m_ * k_ + baseOff, 1);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WriteOut(const LocalTensor<int64_t> &src, int64_t gmOff, int32_t which) {
        (void)which;
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(k_ * 8);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(oGm_[static_cast<int32_t>(gmOff)], src, p);
    }

    __aicore__ inline LocalTensor<T> LoadChunk(int64_t cs, int32_t cn) {
        LocalTensor<T> raw = rawQue_.AllocTensor<T>();
        const int64_t elems = static_cast<int64_t>(cn) * d_;
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(elems * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<T> pp;
        DataCopyPad(raw, xGm_[static_cast<int32_t>(cs * d_)], p, pp);
        rawQue_.EnQue(raw);
        LocalTensor<T> rd = rawQue_.DeQue<T>();
        if constexpr (sizeof(T) == 2) {
            LocalTensor<float> fb = fBuf_.Get<float>();
            Cast(fb, rd, RoundMode::CAST_NONE, static_cast<int32_t>(elems));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            fPtr_ = (const __ubuf__ float *)fb.GetPhyAddr();
        } else {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            fPtr_ = (const __ubuf__ float *)rd.GetPhyAddr();
        }
        return rd;
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> rawQue_;
    TBuf<TPosition::VECCALC> fBuf_;
    TBuf<TPosition::VECCALC> oBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int64_t> pxGm_;
    GlobalTensor<int64_t> pyGm_;
    GlobalTensor<int64_t> oGm_;
    const __ubuf__ float *fPtr_ = nullptr;
    int64_t n_ = 0, m_ = 0, k_ = 0, chunk_ = 0, q0_ = 0, q1_ = 0;
    int32_t d_ = 3;
    float r2_ = 1.0f;
    bool ignoreSame_ = false;
};

template <typename DT_X>
__global__ __aicore__ void radius(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR out, GM_ADDR workspace,
                                  GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(RadiusTilingData);
    GET_TILING_DATA_WITH_STRUCT(RadiusTilingData, tiling_data, tiling);
    KernelRadius<DT_X> op;
    op.Init(x, y, ptr_x, ptr_y, out, tiling_data);
    op.Process();
}

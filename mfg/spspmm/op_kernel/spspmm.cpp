// Kernel侧核函数实现  spspmm: C = A(M,K) * B(K,N), COO 稀疏×稀疏, 输出 COO 行主序压实
//
// 输出: indexC(2,U) int64, valueC(U);  U = m*n;  前 E3 槽位为真实条目(行升序,行内列升序),
//       其余 indexC=(-1,-1), valueC=0.
//
// 算法: 逐输出行 i 累加稠密行缓冲 acc[n]/hit[n] (n 方向分块 accN):
//   for each A 项 (i,k,a): 取 B 第 k 行的连续 run [start,len) -> acc[n] += a*b, hit[n]=1
// 之后按 n 升序扫描 hit 压实写出. B 第 k 行的 run 由 A 的 k 值二分查找 B 的行索引数组得到.
// 单核(blockDim=1)实现: 无 workspace、无 SyncAll.
#include "kernel_operator.h"

#include "spspmm_tiling.h"
#include "tiling_key_spspmm.h"

using namespace AscendC;

namespace spk {
constexpr int32_t CH = 64;      // A 分块
constexpr int32_t RC = 1024;    // B run 子块
constexpr int32_t BM = 128;     // emit 批条目
constexpr int32_t FILL = 1024;  // 填充 tile
constexpr int32_t AROW = 512;   // A 行索引缓存
constexpr int32_t LA = 64;
}  // namespace spk

template <typename T>
class KernelSpspmm {
public:
    __aicore__ inline KernelSpspmm() {}

    __aicore__ inline void Init(GM_ADDR indexA, GM_ADDR valueA, GM_ADDR indexB, GM_ADDR valueB,
                                GM_ADDR indexC, GM_ADDR valueC, const SpspmmTilingData &t) {
        m_ = static_cast<int64_t>(t.m);
        n_ = static_cast<int64_t>(t.n);
        k_ = static_cast<int64_t>(t.k);
        U_ = static_cast<int64_t>(t.U);
        e1_ = static_cast<int64_t>(t.e1);
        e2_ = static_cast<int64_t>(t.e2);
        accN_ = static_cast<int64_t>(t.accN);
        chN_ = static_cast<int64_t>(t.chN);
        S_ = 0;

        iaGm_.SetGlobalBuffer((__gm__ int64_t *)indexA, (uint64_t)(2 * e1_ > 0 ? 2 * e1_ : 1));
        vaGm_.SetGlobalBuffer((__gm__ T *)valueA, (uint64_t)(e1_ > 0 ? e1_ : 1));
        ibGm_.SetGlobalBuffer((__gm__ int64_t *)indexB, (uint64_t)(2 * e2_ > 0 ? 2 * e2_ : 1));
        vbGm_.SetGlobalBuffer((__gm__ T *)valueB, (uint64_t)(e2_ > 0 ? e2_ : 1));
        icGm_.SetGlobalBuffer((__gm__ int32_t *)indexC, (uint64_t)(4 * U_ > 0 ? 4 * U_ : 1));
        vcGm_.SetGlobalBuffer((__gm__ T *)valueC, (uint64_t)(U_ > 0 ? U_ : 1));

        const int32_t accN = static_cast<int32_t>(accN_);
        pipe_.InitBuffer(bAI64_, spk::CH * 8);
        pipe_.InitBuffer(bAV_, spk::CH * (int32_t)sizeof(T));
        pipe_.InitBuffer(bAValF_, (spk::CH + 8) * 4);
        pipe_.InitBuffer(bAcc_, (accN + 8) * 4);
        pipe_.InitBuffer(bHit_, (accN + 8) * 4);
        pipe_.InitBuffer(bRI64_, spk::RC * 8);
        pipe_.InitBuffer(bRV_, spk::RC * (int32_t)sizeof(T));
        pipe_.InitBuffer(bRVF_, spk::RC * 4);
        pipe_.InitBuffer(bEN_, 2 * spk::BM * 4);
        pipe_.InitBuffer(bEI_, 2 * spk::BM * 4);
        pipe_.InitBuffer(bEN2_, 2 * spk::BM * 4);
        pipe_.InitBuffer(bEI2_, 2 * spk::BM * 4);
        pipe_.InitBuffer(bEV_, spk::BM * 4);
        pipe_.InitBuffer(bEVT_, spk::BM * (int32_t)sizeof(T));
        pipe_.InitBuffer(bOffI_, 2 * spk::BM * 4);
        pipe_.InitBuffer(bFill32_, spk::FILL * 4);
        pipe_.InitBuffer(bFillH_, spk::FILL * 2);
        pipe_.InitBuffer(bARow_, spk::AROW * 8);
    }

    __aicore__ inline void Process() {
        if (m_ <= 0 || n_ <= 0 || U_ <= 0) {
            return;
        }
        InitTables();
        arowSt_ = -1;
        arowCnt_ = 0;
        int64_t ap = 0;
        for (int64_t i = 0; i < m_; ++i) {
            const int64_t a0 = ap;
            while (ap < e1_ && RowAt(ap) == i) {
                ++ap;
            }
            const int64_t a1 = ap;
            if (a1 > a0) {
                ProcessRow(i, a0, a1);
            }
        }
        MergeFill();
    }

private:
    __aicore__ inline void InitTables() {
        LocalTensor<uint32_t> offIU = bOffI_.Get<uint32_t>();
        LocalTensor<int32_t> offII = offIU.ReinterpretCast<int32_t>();
        for (int32_t t = 0; t < 2 * spk::BM; ++t) {
            offII.SetValue(t, 4 * ((t >> 1) + (t & 1) * spk::BM));
        }
        LocalTensor<int32_t> en = bEN_.Get<int32_t>();
        LocalTensor<int32_t> ei = bEI_.Get<int32_t>();
        for (int32_t t = spk::BM; t < 2 * spk::BM; ++t) {
            en.SetValue(t, 0);
            ei.SetValue(t, 0);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline int64_t RowAt(int64_t j) {
        if (j < arowSt_ || j >= arowSt_ + arowCnt_) {
            const int64_t cnt = (e1_ - j) < spk::AROW ? (e1_ - j) : spk::AROW;
            LocalTensor<int64_t> buf = bARow_.Get<int64_t>();
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cnt * 8);
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPadExtParams<int64_t> pp;
            DataCopyPad(buf, iaGm_[j], p, pp);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            arowSt_ = j;
            arowCnt_ = cnt;
        }
        __ubuf__ int64_t *p = (__ubuf__ int64_t *)bARow_.Get<int64_t>().GetPhyAddr();
        return p[j - arowSt_];
    }

    // 在 B 的行索引数组(indexB[0..e2), 升序)中找第一个 >= key 的位置
    __aicore__ inline int32_t LowerBoundGm(int32_t key) {
        int64_t lo = 0;
        int64_t hi = e2_;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (ibGm_.GetValue(mid) < static_cast<int64_t>(key)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return static_cast<int32_t>(lo);
    }

    // 取 A 项块 [j0, j0+cnt) 的 (k, a) 并处理各自的 B run
    __aicore__ inline void LookupRuns(int64_t j0, int32_t cnt, int32_t nb, int32_t cn) {
        LocalTensor<int64_t> ai64 = bAI64_.Get<int64_t>();
        LocalTensor<T> av = bAV_.Get<T>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(cnt * 8);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<int64_t> pp;
        DataCopyPad(ai64, iaGm_[e1_ + j0], p, pp);
        DataCopyExtParams q;
        q.blockCount = 1;
        q.blockLen = static_cast<uint32_t>(cnt * (int32_t)sizeof(T));
        q.srcStride = 0;
        q.dstStride = 0;
        q.rsv = 0;
        DataCopyPadExtParams<T> pq;
        DataCopyPad(av, vaGm_[j0], q, pq);
        LocalTensor<float> af = bAValF_.Get<float>();
        if constexpr (sizeof(T) == 2) {
            // 显式 MTE2->S：标量随后要读 ai64(列索引, int64)，不能只靠 MTE2->V->V_S 传递
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(af, av, RoundMode::CAST_NONE, spk::LA);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        __ubuf__ int32_t *kp = (__ubuf__ int32_t *)ai64.GetPhyAddr();
        __ubuf__ float *vp = (__ubuf__ float *)af.GetPhyAddr();
        if constexpr (sizeof(T) != 2) {
            vp = (__ubuf__ float *)av.GetPhyAddr();
        }
        for (int32_t t = 0; t < cnt; ++t) {
            const int32_t kk = kp[2 * t];
            const int32_t st = LowerBoundGm(kk);
            if (st >= e2_) {
                continue;
            }
            const int32_t en = LowerBoundGm(kk + 1);
            const int32_t ln = en - st;
            if (ln > 0) {
                ConsumeRun(st, ln, vp[t], nb, cn);
            }
        }
        if constexpr (sizeof(T) == 2) {
            // Cast(V) 读过 av，下一轮 DataCopyPad 覆盖前要等 V 完成
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    }

    // --------------------------------------------------------------- one row
    __aicore__ inline void ProcessRow(int64_t i, int64_t a0, int64_t a1) {
        const int32_t accN = static_cast<int32_t>(accN_);
        for (int64_t nc = 0; nc < chN_; ++nc) {
            const int32_t nb = static_cast<int32_t>(nc * accN);
            const int32_t cn = static_cast<int32_t>(((nc + 1) * accN <= n_) ? accN : (n_ - nb));
            const int32_t cnOp = ((cn + spk::LA - 1) / spk::LA) * spk::LA;
            LocalTensor<float> acc = bAcc_.Get<float>();
            LocalTensor<float> hit = bHit_.Get<float>();
            Duplicate(acc, 0.0f, cnOp);
            Duplicate(hit, 0.0f, cnOp);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (int64_t j0 = a0; j0 < a1; j0 += spk::CH) {
                const int32_t cnt = static_cast<int32_t>(((j0 + spk::CH) <= a1) ? spk::CH : (a1 - j0));
                LookupRuns(j0, cnt, nb, cn);
            }
            EmitRow(i, nb, cn);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void ConsumeRun(int32_t st, int32_t ln, float a, int32_t nb, int32_t cn) {
        const int32_t nend = nb + cn;
        LocalTensor<float> acc = bAcc_.Get<float>();
        LocalTensor<float> hit = bHit_.Get<float>();
        __ubuf__ float *ap = (__ubuf__ float *)acc.GetPhyAddr();
        __ubuf__ float *hp = (__ubuf__ float *)hit.GetPhyAddr();
        for (int32_t su = 0; su < ln; su += spk::RC) {
            const int32_t cnt = ((su + spk::RC) <= ln) ? spk::RC : (ln - su);
            LocalTensor<int64_t> cols = bRI64_.Get<int64_t>();
            LocalTensor<T> rv = bRV_.Get<T>();
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cnt * 8);
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPadExtParams<int64_t> pp;
            DataCopyPad(cols, ibGm_[e2_ + st + su], p, pp);
            DataCopyExtParams q;
            q.blockCount = 1;
            q.blockLen = static_cast<uint32_t>(cnt * (int32_t)sizeof(T));
            q.srcStride = 0;
            q.dstStride = 0;
            q.rsv = 0;
            DataCopyPadExtParams<T> pq;
            DataCopyPad(rv, vbGm_[st + su], q, pq);
            __ubuf__ int32_t *cp = (__ubuf__ int32_t *)cols.GetPhyAddr();
            __ubuf__ float *vp = nullptr;
            const int32_t cntOp = ((cnt + spk::LA - 1) / spk::LA) * spk::LA;
            if constexpr (sizeof(T) == 2) {
                // 显式 MTE2->S：标量随后要读 cols(int64 列索引)，不能只靠 MTE2->V->V_S 传递
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                LocalTensor<float> rf = bRVF_.Get<float>();
                Cast(rf, rv, RoundMode::CAST_NONE, cntOp);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                vp = (__ubuf__ float *)rf.GetPhyAddr();
            } else {
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                vp = (__ubuf__ float *)rv.GetPhyAddr();
            }
            for (int32_t u = 0; u < cnt; ++u) {
                const int32_t nn = cp[2 * u];
                if (nn >= nb && nn < nend) {
                    const int32_t id = nn - nb;
                    ap[id] += a * vp[u];
                    hp[id] = 1.0f;
                }
            }
            if constexpr (sizeof(T) == 2) {
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                // 标量读过 cols，下一轮 DataCopyPad 覆盖前显式等 S 完成
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            }
        }
    }

    // --------------------------------------------------------------- emit
    __aicore__ inline void Flush(int32_t t) {
        LocalTensor<int32_t> en = bEN_.Get<int32_t>();
        LocalTensor<int32_t> ei = bEI_.Get<int32_t>();
        LocalTensor<int32_t> en2 = bEN2_.Get<int32_t>();
        LocalTensor<int32_t> ei2 = bEI2_.Get<int32_t>();
        LocalTensor<float> ev = bEV_.Get<float>();
        LocalTensor<uint32_t> offIU = bOffI_.Get<uint32_t>();
        const int32_t c2 = 2 * t;
        const int32_t tOp = ((t + spk::LA - 1) / spk::LA) * spk::LA;
        int32_t c2op = ((c2 + spk::LA - 1) / spk::LA) * spk::LA;
        if (c2op > 2 * spk::BM) {
            c2op = 2 * spk::BM;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Gather(en2, en, offIU, 0, c2op);
        Gather(ei2, ei, offIU, 0, c2op);
        if constexpr (sizeof(T) == 2) {
            LocalTensor<T> evt = bEVT_.Get<T>();
            Cast(evt, ev, RoundMode::CAST_NONE, tOp);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams q;
            q.blockCount = 1;
            q.blockLen = static_cast<uint32_t>(t * (int32_t)sizeof(T));
            q.srcStride = 0;
            q.dstStride = 0;
            q.rsv = 0;
            DataCopyPad(vcGm_[S_], evt, q);
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams q;
            q.blockCount = 1;
            q.blockLen = static_cast<uint32_t>(t * 4);
            q.srcStride = 0;
            q.dstStride = 0;
            q.rsv = 0;
            DataCopyPad(vcGm_[S_], ev, q);
        }
        DataCopyExtParams pn;
        pn.blockCount = 1;
        pn.blockLen = static_cast<uint32_t>(c2 * 4);
        pn.srcStride = 0;
        pn.dstStride = 0;
        pn.rsv = 0;
        DataCopyPad(icGm_[2 * S_], ei2, pn);
        DataCopyPad(icGm_[2 * (U_ + S_)], en2, pn);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        S_ += t;
    }

    __aicore__ inline void EmitRow(int64_t i, int32_t nb, int32_t cn) {
        LocalTensor<float> acc = bAcc_.Get<float>();
        LocalTensor<float> hit = bHit_.Get<float>();
        LocalTensor<int32_t> en = bEN_.Get<int32_t>();
        LocalTensor<int32_t> ei = bEI_.Get<int32_t>();
        LocalTensor<float> ev = bEV_.Get<float>();
        __ubuf__ float *ap = (__ubuf__ float *)acc.GetPhyAddr();
        __ubuf__ float *hp = (__ubuf__ float *)hit.GetPhyAddr();
        int32_t t = 0;
        for (int32_t j = 0; j < cn; ++j) {
            if (hp[j] > 0.5f) {
                en.SetValue(t, nb + j);
                ei.SetValue(t, static_cast<int32_t>(i));
                ev.SetValue(t, ap[j]);
                ++t;
                if (t == spk::BM) {
                    Flush(t);
                    t = 0;
                }
            }
        }
        if (t > 0) {
            Flush(t);
        }
    }

    __aicore__ inline void FillWords(const LocalTensor<int32_t> &src, int64_t w, const int64_t wEnd) {
        while (w < wEnd) {
            const int32_t cnt = ((wEnd - w) < spk::FILL) ? static_cast<int32_t>(wEnd - w) : spk::FILL;
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cnt * 4);
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPad(icGm_[w], src, p);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            w += cnt;
        }
    }

    // --------------------------------------------------------------- fill
    __aicore__ inline void MergeFill() {
        if (S_ >= U_) {
            return;
        }
        LocalTensor<int32_t> f32v = bFill32_.Get<int32_t>();
        Duplicate(f32v, (int32_t)-1, spk::FILL);
        // indexC[0][S..U) -> 词 [2S, 2U) ; indexC[1][S..U) -> 词 [2U+2S, 4U)
        FillWords(f32v, 2 * S_, 2 * U_);
        FillWords(f32v, 2 * U_ + 2 * S_, 4 * U_);
        if constexpr (sizeof(T) == 2) {
            LocalTensor<half> fh = bFillH_.Get<half>();
            Duplicate(fh, (half)0.0, spk::FILL);
            int64_t s = S_;
            while (s < U_) {
                const int32_t cnt = ((U_ - s) < spk::FILL) ? static_cast<int32_t>(U_ - s) : spk::FILL;
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams q;
                q.blockCount = 1;
                q.blockLen = static_cast<uint32_t>(cnt * 2);
                q.srcStride = 0;
                q.dstStride = 0;
                q.rsv = 0;
                DataCopyPad(vcGm_[s], fh, q);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                s += cnt;
            }
        } else {
            LocalTensor<float> ff = bFill32_.Get<float>();
            Duplicate(ff, 0.0f, spk::FILL);
            int64_t s = S_;
            while (s < U_) {
                const int32_t cnt = ((U_ - s) < spk::FILL) ? static_cast<int32_t>(U_ - s) : spk::FILL;
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams q;
                q.blockCount = 1;
                q.blockLen = static_cast<uint32_t>(cnt * 4);
                q.srcStride = 0;
                q.dstStride = 0;
                q.rsv = 0;
                DataCopyPad(vcGm_[s], ff, q);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                s += cnt;
            }
        }
    }

    // --------------------------------------------------------------- members
    TPipe pipe_;
    TBuf<TPosition::VECCALC> bAI64_;
    TBuf<TPosition::VECCALC> bAV_;
    TBuf<TPosition::VECCALC> bAValF_;
    TBuf<TPosition::VECCALC> bAcc_;
    TBuf<TPosition::VECCALC> bHit_;
    TBuf<TPosition::VECCALC> bRI64_;
    TBuf<TPosition::VECCALC> bRV_;
    TBuf<TPosition::VECCALC> bRVF_;
    TBuf<TPosition::VECCALC> bEN_;
    TBuf<TPosition::VECCALC> bEI_;
    TBuf<TPosition::VECCALC> bEN2_;
    TBuf<TPosition::VECCALC> bEI2_;
    TBuf<TPosition::VECCALC> bEV_;
    TBuf<TPosition::VECCALC> bEVT_;
    TBuf<TPosition::VECCALC> bOffI_;
    TBuf<TPosition::VECCALC> bFill32_;
    TBuf<TPosition::VECCALC> bFillH_;
    TBuf<TPosition::VECCALC> bARow_;

    GlobalTensor<int64_t> iaGm_;
    GlobalTensor<T> vaGm_;
    GlobalTensor<int64_t> ibGm_;
    GlobalTensor<T> vbGm_;
    GlobalTensor<int32_t> icGm_;
    GlobalTensor<T> vcGm_;

    int64_t m_ = 0, n_ = 0, k_ = 0, U_ = 0, e1_ = 0, e2_ = 0, accN_ = 0, chN_ = 0;
    int64_t S_ = 0;
    int64_t arowSt_ = -1;
    int64_t arowCnt_ = 0;
};

template <typename DT_VALUEA>
__global__ __aicore__ void spspmm(GM_ADDR indexA, GM_ADDR valueA, GM_ADDR indexB, GM_ADDR valueB,
                                  GM_ADDR indexC, GM_ADDR valueC, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SpspmmTilingData);
    GET_TILING_DATA_WITH_STRUCT(SpspmmTilingData, tiling_data, tiling);
    KernelSpspmm<DT_VALUEA> op;
    op.Init(indexA, valueA, indexB, valueB, indexC, valueC, tiling_data);
    op.Process();
}

// Kernel侧核函数实现
// ComplexDiv: out[i,j] = dividend[aRow, aCol] / divisor[bRow, bCol]
//   广播: 行方向 aN==1 时固定取第 0 行; 列方向 aM==1 时整行用同一个标量广播。
//   实数: 逐点除法 (f16 先升 f32 再除, 满足 1e-3)。
//   复数: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
//         A/B 分块搬入后用 Gather 按字节偏移去交错出 a,b,c,d;
//         结果放 W=[re|im] 再用 Gather 交织回 [re0,im0,re1,im1,...]。
//   UB 说明: out 用 TQue<VECOUT,2> + EnQue/DeQue, 保证 MTE3 写完后下一轮才能覆盖同一 buffer。
#include "kernel_operator.h"

#include "complex_div_tiling.h"
#include "tiling_key_complex_div.h"

using namespace AscendC;

namespace {
constexpr int32_t ALIGN_ELEM = 64;
constexpr int32_t UB_PAD = 64;
}  // namespace

// ============================ 实数路径 ============================
template <typename T>
class KernelDivReal {
public:
    __aicore__ inline KernelDivReal() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out, const ComplexDivTilingData &t) {
        aN_ = static_cast<int64_t>(t.aN);
        aM_ = static_cast<int64_t>(t.aM);
        bN_ = static_cast<int64_t>(t.bN);
        bM_ = static_cast<int64_t>(t.bM);
        outM_ = static_cast<int64_t>(t.outM);
        chunk_ = static_cast<int64_t>(t.chunk);
        const int64_t per = static_cast<int64_t>(t.rowsPerCore);
        const int64_t outN = static_cast<int64_t>(t.outN);
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        int64_t s = blk * per;
        int64_t en = s + per;
        if (s > outN) {
            s = outN;
        }
        if (en > outN) {
            en = outN;
        }
        startRow_ = s;
        endRow_ = en;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a), aN_ * aM_);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b), bN_ * bM_);
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), outN * outM_);

        const uint32_t bytes = static_cast<uint32_t>((chunk_ + UB_PAD) * sizeof(T));
        pipe_.InitBuffer(inQa_, 2, bytes);
        pipe_.InitBuffer(inQb_, 2, bytes);
        pipe_.InitBuffer(outQ_, 2, bytes);
        if constexpr (sizeof(T) == 2) {
            const uint32_t fb = static_cast<uint32_t>((chunk_ + UB_PAD) * sizeof(float));
            pipe_.InitBuffer(fa_, fb);
            pipe_.InitBuffer(fbb_, fb);
            pipe_.InitBuffer(fcc_, fb);
        }
    }

    __aicore__ inline void Process() {
        for (int64_t row = startRow_; row < endRow_; ++row) {
            const int64_t aRow = (aN_ == 1) ? 0 : row;
            const int64_t bRow = (bN_ == 1) ? 0 : row;
            for (int64_t col = 0; col < outM_; col += chunk_) {
                const int64_t c = (col + chunk_ < outM_) ? chunk_ : (outM_ - col);
                const int32_t cn = static_cast<int32_t>(c);
                LocalTensor<T> ta = inQa_.AllocTensor<T>();
                if (aM_ == 1) {
                    Duplicate(ta, static_cast<T>(static_cast<float>(aGm_.GetValue(static_cast<int32_t>(aRow)))), cn);
                } else {
                    LoadSpan(ta, aGm_, aRow * aM_ + col, c);
                }
                inQa_.EnQue(ta);
                LocalTensor<T> tb = inQb_.AllocTensor<T>();
                if (bM_ == 1) {
                    Duplicate(tb, static_cast<T>(static_cast<float>(bGm_.GetValue(static_cast<int32_t>(bRow)))), cn);
                } else {
                    LoadSpan(tb, bGm_, bRow * bM_ + col, c);
                }
                inQb_.EnQue(tb);

                LocalTensor<T> ta2 = inQa_.DeQue<T>();
                LocalTensor<T> tb2 = inQb_.DeQue<T>();
                LocalTensor<T> o = outQ_.AllocTensor<T>();
                if constexpr (sizeof(T) == 2) {
                    LocalTensor<float> fa = fa_.Get<float>();
                    LocalTensor<float> fbb = fbb_.Get<float>();
                    LocalTensor<float> fcc = fcc_.Get<float>();
                    Cast(fa, ta2, RoundMode::CAST_NONE, cn);
                    Cast(fbb, tb2, RoundMode::CAST_NONE, cn);
                    Div(fcc, fa, fbb, cn);
                    Cast(o, fcc, RoundMode::CAST_NONE, cn);
                } else {
                    Div(o, ta2, tb2, cn);
                }
                inQa_.FreeTensor(ta2);
                inQb_.FreeTensor(tb2);
                outQ_.EnQue(o);

                LocalTensor<T> od = outQ_.DeQue<T>();
                const int64_t off = row * outM_ + col;
                if (((c * static_cast<int64_t>(sizeof(T))) % 32) == 0 &&
                    ((off * static_cast<int64_t>(sizeof(T))) % 32) == 0) {
                    DataCopy(oGm_[static_cast<int32_t>(off)], od, static_cast<uint32_t>(c));
                } else {
                    DataCopyExtParams p;
                    p.blockCount = 1;
                    p.blockLen = static_cast<uint32_t>(c * sizeof(T));
                    p.srcStride = 0;
                    p.dstStride = 0;
                    p.rsv = 0;
                    DataCopyPad(oGm_[static_cast<int32_t>(off)], od, p);
                }
                outQ_.FreeTensor(od);
            }
        }
    }

private:
    __aicore__ inline void LoadSpan(const LocalTensor<T> &dst, const GlobalTensor<T> &src, int64_t off, int64_t cnt) {
        if (((cnt * static_cast<int64_t>(sizeof(T))) % 32) == 0 &&
            ((off * static_cast<int64_t>(sizeof(T))) % 32) == 0) {
            DataCopy(dst, src[static_cast<int32_t>(off)], static_cast<uint32_t>(cnt));
        } else {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cnt * sizeof(T));
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPadExtParams<T> pp;
            DataCopyPad(dst, src[static_cast<int32_t>(off)], p, pp);
        }
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQa_;
    TQue<TPosition::VECIN, 2> inQb_;
    TQue<TPosition::VECOUT, 2> outQ_;
    TBuf<TPosition::VECCALC> fa_;
    TBuf<TPosition::VECCALC> fbb_;
    TBuf<TPosition::VECCALC> fcc_;
    GlobalTensor<T> aGm_;
    GlobalTensor<T> bGm_;
    GlobalTensor<T> oGm_;
    int64_t aN_ = 0, aM_ = 0, bN_ = 0, bM_ = 0, outM_ = 0, chunk_ = 0, startRow_ = 0, endRow_ = 0;
};

// ============================ 复数路径 ============================
class KernelDivC64 {
public:
    __aicore__ inline KernelDivC64() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out, const ComplexDivTilingData &t) {
        aN_ = static_cast<int64_t>(t.aN);
        aM_ = static_cast<int64_t>(t.aM);
        bN_ = static_cast<int64_t>(t.bN);
        bM_ = static_cast<int64_t>(t.bM);
        outM_ = static_cast<int64_t>(t.outM);
        chunk_ = static_cast<int64_t>(t.chunk);
        nOp_ = static_cast<int32_t>(t.chunk);
        if (nOp_ < 64) {
            nOp_ = 64;
        }
        const int64_t per = static_cast<int64_t>(t.rowsPerCore);
        const int64_t outN = static_cast<int64_t>(t.outN);
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        int64_t s = blk * per;
        int64_t en = s + per;
        if (s > outN) {
            s = outN;
        }
        if (en > outN) {
            en = outN;
        }
        startRow_ = s;
        endRow_ = en;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), aN_ * aM_ * 2);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), bN_ * bM_ * 2);
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), outN * outM_ * 2);

        const int32_t ch = static_cast<int32_t>(chunk_);
        const int32_t sz = ch + UB_PAD;
        pipe_.InitBuffer(bufA_, static_cast<uint32_t>((2 * ch + 2 * UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(bufB_, static_cast<uint32_t>((2 * ch + 2 * UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tI_, static_cast<uint32_t>(4 * sz * sizeof(float)));  // a,b,c,d
        pipe_.InitBuffer(tJ_, static_cast<uint32_t>(6 * sz * sizeof(float)));  // den,t1,t2,W(2*sz)
        pipe_.InitBuffer(outQ_, 2, static_cast<uint32_t>((2 * ch + 2 * UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>(6 * sz * sizeof(uint32_t)));

        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<int32_t> offU = oi;
        LocalTensor<int32_t> offV = oi[sz];
        LocalTensor<int32_t> offI = oi[2 * sz];
        LocalTensor<int32_t> scratch = oi[4 * sz];
        LocalTensor<float> fw = bufA_.Get<float>();
        CreateVecIndex(offU, static_cast<int32_t>(0), static_cast<uint32_t>(nOp_));
        Muls(offU, offU, static_cast<int32_t>(2 * sizeof(float)), nOp_);
        Adds(offV, offU, static_cast<int32_t>(sizeof(float)), nOp_);

        // 交织偏移 offI[k] = 4n*k - (8n-4)*floor(k/2), n = nOp_
        const int32_t twoN = 2 * nOp_;
        CreateVecIndex(offI, static_cast<int32_t>(0), static_cast<uint32_t>(twoN));
        Cast(fw, offI, RoundMode::CAST_NONE, twoN);
        Muls(fw, fw, 0.5f, twoN);
        Cast(scratch, fw, RoundMode::CAST_FLOOR, twoN);
        Muls(offI, offI, static_cast<int32_t>(4 * nOp_), twoN);
        Muls(scratch, scratch, -static_cast<int32_t>(8 * nOp_ - 4), twoN);
        Add(offI, offI, scratch, twoN);
    }

    __aicore__ inline void Process() {
        for (int64_t row = startRow_; row < endRow_; ++row) {
            const int64_t aRow = (aN_ == 1) ? 0 : row;
            const int64_t bRow = (bN_ == 1) ? 0 : row;
            for (int64_t col = 0; col < outM_; col += chunk_) {
                const int64_t c = (col + chunk_ < outM_) ? chunk_ : (outM_ - col);
                DoChunk(row, col, c, aRow, bRow);
            }
        }
    }

private:
    __aicore__ inline void LoadSpan(const LocalTensor<float> &dst, const GlobalTensor<float> &src, int64_t elemOff,
                                    int64_t floats) {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(floats * sizeof(float));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<float> pp;
        DataCopyPad(dst, src[static_cast<int32_t>(elemOff)], p, pp);
    }

    __aicore__ inline void DoChunk(int64_t row, int64_t col, int64_t c, int64_t aRow, int64_t bRow) {
        const int32_t cnt = static_cast<int32_t>(c);
        int32_t cntOp = ((cnt + 63) / 64) * 64;
        if (cntOp > nOp_) {
            cntOp = nOp_;
        }
        const int32_t sz = static_cast<int32_t>(chunk_) + UB_PAD;
        LocalTensor<float> A = bufA_.Get<float>();
        LocalTensor<float> B = bufB_.Get<float>();
        LocalTensor<float> ti = tI_.Get<float>();
        LocalTensor<float> tj = tJ_.Get<float>();
        LocalTensor<float> aa = ti;
        LocalTensor<float> bb = ti[sz];
        LocalTensor<float> cc = ti[2 * sz];
        LocalTensor<float> dd = ti[3 * sz];
        LocalTensor<float> den = tj;
        LocalTensor<float> t1 = tj[sz];
        LocalTensor<float> t2 = tj[2 * sz];
        LocalTensor<float> w = tj[3 * sz];

        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<uint32_t> offU = oi.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offV = oi[sz].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offI = oi[2 * sz].template ReinterpretCast<uint32_t>();

        // 上一轮 V 已读过 A/B, 本轮 MTE2 覆盖前必须先等它做完 (TBuf 不会被 TQue 自动同步)
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        // dividend -> aa/bb
        if (aM_ == 1) {
            const float v0 = aGm_.GetValue(static_cast<int32_t>(aRow * 2));
            const float v1 = aGm_.GetValue(static_cast<int32_t>(aRow * 2 + 1));
            Duplicate(aa, v0, cntOp);
            Duplicate(bb, v1, cntOp);
        } else {
            LoadSpan(A, aGm_, (aRow * aM_ + col) * 2, 2 * c);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Gather(aa, A, offU, 0, static_cast<uint32_t>(cntOp));
            Gather(bb, A, offV, 0, static_cast<uint32_t>(cntOp));
        }
        // divisor -> cc/dd
        if (bM_ == 1) {
            const float v0 = bGm_.GetValue(static_cast<int32_t>(bRow * 2));
            const float v1 = bGm_.GetValue(static_cast<int32_t>(bRow * 2 + 1));
            Duplicate(cc, v0, cntOp);
            Duplicate(dd, v1, cntOp);
        } else {
            LoadSpan(B, bGm_, (bRow * bM_ + col) * 2, 2 * c);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Gather(cc, B, offU, 0, static_cast<uint32_t>(cntOp));
            Gather(dd, B, offV, 0, static_cast<uint32_t>(cntOp));
        }

        Mul(den, cc, cc, cntOp);
        Mul(t1, dd, dd, cntOp);
        Add(den, den, t1, cntOp);  // den = c^2 + d^2

        Mul(t1, aa, cc, cntOp);
        Mul(t2, bb, dd, cntOp);
        Add(w, t1, t2, cntOp);     // re 分子
        Div(w, w, den, cntOp);     // re = 分子/den

        Mul(t1, bb, cc, cntOp);
        Mul(t2, aa, dd, cntOp);
        Sub(t2, t1, t2, cntOp);    // im 分子
        Div(t2, t2, den, cntOp);   // im = 分子/den
        Adds(w[static_cast<int32_t>(chunk_)], t2, 0.0f, cntOp);

        LocalTensor<float> o = outQ_.AllocTensor<float>();
        Gather(o, w, offI, 0, static_cast<uint32_t>(2 * cntOp));
        outQ_.EnQue(o);

        LocalTensor<float> od = outQ_.DeQue<float>();
        const int64_t off = row * outM_ * 2 + col * 2;
        const int64_t elems = 2 * c;
        if (((elems * 4) % 32) == 0 && ((off * 4) % 32) == 0) {
            DataCopy(oGm_[static_cast<int32_t>(off)], od, static_cast<uint32_t>(elems));
        } else {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(elems * 4);
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPad(oGm_[static_cast<int32_t>(off)], od, p);
        }
        outQ_.FreeTensor(od);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufB_;
    TBuf<TPosition::VECCALC> tI_;
    TBuf<TPosition::VECCALC> tJ_;
    TQue<TPosition::VECOUT, 2> outQ_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> oGm_;
    int64_t aN_ = 0, aM_ = 0, bN_ = 0, bM_ = 0, outM_ = 0, chunk_ = 0, startRow_ = 0, endRow_ = 0;
    int32_t nOp_ = 0;
};

template <typename DT>
__global__ __aicore__ void complex_div(GM_ADDR dividend, GM_ADDR divisor, GM_ADDR out, GM_ADDR workspace,
                                       GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ComplexDivTilingData);
    GET_TILING_DATA_WITH_STRUCT(ComplexDivTilingData, tiling_data, tiling);
    const uint32_t mode = tiling_data.dtypeMode;
    if (mode == 2u) {
        KernelDivC64 op;
        op.Init(dividend, divisor, out, tiling_data);
        op.Process();
    } else if (mode == 0u) {
        KernelDivReal<half> op;
        op.Init(dividend, divisor, out, tiling_data);
        op.Process();
    } else {
        KernelDivReal<float> op;
        op.Init(dividend, divisor, out, tiling_data);
        op.Process();
    }
}

// Kernel侧核函数实现
// ComplexExp:
//   实数:  out = e^x
//   复数:  x = a + b*i -> out = e^a*cos b + i*e^a*sin b
//
// complex64 在 GM 中按 [a, b] 交错存放, 需要先去掉交错才能分别做 Exp/Cos/Sin,
// 结果还要再交织回 [re, im]。两个方向都用 vector Gather(按字节偏移取元素)实现,
// 偏移张量在 Init 里按 tileLength 生成一次 (910B 上 Scatter 不支持)。
// 多核: 每核一段连续元素, 段内按 tileLength 分块走 TQue 双缓冲流水。
#include "kernel_operator.h"

#include "complex_exp_tiling.h"
#include "tiling_key_complex_exp.h"

using namespace AscendC;

namespace {
constexpr int32_t ALIGN_ELEM = 64;  // 一次 repeat 的对齐粒度 (float32: 64 个/256B)
constexpr int32_t UB_PAD = 64;      // 缓冲尾部预留
}  // namespace

// ============================ 实数路径: out = exp(x) ============================
template <typename T>
class KernelExpReal {
public:
    __aicore__ inline KernelExpReal() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, const ComplexExpTilingData &t) {
        const int64_t total = static_cast<int64_t>(t.length);
        int64_t s = static_cast<int64_t>(GetBlockIdx()) * static_cast<int64_t>(t.blockFactor);
        int64_t e = s + static_cast<int64_t>(t.blockFactor);
        if (s > total) {
            s = total;
        }
        if (e > total) {
            e = total;
        }
        start_ = s;
        end_ = e;
        tile_ = static_cast<int64_t>(t.tileLength);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), total);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), total);

        pipe_.InitBuffer(inQue_, 2, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(T)));
        pipe_.InitBuffer(outQue_, 2, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(T)));
        if constexpr (sizeof(T) == 2) {
            pipe_.InitBuffer(fBuf_, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        }
    }

    __aicore__ inline void Process() {
        if (start_ >= end_) {
            return;
        }
        int64_t off = start_;
        int64_t rem = end_ - start_;
        while (rem > 0) {
            const int64_t n = rem > tile_ ? tile_ : rem;
            CopyIn(off, n);
            Compute(n);
            CopyOut(off, n);
            off += n;
            rem -= n;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t off, int64_t n) {
        LocalTensor<T> in = inQue_.AllocTensor<T>();
        if (((n % ALIGN_ELEM) == 0) && (((off * static_cast<int64_t>(sizeof(T))) % 32) == 0)) {
            DataCopy(in, xGm_[static_cast<int32_t>(off)], static_cast<uint32_t>(n));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(n * sizeof(T));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<T> padParams;
            DataCopyPad(in, xGm_[static_cast<int32_t>(off)], params, padParams);
        }
        inQue_.EnQue(in);
    }

    __aicore__ inline void Compute(int64_t n) {
        LocalTensor<T> in = inQue_.DeQue<T>();
        LocalTensor<T> out = outQue_.AllocTensor<T>();
        if constexpr (sizeof(T) == 2) {
            LocalTensor<float> fb = fBuf_.Get<float>();
            Cast(fb, in, RoundMode::CAST_NONE, static_cast<int32_t>(n));
            Exp(fb, fb, static_cast<int32_t>(n));
            Cast(out, fb, RoundMode::CAST_NONE, static_cast<int32_t>(n));
        } else {
            Exp(out, in, static_cast<int32_t>(n));
        }
        inQue_.FreeTensor(in);
        outQue_.EnQue(out);
    }

    __aicore__ inline void CopyOut(int64_t off, int64_t n) {
        LocalTensor<T> out = outQue_.DeQue<T>();
        if (((n % ALIGN_ELEM) == 0) && (((off * static_cast<int64_t>(sizeof(T))) % 32) == 0)) {
            DataCopy(yGm_[static_cast<int32_t>(off)], out, static_cast<uint32_t>(n));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(n * sizeof(T));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPad(yGm_[static_cast<int32_t>(off)], out, params);
        }
        outQue_.FreeTensor(out);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;
    TBuf<TPosition::VECCALC> fBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t start_ = 0;
    int64_t end_ = 0;
    int64_t tile_ = 0;
};

// ============================ 复数路径: out = e^a(cos b + i sin b) ============================
class KernelComplexExpC64 {
public:
    __aicore__ inline KernelComplexExpC64() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, const ComplexExpTilingData &t) {
        const int64_t total = static_cast<int64_t>(t.length);  // 复数个数
        int64_t s = static_cast<int64_t>(GetBlockIdx()) * static_cast<int64_t>(t.blockFactor);
        int64_t e = s + static_cast<int64_t>(t.blockFactor);
        if (s > total) {
            s = total;
        }
        if (e > total) {
            e = total;
        }
        start_ = s;
        end_ = e;
        tile_ = static_cast<int64_t>(t.tileLength);
        nOp_ = static_cast<int32_t>(tile_);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), total * 2);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), total * 2);

        pipe_.InitBuffer(inQue_, 2, static_cast<uint32_t>((2 * tile_ + 2 * UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(outQue_, 2, static_cast<uint32_t>((2 * tile_ + 2 * UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tA_, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tB_, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tC_, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tW_, static_cast<uint32_t>((2 * tile_ + 2 * UB_PAD) * sizeof(float)));

        // 4 段 uint32: offU(n) / offV(n) / offI(2n) / scratch(2n)
        const int32_t stride = static_cast<int32_t>(tile_ + UB_PAD);
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>(6 * stride * sizeof(uint32_t)));

        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<int32_t> offU = oi;
        LocalTensor<int32_t> offV = oi[stride];
        LocalTensor<int32_t> offI = oi[2 * stride];
        LocalTensor<int32_t> scratch = oi[4 * stride];
        LocalTensor<float> fw = tW_.Get<float>();  // 初始化阶段当作 2n 浮点暂存

        // 输入去交错偏移(vgather 的 offset 单位是**字节**):
        //   offU[j] = 2j*4 = 8j   (实部 a) ; offV[j] = 8j+4 (虚部 b)
        CreateVecIndex(offU, static_cast<int32_t>(0), static_cast<uint32_t>(nOp_));
        Muls(offU, offU, static_cast<int32_t>(2 * sizeof(float)), nOp_);
        Adds(offV, offU, static_cast<int32_t>(sizeof(float)), nOp_);

        // 输出交织偏移: 设 W = [Re | Im] (各 n 个 float), 则 out[k] = W[floor(k/2) + (k%2)*n]
        //   字节偏移 offI[k] = 4n*k - (8n-4)*floor(k/2)
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
        if (start_ >= end_) {
            return;
        }
        int64_t off = start_;
        int64_t rem = end_ - start_;
        while (rem > 0) {
            const int64_t n = rem > tile_ ? tile_ : rem;
            CopyIn(off, n);
            Compute(n);
            CopyOut(off, n);
            off += n;
            rem -= n;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t off, int64_t n) {
        LocalTensor<float> in = inQue_.AllocTensor<float>();
        const int64_t floats = n * 2;
        if (((n % ALIGN_ELEM) == 0) && (((off * 8) % 32) == 0)) {
            DataCopy(in, xGm_[static_cast<int32_t>(off * 2)], static_cast<uint32_t>(floats));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(floats * sizeof(float));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<float> padParams;
            DataCopyPad(in, xGm_[static_cast<int32_t>(off * 2)], params, padParams);
        }
        inQue_.EnQue(in);
    }

    __aicore__ inline void Compute(int64_t n) {
        int64_t nProc = ((n + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
        if (nProc > tile_) {
            nProc = tile_;
        }
        const int32_t np = static_cast<int32_t>(nProc);
        LocalTensor<float> in = inQue_.DeQue<float>();
        LocalTensor<float> out = outQue_.AllocTensor<float>();
        LocalTensor<float> a = tA_.Get<float>();
        LocalTensor<float> b = tB_.Get<float>();
        LocalTensor<float> c = tC_.Get<float>();
        LocalTensor<float> w = tW_.Get<float>();
        const int32_t stride = static_cast<int32_t>(tile_ + UB_PAD);
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<uint32_t> offU = oi.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offV = oi[stride].template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offI = oi[2 * stride].template ReinterpretCast<uint32_t>();

        Gather(a, in, offU, 0, static_cast<uint32_t>(np));  // a = 实部
        Gather(b, in, offV, 0, static_cast<uint32_t>(np));  // b = 虚部
        Exp(a, a, np);                                      // e^a
        Cos(c, b, np);                                      // cos b
        Sin(b, b, np);                                      // sin b
        Mul(w, a, c, np);                                   // W[0..n)  = 实部结果
        Mul(w[nOp_], a, b, np);                             // W[nOp_..) = 虚部结果 (offI 按 nOp_ 生成)
        Gather(out, w, offI, 0, static_cast<uint32_t>(2 * np));  // 交织回 [re, im]

        inQue_.FreeTensor(in);
        outQue_.EnQue(out);
    }

    __aicore__ inline void CopyOut(int64_t off, int64_t n) {
        LocalTensor<float> out = outQue_.DeQue<float>();
        if (((n % ALIGN_ELEM) == 0) && (((off * 8) % 32) == 0)) {
            DataCopy(yGm_[static_cast<int32_t>(off * 2)], out, static_cast<uint32_t>(n * 2));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(n * 2 * sizeof(float));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPad(yGm_[static_cast<int32_t>(off * 2)], out, params);
        }
        outQue_.FreeTensor(out);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;
    TBuf<TPosition::VECCALC> tA_;
    TBuf<TPosition::VECCALC> tB_;
    TBuf<TPosition::VECCALC> tC_;
    TBuf<TPosition::VECCALC> tW_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    int64_t start_ = 0;
    int64_t end_ = 0;
    int64_t tile_ = 0;
    int32_t nOp_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void complex_exp(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ComplexExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ComplexExpTilingData, tiling_data, tiling);
    const uint32_t mode = tiling_data.dtypeMode;
    if (mode == 2u) {
        KernelComplexExpC64 op;
        op.Init(x, out, tiling_data);
        op.Process();
    } else if (mode == 0u) {
        KernelExpReal<half> op;
        op.Init(x, out, tiling_data);
        op.Process();
    } else {
        KernelExpReal<float> op;
        op.Init(x, out, tiling_data);
        op.Process();
    }
}

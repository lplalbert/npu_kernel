// Kernel侧核函数实现
// ComplexAbs: out[i] = sqrt(re[i]^2 + im[i]^2)
//
// 数据布局: GM 中 src 为 complex64, 即 [re0, im0, re1, im1, ...] 连续 float32 交错。
// 做法(纯逐点访存受限, 目标 = 打满 HBM 带宽):
//   1) DataCopy/DataCopyPad 把 2*n 个 float32 (n 个复数) 搬进 UB;
//   2) 用 vector Gather 按字节偏移取出实部 A 与虚部 B (一次指令完成去交错);
//   3) A = A*A ; B = B*B ; A = A+B ; out = Sqrt(A);
//   4) DataCopy/DataCopyPad 写回 GM。
// 说明: 910B(CANN 9.0) 上 vector Scatter 不支持, pair-reduce 的模板签名各版本不一致,
//       因此统一使用 Gather(按字节偏移取元素), 版本兼容性最好。
// 多核: 每核负责一段连续的复数元素(blockFactor, 64 的整数倍), 段内按 tileLength 分块,
//       每块走 TQue double buffer 流水。
#include "kernel_operator.h"

#include "complex_abs_tiling.h"
#include "tiling_key_complex_abs.h"

using namespace AscendC;

namespace {
constexpr int32_t ALIGN_COMPLEX = 64;  // 对齐粒度: 64 个复数 = 128 float (一次 repeat 的两倍)
constexpr int32_t UB_PAD = 64;         // 缓冲尾部预留 (尾部块按整 tile 处理)
}  // namespace

template <class DT_SRC>
class KernelComplexAbs {
public:
    __aicore__ inline KernelComplexAbs() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, const ComplexAbsTilingData &t) {
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
        nOp_ = static_cast<int32_t>(tile_);

        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), total * 2);
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), total);

        pipe_.InitBuffer(inQue_, 2, static_cast<uint32_t>((2 * tile_ + 2 * UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(outQue_, 2, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tA_, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        pipe_.InitBuffer(tB_, static_cast<uint32_t>((tile_ + UB_PAD) * sizeof(float)));
        const int32_t stride = static_cast<int32_t>(tile_ + UB_PAD);
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>(2 * stride * sizeof(uint32_t)));

        // 去交错偏移 (vgather 的 offset 单位是**字节**):
        //   offE[j] = 2j*4 = 8j  (实部 re) ; offO[j] = 8j+4 (虚部 im)
        // 只与 tileLength 有关, 每个 kernel 只生成一次; 尾部块按整 tile 处理, 垃圾结果不写回 GM。
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<int32_t> offE = oi;
        LocalTensor<int32_t> offO = oi[stride];
        CreateVecIndex(offE, static_cast<int32_t>(0), static_cast<uint32_t>(nOp_));
        Muls(offE, offE, static_cast<int32_t>(2 * sizeof(float)), nOp_);
        Adds(offO, offE, static_cast<int32_t>(sizeof(float)), nOp_);
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
        if (((n % ALIGN_COMPLEX) == 0) && (((off * 8) % 32) == 0)) {
            DataCopy(in, srcGm_[static_cast<int32_t>(off * 2)], static_cast<uint32_t>(floats));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(floats * sizeof(float));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<float> padParams;
            DataCopyPad(in, srcGm_[static_cast<int32_t>(off * 2)], params, padParams);
        }
        inQue_.EnQue(in);
    }

    __aicore__ inline void Compute(int64_t n) {
        // 只处理本块实际需要的元素数(按 64 对齐), 避免小 shape 下做整 tile 的无效向量运算
        int64_t nProc = ((n + ALIGN_COMPLEX - 1) / ALIGN_COMPLEX) * ALIGN_COMPLEX;
        if (nProc > tile_) {
            nProc = tile_;
        }
        const int32_t np = static_cast<int32_t>(nProc);
        LocalTensor<float> in = inQue_.DeQue<float>();
        LocalTensor<float> out = outQue_.AllocTensor<float>();
        LocalTensor<float> a = tA_.Get<float>();
        LocalTensor<float> b = tB_.Get<float>();
        const int32_t stride = static_cast<int32_t>(tile_ + UB_PAD);
        LocalTensor<int32_t> oi = offBuf_.Get<int32_t>();
        LocalTensor<uint32_t> offE = oi.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offO = oi[stride].template ReinterpretCast<uint32_t>();

        Gather(a, in, offE, 0, static_cast<uint32_t>(np));  // a = 实部
        Gather(b, in, offO, 0, static_cast<uint32_t>(np));  // b = 虚部
        Mul(a, a, a, np);
        Mul(b, b, b, np);
        Add(a, a, b, np);
        Sqrt(out, a, np);

        inQue_.FreeTensor(in);
        outQue_.EnQue(out);
    }

    __aicore__ inline void CopyOut(int64_t off, int64_t n) {
        LocalTensor<float> out = outQue_.DeQue<float>();
        if (((n % ALIGN_COMPLEX) == 0) && (((off * 4) % 32) == 0)) {
            DataCopy(dstGm_[static_cast<int32_t>(off)], out, static_cast<uint32_t>(n));
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(n * sizeof(float));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPad(dstGm_[static_cast<int32_t>(off)], out, params);
        }
        outQue_.FreeTensor(out);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;
    TBuf<TPosition::VECCALC> tA_;
    TBuf<TPosition::VECCALC> tB_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<float> srcGm_;
    GlobalTensor<float> dstGm_;
    int64_t start_ = 0;
    int64_t end_ = 0;
    int64_t tile_ = 0;
    int32_t nOp_ = 0;
};

template <typename DT_SRC>
__global__ __aicore__ void complex_abs(GM_ADDR src, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ComplexAbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(ComplexAbsTilingData, tiling_data, tiling);
    KernelComplexAbs<DT_SRC> op;
    op.Init(src, out, tiling_data);
    op.Process();
}

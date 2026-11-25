// Kernel侧核函数实现
// GatherCoo: out[i, :] = src[index[i], :],  i in [0, E)
//
// 纯带宽算子: 输出 E*K 连续写; 输入按 index 随机读整行(K*sizeof(T) 字节)。
//
// 对齐要点(910B 真机): DataCopyPad 的 **UB 侧地址必须 32B 对齐**(GM 侧可任意),
// 所以 UB 里每行按 rowStride = alignUp(rowBytes,32) 摆放:
//   - rowBytes 是 32B 整数倍时: UB 内行首天然对齐, 一批行可以一次连续写回 GM(最快);
//   - 否则: 逐行 DataCopyPad 写回(UB 行首对齐, GM 行首可任意)。
// 多核按输出行切分(每核 rowsPerCore 行), 每批 groupRows 行, TQue 双缓冲流水。
#include "kernel_operator.h"

#include "gather_coo_tiling.h"
#include "tiling_key_gather_coo.h"

using namespace AscendC;

template <typename T>
class KernelGatherCoo {
public:
    __aicore__ inline KernelGatherCoo() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, const GatherCooTilingData &t) {
        m_ = static_cast<int64_t>(t.m);
        k_ = static_cast<int64_t>(t.k);
        e_ = static_cast<int64_t>(t.e);
        group_ = static_cast<int64_t>(t.groupRows);
        rowBytes_ = k_ * static_cast<int64_t>(sizeof(T));
        rowStride_ = ((rowBytes_ + 31) / 32) * 32;
        rowElems_ = rowStride_ / static_cast<int64_t>(sizeof(T));
        coalesce_ = ((rowBytes_ % 32) == 0) ? 1 : 0;
        const int64_t per = static_cast<int64_t>(t.rowsPerCore);
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        int64_t s = blk * per;
        int64_t en = s + per;
        if (s > e_) {
            s = e_;
        }
        if (en > e_) {
            en = e_;
        }
        start_ = s;
        end_ = en;

        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), m_ * k_);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index), e_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), e_ * k_);

        pipe_.InitBuffer(rowQue_, 2, static_cast<uint32_t>(group_ * rowStride_ + 64));
    }

    __aicore__ inline void Process() {
        if (start_ >= end_) {
            return;
        }
        for (int64_t i = start_; i < end_; i += group_) {
            const int64_t g = (i + group_ < end_) ? group_ : (end_ - i);
            CopyGroup(i, g);
        }
    }

private:
    __aicore__ inline void CopyGroup(int64_t rowStart, int64_t rows) {
        // 1) 逐行读 src 到 UB (UB 行首按 rowStride 对齐); index 用标量 GM 读, 避免 UB 标量读的同步风险
        LocalTensor<T> rowBuf = rowQue_.AllocTensor<T>();
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t srcRow = static_cast<int64_t>(idxGm_.GetValue(static_cast<int32_t>(rowStart + r)));
            const int32_t dstOff = static_cast<int32_t>(r * rowElems_);
            const int32_t srcOff = static_cast<int32_t>(srcRow * k_);
            if (coalesce_ != 0) {
                DataCopy(rowBuf[dstOff], srcGm_[srcOff], static_cast<uint32_t>(k_));
            } else {
                DataCopyExtParams p;
                p.blockCount = 1;
                p.blockLen = static_cast<uint32_t>(rowBytes_);
                p.srcStride = 0;
                p.dstStride = 0;
                p.rsv = 0;
                DataCopyPadExtParams<T> pp;
                DataCopyPad(rowBuf[dstOff], srcGm_[srcOff], p, pp);
            }
        }
        rowQue_.EnQue(rowBuf);

        // 3) 写回 out
        LocalTensor<T> o = rowQue_.DeQue<T>();
        if (coalesce_ != 0) {
            DataCopy(outGm_[static_cast<int32_t>(rowStart * k_)], o, static_cast<uint32_t>(rows * k_));
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams p;
                p.blockCount = 1;
                p.blockLen = static_cast<uint32_t>(rowBytes_);
                p.srcStride = 0;
                p.dstStride = 0;
                p.rsv = 0;
                DataCopyPad(outGm_[static_cast<int32_t>((rowStart + r) * k_)],
                            o[static_cast<int32_t>(r * rowElems_)], p);
            }
        }
        rowQue_.FreeTensor(o);
    }

    TPipe pipe_;
    // 纯 GM->UB->GM 搬运, 必须用 TQueBind 才会插入 MTE2->MTE3 依赖
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> rowQue_;
    GlobalTensor<T> srcGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<T> outGm_;
    int64_t m_ = 0;
    int64_t k_ = 0;
    int64_t e_ = 0;
    int64_t group_ = 0;
    int64_t rowBytes_ = 0;
    int64_t rowStride_ = 0;
    int64_t rowElems_ = 0;
    int64_t start_ = 0;
    int64_t end_ = 0;
    int32_t coalesce_ = 1;
};

template <typename DT_SRC>
__global__ __aicore__ void gather_coo(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GatherCooTilingData);
    GET_TILING_DATA_WITH_STRUCT(GatherCooTilingData, tiling_data, tiling);
    KernelGatherCoo<DT_SRC> op;
    op.Init(src, index, out, tiling_data);
    op.Process();
}

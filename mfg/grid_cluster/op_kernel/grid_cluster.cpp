// Kernel侧核函数实现
// GridCluster: cluster[i] = sum_d trunc((pos[i,d]-start[d])/size[d]) * stride[d]
//
// 做法(逐点带宽型):
//   1) 多核按点均分, 段内按 chunk 分块, TQue 双缓冲流水;
//   2) pos 按 [N,D] 连续搬入, 用 Gather 按字节偏移去交错出各维 x_d (f16 先升 f32);
//   3) 每维: Adds(-start_d) -> Div(size_d 向量) -> Cast(TRUNC) 得 int32 体素坐标, 再乘 stride_d 累加;
//   4) int32 结果 Cast 成 int64 (CAST_NONE, 910B 支持 int32->int64) 后写回。
#include "kernel_operator.h"

#include "grid_cluster_tiling.h"
#include "tiling_key_grid_cluster.h"

using namespace AscendC;

namespace {
constexpr int32_t UB_PAD = 64;
}  // namespace

template <typename T>
class KernelGridCluster {
public:
    __aicore__ inline KernelGridCluster() {}

    __aicore__ inline void Init(GM_ADDR pos, GM_ADDR size, GM_ADDR start, GM_ADDR end, GM_ADDR out,
                                const GridClusterTilingData &t) {
        n_ = static_cast<int64_t>(t.n);
        d_ = static_cast<int32_t>(t.dim);
        chunk_ = static_cast<int64_t>(t.chunk);
        const int64_t per = static_cast<int64_t>(t.blockFactor);
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        int64_t s = blk * per;
        int64_t en = s + per;
        if (s > n_) {
            s = n_;
        }
        if (en > n_) {
            en = n_;
        }
        startRow_ = s;
        endRow_ = en;

        posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(pos), n_ * d_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(out), n_ * 2);
        GlobalTensor<T> sizeGm;
        GlobalTensor<T> startGm;
        GlobalTensor<T> endGm;
        sizeGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(size), d_);
        startGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(start), d_);
        endGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(end), d_);

        int64_t stride = 1;
        for (int32_t d = 0; d < d_; ++d) {
            const float sv = static_cast<float>(sizeGm.GetValue(d));
            const float stv = static_cast<float>(startGm.GetValue(d));
            const float edv = static_cast<float>(endGm.GetValue(d));
            startF_[d] = stv;
            sizeF_[d] = sv;
            int64_t nv = static_cast<int64_t>((edv - stv) / sv) + 1;  // trunc (非负)
            if (nv < 1) {
                nv = 1;
            }
            strideF_[d] = stride;
            stride *= nv;
        }

        const int32_t ch = static_cast<int32_t>(chunk_);
        const int32_t sz = ch + UB_PAD;
        pipe_.InitBuffer(rawQue_, 2, static_cast<uint32_t>((d_ * chunk_ + UB_PAD) * sizeof(T)));
        if constexpr (sizeof(T) == 2) {
            pipe_.InitBuffer(fBuf_, static_cast<uint32_t>((d_ * chunk_ + UB_PAD) * sizeof(float)));
        }
        pipe_.InitBuffer(dimBuf_, static_cast<uint32_t>(sz * sizeof(float)));
        pipe_.InitBuffer(sizeBuf_, static_cast<uint32_t>(sz * sizeof(float)));
        pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(sz * sizeof(int32_t)));
        pipe_.InitBuffer(vBuf_, static_cast<uint32_t>(sz * sizeof(int32_t)));
        pipe_.InitBuffer(obQue_, 2, static_cast<uint32_t>((2 * ch + 2 * UB_PAD) * sizeof(int32_t)));
        pipe_.InitBuffer(wBuf_, static_cast<uint32_t>((2 * ch + 2 * UB_PAD) * sizeof(int32_t)));
        pipe_.InitBuffer(offBuf_, static_cast<uint32_t>((6 * sz) * sizeof(uint32_t)));

        // 各维去交错基准偏移 (vgather offset 单位=字节): base[k] = k*(D*4)
        LocalTensor<int32_t> base = offBuf_.Get<int32_t>();
        CreateVecIndex(base, static_cast<int32_t>(0), static_cast<uint32_t>(ch));
        Muls(base, base, static_cast<int32_t>(d_ * static_cast<int32_t>(sizeof(float))), ch);

        // int64 交织: out[k] = (k%2==0) ? v[k/2] : 0, 即 W=[v|0] 后 Gather
        LocalTensor<int32_t> offI = offBuf_.Get<int32_t>()[2 * sz];
        LocalTensor<int32_t> scratch = offBuf_.Get<int32_t>()[4 * sz];
        LocalTensor<float> fw = wBuf_.Get<float>();
        const int32_t twoN = 2 * ch;
        CreateVecIndex(offI, static_cast<int32_t>(0), static_cast<uint32_t>(twoN));
        Cast(fw, offI, RoundMode::CAST_NONE, twoN);
        Muls(fw, fw, 0.5f, twoN);
        Cast(scratch, fw, RoundMode::CAST_FLOOR, twoN);
        Muls(offI, offI, static_cast<int32_t>(4 * ch), twoN);
        Muls(scratch, scratch, -static_cast<int32_t>(8 * ch - 4), twoN);
        Add(offI, offI, scratch, twoN);
        // W 的高半段恒 0 (float 视图清零, 全 0 位即 int32 的 0)
        Muls(fw[ch], fw[ch], 0.0f, ch);
    }

    __aicore__ inline void Process() {
        if (startRow_ >= endRow_) {
            return;
        }
        for (int64_t i = startRow_; i < endRow_; i += chunk_) {
            const int64_t c = (i + chunk_ < endRow_) ? chunk_ : (endRow_ - i);
            DoChunk(i, c);
        }
    }

private:
    __aicore__ inline void DoChunk(int64_t rowStart, int64_t cnt64) {
        const int32_t cnt = static_cast<int32_t>(cnt64);
        // vgather 的 count 需按 64 对齐(小 count 会触发 vector core exception), 多算的部分不写回
        int32_t cntOp = ((cnt + 63) / 64) * 64;
        const int32_t ch0 = static_cast<int32_t>(chunk_);
        if (cntOp > ch0) {
            cntOp = ch0;
        }
        const int32_t ch = static_cast<int32_t>(chunk_);
        const int32_t sz = ch + UB_PAD;

        LocalTensor<T> raw = rawQue_.AllocTensor<T>();
        {
            const int64_t off = rowStart * d_;
            const int64_t elems = cnt64 * d_;
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(elems * sizeof(T));
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPadExtParams<T> pp;
            DataCopyPad(raw, posGm_[static_cast<int32_t>(off)], p, pp);
        }
        rawQue_.EnQue(raw);
        LocalTensor<T> rawD = rawQue_.DeQue<T>();

        LocalTensor<float> src;
        if constexpr (sizeof(T) == 2) {
            LocalTensor<float> fb = fBuf_.Get<float>();
            Cast(fb, rawD, RoundMode::CAST_NONE, cnt * d_);
            src = fb;
        } else {
            src = rawD.template ReinterpretCast<float>();
        }

        LocalTensor<int32_t> base = offBuf_.Get<int32_t>();
        LocalTensor<int32_t> offCur = offBuf_.Get<int32_t>()[sz];
        LocalTensor<float> dimL = dimBuf_.Get<float>();
        LocalTensor<float> szL = sizeBuf_.Get<float>();
        LocalTensor<int32_t> acc = accBuf_.Get<int32_t>();
        LocalTensor<int32_t> vL = vBuf_.Get<int32_t>();

        for (int32_t d = 0; d < d_; ++d) {
            Adds(offCur, base, static_cast<int32_t>(d * static_cast<int32_t>(sizeof(float))), ch);
            Gather(dimL, src, offCur.template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(cntOp));
            Adds(dimL, dimL, -startF_[d], cntOp);
            Duplicate(szL, sizeF_[d], cntOp);
            Div(dimL, dimL, szL, cntOp);
            Cast(vL, dimL, RoundMode::CAST_TRUNC, cntOp);
            if (d == 0) {
                Muls(acc, vL, static_cast<int32_t>(strideF_[0]), cntOp);
            } else {
                Muls(vL, vL, static_cast<int32_t>(strideF_[d]), cntOp);
                Add(acc, acc, vL, cntOp);
            }
        }

        LocalTensor<int32_t> w = wBuf_.Get<int32_t>();
        Adds(w, acc, static_cast<int32_t>(0), cntOp);
        LocalTensor<int32_t> ob = obQue_.AllocTensor<int32_t>();
        Gather(ob, w, offBuf_.Get<int32_t>()[2 * sz].template ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(2 * cntOp));
        obQue_.EnQue(ob);

        rawQue_.FreeTensor(rawD);

        LocalTensor<int32_t> obd = obQue_.DeQue<int32_t>();

        // 写回 int64
        if ((((cnt64 * 8) % 32) == 0) && (((rowStart * 8) % 32) == 0)) {
            DataCopy(outGm_[static_cast<int32_t>(rowStart * 2)], obd, static_cast<uint32_t>(2 * cnt));
        } else {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(cnt64 * 8);
            p.srcStride = 0;
            p.dstStride = 0;
            p.rsv = 0;
            DataCopyPad(outGm_[static_cast<int32_t>(rowStart * 2)], obd, p);
        }
        obQue_.FreeTensor(obd);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> rawQue_;
    TBuf<TPosition::VECCALC> fBuf_;
    TBuf<TPosition::VECCALC> dimBuf_;
    TBuf<TPosition::VECCALC> sizeBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TQue<TPosition::VECOUT, 2> obQue_;
    TBuf<TPosition::VECCALC> wBuf_;
    TBuf<TPosition::VECCALC> offBuf_;
    GlobalTensor<T> posGm_;
    GlobalTensor<int32_t> outGm_;
    float startF_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sizeF_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    int64_t strideF_[4] = {1, 1, 1, 1};
    int64_t n_ = 0;
    int64_t chunk_ = 0;
    int64_t startRow_ = 0;
    int64_t endRow_ = 0;
    int64_t outBytesPer_ = 8;
    int32_t d_ = 3;
};

template <typename DT_POS>
__global__ __aicore__ void grid_cluster(GM_ADDR pos, GM_ADDR size, GM_ADDR start, GM_ADDR end, GM_ADDR out,
                                        GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GridClusterTilingData);
    GET_TILING_DATA_WITH_STRUCT(GridClusterTilingData, tiling_data, tiling);
    KernelGridCluster<DT_POS> op;
    op.Init(pos, size, start, end, out, tiling_data);
    op.Process();
}

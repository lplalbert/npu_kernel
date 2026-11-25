/*!
 * \file index.h
 * \brief Index 算子 kernel 类定义
 *
 * 语义（与官方 golden gen_data.py::impl 一致）：indices 为单一索引张量，对每个被
 * indexed_sizes 标记为索引的维度使用【同一个】下标值（torch 高级索引 x[idx, idx, ...]）。
 *   out[k, kept...] = x[..., (v_k for each indexed dim), kept...]，k 为 indices 展平下标。
 *
 * 布局：
 *   - 连续轴：out = [前缀保留维] + [indices.shape] + [后缀保留维]
 *   - 非连续轴：out = [indices.shape] + [按原顺序的保留维]
 * innerLen = 最后一个被索引维之后的连续块（这些维全部保留且内存连续）；
 * 其余保留维（< lastIdx）为 outer 维，共 outerCount 个组合。
 * 每个 (k, o) 对应从 x[src .. src+innerLen) 连续搬 innerLen 个元素到 y[dst .. dst+innerLen)。
 *
 * 快速路径：非连续轴布局下 dst = r*innerLen（r 为行号，全局连续），且 innerLen 32B 对齐时，
 *   一次批量搬 B 行（每行一次 MTE2 到 UB 连续槽），再一次 DataCopyPad 连续写出，降低指令数。
 * 所有路径都只有 DataCopyPad(MTE2) + DataCopyPad(MTE3)，UB 缓冲由 TQueBind<VECIN,VECOUT> 双向复用，
 * 不做任何 LocalTensor->LocalTensor 的 DataCopy（该路径在 CANN 9.0 非受支持拷贝路径）。
 */

#ifndef INDEX_H
#define INDEX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "index_tiling_data.h"
#include "index_tiling_key.h"

namespace NsIndex {

using namespace AscendC;

template <typename T>
class Index {
public:
    __aicore__ inline Index(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indexed_sizes, GM_ADDR indices, GM_ADDR y, const IndexTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t GetIdxVal(int64_t k);
    __aicore__ inline int64_t ComputeSrc(int64_t r);
    __aicore__ inline int64_t Gcd(int64_t a, int64_t b);
    __aicore__ inline void CopyRow(int64_t src, int64_t dst, int64_t n);
    __aicore__ inline void ProcessFast(int64_t s, int64_t e);
    __aicore__ inline void ProcessContig2D(int64_t blockId, int64_t blockNum);

private:
    TPipe pipe;
    // Pure data-movement operator: GM -> UB -> GM through ONE bound queue (VECIN+VECOUT share the
    // same UB buffer).  This is the CANN-recommended pattern for copy-only operators and avoids any
    // LocalTensor -> LocalTensor DataCopy (VECIN -> VECOUT is not a supported local copy path on
    // CANN 9.0, where it can raise a vector-core exception / produce corrupted data).
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> ioQ;
    TBuf<QuePosition::VECCALC> maskUb_;   // indexed_sizes staging (int32 words)

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int64_t> maskGm;
    GlobalTensor<int32_t> maskGm32;   // same bytes as maskGm, read 32 bits at a time
    GlobalTensor<int64_t> idx64Gm;
    GlobalTensor<int32_t> idx32Gm;

    IndexTilingData td_;
    int64_t shape_[INDEX_MAX_DIM] = {0};
    int64_t stride_[INDEX_MAX_DIM] = {0};
    int64_t mask_[INDEX_MAX_DIM] = {0};
    int64_t outerDims_[INDEX_MAX_DIM] = {0};

    int64_t dimNum_ = 0;
    int64_t idxNum_ = 0;
    int64_t firstIdx_ = -1;
    int64_t lastIdx_ = -1;
    int64_t innerLen_ = 1;
    int64_t outerNum_ = 0;
    int64_t outerCount_ = 1;
    int64_t kTotal_ = 1;
    int64_t outContig_ = 0;
    int64_t slotElems_ = 1;
    int64_t alignElem_ = 1;
    int64_t batchRows_ = 1;
    int64_t chunkElems_ = 1;
    bool idxIs64_ = true;
    bool noIndex_ = false;
    int64_t idxMax_ = 0;   // flat index space size P (clamp bound)
};

template <typename T>
__aicore__ inline int64_t Index<T>::GetIdxVal(int64_t k)
{
    int64_t v;
    if (idxIs64_) {
        v = idx64Gm.GetValue(k);
    } else {
        v = static_cast<int64_t>(idx32Gm.GetValue(k));
    }
    // Defensive clamp: valid flat indices are 0 <= v < P; clamping keeps every source offset in x
    // even if a caller hands us a value outside the contract (protects against OOB reads).
    if (v < 0) {
        v = 0;
    }
    if (idxMax_ > 0 && v >= idxMax_) {
        v = idxMax_ - 1;
    }
    return v;
}

template <typename T>
__aicore__ inline int64_t Index<T>::ComputeSrc(int64_t r)
{
    int64_t k = (outerCount_ > 0) ? (r / outerCount_) : r;
    int64_t o = (outerCount_ > 0) ? (r % outerCount_) : 0;
    int64_t v = GetIdxVal(k);
    // Semantics of the reference impl (spec 3.1): the single indices tensor is used for EVERY
    // indexed dim (x[idx, idx, ...]); v is contractually < min(size of indexed dims).
    int64_t src = 0;
    for (int64_t i = 0; i < dimNum_; ++i) {
        if (mask_[i]) {
            src += v * stride_[i];
        }
    }
    int64_t rem = o;
    for (int64_t j = outerNum_ - 1; j >= 0; --j) {
        int64_t d = outerDims_[j];
        int64_t c = rem % shape_[d];
        rem /= shape_[d];
        src += c * stride_[d];
    }
    return src;
}

template <typename T>
__aicore__ inline void Index<T>::Init(GM_ADDR x, GM_ADDR indexed_sizes, GM_ADDR indices, GM_ADDR y, const IndexTilingData* tilingData)
{
    td_ = *tilingData;
    dimNum_ = td_.dimNum;
    if (dimNum_ > INDEX_MAX_DIM) {
        dimNum_ = INDEX_MAX_DIM;
    }
    idxNum_ = td_.idxNum;
    idxIs64_ = (td_.idxBytes == 8);
    for (int64_t i = 0; i < dimNum_; ++i) {
        shape_[i] = td_.shape[i];
        stride_[i] = td_.stride[i];
    }

    xGm.SetGlobalBuffer((__gm__ T*)x, td_.xElems > 0 ? td_.xElems : 1);
    int64_t yElems = td_.outElems;
    if (yElems < 0) {
        yElems = 0;
    }
    yGm.SetGlobalBuffer((__gm__ T*)y, yElems > 0 ? yElems : 1);
    maskGm.SetGlobalBuffer((__gm__ int64_t*)indexed_sizes, dimNum_ > 0 ? dimNum_ : 1);
    if (idxIs64_) {
        idx64Gm.SetGlobalBuffer((__gm__ int64_t*)indices, idxNum_ > 0 ? idxNum_ : 1);
    } else {
        idx32Gm.SetGlobalBuffer((__gm__ int32_t*)indices, idxNum_ > 0 ? idxNum_ : 1);
    }

    maskGm32.SetGlobalBuffer((__gm__ int32_t*)indexed_sizes, (dimNum_ > 0 ? 2 * dimNum_ : 2));
    // Read indexed_sizes through UB (MTE2) instead of 64-bit GM scalar loads: we only need the
    // "is non-zero" bit, so copy the raw bytes as int32 words and OR the two halves of each int64.
    pipe.InitBuffer(maskUb_, 256);
    {
        const int64_t nWords = (dimNum_ > 0) ? (2 * dimNum_) : 2;
        LocalTensor<int32_t> mb = maskUb_.Get<int32_t>();
        DataCopyExtParams mp;
        mp.blockCount = 1;
        mp.blockLen = static_cast<uint32_t>(nWords * 4);
        mp.srcStride = 0;
        mp.dstStride = 0;
        mp.rsv = 0;
        DataCopyPad(mb, maskGm32, mp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        for (int64_t i = 0; i < dimNum_; ++i) {
            mask_[i] = (mb.GetValue(2 * i) != 0 || mb.GetValue(2 * i + 1) != 0) ? 1 : 0;
        }
        PipeBarrier<PIPE_ALL>();
    }

    firstIdx_ = -1;
    lastIdx_ = -1;
    for (int64_t i = 0; i < dimNum_; ++i) {
        if (mask_[i]) {
            if (firstIdx_ < 0) {
                firstIdx_ = i;
            }
            lastIdx_ = i;
        }
    }

    // same-index semantics: the largest contract-valid index is min(size of indexed dims) - 1
    idxMax_ = 0;
    for (int64_t i = 0; i < dimNum_; ++i) {
        if (mask_[i] && (idxMax_ == 0 || shape_[i] < idxMax_)) {
            idxMax_ = shape_[i];
        }
    }
    if (idxMax_ <= 0) {
        idxMax_ = 0;
    }

    if (firstIdx_ < 0) {
        noIndex_ = true;
        innerLen_ = td_.xElems;
        outerCount_ = 1;
        outerNum_ = 0;
        kTotal_ = td_.xElems;
        outContig_ = 0;
    } else {
        innerLen_ = 1;
        for (int64_t i = lastIdx_ + 1; i < dimNum_; ++i) {
            innerLen_ *= shape_[i];
        }
        outerNum_ = 0;
        outerCount_ = 1;
        for (int64_t i = 0; i < lastIdx_; ++i) {
            if (!mask_[i]) {
                outerDims_[outerNum_++] = i;
                outerCount_ *= shape_[i];
            }
        }
        kTotal_ = 1;
        for (int64_t i = 0; i < dimNum_; ++i) {
            if (!mask_[i]) {
                kTotal_ *= shape_[i];
            }
        }
        outContig_ = 1;
        for (int64_t i = 0; i < dimNum_; ++i) {
            bool inBlock = (i >= firstIdx_ && i <= lastIdx_);
            if (inBlock && !mask_[i]) {
                outContig_ = 0;
            }
            if (!inBlock && mask_[i]) {
                outContig_ = 0;
            }
        }
    }
    if (innerLen_ < 1) {
        innerLen_ = 1;
    }
    if (outerCount_ < 1 && !noIndex_) {
        outerCount_ = 1;
    }

    int64_t al = 32 / static_cast<int64_t>(sizeof(T));
    if (al < 1) {
        al = 1;
    }
    alignElem_ = al;
    int64_t cap = 16384 / static_cast<int64_t>(sizeof(T));
    if (cap < al) {
        cap = al;
    }
    int64_t sl = innerLen_;
    if (sl > cap) {
        sl = cap;
    }
    slotElems_ = ((sl + al - 1) / al) * al;
    if (slotElems_ < al) {
        slotElems_ = al;
    }

    int64_t targetElems = 32768 / static_cast<int64_t>(sizeof(T));
    if (targetElems < slotElems_) {
        targetElems = slotElems_;
    }
    batchRows_ = targetElems / slotElems_;
    if (batchRows_ < 1) {
        batchRows_ = 1;
    }
    chunkElems_ = slotElems_ * batchRows_;

    pipe.InitBuffer(ioQ, 2, static_cast<uint32_t>(chunkElems_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Index<T>::CopyRow(int64_t src, int64_t dst, int64_t n)
{
    int64_t done = 0;
    while (done < n) {
        int64_t c = n - done;
        if (c > slotElems_) {
            c = slotElems_;
        }
        LocalTensor<T> t = ioQ.template AllocTensor<T>();
        DataCopyExtParams ip;
        ip.blockCount = 1;
        ip.blockLen = static_cast<uint32_t>(c * static_cast<int64_t>(sizeof(T)));
        ip.srcStride = 0;
        ip.dstStride = 0;
        ip.rsv = 0;
        DataCopyPad(t, xGm[src + done], ip, DataCopyPadExtParams<T>{false, 0, 0, 0});
        ioQ.EnQue(t);

        LocalTensor<T> tv = ioQ.template DeQue<T>();
        DataCopyExtParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(c * static_cast<int64_t>(sizeof(T)));
        op.srcStride = 0;
        op.dstStride = 0;
        op.rsv = 0;
        DataCopyPad(yGm[dst + done], tv, op);
        ioQ.EnQue(tv);
        tv = ioQ.template DeQue<T>();
        ioQ.FreeTensor(tv);

        done += c;
    }
}

template <typename T>
__aicore__ inline void Index<T>::ProcessFast(int64_t s, int64_t e)
{
    int64_t r = s;
    while (r < e) {
        int64_t b = e - r;
        if (b > batchRows_) {
            b = batchRows_;
        }
        LocalTensor<T> big = ioQ.template AllocTensor<T>();
        for (int64_t i = 0; i < b; ++i) {
            int64_t src = ComputeSrc(r + i);
            DataCopyExtParams ip;
            ip.blockCount = 1;
            ip.blockLen = static_cast<uint32_t>(innerLen_ * static_cast<int64_t>(sizeof(T)));
            ip.srcStride = 0;
            ip.dstStride = 0;
            ip.rsv = 0;
            DataCopyPad(big[static_cast<int32_t>(i * slotElems_)], xGm[src], ip, DataCopyPadExtParams<T>{false, 0, 0, 0});
        }
        ioQ.EnQue(big);

        LocalTensor<T> bv = ioQ.template DeQue<T>();
        DataCopyExtParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(b * innerLen_ * static_cast<int64_t>(sizeof(T)));
        op.srcStride = 0;
        op.dstStride = 0;
        op.rsv = 0;
        DataCopyPad(yGm[r * innerLen_], bv, op);
        ioQ.EnQue(bv);
        bv = ioQ.template DeQue<T>();
        ioQ.FreeTensor(bv);

        r += b;
    }
}

template <typename T>
__aicore__ inline void Index<T>::ProcessContig2D(int64_t blockId, int64_t blockNum)
{
    // 连续轴布局且 outerCount>1：固定 k 时，outer(前缀保留维) 连续，源地址步长恒为 BlockStride，
    // 目标块步长 M*innerLen。用 2D 跨步 DataCopyPad 一次搬 outerCount 个内块。
    int64_t M = idxNum_;
    int64_t BlockStride = 1;
    for (int64_t i = firstIdx_; i < dimNum_; ++i) {
        BlockStride *= shape_[i];
    }
    int64_t groupMax = chunkElems_ / innerLen_;
    if (groupMax < 1) {
        groupMax = 1;
    }
    int64_t perK = (M + blockNum - 1) / blockNum;
    int64_t ks = blockId * perK;
    int64_t ke = ks + perK;
    if (ks > M) {
        ks = M;
    }
    if (ke > M) {
        ke = M;
    }
    for (int64_t k = ks; k < ke; ++k) {
        int64_t v = GetIdxVal(k);
        int64_t base = 0;
        for (int64_t i = 0; i < dimNum_; ++i) {
            if (mask_[i]) {
                base += v * stride_[i];
            }
        }
        int64_t o = 0;
        while (o < outerCount_) {
            int64_t g = outerCount_ - o;
            if (g > groupMax) {
                g = groupMax;
            }
            LocalTensor<T> ub = ioQ.template AllocTensor<T>();
            DataCopyExtParams ip;
            ip.blockCount = static_cast<uint16_t>(g);
            ip.blockLen = static_cast<uint32_t>(innerLen_ * static_cast<int64_t>(sizeof(T)));
            ip.srcStride = static_cast<uint32_t>((BlockStride - innerLen_) * static_cast<int64_t>(sizeof(T)));
            ip.dstStride = 0;
            ip.rsv = 0;
            DataCopyPad(ub, xGm[base + o * BlockStride], ip, DataCopyPadExtParams<T>{false, 0, 0, 0});
            ioQ.EnQue(ub);

            LocalTensor<T> sv = ioQ.template DeQue<T>();
            DataCopyExtParams op;
            op.blockCount = static_cast<uint16_t>(g);
            op.blockLen = static_cast<uint32_t>(innerLen_ * static_cast<int64_t>(sizeof(T)));
            op.srcStride = 0;
            op.dstStride = static_cast<uint32_t>((M * innerLen_ - innerLen_) * static_cast<int64_t>(sizeof(T)));
            op.rsv = 0;
            DataCopyPad(yGm[k * innerLen_ + o * (M * innerLen_)], sv, op);
            ioQ.EnQue(sv);
            sv = ioQ.template DeQue<T>();
            ioQ.FreeTensor(sv);

            o += g;
        }
    }
}

template <typename T>
__aicore__ inline int64_t Index<T>::Gcd(int64_t a, int64_t b)
{
    a = (a < 0) ? -a : a;
    b = (b < 0) ? -b : b;
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

template <typename T>
__aicore__ inline void Index<T>::Process()
{
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum < 1) {
        blockNum = 1;
    }
    int64_t blockId = static_cast<int64_t>(GetBlockIdx());

    if (noIndex_ || firstIdx_ < 0) {
        if (blockId == 0 && td_.outElems > 0 && innerLen_ > 0) {
            CopyRow(0, 0, innerLen_);
        }
        return;
    }

    // 空输出 / 空张量 / 空 indices：直接跳过，绝不兜底成 1 行去读不存在的索引或写越界
    if (td_.outElems <= 0 || td_.xElems <= 0 || idxNum_ <= 0 || outerCount_ <= 0 || innerLen_ <= 0) {
        return;
    }

    int64_t rows = idxNum_ * outerCount_;
    int64_t esz = static_cast<int64_t>(sizeof(T));
    // 行内块长度是否为 32B 整数倍：若为真，任一行 [dst, dst+innerLen) 独占若干完整 32B 扇区，
    // 且任一行 dst 都是 innerLen 的整数倍 => 多核按行切分也不会让两核共享同一 32B 扇区。
    bool aligned = ((innerLen_ * esz) % 32) == 0;
    // dst 是否随行号线性连续：非连续轴布局恒为连续；连续轴且 outerCount==1 也连续。
    bool rowContig = (outContig_ == 0) || (outerCount_ == 1);

    bool fast = (slotElems_ == innerLen_) && rowContig; // slotElems==innerLen 蕴含 aligned
    if (fast) {
        int64_t per = (rows + blockNum - 1) / blockNum;
        int64_t s = blockId * per;
        int64_t e = s + per;
        if (s > rows) {
            s = rows;
        }
        if (e > rows) {
            e = rows;
        }
        ProcessFast(s, e);
        return;
    }
    if (outContig_ && outerCount_ > 1 && slotElems_ == innerLen_) {
        ProcessContig2D(blockId, blockNum);
        return;
    }

    // ---- 逐行回退路径（innerLen 非 32B 对齐）----
    int64_t s;
    int64_t e;
    if (!rowContig) {
        // 连续轴且 outerCount>1：dst 被置换，非对齐块可能让不同核落在同一 32B 扇区 -> 单核保证正确/确定
        if (blockId != 0) {
            return;
        }
        s = 0;
        e = rows;
    } else {
        // dst = r*innerLen 连续：把每核负责的行数取为 unit 的整数倍，使每核写起点 dst 字节数 32B 对齐，
        // 从而相邻核不会共享同一 32B 扇区（消除潜在读改写竞争），同时保留多核并行。
        int64_t g = Gcd(32, innerLen_ * esz);
        int64_t unit = 32 / g; // 非对齐时 g<32 => unit>=2
        if (unit < 1) {
            unit = 1;
        }
        int64_t per = (rows + blockNum - 1) / blockNum;
        per = ((per + unit - 1) / unit) * unit;
        if (per < 1) {
            per = 1;
        }
        s = blockId * per;
        e = s + per;
        if (s > rows) {
            s = rows;
        }
        if (e > rows) {
            e = rows;
        }
    }

    for (int64_t r = s; r < e; ++r) {
        int64_t src = ComputeSrc(r);
        int64_t dst;
        if (outContig_) {
            int64_t k = r / outerCount_;
            int64_t o = r % outerCount_;
            dst = o * (idxNum_ * innerLen_) + k * innerLen_;
        } else {
            dst = r * innerLen_;
        }
        CopyRow(src, dst, innerLen_);
    }
}

} // namespace NsIndex
#endif // INDEX_H

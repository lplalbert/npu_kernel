/*!
 * \file confusion_mul_grad.h
 * \brief ConfusionMulGrad kernel class
 *
 *   out0 = fp32(in0) * fp32(in1)                -> cast back to T
 *   p    = fp32(in1) * fp32(in2)
 *   out1 = ReduceSum(p, axis, keep_dims)        fp32 accumulation
 *
 * axis in {[1],[0],[0,1]} => three data-parallel decompositions:
 *   mode 0 (axis=[1]) : rows split across cores, per-row fold reduce (no cross-core)
 *   mode 1 (axis=[0]) : columns split across cores, each core reduces all rows (no cross-core)
 *   mode 2 (axis=[0,1]): flat split; partial sums merged by a 32B atomic add with the
 *                        (P0 - g) offset trick (avoids SyncAll which faults 507035 here)
 */

#ifndef CONFUSIONMULGRAD_H
#define CONFUSIONMULGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "confusion_mul_grad_tiling_data.h"
#include "confusion_mul_grad_tiling_key.h"

namespace NsConfusionMulGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t A64 = 64;
constexpr int32_t ROWSUM_B = 512;

__aicore__ inline int64_t RoundUp64(int64_t x) { return ((x + A64 - 1) >> 6) << 6; }

template <typename T>
class ConfusionMulGrad {
public:
    __aicore__ inline ConfusionMulGrad(){};

    __aicore__ inline void Init(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR output0,
                                GM_ADDR output1, const ConfusionMulGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t Align32Elems(int64_t cnt) const {
        const int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
        return ((bytes + 31) / 32 * 32) / static_cast<int64_t>(sizeof(T));
    }
    __aicore__ inline LocalTensor<float> GetOF() {
        if constexpr (std::is_same<T, float>::value) {
            return oq_.AllocTensor<float>();
        } else {
            return tO_.Get<float>();
        }
    }
    __aicore__ inline void CopyIn1D(LocalTensor<T> &dst, GlobalTensor<T> &gm, int64_t off, int64_t cnt);
    __aicore__ inline LocalTensor<T> Load1D(TQue<TPosition::VECIN, BUFFER_NUM> &q, GlobalTensor<T> &gm,
                                            int64_t off, int64_t cnt, int64_t np);
    __aicore__ inline LocalTensor<T> Load2D(TQue<TPosition::VECIN, BUFFER_NUM> &q, GlobalTensor<T> &gm,
                                            int64_t off, int64_t rows, int64_t cw, int64_t N);
    template <typename LT>
    __aicore__ inline void Store1D(LT &src, GlobalTensor<T> &gm, int64_t off, int64_t cnt);
    template <typename LT>
    __aicore__ inline void Store2D(LT &src, GlobalTensor<T> &gm, int64_t off, int64_t rows, int64_t cw, int64_t N);
    __aicore__ inline void Prep(LocalTensor<T> &a0, LocalTensor<T> &a1, LocalTensor<T> &a2,
                                LocalTensor<float> &aF, LocalTensor<float> &bF, LocalTensor<float> &cF,
                                int64_t cnt);
    __aicore__ inline void EmitOut0(LocalTensor<float> &oF, int64_t off, int64_t cnt, int64_t np);
    __aicore__ inline void EmitOut0_2D(LocalTensor<float> &oF, int64_t off, int64_t rows, int64_t cw, int64_t N);
    __aicore__ inline float FoldSum(LocalTensor<float> &a, int64_t n);
    __aicore__ inline void KahanAdd(LocalTensor<float> &acc, LocalTensor<float> &comp,
                                    LocalTensor<float> &y, uint32_t n);
    __aicore__ inline void ProcessRow();
    __aicore__ inline void ProcessRowOld();
    __aicore__ inline void ProcessRowNpu();
    __aicore__ inline void ProcessColOld();
    __aicore__ inline void ColGroupOld(int64_t col, int64_t ncols);
    __aicore__ inline void ProcessCol();
    __aicore__ inline void ProcessColWide32(int64_t col);
    __aicore__ inline void ProcessColTail8(int64_t col, int64_t ncols);

    __aicore__ inline void ProcessColScalar();
    __aicore__ inline void ProcessColChunk(int64_t c0, int64_t c1);
    __aicore__ inline void ProcessColRowBlock();
    __aicore__ inline void ProcessAll();
    __aicore__ inline void FlushOut1(LocalTensor<float> &rsB, int64_t rowStart, int64_t n);
    // torch CPU cascade_sum parameters (SumKernel.cpp, Vectorized<float>::size()==8)
    static __aicore__ inline int64_t LevelPower(int64_t size) {
        int64_t cl = 0;
        while ((static_cast<int64_t>(1) << cl) < size) {
            ++cl;
        }
        const int64_t t = cl / 4;
        return (t > 4) ? t : 4;
    }
    // torch CPU sum (aten/src/ATen/native/cpu/SumKernel.cpp cascade_sum) constants.
    // kV = Vectorized<float>::size() on the machine that generates the golden:
    //   8  -> AVX2 / this aarch64 build (cascade block = 4*kV = 32 elements)
    //   16 -> AVX512 (cascade block = 64 elements)
    // Only this constant (plus nothing else) distinguishes the V8 / V16 packages.
    static constexpr int64_t kV = 8;
    // false -> cascade_sum (torch >= 2.1); true -> pre-cascade binary_kernel_reduce_vec
    // structure: 4 vector accumulators, (a0+a1)+(a2+a3) combine, tail after, no cascade.
    static constexpr bool kOldKernel = false;
    // true -> axis=[1] uses the torch_npu / aclnn ReduceSum order measured on 910B2:
    //   acc[j] += x[tile*4096 + j]  (tiles of 4096 elements)
    //   then halving tree fold 4096 -> 8 lanes, then pairwise over the last 8 lanes.
    static constexpr bool kNpuKernel = false;
    static constexpr int64_t kNpuTile = 4096;
    static constexpr int64_t kBlk = 4 * kV;        // elements per cascade block
    static constexpr int64_t kAcc = 16 * kV;       // 4 levels x kBlk lanes
    static __aicore__ inline int64_t LoadCols() {
        // MTE needs a 32B row block: max(kV, 16) columns
        return (kV >= 16) ? kV : 16;
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> q0_;
    TQue<QuePosition::VECIN, BUFFER_NUM> q1_;
    TQue<QuePosition::VECIN, BUFFER_NUM> q2_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> oq_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> tP_;
    TBuf<QuePosition::VECCALC> tA_;
    TBuf<QuePosition::VECCALC> tB_;
    TBuf<QuePosition::VECCALC> tC_;
    TBuf<QuePosition::VECCALC> tO_;
    TBuf<QuePosition::VECCALC> rsvBuf_;
    TBuf<QuePosition::VECCALC> rsBuf_;
    TBuf<QuePosition::VECCALC> tOutH_;
    TBuf<QuePosition::VECCALC> tComp_;
    TBuf<QuePosition::VECCALC> gBuf_;
    TBuf<QuePosition::VECCALC> tK_;
    TBuf<QuePosition::VECCALC> tKT_;

    GlobalTensor<T> in0Gm_;
    GlobalTensor<T> in1Gm_;
    GlobalTensor<T> in2Gm_;
    GlobalTensor<T> out0Gm_;
    GlobalTensor<T> out1Gm_;

    ConfusionMulGradTilingData t_;
    int64_t tile_ = 0;
    int64_t blockIdx_ = 0;
    float gRead_ = 0.0f;
};

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::Init(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                 GM_ADDR output0, GM_ADDR output1,
                                                 const ConfusionMulGradTilingData *tilingData) {
    t_ = *tilingData;
    tile_ = static_cast<int64_t>(t_.tileLength);
    in0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input0), t_.totalNum);
    in1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input1), t_.totalNum);
    in2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input2), t_.totalNum);
    out0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output0), t_.totalNum);
    int64_t out1Len = 1;
    if (t_.axisMode == 0) {
        out1Len = t_.M;
    } else if (t_.axisMode == 1) {
        out1Len = t_.N;
    }
    out1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output1), out1Len);
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if constexpr (std::is_same<T, float>::value) {
        if (t_.axisMode == 2 && t_.usedCores > 1) {
            gRead_ = out1Gm_.GetValue(0);
        }
    }
    pipe_.InitBuffer(q0_, BUFFER_NUM, static_cast<uint32_t>(tile_ * sizeof(T)));
    pipe_.InitBuffer(q1_, BUFFER_NUM, static_cast<uint32_t>(tile_ * sizeof(T)));
    pipe_.InitBuffer(q2_, BUFFER_NUM, static_cast<uint32_t>(tile_ * sizeof(T)));
    pipe_.InitBuffer(oq_, BUFFER_NUM, static_cast<uint32_t>(tile_ * sizeof(T)));
    pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(tile_ * sizeof(float)));
    pipe_.InitBuffer(tP_, static_cast<uint32_t>(tile_ * sizeof(float)));
    if constexpr (!std::is_same<T, float>::value) {
        pipe_.InitBuffer(tA_, static_cast<uint32_t>(tile_ * sizeof(float)));
        pipe_.InitBuffer(tB_, static_cast<uint32_t>(tile_ * sizeof(float)));
        pipe_.InitBuffer(tC_, static_cast<uint32_t>(tile_ * sizeof(float)));
        pipe_.InitBuffer(tO_, static_cast<uint32_t>(tile_ * sizeof(float)));
    }
    if (t_.axisMode != 0) {
        pipe_.InitBuffer(tComp_, static_cast<uint32_t>(tile_ * sizeof(float)));
        pipe_.InitBuffer(tK_, static_cast<uint32_t>(tile_ * sizeof(float)));
        pipe_.InitBuffer(tKT_, static_cast<uint32_t>(tile_ * sizeof(float)));
    }
    pipe_.InitBuffer(gBuf_, static_cast<uint32_t>(tile_ * sizeof(float)));
    if constexpr (std::is_same<T, float>::value) {
        if (t_.axisMode == 1 && t_.useRowBlock == 1 && blockIdx_ == 0) {
            LocalTensor<float> gb = gBuf_.Get<float>();
            const int64_t padE = Align32Elems(t_.N) - t_.N;
            // Zero the whole landing buffer first so that the [Align8(N), accLen) tail
            // that DataCopyPad does not touch is still a defined 0 when Sub reads it.
            Duplicate(gb, 0.0f, static_cast<uint32_t>(t_.accLen));
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            DataCopyExtParams gp;
            gp.blockCount = 1;
            gp.blockLen = static_cast<uint32_t>(t_.N * sizeof(float));
            gp.srcStride = 0;
            gp.dstStride = 0;
            gp.rsv = 0;
            DataCopyPadExtParams<float> gpp{padE > 0, 0, static_cast<uint8_t>(padE), 0.0f};
            DataCopyPad(gb, out1Gm_[0], gp, gpp);
            // gb is a DataCopyPad (MTE2) landing buffer that is later read by the
            // vector Sub in ProcessColRowBlock -> needs an explicit MTE2_V event.
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            PipeBarrier<PIPE_ALL>();
        }
    }
    pipe_.InitBuffer(rsvBuf_, static_cast<uint32_t>(A64 * sizeof(float)));
    pipe_.InitBuffer(rsBuf_, static_cast<uint32_t>(ROWSUM_B * sizeof(float)));
    pipe_.InitBuffer(tOutH_, static_cast<uint32_t>(ROWSUM_B * sizeof(T)));
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::CopyIn1D(LocalTensor<T> &dst, GlobalTensor<T> &gm, int64_t off,
                                                     int64_t cnt) {
    const int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
    const int64_t gmb = off * static_cast<int64_t>(sizeof(T));
    if ((bytes & 31) == 0 && (gmb & 31) == 0) {
        DataCopy(dst, gm[off], static_cast<uint32_t>(cnt));
    } else {
        const int64_t padElems = Align32Elems(cnt) - cnt;
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(bytes);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<T> pp{padElems > 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        DataCopyPad(dst, gm[off], p, pp);
    }
}

template <typename T>
__aicore__ inline LocalTensor<T> ConfusionMulGrad<T>::Load1D(TQue<TPosition::VECIN, BUFFER_NUM> &q,
                                                             GlobalTensor<T> &gm, int64_t off, int64_t cnt,
                                                             int64_t np) {
    LocalTensor<T> t = q.AllocTensor<T>();
    if (np > Align32Elems(cnt)) {
        Duplicate(t, static_cast<T>(0), static_cast<uint32_t>(np));
        // The zero fill is a vector write; the DataCopyPad below is MTE2 and writes
        // the same buffer, so drain the vector pipe first (WAR).
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
    }
    CopyIn1D(t, gm, off, cnt);
    q.EnQue(t);
    return q.DeQue<T>();
}

template <typename T>
__aicore__ inline LocalTensor<T> ConfusionMulGrad<T>::Load2D(TQue<TPosition::VECIN, BUFFER_NUM> &q,
                                                             GlobalTensor<T> &gm, int64_t off, int64_t rows,
                                                             int64_t cw, int64_t N) {
    LocalTensor<T> t = q.AllocTensor<T>();
    DataCopyExtParams p;
    p.blockCount = static_cast<uint32_t>(rows);
    p.blockLen = static_cast<uint32_t>(cw * static_cast<int64_t>(sizeof(T)));
    p.srcStride = static_cast<uint32_t>((N - cw) * static_cast<int64_t>(sizeof(T)));
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(t, gm[off], p, pp);
    q.EnQue(t);
    return q.DeQue<T>();
}

template <typename T>
template <typename LT>
__aicore__ inline void ConfusionMulGrad<T>::Store1D(LT &src, GlobalTensor<T> &gm, int64_t off, int64_t cnt) {
    const int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
    const int64_t gmb = off * static_cast<int64_t>(sizeof(T));
    if ((bytes & 31) == 0 && (gmb & 31) == 0) {
        DataCopy(gm[off], src, static_cast<uint32_t>(cnt));
    } else {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(bytes);
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPad(gm[off], src, p);
    }
}

template <typename T>
template <typename LT>
__aicore__ inline void ConfusionMulGrad<T>::Store2D(LT &src, GlobalTensor<T> &gm, int64_t off, int64_t rows,
                                                    int64_t cw, int64_t N) {
    DataCopyExtParams p;
    p.blockCount = static_cast<uint32_t>(rows);
    p.blockLen = static_cast<uint32_t>(cw * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = static_cast<uint32_t>((N - cw) * static_cast<int64_t>(sizeof(T)));
    p.rsv = 0;
    DataCopyPad(gm[off], src, p);
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::Prep(LocalTensor<T> &a0, LocalTensor<T> &a1, LocalTensor<T> &a2,
                                                 LocalTensor<float> &aF, LocalTensor<float> &bF,
                                                 LocalTensor<float> &cF, int64_t cnt) {
    if constexpr (std::is_same<T, float>::value) {
        aF = a0;
        bF = a1;
        cF = a2;
    } else {
        aF = tA_.Get<float>();
        bF = tB_.Get<float>();
        cF = tC_.Get<float>();
        Cast(aF, a0, RoundMode::CAST_NONE, static_cast<uint32_t>(cnt));
        Cast(bF, a1, RoundMode::CAST_NONE, static_cast<uint32_t>(cnt));
        Cast(cF, a2, RoundMode::CAST_NONE, static_cast<uint32_t>(cnt));
    }
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::EmitOut0(LocalTensor<float> &oF, int64_t off, int64_t cnt,
                                                     int64_t np) {
    if constexpr (std::is_same<T, float>::value) {
        oq_.EnQue(oF);
        LocalTensor<T> o = oq_.DeQue<T>();
        Store1D(o, out0Gm_, off, cnt);
        oq_.FreeTensor(o);
    } else {
        LocalTensor<T> o = oq_.AllocTensor<T>();
        Cast(o, oF, RoundMode::CAST_RINT, static_cast<uint32_t>(np));
        oq_.EnQue(o);
        o = oq_.DeQue<T>();
        Store1D(o, out0Gm_, off, cnt);
        oq_.FreeTensor(o);
    }
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::EmitOut0_2D(LocalTensor<float> &oF, int64_t off, int64_t rows,
                                                        int64_t cw, int64_t N) {
    if constexpr (std::is_same<T, float>::value) {
        oq_.EnQue(oF);
        LocalTensor<T> o = oq_.DeQue<T>();
        Store2D(o, out0Gm_, off, rows, cw, N);
        oq_.FreeTensor(o);
    } else {
        LocalTensor<T> o = oq_.AllocTensor<T>();
        Cast(o, oF, RoundMode::CAST_RINT, static_cast<uint32_t>(rows * cw));
        oq_.EnQue(o);
        o = oq_.DeQue<T>();
        Store2D(o, out0Gm_, off, rows, cw, N);
        oq_.FreeTensor(o);
    }
}

template <typename T>
__aicore__ inline float ConfusionMulGrad<T>::FoldSum(LocalTensor<float> &a, int64_t n) {
    LocalTensor<float> rsv = rsvBuf_.Get<float>();
    while (n > A64) {
        const int64_t h = n >> 1;
        Add(a, a, a[static_cast<int32_t>(h)], static_cast<uint32_t>(h));
        n = h;
    }
    WholeReduceSum(rsv, a, static_cast<int32_t>(n), 1, 1, 1, 1);
    PipeBarrier<PIPE_ALL>();
    return rsv.GetValue(0);
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::KahanAdd(LocalTensor<float> &acc, LocalTensor<float> &comp,
                                                     LocalTensor<float> &y, uint32_t n) {
    LocalTensor<float> t = tK_.Get<float>();
    LocalTensor<float> ct = tKT_.Get<float>();
    Sub(y, y, comp, n);        // y = y - comp
    Add(t, acc, y, n);         // t = acc + y
    Sub(ct, t, acc, n);        // ct = t - acc
    Sub(comp, ct, y, n);       // comp = (t - acc) - y
    Adds(acc, t, static_cast<float>(0), n);  // acc = t
}
// axis=[1] with the torch_npu / aclnn ReduceSum order (measured on 910B2, CANN 9.2):
//   * row is walked in tiles of 4096 elements, acc[j] += x[tile*4096 + j]
//   * then a halving tree fold 4096 -> 8, then a pairwise fold of the last 8 lanes
// (bit-exact against torch_npu for N=1024/2053/4099/65536 etc.; N=12289 can differ 1 ulp)
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessRowNpu() {
    const int64_t N = t_.N;
    const int64_t rpc = t_.rowPerCore;
    int64_t rs = blockIdx_ * rpc;
    if (rs > t_.M) rs = t_.M;
    int64_t re = rs + rpc;
    if (re > t_.M) re = t_.M;
    if (rs >= re) return;

    const int64_t KB = kNpuTile;
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    LocalTensor<float> rsB = rsBuf_.Get<float>();
    int64_t bi = 0;
    int64_t rowBase = rs;
    for (int64_t r = rs; r < re; ++r) {
        const int64_t rowOff = r * N;
        Duplicate(acc, 0.0f, static_cast<uint32_t>(KB));
        for (int64_t c0 = 0; c0 < N; c0 += KB) {
            int64_t cnt = N - c0;
            if (cnt > KB) cnt = KB;
            const int64_t np = RoundUp64(cnt);
            LocalTensor<T> t0 = Load1D(q0_, in0Gm_, rowOff + c0, cnt, np);
            LocalTensor<T> t1 = Load1D(q1_, in1Gm_, rowOff + c0, cnt, np);
            LocalTensor<T> t2 = Load1D(q2_, in2Gm_, rowOff + c0, cnt, np);
            LocalTensor<float> aF;
            LocalTensor<float> bF;
            LocalTensor<float> cF;
            Prep(t0, t1, t2, aF, bF, cF, np);
            LocalTensor<float> oF = GetOF();
            Mul(oF, aF, bF, static_cast<uint32_t>(np));
            EmitOut0(oF, rowOff + c0, cnt, np);
            Mul(pf, bF, cF, static_cast<uint32_t>(np));
            Add(acc, acc, pf, static_cast<uint32_t>(cnt));
            q0_.FreeTensor(t0);
            q1_.FreeTensor(t1);
            q2_.FreeTensor(t2);
        }
        int64_t n = KB;
        while (n > 8) {
            const int64_t h = n >> 1;
            Add(acc, acc, acc[static_cast<int32_t>(h)], static_cast<uint32_t>(h));
            n = h;
        }
        PipeBarrier<PIPE_ALL>();
        const float a0 = acc.GetValue(0);
        const float a1 = acc.GetValue(1);
        const float a2 = acc.GetValue(2);
        const float a3 = acc.GetValue(3);
        const float a4 = acc.GetValue(4);
        const float a5 = acc.GetValue(5);
        const float a6 = acc.GetValue(6);
        const float a7 = acc.GetValue(7);
        const float fin = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        rsB.SetValue(static_cast<int32_t>(bi), fin);
        ++bi;
        if (bi == ROWSUM_B) {
            FlushOut1(rsB, rowBase, bi);
            rowBase += bi;
            bi = 0;
        }
    }
    if (bi > 0) {
        FlushOut1(rsB, rowBase, bi);
    }
}

// ---------------------------------------------------------------------------
// Pre-cascade torch CPU sum kernel (aten/src/ATen/native/cpu/Reduce.h
// binary_kernel_reduce_vec -> vectorized_inner_reduction / vectorized_outer_reduction):
//   * 4 vector accumulators (kV lanes each), blocks of 4*kV elements
//   * combine  acc0 = (acc0 + acc1) + (acc2 + acc3)      [pairwise, NOT left fold]
//   * lane fold over the kV lanes, then the scalar tail (N % 4*kV) is added AFTER
//   * no base-16 carry cascade at all
//   * axis=[0]: one plain linear accumulation per column, groups of 4*kV columns,
//     remaining columns are summed sequentially one by one.
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessRowOld() {
    const int64_t N = t_.N;
    const int64_t rpc = t_.rowPerCore;
    int64_t rs = blockIdx_ * rpc;
    if (rs > t_.M) rs = t_.M;
    int64_t re = rs + rpc;
    if (re > t_.M) re = t_.M;
    if (rs >= re) return;

    const int64_t nblk = N / kBlk;     // full 4*kV blocks
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    LocalTensor<float> rsB = rsBuf_.Get<float>();
    int64_t bi = 0;
    int64_t rowBase = rs;
    for (int64_t r = rs; r < re; ++r) {
        const int64_t rowOff = r * N;
        Duplicate(acc, 0.0f, static_cast<uint32_t>(kBlk));
        int64_t i = 0;
        int64_t lastStart = 0;
        for (int64_t c0 = 0; c0 < N; c0 += tile_) {
            int64_t cnt = N - c0;
            if (cnt > tile_) cnt = tile_;
            const int64_t np = RoundUp64(cnt);
            LocalTensor<T> t0 = Load1D(q0_, in0Gm_, rowOff + c0, cnt, np);
            LocalTensor<T> t1 = Load1D(q1_, in1Gm_, rowOff + c0, cnt, np);
            LocalTensor<T> t2 = Load1D(q2_, in2Gm_, rowOff + c0, cnt, np);
            LocalTensor<float> aF;
            LocalTensor<float> bF;
            LocalTensor<float> cF;
            Prep(t0, t1, t2, aF, bF, cF, np);
            LocalTensor<float> oF = GetOF();
            Mul(oF, aF, bF, static_cast<uint32_t>(np));
            EmitOut0(oF, rowOff + c0, cnt, np);
            Mul(pf, bF, cF, static_cast<uint32_t>(np));
            lastStart = c0;
            int64_t blk = cnt / kBlk;
            const int64_t avail = nblk - i;
            if (blk > avail) blk = avail;
            for (int64_t b = 0; b < blk; ++b) {
                for (int j = 0; j < 4; ++j) {
                    Add(acc[j * kV], acc[j * kV],
                        pf[static_cast<int32_t>(b * kBlk + j * kV)], static_cast<uint32_t>(kV));
                }
                ++i;
            }
            q0_.FreeTensor(t0);
            q1_.FreeTensor(t1);
            q2_.FreeTensor(t2);
        }
        // acc0 = (acc0 + acc1) + (acc2 + acc3)
        Add(acc, acc, acc[kV], static_cast<uint32_t>(kV));
        Add(acc[2 * kV], acc[2 * kV], acc[3 * kV], static_cast<uint32_t>(kV));
        Add(acc, acc, acc[2 * kV], static_cast<uint32_t>(kV));
        PipeBarrier<PIPE_ALL>();
        float fin = 0.0f;
        for (int l = 0; l < static_cast<int>(kV); ++l) {
            fin += acc.GetValue(l);
        }
        for (int64_t a = nblk * kBlk; a < N; ++a) {
            fin += pf.GetValue(static_cast<int32_t>(a - lastStart));
        }
        rsB.SetValue(static_cast<int32_t>(bi), fin);
        ++bi;
        if (bi == ROWSUM_B) {
            FlushOut1(rsB, rowBase, bi);
            rowBase += bi;
            bi = 0;
        }
    }
    if (bi > 0) {
        FlushOut1(rsB, rowBase, bi);
    }
}

// one group of kBlk columns, plain linear accumulation over rows
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ColGroupOld(int64_t col, int64_t ncols) {
    const int64_t N = t_.N;
    const int64_t M = t_.M;
    const int64_t win = kBlk;
    int64_t loadCol = col;
    if (loadCol + win > N) {
        loadCol = N - win;
    }
    const int64_t laneBase = col - loadCol;
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    Duplicate(acc, 0.0f, static_cast<uint32_t>(kBlk));
    int64_t R = tile_ / win;
    if (R < 1) R = 1;
    if (R > M) R = M;
    for (int64_t r0 = 0; r0 < M; r0 += R) {
        int64_t rr = M - r0;
        if (rr > R) rr = R;
        const int64_t np = RoundUp64(rr * win);
        LocalTensor<T> t0 = Load2D(q0_, in0Gm_, r0 * N + loadCol, rr, win, N);
        LocalTensor<T> t1 = Load2D(q1_, in1Gm_, r0 * N + loadCol, rr, win, N);
        LocalTensor<T> t2 = Load2D(q2_, in2Gm_, r0 * N + loadCol, rr, win, N);
        LocalTensor<float> aF;
        LocalTensor<float> bF;
        LocalTensor<float> cF;
        Prep(t0, t1, t2, aF, bF, cF, np);
        LocalTensor<float> oF = GetOF();
        Mul(oF, aF, bF, static_cast<uint32_t>(np));
        EmitOut0_2D(oF, r0 * N + loadCol, rr, win, N);
        Mul(pf, bF, cF, static_cast<uint32_t>(np));
        for (int64_t r = 0; r < rr; ++r) {
            Add(acc, acc, pf[static_cast<int32_t>(r * win)], static_cast<uint32_t>(win));
        }
        q0_.FreeTensor(t0);
        q1_.FreeTensor(t1);
        q2_.FreeTensor(t2);
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> tmp = rsvBuf_.Get<float>();
    Duplicate(tmp, 0.0f, 64);
    PipeBarrier<PIPE_ALL>();
    for (int64_t j = 0; j < ncols; ++j) {
        tmp.SetValue(static_cast<int32_t>(j), acc.GetValue(static_cast<int32_t>(laneBase + j)));
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<T> o = oq_.AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        Adds(o, tmp, static_cast<float>(0), 64);
    } else {
        Cast(o, tmp, RoundMode::CAST_RINT, 64);
    }
    oq_.EnQue(o);
    o = oq_.DeQue<T>();
    Store1D(o, out1Gm_, col, ncols);
    oq_.FreeTensor(o);
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessColOld() {
    const int64_t N = t_.N;
    int64_t c0 = blockIdx_ * t_.colPerCore;
    if (c0 >= N) return;
    int64_t c1 = c0 + t_.colPerCore;
    if (c1 > N) c1 = N;
    if (N < kBlk) {
        // whole tensor narrower than one vector group: torch sums each column scalar-wise
        for (int64_t c = c0; c < c1; ++c) {
            const int64_t np = RoundUp64(N);
            LocalTensor<float> pf = tP_.Get<float>();
            float s = 0.0f;
            for (int64_t r = 0; r < t_.M; ++r) {
                const int64_t ro = r * N;
                LocalTensor<T> t0 = Load1D(q0_, in0Gm_, ro, N, np);
                LocalTensor<T> t1 = Load1D(q1_, in1Gm_, ro, N, np);
                LocalTensor<T> t2 = Load1D(q2_, in2Gm_, ro, N, np);
                LocalTensor<float> aF;
                LocalTensor<float> bF;
                LocalTensor<float> cF;
                Prep(t0, t1, t2, aF, bF, cF, np);
                Mul(pf, bF, cF, static_cast<uint32_t>(np));
                LocalTensor<float> oF = GetOF();
                Mul(oF, aF, bF, static_cast<uint32_t>(np));
                EmitOut0(oF, ro, N, np);
                q0_.FreeTensor(t0);
                q1_.FreeTensor(t1);
                q2_.FreeTensor(t2);
                PipeBarrier<PIPE_ALL>();
                s += pf.GetValue(static_cast<int32_t>(c));
            }
            LocalTensor<T> o = oq_.AllocTensor<T>();
            o.SetValue(0, static_cast<T>(s));
            oq_.EnQue(o);
            o = oq_.DeQue<T>();
            Store1D(o, out1Gm_, c, 1);
            oq_.FreeTensor(o);
        }
        return;
    }
    int64_t col = c0;
    for (; col + kBlk <= N; col += kBlk) {
        ColGroupOld(col, kBlk);
    }
    if (col < N) {
        ColGroupOld(col, N - col);
    }
}

// ---------------------------------------------------------------------------
// axis=[1] (mode 0): bit-exact replica of torch CPU `sum(dim=1)`.
//
// torch CPU sum uses cascade_sum (aten/src/ATen/native/cpu/SumKernel.cpp):
//   * Vectorized<float>::size() == 8 (verified bit-exact on 8/9/16/.../65536)
//   * a row of N elements is cut into 32-element blocks (4 vectors of 8 lanes);
//     `row_sum` keeps 4 vector accumulators (one per 8-lane vector of the block)
//     and `multi_row_sum` runs a base-16 carry cascade over the blocks:
//        level0 += block                      (every block)
//        level1 += level0; level0 = 0         (every 16 blocks == 512 elems)
//        level2 += level1; level1 = 0         (every 256 blocks)
//        level3 += level2; level2 = 0         (every 4096 blocks)
//     leftover blocks (< 16) are folded into level0 without cascading;
//     then level0 = ((level0+level1)+level2)+level3
//   * the tail vectors (nvec % 4 vectors of 8) are added to lane group 0, then
//     group0 = ((g0+g1)+g2)+g3
//   * the trailing scalars (N % 8) are summed FIRST (ascending), then the 8
//     lanes of group0 are added one by one.
// The 128-lane accumulator below holds the four 32-lane levels contiguously:
//   [0:32)=level0 [32:64)=level1 [64:96)=level2 [96:128)=level3
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessRow() {
    if constexpr (kNpuKernel) {
        ProcessRowNpu();
        return;
    }
    if constexpr (kOldKernel) {
        ProcessRowOld();
        return;
    }
    const int64_t N = t_.N;
    const int64_t rpc = t_.rowPerCore;
    int64_t rs = blockIdx_ * rpc;
    if (rs > t_.M) rs = t_.M;
    int64_t re = rs + rpc;
    if (re > t_.M) re = t_.M;
    if (rs >= re) return;

    const int64_t nvec = N / kV;               // kV-lane vectors
    const int64_t nblocks = nvec / 4;          // kBlk-element blocks (4 vectors)
    const int64_t tailStart = nblocks * kBlk;
    const bool smallN = (N < kV);
    // level_power = max(4, ceil_log2(nblocks) / 4); level_step = 1 << level_power
    int64_t lp = 4;
    {
        int64_t cl = 0;
        while ((static_cast<int64_t>(1) << cl) < nblocks) {
            ++cl;
        }
        const int64_t t = cl / 4;
        if (t > lp) lp = t;
    }
    const int64_t step0 = static_cast<int64_t>(1) << lp;   // blocks per level-0 group

    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    LocalTensor<float> rsB = rsBuf_.Get<float>();
    int64_t bi = 0;
    int64_t rowBase = rs;
    for (int64_t r = rs; r < re; ++r) {
        const int64_t rowOff = r * N;
        float fin = 0.0f;

        if (smallN) {
            // torch takes the scalar inner path: 4-element blocks, no vector level
            const int64_t np = RoundUp64(N);
            LocalTensor<T> t0 = Load1D(q0_, in0Gm_, rowOff, N, np);
            LocalTensor<T> t1 = Load1D(q1_, in1Gm_, rowOff, N, np);
            LocalTensor<T> t2 = Load1D(q2_, in2Gm_, rowOff, N, np);
            LocalTensor<float> aF;
            LocalTensor<float> bF;
            LocalTensor<float> cF;
            Prep(t0, t1, t2, aF, bF, cF, np);
            LocalTensor<float> oF = GetOF();
            Mul(oF, aF, bF, static_cast<uint32_t>(np));
            EmitOut0(oF, rowOff, N, np);
            Mul(pf, bF, cF, static_cast<uint32_t>(np));
            q0_.FreeTensor(t0);
            q1_.FreeTensor(t1);
            q2_.FreeTensor(t2);
            PipeBarrier<PIPE_ALL>();
            const int64_t sizeIlp = N >> 2;   // < 4 here
            float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int64_t i = 0; i < sizeIlp; ++i) {
                for (int k = 0; k < 4; ++k) {
                    part[k] += pf.GetValue(static_cast<int32_t>((i << 2) + k));
                }
            }
            // partial[0] keeps the last block sums; the trailing scalars are added
            // to it first, then partial[1..3] are folded in (torch row_sum order).
            fin = part[0];
            for (int64_t a = sizeIlp << 2; a < N; ++a) {
                fin += pf.GetValue(static_cast<int32_t>(a));
            }
            fin += part[1];
            fin += part[2];
            fin += part[3];
        } else {
            Duplicate(acc, 0.0f, static_cast<uint32_t>(kAcc));
            int64_t i = 0;             // block counter (whole row)
            int64_t lastStart = 0;     // start element of the last loaded tile
            for (int64_t c0 = 0; c0 < N; c0 += tile_) {
                int64_t cnt = N - c0;
                if (cnt > tile_) cnt = tile_;
                const int64_t np = RoundUp64(cnt);
                LocalTensor<T> t0 = Load1D(q0_, in0Gm_, rowOff + c0, cnt, np);
                LocalTensor<T> t1 = Load1D(q1_, in1Gm_, rowOff + c0, cnt, np);
                LocalTensor<T> t2 = Load1D(q2_, in2Gm_, rowOff + c0, cnt, np);
                LocalTensor<float> aF;
                LocalTensor<float> bF;
                LocalTensor<float> cF;
                Prep(t0, t1, t2, aF, bF, cF, np);
                LocalTensor<float> oF = GetOF();
                Mul(oF, aF, bF, static_cast<uint32_t>(np));
                EmitOut0(oF, rowOff + c0, cnt, np);
                Mul(pf, bF, cF, static_cast<uint32_t>(np));
                lastStart = c0;
                int64_t blk = cnt / kBlk;
                const int64_t avail = nblocks - i;
                if (blk > avail) blk = avail;
                for (int64_t b = 0; b < blk; ++b) {
                    Add(acc, acc, pf[static_cast<int32_t>(b * kBlk)], static_cast<uint32_t>(kBlk));
                    ++i;
                    if ((i & (step0 - 1)) == 0) {
                        Add(acc[kBlk], acc[kBlk], acc, static_cast<uint32_t>(kBlk));
                        Duplicate(acc, 0.0f, static_cast<uint32_t>(kBlk));
                        if ((i & ((step0 << lp) - 1)) == 0) {
                            Add(acc[2 * kBlk], acc[2 * kBlk], acc[kBlk], static_cast<uint32_t>(kBlk));
                            Duplicate(acc[kBlk], 0.0f, static_cast<uint32_t>(kBlk));
                            if ((i & ((step0 << (2 * lp)) - 1)) == 0) {
                                Add(acc[3 * kBlk], acc[3 * kBlk], acc[2 * kBlk], static_cast<uint32_t>(kBlk));
                                Duplicate(acc[2 * kBlk], 0.0f, static_cast<uint32_t>(kBlk));
                            }
                        }
                    }
                }
                q0_.FreeTensor(t0);
                q1_.FreeTensor(t1);
                q2_.FreeTensor(t2);
            }
            // level0 = ((level0 + level1) + level2) + level3
            Add(acc, acc, acc[kBlk], static_cast<uint32_t>(kBlk));
            Add(acc, acc, acc[2 * kBlk], static_cast<uint32_t>(kBlk));
            Add(acc, acc, acc[3 * kBlk], static_cast<uint32_t>(kBlk));
            // tail vectors -> lane group 0 (ascending)
            for (int64_t a = tailStart; a + kV <= nvec * kV; a += kV) {
                Add(acc, acc, pf[static_cast<int32_t>(a - lastStart)], static_cast<uint32_t>(kV));
            }
            // group0 = ((g0 + g1) + g2) + g3
            Add(acc, acc, acc[kV], static_cast<uint32_t>(kV));
            Add(acc, acc, acc[2 * kV], static_cast<uint32_t>(kV));
            Add(acc, acc, acc[3 * kV], static_cast<uint32_t>(kV));
            PipeBarrier<PIPE_ALL>();
            for (int64_t a = nvec * kV; a < N; ++a) {
                fin += pf.GetValue(static_cast<int32_t>(a - lastStart));
            }
            for (int k = 0; k < static_cast<int>(kV); ++k) {
                fin += acc.GetValue(k);
            }
        }
        rsB.SetValue(static_cast<int32_t>(bi), fin);
        ++bi;
        if (bi == ROWSUM_B) {
            FlushOut1(rsB, rowBase, bi);
            rowBase += bi;
            bi = 0;
        }
    }
    if (bi > 0) {
        FlushOut1(rsB, rowBase, bi);
    }
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessCol() {
    if constexpr (kOldKernel) {
        ProcessColOld();
        return;
    }
    const int64_t N = t_.N;
    int64_t c0 = blockIdx_ * t_.colPerCore;
    if (c0 >= N) return;
    int64_t c1 = c0 + t_.colPerCore;
    if (c1 > N) c1 = N;
    const int64_t loadCols = LoadCols();
    if (N < loadCols) {
        ProcessColScalar();
        return;
    }
    // torch chunk order over the *whole* tensor: 32-column chunks first, then
    // 8-column chunks, then single columns. colPerCore is 64-aligned so our
    // per-core sequence continues the global one unchanged.
    int64_t col = c0;
    for (; col + kBlk <= N; col += kBlk) {
        ProcessColWide32(col);
    }
    for (; col + kV <= N; col += kV) {
        ProcessColTail8(col, kV);
    }
    if (col < N) {
        ProcessColTail8(col, N - col);
    }
}

// axis=[0] kBlk-column chunk: 1 row per cascade block (torch multi_row_sum over rows)
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessColWide32(int64_t col) {
    const int64_t N = t_.N;
    const int64_t M = t_.M;
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    Duplicate(acc, 0.0f, static_cast<uint32_t>(kAcc));
    const int64_t lp = LevelPower(M);
    const int64_t step0 = static_cast<int64_t>(1) << lp;
    int64_t R = tile_ / kBlk;
    R = (R / 16) * 16;
    if (R < 16) R = 16;
    if (R > M) R = M;
    int64_t i = 0;
    for (int64_t r0 = 0; r0 < M; r0 += R) {
        int64_t rr = M - r0;
        if (rr > R) rr = R;
        const int64_t np = RoundUp64(rr * kBlk);
        LocalTensor<T> t0 = Load2D(q0_, in0Gm_, r0 * N + col, rr, kBlk, N);
        LocalTensor<T> t1 = Load2D(q1_, in1Gm_, r0 * N + col, rr, kBlk, N);
        LocalTensor<T> t2 = Load2D(q2_, in2Gm_, r0 * N + col, rr, kBlk, N);
        LocalTensor<float> aF;
        LocalTensor<float> bF;
        LocalTensor<float> cF;
        Prep(t0, t1, t2, aF, bF, cF, np);
        LocalTensor<float> oF = GetOF();
        Mul(oF, aF, bF, static_cast<uint32_t>(np));
        EmitOut0_2D(oF, r0 * N + col, rr, kBlk, N);
        Mul(pf, bF, cF, static_cast<uint32_t>(np));
        for (int64_t r = 0; r < rr; ++r) {
            Add(acc, acc, pf[static_cast<int32_t>(r * kBlk)], static_cast<uint32_t>(kBlk));
            ++i;
            if ((i & (step0 - 1)) == 0) {
                Add(acc[kBlk], acc[kBlk], acc, static_cast<uint32_t>(kBlk));
                Duplicate(acc, 0.0f, static_cast<uint32_t>(kBlk));
                if ((i & ((step0 << lp) - 1)) == 0) {
                    Add(acc[2 * kBlk], acc[2 * kBlk], acc[kBlk], static_cast<uint32_t>(kBlk));
                    Duplicate(acc[kBlk], 0.0f, static_cast<uint32_t>(kBlk));
                    if ((i & ((step0 << (2 * lp)) - 1)) == 0) {
                        Add(acc[3 * kBlk], acc[3 * kBlk], acc[2 * kBlk], static_cast<uint32_t>(kBlk));
                        Duplicate(acc[2 * kBlk], 0.0f, static_cast<uint32_t>(kBlk));
                    }
                }
            }
        }
        q0_.FreeTensor(t0);
        q1_.FreeTensor(t1);
        q2_.FreeTensor(t2);
    }
    Add(acc, acc, acc[kBlk], static_cast<uint32_t>(kBlk));
    Add(acc, acc, acc[2 * kBlk], static_cast<uint32_t>(kBlk));
    Add(acc, acc, acc[3 * kBlk], static_cast<uint32_t>(kBlk));
    LocalTensor<T> o = oq_.AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        Adds(o, acc, static_cast<float>(0), static_cast<uint32_t>(kBlk));
    } else {
        Cast(o, acc, RoundMode::CAST_RINT, static_cast<uint32_t>(kBlk));
    }
    oq_.EnQue(o);
    o = oq_.DeQue<T>();
    Store1D(o, out1Gm_, col, kBlk);
    oq_.FreeTensor(o);
}

// axis=[0] trailing kV-column chunks: torch row_sum over rows, 4-row blocks
// (ilp_factor=4). The load window is LoadCols() columns (>= kV, 32B rows on the
// MTE side); `sets` accumulator banks cover it. Results are pulled out with scalar
// reads so no vector operand is ever misaligned.
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessColTail8(int64_t col, int64_t ncols) {
    const int64_t N = t_.N;
    const int64_t M = t_.M;
    const int64_t loadCols = LoadCols();
    const int64_t sets = loadCols / kV;
    int64_t loadCol = col;
    if (loadCol + loadCols > N) {
        loadCol = N - loadCols;
    }
    const int64_t laneBase = col - loadCol;
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    for (int64_t s = 0; s < sets; ++s) {
        Duplicate(acc[static_cast<int32_t>(s * kAcc)], 0.0f, static_cast<uint32_t>(kAcc));
    }
    int64_t tailRows = 0;
    int64_t tailOff = 0;
    const int64_t sizeIlp = M >> 2;
    const int64_t lp = LevelPower(sizeIlp);
    const int64_t step0 = static_cast<int64_t>(1) << lp;
    int64_t R = tile_ / loadCols;
    R = (R / 4) * 4;
    if (R < 4) R = 4;
    if (R > M) R = M;
    int64_t i = 0;
    for (int64_t r0 = 0; r0 < M; r0 += R) {
        int64_t rr = M - r0;
        if (rr > R) rr = R;
        const int64_t np = RoundUp64(rr * loadCols);
        LocalTensor<T> t0 = Load2D(q0_, in0Gm_, r0 * N + loadCol, rr, loadCols, N);
        LocalTensor<T> t1 = Load2D(q1_, in1Gm_, r0 * N + loadCol, rr, loadCols, N);
        LocalTensor<T> t2 = Load2D(q2_, in2Gm_, r0 * N + loadCol, rr, loadCols, N);
        LocalTensor<float> aF;
        LocalTensor<float> bF;
        LocalTensor<float> cF;
        Prep(t0, t1, t2, aF, bF, cF, np);
        LocalTensor<float> oF = GetOF();
        Mul(oF, aF, bF, static_cast<uint32_t>(np));
        EmitOut0_2D(oF, r0 * N + loadCol, rr, loadCols, N);
        Mul(pf, bF, cF, static_cast<uint32_t>(np));
        int64_t b = 0;
        for (; b + 4 <= rr; b += 4) {
            for (int k = 0; k < 4; ++k) {
                const int32_t off = static_cast<int32_t>((b + k) * loadCols);
                for (int64_t s = 0; s < sets; ++s) {
                    Add(acc[static_cast<int32_t>(s * kAcc + k * kV)],
                        acc[static_cast<int32_t>(s * kAcc + k * kV)],
                        pf[static_cast<int32_t>(off + s * kV)], static_cast<uint32_t>(kV));
                }
            }
            ++i;
            if ((i & (step0 - 1)) == 0) {
                for (int64_t s = 0; s < sets; ++s) {
                    const int32_t base = static_cast<int32_t>(s * kAcc);
                    Add(acc[base + kBlk], acc[base + kBlk], acc[base], static_cast<uint32_t>(kBlk));
                    Duplicate(acc[base], 0.0f, static_cast<uint32_t>(kBlk));
                }
                if ((i & ((step0 << lp) - 1)) == 0) {
                    for (int64_t s = 0; s < sets; ++s) {
                        const int32_t base = static_cast<int32_t>(s * kAcc);
                        Add(acc[base + 2 * kBlk], acc[base + 2 * kBlk], acc[base + kBlk],
                            static_cast<uint32_t>(kBlk));
                        Duplicate(acc[base + kBlk], 0.0f, static_cast<uint32_t>(kBlk));
                    }
                    if ((i & ((step0 << (2 * lp)) - 1)) == 0) {
                        for (int64_t s = 0; s < sets; ++s) {
                            const int32_t base = static_cast<int32_t>(s * kAcc);
                            Add(acc[base + 3 * kBlk], acc[base + 3 * kBlk], acc[base + 2 * kBlk],
                                static_cast<uint32_t>(kBlk));
                            Duplicate(acc[base + 2 * kBlk], 0.0f, static_cast<uint32_t>(kBlk));
                        }
                    }
                }
            }
        }
        tailRows = rr - b;
        tailOff = b * loadCols;
        q0_.FreeTensor(t0);
        q1_.FreeTensor(t1);
        q2_.FreeTensor(t2);
    }
    for (int64_t s = 0; s < sets; ++s) {
        const int32_t base = static_cast<int32_t>(s * kAcc);
        Add(acc[base], acc[base], acc[base + kBlk], static_cast<uint32_t>(kBlk));
        Add(acc[base], acc[base], acc[base + 2 * kBlk], static_cast<uint32_t>(kBlk));
        Add(acc[base], acc[base], acc[base + 3 * kBlk], static_cast<uint32_t>(kBlk));
    }
    for (int64_t k = 0; k < tailRows; ++k) {
        const int32_t off = static_cast<int32_t>(tailOff + k * loadCols);
        for (int64_t s = 0; s < sets; ++s) {
            Add(acc[static_cast<int32_t>(s * kAcc)], acc[static_cast<int32_t>(s * kAcc)],
                pf[static_cast<int32_t>(off + s * kV)], static_cast<uint32_t>(kV));
        }
    }
    for (int64_t s = 0; s < sets; ++s) {
        const int32_t base = static_cast<int32_t>(s * kAcc);
        Add(acc[base], acc[base], acc[base + kV], static_cast<uint32_t>(kV));
        Add(acc[base], acc[base], acc[base + 2 * kV], static_cast<uint32_t>(kV));
        Add(acc[base], acc[base], acc[base + 3 * kV], static_cast<uint32_t>(kV));
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> tmp = rsvBuf_.Get<float>();
    Duplicate(tmp, 0.0f, 64);
    PipeBarrier<PIPE_ALL>();
    for (int64_t j = 0; j < ncols; ++j) {
        const int64_t lane = laneBase + j;
        const int64_t s = lane / kV;
        const int64_t idx = lane - s * kV;
        const float v = acc.GetValue(static_cast<int32_t>(s * kAcc + idx));
        tmp.SetValue(static_cast<int32_t>(j), v);
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<T> o = oq_.AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        Adds(o, tmp, static_cast<float>(0), 64);
    } else {
        Cast(o, tmp, RoundMode::CAST_RINT, 64);
    }
    oq_.EnQue(o);
    o = oq_.DeQue<T>();
    Store1D(o, out1Gm_, col, ncols);
    oq_.FreeTensor(o);
}

// Fallback for N < 8 (fp32) / N < 16 (fp16): per-row scalar accumulation following
// torch's row_sum scalar order (4-row blocks + base-16 cascade + tail rows).
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessColScalar() {
    const int64_t N = t_.N;
    const int64_t M = t_.M;
    int64_t c0 = blockIdx_ * t_.colPerCore;
    if (c0 >= N) return;
    int64_t c1 = c0 + t_.colPerCore;
    if (c1 > N) c1 = N;
    const int64_t np = RoundUp64(N);
    LocalTensor<float> pf = tP_.Get<float>();
    for (int64_t c = c0; c < c1; ++c) {
        const int64_t sizeIlp = M >> 2;
        const int64_t lp = LevelPower(sizeIlp);
        const int64_t step0 = static_cast<int64_t>(1) << lp;
        float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float a2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float a3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float tailv = 0.0f;
        int64_t i = 0;
        for (int64_t r = 0; r < M; r += 4) {
            int64_t rr = M - r;
            if (rr > 4) rr = 4;
            for (int64_t k = 0; k < rr; ++k) {
                const int64_t ro = (r + k) * N;
                LocalTensor<T> t0 = Load1D(q0_, in0Gm_, ro, N, np);
                LocalTensor<T> t1 = Load1D(q1_, in1Gm_, ro, N, np);
                LocalTensor<T> t2 = Load1D(q2_, in2Gm_, ro, N, np);
                LocalTensor<float> aF;
                LocalTensor<float> bF;
                LocalTensor<float> cF;
                Prep(t0, t1, t2, aF, bF, cF, np);
                Mul(pf, bF, cF, static_cast<uint32_t>(np));
                LocalTensor<float> oF = GetOF();
                Mul(oF, aF, bF, static_cast<uint32_t>(np));
                EmitOut0(oF, ro, N, np);
                q0_.FreeTensor(t0);
                q1_.FreeTensor(t1);
                q2_.FreeTensor(t2);
                PipeBarrier<PIPE_ALL>();
                const float v = pf.GetValue(static_cast<int32_t>(c));
                if ((r + k) < sizeIlp * 4) {
                    a0[k] += v;
                } else {
                    tailv += v;
                }
            }
            if (r + rr <= sizeIlp * 4) {
                ++i;
                if ((i & (step0 - 1)) == 0) {
                    for (int k = 0; k < 4; ++k) {
                        a1[k] += a0[k];
                        a0[k] = 0.0f;
                    }
                    if ((i & ((step0 << lp) - 1)) == 0) {
                        for (int k = 0; k < 4; ++k) {
                            a2[k] += a1[k];
                            a1[k] = 0.0f;
                        }
                        if ((i & ((step0 << (2 * lp)) - 1)) == 0) {
                            for (int k = 0; k < 4; ++k) {
                                a3[k] += a2[k];
                                a2[k] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
        float p[4];
        for (int k = 0; k < 4; ++k) {
            p[k] = a0[k];
            p[k] += a1[k];
            p[k] += a2[k];
            p[k] += a3[k];
        }
        p[0] += tailv;
        p[0] += p[1];
        p[0] += p[2];
        p[0] += p[3];
        const float fin = p[0];
        LocalTensor<T> o = oq_.AllocTensor<T>();
        o.SetValue(0, static_cast<T>(fin));
        oq_.EnQue(o);
        o = oq_.DeQue<T>();
        Store1D(o, out1Gm_, c, 1);
        oq_.FreeTensor(o);
    }
}


template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessColChunk(int64_t c0, int64_t c1) {
    const int64_t N = t_.N;
    const int64_t M = t_.M;
    const int64_t cw = c1 - c0;
    const bool wide = (cw == t_.colPerCore) && (cw >= A64);

    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    LocalTensor<float> comp = tComp_.Get<float>();
    const int64_t cwp = RoundUp64(cw);
    Duplicate(acc, 0.0f, static_cast<uint32_t>(cwp));
    Duplicate(comp, 0.0f, static_cast<uint32_t>(cwp));

    if (wide) {
        int64_t R = tile_ / cw;
        if (R < 1) R = 1;
        if (R > M) R = M;
        for (int64_t r0 = 0; r0 < M; r0 += R) {
            int64_t rr = M - r0;
            if (rr > R) rr = R;
            LocalTensor<T> t0 = Load2D(q0_, in0Gm_, r0 * N + c0, rr, cw, N);
            LocalTensor<T> t1 = Load2D(q1_, in1Gm_, r0 * N + c0, rr, cw, N);
            LocalTensor<T> t2 = Load2D(q2_, in2Gm_, r0 * N + c0, rr, cw, N);
            LocalTensor<float> aF;
            LocalTensor<float> bF;
            LocalTensor<float> cF;
            Prep(t0, t1, t2, aF, bF, cF, rr * cw);
            LocalTensor<float> oF = GetOF();
            Mul(oF, aF, bF, static_cast<uint32_t>(rr * cw));
            EmitOut0_2D(oF, r0 * N + c0, rr, cw, N);
            // p = b*c over the whole (rr x cw) tile, then tree-fold the row dimension:
            // log2(rr) wide Adds instead of rr narrow ones (major win: 8191 rows).
            Mul(pf, bF, cF, static_cast<uint32_t>(rr * cw));
            int64_t n = rr;
            while (n > 1) {
                const int64_t h = (n + 1) >> 1;
                const int64_t cc = n >> 1;
                Add(pf, pf, pf[static_cast<int32_t>(h * cw)], static_cast<uint32_t>(cc * cw));
                n = h;
            }
            KahanAdd(acc, comp, pf, static_cast<uint32_t>(cw));
            q0_.FreeTensor(t0);
            q1_.FreeTensor(t1);
            q2_.FreeTensor(t2);
        }
    } else {
        for (int64_t r = 0; r < M; ++r) {
            const int64_t ro = r * N + c0;
            LocalTensor<T> t0 = Load1D(q0_, in0Gm_, ro, cw, cwp);
            LocalTensor<T> t1 = Load1D(q1_, in1Gm_, ro, cw, cwp);
            LocalTensor<T> t2 = Load1D(q2_, in2Gm_, ro, cw, cwp);
            LocalTensor<float> aF;
            LocalTensor<float> bF;
            LocalTensor<float> cF;
            Prep(t0, t1, t2, aF, bF, cF, cwp);
            LocalTensor<float> oF = GetOF();
            Mul(oF, aF, bF, static_cast<uint32_t>(cwp));
            EmitOut0(oF, ro, cw, cwp);
            Mul(pf, bF, cF, static_cast<uint32_t>(cwp));
            KahanAdd(acc, comp, pf, static_cast<uint32_t>(cwp));
            q0_.FreeTensor(t0);
            q1_.FreeTensor(t1);
            q2_.FreeTensor(t2);
        }
    }

    LocalTensor<T> o = oq_.AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        Adds(o, acc, static_cast<float>(0), static_cast<uint32_t>(cwp));
    } else {
        Cast(o, acc, RoundMode::CAST_RINT, static_cast<uint32_t>(cwp));
    }
    oq_.EnQue(o);
    o = oq_.DeQue<T>();
    Store1D(o, out1Gm_, c0, cw);
    oq_.FreeTensor(o);
}

// ---------------- mode 1 fast path: row-block + contiguous rows + atomic merge ----
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessColRowBlock() {
    if constexpr (!std::is_same<T, float>::value) {
        ProcessCol();  // fp16 never reaches here (tiling keeps useRowBlock=0)
    } else {
        const int64_t N = t_.N;
        const int64_t npv = t_.accLen;                 // AlignI(N, 64) -> 64-aligned vector counts
        const int64_t nw = (N + 7) / 8 * 8;            // AlignI(N, 8) -> 32B-multiple atomic write
        const int64_t rpc = t_.rowPerCore;
        int64_t rs = blockIdx_ * rpc;
        if (rs > t_.M) rs = t_.M;
        int64_t re = rs + rpc;
        if (re > t_.M) re = t_.M;

        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> comp = tComp_.Get<float>();
        LocalTensor<float> pf = tP_.Get<float>();
        Duplicate(acc, 0.0f, static_cast<uint32_t>(npv));
        Duplicate(comp, 0.0f, static_cast<uint32_t>(npv));

        for (int64_t r = rs; r < re; ++r) {
            const int64_t ro = r * N;
            LocalTensor<T> t0 = Load1D(q0_, in0Gm_, ro, N, npv);
            LocalTensor<T> t1 = Load1D(q1_, in1Gm_, ro, N, npv);
            LocalTensor<T> t2 = Load1D(q2_, in2Gm_, ro, N, npv);
            LocalTensor<float> aF;
            LocalTensor<float> bF;
            LocalTensor<float> cF;
            Prep(t0, t1, t2, aF, bF, cF, npv);
            LocalTensor<float> oF = GetOF();
            Mul(oF, aF, bF, static_cast<uint32_t>(npv));
            EmitOut0(oF, ro, N, npv);
            Mul(pf, bF, cF, static_cast<uint32_t>(npv));
            KahanAdd(acc, comp, pf, static_cast<uint32_t>(npv));
            q0_.FreeTensor(t0);
            q1_.FreeTensor(t1);
            q2_.FreeTensor(t2);
        }

        if (t_.usedCores > 1) {
            if (blockIdx_ == 0) {
                LocalTensor<float> gb = gBuf_.Get<float>();
                // (P0 - g): cancel whatever output1 contained before the atomic merge.
                Sub(acc, acc, gb, static_cast<uint32_t>(npv));
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();
            // acc was produced by vector ops; drain V before the MTE3 atomic store.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<float>();
            DataCopy(out1Gm_[0], acc, static_cast<uint32_t>(nw));
            SetAtomicNone();
            PipeBarrier<PIPE_ALL>();
        } else {
            LocalTensor<T> o = oq_.AllocTensor<T>();
            Adds(o, acc, static_cast<float>(0), static_cast<uint32_t>(npv));
            oq_.EnQue(o);
            o = oq_.DeQue<T>();
            Store1D(o, out1Gm_, 0, N);
            oq_.FreeTensor(o);
        }
    }
}
template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::ProcessAll() {
    const int64_t accLen = t_.accLen;
    int64_t s = blockIdx_ * t_.rowPerCore;
    int64_t e = s + t_.rowPerCore;
    if (s > t_.totalNum) s = t_.totalNum;
    if (e > t_.totalNum) e = t_.totalNum;
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> pf = tP_.Get<float>();
    LocalTensor<float> comp = tComp_.Get<float>();
    Duplicate(acc, 0.0f, static_cast<uint32_t>(accLen));
    Duplicate(comp, 0.0f, static_cast<uint32_t>(accLen));
    for (int64_t off = s; off < e; off += accLen) {
        int64_t cnt = e - off;
        if (cnt > accLen) cnt = accLen;
        const int64_t np = RoundUp64(cnt);
        LocalTensor<T> t0 = Load1D(q0_, in0Gm_, off, cnt, np);
        LocalTensor<T> t1 = Load1D(q1_, in1Gm_, off, cnt, np);
        LocalTensor<T> t2 = Load1D(q2_, in2Gm_, off, cnt, np);
        LocalTensor<float> aF;
        LocalTensor<float> bF;
        LocalTensor<float> cF;
        Prep(t0, t1, t2, aF, bF, cF, np);
        LocalTensor<float> oF = GetOF();
        Mul(oF, aF, bF, static_cast<uint32_t>(np));
        EmitOut0(oF, off, cnt, np);
        Mul(pf, bF, cF, static_cast<uint32_t>(np));
        KahanAdd(acc, comp, pf, static_cast<uint32_t>(np));
        q0_.FreeTensor(t0);
        q1_.FreeTensor(t1);
        q2_.FreeTensor(t2);
    }
    float part = FoldSum(acc, accLen);

    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<float> tmp = tP_.Get<float>();
        if (t_.usedCores <= 1) {
            // Single core owns the whole reduction: the partial sum is the result.
            // GlobalTensor::SetValue does NOT write back to GM, so go through UB->GM.
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            tmp.SetValue(0, part);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
            PipeBarrier<PIPE_ALL>();
            Store1D(tmp, out1Gm_, 0, 1);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        float delta = (blockIdx_ == 0) ? (part - gRead_) : part;
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        tmp.SetValue(0, delta);
        for (int32_t j = 1; j < 8; ++j) {
            tmp.SetValue(j, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<float>();
        DataCopy(out1Gm_[0], tmp, 8);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    } else {
        LocalTensor<float> tmp = tP_.Get<float>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        tmp.SetValue(0, part);
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> th = tOutH_.Get<T>();
        Cast(th, tmp, RoundMode::CAST_RINT, static_cast<uint32_t>(A64));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Store1D(th, out1Gm_, 0, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::FlushOut1(LocalTensor<float> &rsB, int64_t rowStart, int64_t n) {
    const int64_t np = RoundUp64(n);
    for (int64_t j = n; j < np; ++j) {
        rsB.SetValue(static_cast<int32_t>(j), 0.0f);
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (std::is_same<T, float>::value) {
        // rsB was filled by scalar SetValue -> make it visible to the MTE3 store.
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        PipeBarrier<PIPE_ALL>();
        Store1D(rsB, out1Gm_, rowStart, n);
    } else {
        // Scalar writes must be visible to the vector Cast that reads them.
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> th = tOutH_.Get<T>();
        Cast(th, rsB, RoundMode::CAST_RINT, static_cast<uint32_t>(np));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Store1D(th, out1Gm_, rowStart, n);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void ConfusionMulGrad<T>::Process() {
    if (t_.axisMode == 0) {
        ProcessRow();
    } else if (t_.axisMode == 1) {
        if (t_.useRowBlock == 1) {
            ProcessColRowBlock();
        } else {
            ProcessCol();
        }
    } else {
        ProcessAll();
    }
}

}  // namespace NsConfusionMulGrad
#endif  // CONFUSIONMULGRAD_H

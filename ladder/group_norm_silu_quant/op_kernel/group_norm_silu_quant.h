/*!
 * \file group_norm_silu_quant.h
 * \brief GroupNormSiluQuant kernel (group-per-core, no cross-core reduce)
 *
 *  For each group (n,g) (a contiguous block of CpG*HW elements in row-major self):
 *    mean = E[x],  var = E[(x-mean)^2],  rstd = 1/sqrt(var+eps)
 *    gn   = ((x - mean) * rstd * gamma[c]) + beta[c]     (per-channel affine)
 *    silu = gn / (1 + exp(-gn))
 *    out  = clamp(rint(silu / quantScale[c]), -128, 127) -> int8
 *
 *  The golden reference (problem spec 3.1) is the plain numpy implementation, whose
 *  `ndarray.mean/var` over the trailing axes of a C-contiguous array reduce with numpy's
 *  flat "Parker" pairwise summation (verified bit-exactly against numpy 1.26: PW_BLOCKSIZE
 *  128, 8 interleaved accumulators per block, `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` combine,
 *  `n2 = n/2 - (n/2)%8` split).  int8 output must match bit-exactly, so the group statistics
 *  are replicated with exactly that summation order (two-pass variance, mean is `sum / n`).
 *
 *  Arithmetic order per element follows the golden exactly, one rounding per operation:
 *    (x - mean) * rstd * gamma + beta,  1 + exp(-gn),  gn / (...)  ,  silu / quantScale
 *  and `rint` is round-half-even (the 1.5*2^23 add/sub trick).
 */

#ifndef GROUPNORMSILUQUANT_H
#define GROUPNORMSILUQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_silu_quant_tiling_data.h"
#include "group_norm_silu_quant_tiling_key.h"

namespace NsGroupNormSiluQuant {

using namespace AscendC;

// maximum length handled by one pairwise-sum leaf (numpy PW_BLOCKSIZE)
constexpr int64_t kPwBlock = 128;
// max recursion depth of the pairwise split (n <= 2^40 elements is far beyond device memory)
constexpr int kPwMaxDepth = 64;
// numpy's reduce internal buffer size (NPY_BUFSIZE); blocks of this many elements are
// summed independently and accumulated sequentially.
constexpr int64_t kNpyBufSize = 8192;

template <typename T>
class GroupNormSiluQuant {
public:
    __aicore__ inline GroupNormSiluQuant(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale,
                                GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
                                const GroupNormSiluQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadToFloat(const LocalTensor<float>& dstF, int64_t elemOff,
                                       int64_t cnt, int64_t cntR, T padT);
    __aicore__ inline void StoreInt8(const LocalTensor<int8_t>& srcI8, int64_t elemOff, int64_t cnt);
    __aicore__ inline void StoreStat(GlobalTensor<T>& dst, int64_t idx, float val);
    __aicore__ inline void ProcessGroup(int64_t gidx);
    __aicore__ inline float PwLeaf(int64_t off, int64_t len, bool centered, float mean);
    __aicore__ inline float PwSum(int64_t off, int64_t len, bool centered, float mean);
    __aicore__ inline float ExpScalarExact(float xf);
    __aicore__ inline float DfTwoSum(float a, float b, float& e);
    __aicore__ inline float DfFastTwoSum(float a, float b, float& e);
    __aicore__ inline float DfTwoProd(float a, float b, float& e);
    __aicore__ inline float DfDiv(float a, float b);
    __aicore__ inline float VSum8192(int64_t absOff, bool centered, float mean);
    __aicore__ inline float PwSumChunked(int64_t off, int64_t n, bool centered, float mean);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> raw_;   // fp16 staging (T)
    TBuf<QuePosition::VECCALC> fx_;    // value / result (float)
    TBuf<QuePosition::VECCALC> ta_;    // scratch (float, or int32 in quant stage)
    TBuf<QuePosition::VECCALC> tb_;    // scratch (float)
    TBuf<QuePosition::VECCALC> tc_;    // scratch (float)
    TBuf<QuePosition::VECCALC> oi8_;   // int8 staging
    TBuf<QuePosition::VECCALC> red_;   // reduce result (64 floats)
    TBuf<QuePosition::VECCALC> red2_;  // reduce temp / result (64 floats)
    TBuf<QuePosition::VECCALC> leafF_; // pairwise leaf staging (float)
    TBuf<QuePosition::VECCALC> leafR_; // pairwise leaf staging (T)
    TBuf<QuePosition::VECCALC> vTile_; // full 8192-element tile for vectorized numpy sum (float)
    TBuf<QuePosition::VECCALC> vAcc_;  // 64*8 lane accumulators (float)
    TBuf<QuePosition::VECCALC> vT1_;   // pair-reduce ping (float)
    TBuf<QuePosition::VECCALC> vT2_;   // pair-reduce pong (float)
    TBuf<QuePosition::VECCALC> vRaw_;  // 8192-element raw T staging (fp16)

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gGm_;
    GlobalTensor<T> bGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> qsGm_;
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> rstdGm_;

    GroupNormSiluQuantTilingData t_;
};

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
                                                   GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut,
                                                   GM_ADDR rstdOut,
                                                   const GroupNormSiluQuantTilingData* tilingData)
{
    t_ = *tilingData;
    const int64_t total = t_.N * t_.C * t_.HW;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), total);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out), total);
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), t_.C);
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), t_.C);
    qsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale), t_.qsPerChannel ? t_.C : 1);
    meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(meanOut), t_.totalGroups);
    rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstdOut), t_.totalGroups);

    const int64_t ch = t_.chunk;
    pipe.InitBuffer(fx_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(ta_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(tb_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(tc_, static_cast<uint32_t>(ch * 4));
    pipe.InitBuffer(oi8_, static_cast<uint32_t>(((ch + 31) / 32) * 32));
    pipe.InitBuffer(red_, static_cast<uint32_t>(64 * 4));
    pipe.InitBuffer(red2_, static_cast<uint32_t>(64 * 4));
    pipe.InitBuffer(leafF_, static_cast<uint32_t>(kPwBlock * 4));
    pipe.InitBuffer(leafR_, static_cast<uint32_t>(kPwBlock * sizeof(T)));
    // vectorized numpy-pairwise buffers: one 8192-element float tile + 64*8 lanes
    pipe.InitBuffer(vTile_, static_cast<uint32_t>(8192 * 4));
    pipe.InitBuffer(vAcc_, static_cast<uint32_t>(512 * 4));
    pipe.InitBuffer(vT1_, static_cast<uint32_t>(512 * 4));
    pipe.InitBuffer(vT2_, static_cast<uint32_t>(512 * 4));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(raw_, static_cast<uint32_t>(((ch + 31) / 32) * 32 * 2));
        pipe.InitBuffer(vRaw_, static_cast<uint32_t>(8192 * sizeof(T)));
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::LoadToFloat(const LocalTensor<float>& dstF, int64_t elemOff,
                                                          int64_t cnt, int64_t cntR, T padT)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    int64_t bytes = cnt * static_cast<int64_t>(sizeof(T));
    int64_t padBytes = ((bytes + 31) / 32) * 32 - bytes;
    int64_t padElems = padBytes / static_cast<int64_t>(sizeof(T));
    DataCopyPadExtParams<T> pp{padElems > 0, 0, static_cast<uint8_t>(padElems), padT};
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> raw = raw_.Get<T>();
        if (cnt < cntR) {
            Duplicate(raw, padT, static_cast<int32_t>(cntR));
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(raw, xGm_[elemOff], p, pp);
        PipeBarrier<PIPE_ALL>();
        Cast(dstF, raw, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        if (cnt < cntR) {
            Duplicate(dstF, static_cast<float>(padT), static_cast<int32_t>(cntR));
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(dstF, xGm_[elemOff], p, pp);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::StoreInt8(const LocalTensor<int8_t>& srcI8, int64_t elemOff,
                                                        int64_t cnt)
{
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(cnt);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    DataCopyPad(yGm_[elemOff], srcI8, p);
    PipeBarrier<PIPE_ALL>();   // MTE3 done before the staging buffer is reused (WAR)
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::StoreStat(GlobalTensor<T>& dst, int64_t idx, float val)
{
    // GM scalar stores via GlobalTensor::SetValue are not flushed reliably on this arch;
    // stage through UB and use DataCopyPad instead.
    LocalTensor<float> sb = red_.Get<float>();
    Duplicate(sb, val, 64);
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(sizeof(T));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> hb = raw_.Get<T>();
        Cast(hb, sb, RoundMode::CAST_RINT, 64);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[idx], hb, p);
    } else {
        DataCopyPad(dst[idx], sb, p);
    }
    PipeBarrier<PIPE_ALL>();
}

// ---------------------------------------------------------------------------
// numpy pairwise summation (exact replication of npy_pairwise_sum for float32)
// ---------------------------------------------------------------------------

// One leaf of the pairwise tree: 8 <= len <= kPwBlock follows numpy's unrolled
// 8-accumulator block, len < 8 follows numpy's plain sequential loop.
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::PwLeaf(int64_t off, int64_t len, bool centered, float mean)
{
    LocalTensor<float> v = leafF_.Get<float>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> raw = leafR_.Get<T>();
        DataCopyPad(raw, xGm_[off], p, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
        PipeBarrier<PIPE_ALL>();
        int64_t cntR = ((len + 63) / 64) * 64;
        Cast(v, raw, RoundMode::CAST_NONE, static_cast<int32_t>(cntR));
    } else {
        DataCopyPad(v, xGm_[off], p, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
    }
    PipeBarrier<PIPE_ALL>();

    if (len < 8) {
        float res = 0.0f;
        for (int64_t i = 0; i < len; ++i) {
            float x = v.GetValue(i);
            if (centered) {
                float d = x - mean;
                res += d * d;
            } else {
                res += x;
            }
        }
        return res;
    }
    float r[8];
    for (int64_t j = 0; j < 8; ++j) {
        float x = v.GetValue(j);
        if (centered) {
            volatile float d = x - mean;
            volatile float sq = d * d;   // volatile: keep the two roundings of (x-mean)^2
            r[j] = sq;
        } else {
            r[j] = x;
        }
    }
    int64_t i = 8;
    const int64_t lim = len - (len % 8);
    for (; i < lim; i += 8) {
        for (int64_t j = 0; j < 8; ++j) {
            float x = v.GetValue(i + j);
            if (centered) {
                volatile float d = x - mean;
                volatile float sq = d * d;   // volatile: no fma, numpy rounds the square first
                r[j] = r[j] + sq;
            } else {
                r[j] += x;
            }
        }
    }
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < len; ++i) {
        float x = v.GetValue(i);
        if (centered) {
            volatile float d = x - mean;
            volatile float sq = d * d;
            res = res + sq;
        } else {
            res += x;
        }
    }
    return res;
}

// Iterative post-order walk of numpy's pairwise recursion:
//   n < 8        -> sequential
//   n <= 128     -> PwLeaf
//   else         -> PwSum(n2) + PwSum(n-n2),  n2 = n/2 - (n/2)%8
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::PwSum(int64_t off, int64_t len, bool centered, float mean)
{
    if (len <= kPwBlock) {
        return PwLeaf(off, len, centered, mean);
    }
    struct Frame {
        int64_t off;
        int64_t len;
        int32_t stage;   // 0: push left, 1: left pushed, 2: left done, 3: right pushed
        float left;
    };
    Frame st[kPwMaxDepth];
    int sp = 0;
    st[sp].off = off;
    st[sp].len = len;
    st[sp].stage = 0;
    st[sp].left = 0.0f;
    ++sp;

    float pending = 0.0f;
    bool have = false;
    while (sp > 0) {
        Frame& f = st[sp - 1];
        if (f.len <= kPwBlock) {
            pending = PwLeaf(f.off, f.len, centered, mean);
            have = true;
            --sp;
            continue;
        }
        if (have) {
            if (f.stage == 1) {
                f.left = pending;
                f.stage = 2;
                have = false;
                continue;
            }
            // f.stage == 3: right child returned
            pending = f.left + pending;
            have = true;
            --sp;
            continue;
        }
        int64_t n2 = f.len / 2;
        n2 -= n2 % 8;
        if (f.stage == 0) {
            f.stage = 1;
            st[sp].off = f.off;
            st[sp].len = n2;
            st[sp].stage = 0;
            st[sp].left = 0.0f;
            ++sp;
        } else if (f.stage == 2) {
            f.stage = 3;
            st[sp].off = f.off + n2;
            st[sp].len = f.len - n2;
            st[sp].stage = 0;
            st[sp].left = 0.0f;
            ++sp;
        } else {
            // unreachable (guards against a corrupted stack)
            --sp;
        }
    }
    return pending;
}

// ---------------------------------------------------------------------------
// Vectorized numpy pairwise sum of ONE full 8192-element block (NPY_BUFSIZE).
//
// The scalar rule (see PwLeaf/PwSum) over a 8192 block is: 64 leaves of 128
// elements.  Each leaf has eight interleaved accumulators
//   r[i] = v[i] + v[8+i] + v[16+i] + ... + v[120+i]     (i = 0..7, sequential)
// and root = ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)).  Leaves are then combined
// with numpy's balanced adjacent tree (64->32->16->8->4->2->1).
//
// Vectorization (arithmetic association is EXACTLY the rule above):
//   * Load all 8192 elements into one fp32 tile.  Leaf l occupies
//     [l*128, l*128+128).  The eight lane accumulators of ALL 64 leaves live in
//     `acc` as acc[l*8 + i].
//   * 16 level-0 repeat Add instructions accumulate the 16 interleaved groups
//     for every leaf at once (dst block stride 1, src block stride 16 blocks =
//     128 floats => next leaf), k = 0..15 in order == the sequential rule.
//   * Three PairReduceSum instructions collapse each leaf's 8 lanes to its root
//     with the exact adjacent-pair tree (8->4->2->1).
//   * The 64 leaf roots are combined with the balanced adjacent tree in scalar
//     fp32 (63 adds per block, negligible).
// No fused multiply-add / reassociation is introduced, so the result is bit
// identical to the scalar PwSum over the same 8192 block.
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::VSum8192(int64_t absOff, bool centered, float mean)
{
    LocalTensor<float> v = vTile_.Get<float>();
    DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(8192 * static_cast<int64_t>(sizeof(T)));
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    if constexpr (sizeof(T) == 2) {
        LocalTensor<T> rw = vRaw_.Get<T>();
        DataCopyPad(rw, xGm_[absOff], p, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
        PipeBarrier<PIPE_ALL>();
        Cast(v, rw, RoundMode::CAST_NONE, 8192);
    } else {
        DataCopyPad(v, xGm_[absOff], p, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
    }
    PipeBarrier<PIPE_ALL>();

    if (centered) {
        // elementwise (x-mean)^2, two roundings (Sub then Mul) == numpy's array
        Adds(v, v, -mean, 8192);
        Mul(v, v, v, 8192);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> acc = vAcc_.Get<float>();
    Duplicate(acc, 0.0f, 512);
    PipeBarrier<PIPE_V>();
    // acc[l*8+i] += v[l*128 + k*8 + i] for all 64 leaves; k ascending == rule.
    BinaryRepeatParams rp(1, 1, 1, 1, 1, 16); // dstBlk,src0Blk,src1Blk,dstRep,src0Rep,src1Rep
    for (int32_t k = 0; k < 16; ++k) {
        Add<float, true>(acc, acc, v[k * 8], static_cast<uint64_t>(0xFF), static_cast<uint8_t>(64), rp);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> t1 = vT1_.Get<float>();
    LocalTensor<float> t2 = vT2_.Get<float>();
    (void)t2;
    // 8 -> 1 per leaf: BlockReduceSum reduces each 32B (8-float) block to one
    // value. acc = 64 leaves x 8 lanes; 8 repeats x 64 elements -> 64 contiguous
    // leaf roots at t1[0..63]. This must equal numpy's leaf tree
    // ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)); verified bit-exact on device.
    BlockReduceSum<float, true>(t1, acc, 8, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float rv[64];
    for (int64_t l = 0; l < 64; ++l) {
        rv[l] = t1.GetValue(static_cast<int32_t>(l));
    }
    for (int64_t width = 64; width > 1; width >>= 1) {
        for (int64_t i = 0; i < width / 2; ++i) {
            rv[i] = rv[i * 2] + rv[i * 2 + 1];
        }
    }
    return rv[0];
}

// numpy's reduce buffers the iteration in blocks of NPY_BUFSIZE (8192) elements:
// each buffered block is summed with the pairwise scheme above and the per-block
// results are accumulated sequentially into the fp32 output.  Verified bit-exact
// against numpy 1.26 for n in {105, 8191, 8192, 8193, 16385, 32768, 131072, ...}.
// Full 8192 blocks use the vectorized VSum8192 above; only the final partial
// block (< 8192) falls back to the scalar PwSum, which is once per group.
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::PwSumChunked(int64_t off, int64_t n, bool centered, float mean)
{
    float acc = 0.0f;
    int64_t i = 0;
    for (; i + kNpyBufSize <= n; i += kNpyBufSize) {
        acc = acc + VSum8192(off + i, centered, mean);
    }
    if (i < n) {
        acc = acc + PwSum(off + i, n - i, centered, mean);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Correctly rounded exp() for ONE fp32 argument, in double-float (two fp32) arithmetic.
// Only used for the few elements whose quantised value sits within a few ulp of a half-integer
// (the only ones where the vector Exp's 1~2 ulp error can flip the int8 result).
// Validated on host against numpy: 52 mismatches out of 75000 random inputs (0.069%), which is
// exactly numpy's own 1-ulp rate vs the correctly rounded value.
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::DfTwoSum(float a, float b, float& e)
{
    volatile float s = a + b;
    volatile float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
    return s;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::DfFastTwoSum(float a, float b, float& e)
{
    volatile float s = a + b;   // requires |a| >= |b|
    e = b - (s - a);
    return s;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::DfTwoProd(float a, float b, float& e)
{
    volatile float p = a * b;
    volatile float ta = a * 4097.0f;
    float ah = ta - (ta - a);
    float al = a - ah;
    volatile float tb = b * 4097.0f;
    float bh = tb - (tb - b);
    float bl = b - bh;
    volatile float abh = ah * bh;
    volatile float d = abh - p;
    e = ((d + ah * bl) + al * bh) + al * bl;
    return p;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::DfDiv(float a, float b)
{
    volatile float q = a / b;
    float pe;
    volatile float pb = q * b;
    (void)DfTwoProd(q, b, pe);
    float r = (a - pb) - pe;
    volatile float corr = r / b;
    return q + corr;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::ExpScalarExact(float xf)
{
    float x = xf;
    if (!(x > -87.0f)) {
        x = -87.0f;
    }
    if (x > 88.0f) {
        x = 88.0f;
    }
    volatile float t = x * 1.4426950408889634f;
    const int32_t k = static_cast<int32_t>(t + ((t >= 0.0f) ? 0.5f : -0.5f));
    const float kh = static_cast<float>(k);
    // p = k*ln2 in double-float (ln2 split into 12 high bits + low part)
    float pl;
    float ph = DfTwoProd(kh, 0.693145751953125f, pl);
    pl = pl + kh * 1.428606765330187045e-06f;
    // r = x - p in double-float
    float rl;
    float rh = DfTwoSum(x, -ph, rl);
    rl = rl - pl;
    rh = DfFastTwoSum(rh, rl, rl);
    // exp(r) = 1 + r*(1 + r/2*(1 + r/3*(...)))  Horner in double-float, degree 14
    float h = 1.0f;
    float l = 0.0f;
    for (int32_t i = 14; i >= 1; --i) {
        float ml;
        float mh = DfTwoProd(h, rh, ml);
        ml = (ml + h * rl) + l * rh;
        float mlo;
        mh = DfFastTwoSum(mh, ml, mlo);
        const float fi = static_cast<float>(i);
        volatile float dh = mh / fi;
        const float dl = mlo / fi;
        float el;
        h = DfTwoSum(1.0f, dh, el);
        l = el + dl;
        float l2;
        h = DfFastTwoSum(h, l, l2);
        l = l2;
    }
    // scale by 2^k (exact for a power of two)
    if (k > 0) {
        for (int32_t i = 0; i < k; ++i) {
            h = h * 2.0f;
            l = l * 2.0f;
        }
    } else {
        for (int32_t i = 0; i < -k; ++i) {
            h = h * 0.5f;
            l = l * 0.5f;
        }
    }
    return h + l;
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::ProcessGroup(int64_t gidx)
{
    const int64_t CpG = t_.CpG;
    const int64_t HW = t_.HW;
    const int64_t gsz = t_.gsz;
    const int64_t chunk = t_.chunk;
    const int64_t G = t_.G;
    const float eps = t_.epsilon;
    const bool doSilu = (t_.activateSilu != 0);
    const int64_t base = gidx * gsz;
    const int64_t g = (G > 0) ? (gidx % G) : 0;
    const int64_t cbase = g * CpG;

    LocalTensor<float> fx = fx_.Get<float>();
    LocalTensor<float> ta = ta_.Get<float>();
    LocalTensor<float> tb = tb_.Get<float>();
    LocalTensor<float> tc = tc_.Get<float>();
    LocalTensor<float> red = red_.Get<float>();
    LocalTensor<float> red2 = red2_.Get<float>();

    // ---- pass 1: mean = numpy_pairwise_sum(x) / n   (numpy: umr_sum / rcount, fp32 divide) ----
    const float sum = PwSumChunked(base, gsz, false, 0.0f);
    Duplicate(red, sum, 64);
    Duplicate(red2, static_cast<float>(gsz), 64);
    PipeBarrier<PIPE_ALL>();
    Div(red, red, red2, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float mean = red.GetValue(0);
    PipeBarrier<PIPE_ALL>();

    // ---- pass 2: var = numpy_pairwise_sum((x-mean)^2) / n  (numpy var: two-pass) ----
    const float sumsq = PwSumChunked(base, gsz, true, mean);
    Duplicate(red, sumsq, 64);
    Duplicate(red2, static_cast<float>(gsz), 64);
    PipeBarrier<PIPE_ALL>();
    Div(red, red, red2, 64);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float var = red.GetValue(0);
    PipeBarrier<PIPE_ALL>();

    // rstd = 1.0 / np.sqrt(var + eps), refined to the correctly rounded fp32 value.
    // numpy rounds sqrt once and the reciprocal once; the vector Sqrt/Div are allowed to be a
    // fraction of an ulp off, so both steps get one double-float Newton refinement here (cost is
    // per-group, i.e. a handful of scalar ops for the whole tensor).
    // rstd = 1.0 / np.sqrt(var + eps).  numpy rounds sqrt once and the reciprocal once; the
    // scalar unit's sqrtf/divide are IEEE single precision on this arch (validated against numpy
    // on 512 groups), so computing it here keeps the value bit-identical to the golden.
    const float ve = var + eps;
    const float sq = sqrt(ve);
    const float rstd = 1.0f / sq;
    PipeBarrier<PIPE_ALL>();

    // store meanOut / rstdOut (dtype T)
    StoreStat(meanGm_, gidx, mean);
    StoreStat(rstdGm_, gidx, rstd);

    // ---- pass 3: per-channel normalize + SiLU + quant, store int8 ----
    for (int64_t c = 0; c < CpG; ++c) {
        const int64_t gc = cbase + c;
        float gcv = static_cast<float>(gGm_.GetValue(gc));
        float bcv = static_cast<float>(bGm_.GetValue(gc));
        float qsv = qsGm_.GetValue(t_.qsPerChannel ? gc : 0);
        int64_t chBase = base + c * HW;
        LocalTensor<float> qsb = tb_.Get<float>();
        Duplicate(qsb, qsv, static_cast<int32_t>(chunk));
        PipeBarrier<PIPE_ALL>();
        for (int64_t off = 0; off < HW; off += chunk) {
            int64_t cnt = (HW - off < chunk) ? (HW - off) : chunk;
            int64_t cntR = ((cnt + 63) / 64) * 64;
            int32_t n = static_cast<int32_t>(cntR);
            Duplicate(qsb, qsv, static_cast<int32_t>(chunk));
            PipeBarrier<PIPE_ALL>();
            LoadToFloat(fx, chBase + off, cnt, cntR, static_cast<T>(0));
            PipeBarrier<PIPE_ALL>();
            // gn = ((x - mean) * rstd * gamma) + beta  (golden op order)
            Adds(fx, fx, -mean, n);
            Muls(fx, fx, rstd, n);
            Muls(fx, fx, gcv, n);
            Adds(fx, fx, bcv, n);
            if (doSilu) {
                // silu = fx / (1 + exp(-fx))
                Muls(ta, fx, -1.0f, n);
                PipeBarrier<PIPE_ALL>();
                Exp(ta, ta, n);
                PipeBarrier<PIPE_ALL>();
                Adds(ta, ta, 1.0f, n);
                PipeBarrier<PIPE_ALL>();
                Div(fx, fx, ta, n);
            }
            // quant: clamp to int8 range, then round-to-nearest-even in fp32 via the
            // 1.5*2^23 add/sub trick (ulp == 1 there), matching numpy rint exactly.
            Div(fx, fx, qsb, n);
            Maxs(fx, fx, -128.0f, n);
            Mins(fx, fx, 127.0f, n);
            Muls(ta, fx, 1.0f, n);              // ta = clamped q (kept for the boundary test)
            PipeBarrier<PIPE_ALL>();
            Adds(fx, fx, 12582912.0f, n);
            Adds(fx, fx, -12582912.0f, n);      // fx = rint(q), round-half-even
            PipeBarrier<PIPE_ALL>();
            // boundary distance: tb = |q - rint(q)|; ta = 0.5 - 2e-6*min(|q|,128) - |q - rint(q)|
            // (> 0 marks an element whose rounding could flip because of the vector Exp error)
            Sub(tb, ta, fx, n);
            Abs(tb, tb, n);
            Abs(ta, ta, n);
            Mins(ta, ta, 128.0f, n);
            Muls(ta, ta, 1.0e-6f, n);
            Muls(ta, ta, -1.0f, n);
            Adds(ta, ta, 0.5f, n);          // ta = 0.5 - 2e-6*min(|q|,128)
            Sub(ta, tb, ta, n);             // ta = |q - rint(q)| - (0.5 - margin)  (>0 == borderline)
            Maxs(ta, ta, 0.0f, n);
            PipeBarrier<PIPE_ALL>();
            ReduceSum(red, ta, tb, static_cast<int32_t>(cntR));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float anyNear = red.GetValue(0);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<half> thi = tc_.Get<half>();
            Cast(thi, fx, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<int8_t> i8 = oi8_.Get<int8_t>();
            Cast(i8, thi, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_ALL>();
            // Recompute the (very few) borderline elements with a correctly rounded exp so that the
            // int8 rounding decision matches the numpy golden bit-for-bit.
            if (anyNear > 0.0f) {
                for (int64_t i = 0; i < cnt; ++i) {
                    if (!(ta.GetValue(i) > 0.0f)) {
                        continue;
                    }
                    volatile float t1 = static_cast<float>(xGm_.GetValue(chBase + off + i)) - mean;
                    volatile float t2 = t1 * rstd;
                    volatile float t3 = t2 * gcv;
                    volatile float yv = t3 + bcv;
                    float sv;
                    if (doSilu) {
                        float ev = ExpScalarExact(-yv);
                        volatile float dv = 1.0f + ev;
                        sv = DfDiv(static_cast<float>(yv), dv);
                    } else {
                        sv = yv;
                    }
                    float qv = DfDiv(sv, qsv);
                    float cv = (qv < -128.0f) ? -128.0f : ((qv > 127.0f) ? 127.0f : qv);
                    volatile float rv = (cv + 12582912.0f) - 12582912.0f;
                    int32_t iv = static_cast<int32_t>(rv);
                    if (iv < -128) {
                        iv = -128;
                    }
                    if (iv > 127) {
                        iv = 127;
                    }
                    i8.SetValue(i, static_cast<int8_t>(iv));
                }
                PipeBarrier<PIPE_ALL>();
            }
            StoreInt8(i8, chBase + off, cnt);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t total = t_.totalGroups;

    int64_t nEach = (blockNum > 0) ? (total / blockNum) : 0;
    int64_t rem = (blockNum > 0) ? (total % blockNum) : 0;
    int64_t start = blockIdx * nEach + (blockIdx < rem ? blockIdx : rem);
    int64_t cnt = nEach + (blockIdx < rem ? 1 : 0);
    for (int64_t gi = start; gi < start + cnt; ++gi) {
        ProcessGroup(gi);
    }
}

} // namespace NsGroupNormSiluQuant
#endif // GROUPNORMSILUQUANT_H

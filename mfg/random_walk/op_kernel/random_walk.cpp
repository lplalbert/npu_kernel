// Kernel side of RandomWalk (node2vec style)
//   out[i, 0] = seed[i]; step j (1..walk_length):
//     neighbours of v are col[rowptr[v] .. rowptr[v+1])
//     uniform (p==q==1): v = col[start + floor(rand * deg)]
//     node2vec bias: weight(x) = (x==prev) ? 1/p : (adjacent(prev,x) ? 1 : 1/q), pick by CDF vs rand
//   isolated node (deg 0): stay, v and prev both unchanged, rand still consumed
//   prev = the node we were at before the move.
//
// Strategy: parallel over walkers (B dimension is naturally balanced); per core a
// chunk of walkers is walled serially with scalar GM reads (CSR random access).
// int64 output is written as int32 pairs (low = node id, high = 0).
#include "kernel_operator.h"

#include "random_walk_tiling.h"
#include "tiling_key_random_walk.h"

using namespace AscendC;

namespace {
constexpr int32_t UB_PAD = 8;
}  // namespace

template <typename T_RAND>
class KernelRandomWalk {
public:
    __aicore__ inline KernelRandomWalk() {}

    __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR seed, GM_ADDR rand, GM_ADDR out,
                                const RandomWalkTilingData &t) {
        n_ = static_cast<int64_t>(t.n);
        e_ = static_cast<int64_t>(t.e);
        b_ = static_cast<int64_t>(t.b);
        wl_ = static_cast<int64_t>(t.wl);
        p_ = t.p;
        q_ = t.q;
        wpc_ = static_cast<int64_t>(t.walkersPerCore);
        cw_ = static_cast<int64_t>(t.chunkW);
        block_ = static_cast<int64_t>(GetBlockIdx());
        L_ = wl_ + 1;

        rowGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rowptr), n_ + 1);
        colGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(col), e_ > 0 ? e_ : 1);
        seedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seed), b_);
        randGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T_RAND *>(rand), b_ * wl_ > 0 ? b_ * wl_ : 1);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(out), b_ * L_ * 2);

        const uint32_t cw = static_cast<uint32_t>(cw_);
        const uint32_t bufInts = cw * static_cast<uint32_t>(L_) * 2u + 2u * UB_PAD;
        pipe_.InitBuffer(outBuf_, bufInts * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (b_ <= 0 || wl_ <= 0) {
            return;
        }
        LocalTensor<int32_t> ob = outBuf_.Get<int32_t>();
        const int64_t L = L_;
        const int64_t wl = wl_;
        const bool uniform = (p_ == 1.0f) && (q_ == 1.0f);
        const float invp = 1.0f / p_;
        const float invq = 1.0f / q_;

        const int64_t w0 = block_ * wpc_;
        if (w0 >= b_) {
            return;
        }
        const int64_t w1 = (w0 + wpc_ < b_) ? (w0 + wpc_) : b_;
        const int64_t cw = cw_;

        for (int64_t ws = w0; ws < w1; ws += cw) {
            const int64_t we = (ws + cw < w1) ? (ws + cw) : w1;
            const int32_t nw = static_cast<int32_t>(we - ws);
            for (int32_t i = 0; i < nw; ++i) {
                const int64_t w = ws + i;
                int64_t v = seedGm_.GetValue(w);
                int64_t prev = -1;
                int32_t base = static_cast<int32_t>(i) * static_cast<int32_t>(L) * 2;
                ob.SetValue(base, static_cast<int32_t>(v));
                ob.SetValue(base + 1, 0);
                for (int64_t j = 0; j < wl; ++j) {
                    const float r = static_cast<float>(randGm_.GetValue(w * wl + j));
                    int64_t s = rowGm_.GetValue(v);
                    int64_t en = rowGm_.GetValue(v + 1);
                    if (s < 0) {
                        s = 0;
                    }
                    if (en > e_) {
                        en = e_;
                    }
                    const int64_t deg = en - s;
                    int64_t nx = v;
                    if (deg > 0) {
                        if (uniform) {
                            int64_t k = static_cast<int64_t>(r * static_cast<float>(deg));
                            if (k < 0) {
                                k = 0;
                            }
                            if (k >= deg) {
                                k = deg - 1;
                            }
                            nx = colGm_.GetValue(s + k);
                        } else {
                            // total weight
                            float total = 0.0f;
                            for (int64_t t = 0; t < deg; ++t) {
                                const int64_t x = colGm_.GetValue(s + t);
                                float wt;
                                if (prev >= 0 && x == prev) {
                                    wt = invp;
                                } else if (prev >= 0 && Adjacent(prev, x)) {
                                    wt = 1.0f;
                                } else {
                                    wt = invq;
                                }
                                total += wt;
                            }
                            const float target = r * total;
                            float cum = 0.0f;
                            nx = colGm_.GetValue(s + deg - 1);
                            for (int64_t t = 0; t < deg; ++t) {
                                const int64_t x = colGm_.GetValue(s + t);
                                float wt;
                                if (prev >= 0 && x == prev) {
                                    wt = invp;
                                } else if (prev >= 0 && Adjacent(prev, x)) {
                                    wt = 1.0f;
                                } else {
                                    wt = invq;
                                }
                                cum += wt;
                                if (cum > target) {
                                    nx = x;
                                    break;
                                }
                            }
                        }
                    }
                    base += 2;
                    ob.SetValue(base, static_cast<int32_t>(nx));
                    ob.SetValue(base + 1, 0);
                    if (deg > 0) {
                        prev = v;
                    }
                    v = nx;
                }
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyExtParams q;
            q.blockCount = 1;
            q.blockLen = static_cast<uint32_t>(static_cast<int64_t>(nw) * L * 2 * sizeof(int32_t));
            q.srcStride = 0;
            q.dstStride = 0;
            q.rsv = 0;
            DataCopyPad(outGm_[static_cast<int32_t>(ws * L * 2)], ob, q);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline bool Adjacent(int64_t prev, int64_t x) const {
        int64_t s = rowGm_.GetValue(prev);
        int64_t en = rowGm_.GetValue(prev + 1);
        if (s < 0) {
            s = 0;
        }
        if (en > e_) {
            en = e_;
        }
        for (int64_t t = s; t < en; ++t) {
            if (colGm_.GetValue(t) == x) {
                return true;
            }
        }
        return false;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> outBuf_;
    GlobalTensor<int64_t> rowGm_;
    GlobalTensor<int64_t> colGm_;
    GlobalTensor<int64_t> seedGm_;
    GlobalTensor<T_RAND> randGm_;
    GlobalTensor<int32_t> outGm_;
    int64_t n_ = 0;
    int64_t e_ = 0;
    int64_t b_ = 0;
    int64_t wl_ = 0;
    int64_t L_ = 0;
    int64_t wpc_ = 0;
    int64_t cw_ = 0;
    int64_t block_ = 0;
    float p_ = 1.0f;
    float q_ = 1.0f;
};

template <typename DT_RAND>
__global__ __aicore__ void random_walk(GM_ADDR rowptr, GM_ADDR col, GM_ADDR seed, GM_ADDR rand, GM_ADDR out,
                                       GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(RandomWalkTilingData);
    GET_TILING_DATA_WITH_STRUCT(RandomWalkTilingData, tiling_data, tiling);
    KernelRandomWalk<DT_RAND> op;
    op.Init(rowptr, col, seed, rand, out, tiling_data);
    op.Process();
}

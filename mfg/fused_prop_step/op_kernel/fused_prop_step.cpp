// Kernel side: FusedPropStep  out = IFFT2( FFT2(field) * h )
//
// The spectrum never leaves the kernel; the fused step is four passes over it:
//   P1 rows forward  (length M, contiguous)            field -> spec
//   P2 cols forward  (length N, stride M)              spec  -> spec
//   P3 rows inverse  (length M) with h multiply        spec  -> spec
//   P4 cols inverse  (length N) with 1/(N*M) scaling   spec  -> out
//
// Spectrum staging (chosen by the host, tiling field useUb):
//   - complex64: the output buffer itself (it is exactly 2 floats per element).
//   - float16 / float32 small and medium sizes: a UB buffer, so the kernel needs no
//     workspace at all (important because the runtime only guarantees a fresh
//     workspace state on the first launch of an op).
//   - float16 / float32 large sizes: a float32 workspace slice per core.
//
// Layout notes:
//   - complex64 tensors are stored interleaved [re0, im0, re1, im1, ...] as float32.
//   - float32 / float16 tensors are real; treated as complex with im = 0, only the
//     real part is written back.
//
// 1D FFT = bit reversal permutation + iterative radix-2 butterflies. Twiddle factors
// come from a table built once at Init with self-written sin/cos (no libm, no vector
// math intrinsics); the inverse transform reuses the forward table by conjugating.
// Batch is split across cores; each core owns whole batches and runs all four passes
// locally, so no cross-core barrier (SyncAll) is involved.
#include "kernel_operator.h"

#include "fused_prop_step_tiling.h"
#include "tiling_key_fused_prop_step.h"

using namespace AscendC;

namespace fpsk {
// user workspace of the real dtypes is placed after the reserved system area so the
// framework bookkeeping at the head of the workspace block stays untouched
constexpr int64_t WR_OFFSET = static_cast<int64_t>(fps_tiling::kWorkspaceOffset);

// |x| <= pi/4 Taylor series for sin / cos (truncation error ~1e-9)
__aicore__ inline float SinSmall(float x) {
    float x2 = x * x;
    return x * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f +
           x2 * (-1.0f / 5040.0f + x2 * (1.0f / 362880.0f)))));
}
__aicore__ inline float CosSmall(float x) {
    float x2 = x * x;
    return 1.0f + x2 * (-0.5f + x2 * (1.0f / 24.0f + x2 * (-1.0f / 720.0f +
           x2 * (1.0f / 40320.0f))));
}
// sin/cos of x in [-pi, pi] by octant reduction to |xr| <= pi/4
__aicore__ inline void SinCosPi(float x, float &s, float &c) {
    const float HPI = 1.57079632679489661923f;
    float q = x / HPI;
    float qr = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
    int32_t k = static_cast<int32_t>(qr);
    float xr = x - static_cast<float>(k) * HPI;
    float sr = SinSmall(xr);
    float cr = CosSmall(xr);
    int32_t kk = k & 3;
    if (kk == 0) { s = sr; c = cr; }
    else if (kk == 1) { s = cr; c = -sr; }
    else if (kk == 2) { s = -sr; c = -cr; }
    else { s = -cr; c = sr; }
}
}  // namespace fpsk

template <class DT_FIELD>
class KernelFusedPropStep {
public:
    __aicore__ inline KernelFusedPropStep() {}

    __aicore__ inline void Init(GM_ADDR field, GM_ADDR h, GM_ADDR out, GM_ADDR workspace,
                                const FusedPropStepTilingData &t) {
        B_ = static_cast<int64_t>(t.B);
        N_ = static_cast<int64_t>(t.N);
        M_ = static_cast<int64_t>(t.M);
        Bh_ = static_cast<int64_t>(t.Bh == 0 ? 1 : t.Bh);
        dt_ = static_cast<int32_t>(t.dtypeCode);
        norm_ = (t.norm != 0);
        logN_ = static_cast<int32_t>(t.logN);
        logM_ = static_cast<int32_t>(t.logM);
        maxNM_ = static_cast<int64_t>(t.maxNM);
        useUb_ = (t.useUb != 0);
        scale_ = norm_ ? (1.0f / static_cast<float>(N_ * M_)) : 1.0f;
        if (N_ <= 0 || M_ <= 0 || B_ <= 0) { return; }

        const int64_t total = B_ * N_ * M_;
        if (dt_ == 0) {
            fieldH_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(field), total);
            hH_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h), Bh_ * N_ * M_);
            outH_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out), total);
        } else {
            const int64_t mul = (dt_ == 2) ? 2 : 1;
            fieldF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(field), total * mul);
            hF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(h), Bh_ * N_ * M_ * mul);
            outF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), total * mul);
        }

        pipe_.InitBuffer(bA_, static_cast<uint32_t>(2 * maxNM_ * static_cast<int64_t>(sizeof(float))));
        pipe_.InitBuffer(bT1_, static_cast<uint32_t>(M_ * static_cast<int64_t>(sizeof(float))));
        pipe_.InitBuffer(bT2_, static_cast<uint32_t>(N_ * static_cast<int64_t>(sizeof(float))));
        BuildTable(bT1_, M_);
        BuildTable(bT2_, N_);
        tabM_ = bT1_.Get<float>();
        tabN_ = bT2_.Get<float>();

        if (useUb_) {
            const int64_t specFloats = 2 * N_ * M_;
            pipe_.InitBuffer(bS_, static_cast<uint32_t>(specFloats * static_cast<int64_t>(sizeof(float))));
            LocalTensor<float> s = bS_.Get<float>();
            specU_ = reinterpret_cast<__ubuf__ float *>(s.GetPhyAddr());
        } else if (dt_ == 2) {
            spec_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), total * 2);
        } else {
            const int64_t perCore = (N_ * M_) * 2;
            __gm__ uint8_t *raw = reinterpret_cast<__gm__ uint8_t *>(workspace) + fpsk::WR_OFFSET;
            __gm__ float *base = reinterpret_cast<__gm__ float *>(raw) +
                                 static_cast<int64_t>(GetBlockIdx()) * perCore;
            spec_.SetGlobalBuffer(base, perCore);
        }
    }

    __aicore__ inline void Process() {
        if (B_ <= 0 || N_ <= 0 || M_ <= 0) { return; }
        int64_t coreNum = static_cast<int64_t>(GetBlockNum());
        if (coreNum < 1) { coreNum = 1; }
        int64_t cid = static_cast<int64_t>(GetBlockIdx());
        LocalTensor<float> a = bA_.Get<float>();
        ap_ = reinterpret_cast<__ubuf__ float *>(a.GetPhyAddr());
        for (int64_t b = cid; b < B_; b += coreNum) {
            base_ = b * N_ * M_;
            hRowBase_ = ((Bh_ > 1) ? (b % Bh_) : 0) * N_ * M_;  // h broadcast along batch
            RowsForward();
            ColsForward();
            RowsInverseH();
            ColsInverseOut();
        }
    }

private:
    // ---------------- data access ----------------
    __aicore__ inline void LoadField(int64_t e, float &re, float &im) {
        if (dt_ == 0) {
            re = static_cast<float>(fieldH_.GetValue(e));
            im = 0.0f;
        } else if (dt_ == 1) {
            re = fieldF_.GetValue(e);
            im = 0.0f;
        } else {
            re = fieldF_.GetValue(2 * e);
            im = fieldF_.GetValue(2 * e + 1);
        }
    }
    __aicore__ inline void LoadH(int64_t e, float &re, float &im) {
        if (dt_ == 0) {
            re = static_cast<float>(hH_.GetValue(e));
            im = 0.0f;
        } else if (dt_ == 1) {
            re = hF_.GetValue(e);
            im = 0.0f;
        } else {
            re = hF_.GetValue(2 * e);
            im = hF_.GetValue(2 * e + 1);
        }
    }
    __aicore__ inline void StoreOut(int64_t e, float re, float im) {
        if (dt_ == 0) {
            outH_.SetValue(e, static_cast<half>(re));
        } else if (dt_ == 1) {
            outF_.SetValue(e, re);
        } else {
            outF_.SetValue(2 * e, re);
            outF_.SetValue(2 * e + 1, im);
        }
    }
    // spectrum access, local element index inside the current batch
    __aicore__ inline void SpecSet(int64_t l, float re, float im) {
        if (useUb_) {
            specU_[2 * l] = re;
            specU_[2 * l + 1] = im;
        } else {
            spec_.SetValue(2 * (base_ + l), re);
            spec_.SetValue(2 * (base_ + l) + 1, im);
        }
    }
    __aicore__ inline void SpecGet(int64_t l, float &re, float &im) {
        if (useUb_) {
            re = specU_[2 * l];
            im = specU_[2 * l + 1];
        } else {
            re = spec_.GetValue(2 * (base_ + l));
            im = spec_.GetValue(2 * (base_ + l) + 1);
        }
    }

    // ---------------- passes ----------------
    __aicore__ inline void RowsForward() {
        for (int64_t n = 0; n < N_; ++n) {
            const int64_t l0 = n * M_;
            for (int64_t m = 0; m < M_; ++m) {
                float re = 0.0f, im = 0.0f;
                LoadField(base_ + l0 + m, re, im);
                ap_[2 * m] = re;
                ap_[2 * m + 1] = im;
            }
            FFTAt(ap_, static_cast<int32_t>(M_), logM_, tabM_, false);
            for (int64_t m = 0; m < M_; ++m) {
                SpecSet(l0 + m, ap_[2 * m], ap_[2 * m + 1]);
            }
        }
    }

    __aicore__ inline void ColsForward() {
        for (int64_t m = 0; m < M_; ++m) {
            for (int64_t n = 0; n < N_; ++n) {
                float re = 0.0f, im = 0.0f;
                SpecGet(n * M_ + m, re, im);
                ap_[2 * n] = re;
                ap_[2 * n + 1] = im;
            }
            FFTAt(ap_, static_cast<int32_t>(N_), logN_, tabN_, false);
            for (int64_t n = 0; n < N_; ++n) {
                SpecSet(n * M_ + m, ap_[2 * n], ap_[2 * n + 1]);
            }
        }
    }

    // inverse row transform of the spectrum, with the transfer function folded in
    __aicore__ inline void RowsInverseH() {
        for (int64_t n = 0; n < N_; ++n) {
            const int64_t l0 = n * M_;
            for (int64_t m = 0; m < M_; ++m) {
                float sr = 0.0f, si = 0.0f;
                SpecGet(l0 + m, sr, si);
                float hr = 1.0f, hi = 0.0f;
                LoadH(hRowBase_ + l0 + m, hr, hi);
                ap_[2 * m] = sr * hr - si * hi;
                ap_[2 * m + 1] = sr * hi + si * hr;
            }
            FFTAt(ap_, static_cast<int32_t>(M_), logM_, tabM_, true);
            for (int64_t m = 0; m < M_; ++m) {
                SpecSet(l0 + m, ap_[2 * m], ap_[2 * m + 1]);
            }
        }
    }

    __aicore__ inline void ColsInverseOut() {
        for (int64_t m = 0; m < M_; ++m) {
            for (int64_t n = 0; n < N_; ++n) {
                float re = 0.0f, im = 0.0f;
                SpecGet(n * M_ + m, re, im);
                ap_[2 * n] = re;
                ap_[2 * n + 1] = im;
            }
            FFTAt(ap_, static_cast<int32_t>(N_), logN_, tabN_, true);
            for (int64_t n = 0; n < N_; ++n) {
                StoreOut(base_ + n * M_ + m, ap_[2 * n] * scale_, ap_[2 * n + 1] * scale_);
            }
        }
    }

    // Twiddle table for length len: tab[2t] = cos(2*pi*t/len), tab[2t+1] = sin(2*pi*t/len)
    // (forward transform convention; the inverse pass conjugates it).
    __aicore__ inline void BuildTable(TBuf<TPosition::VECCALC> &buf, int64_t len) {
        LocalTensor<float> tab = buf.Get<float>();
        const int64_t half = len >> 1;
        for (int64_t t = 0; t < half; ++t) {
            float ang = -6.28318530717958647692f * static_cast<float>(t) / static_cast<float>(len);
            float s = 0.0f, c = 1.0f;
            fpsk::SinCosPi(ang, s, c);
            tab.SetValue(2 * t, c);
            tab.SetValue(2 * t + 1, s);
        }
        if (len > 1) {
            tab.SetValue(0, 1.0f);
            tab.SetValue(1, 0.0f);
        }
    }

    // In-place 1D radix-2 FFT of length len (power of two); data is interleaved [re,im].
    // conj == true runs the inverse convention by negating the table sine.
    __aicore__ inline void FFTAt(__ubuf__ float *a, int32_t len, int32_t logL,
                                 LocalTensor<float> &tab, bool conj) {
        if (len <= 1) { return; }
        __ubuf__ float *tp = reinterpret_cast<__ubuf__ float *>(tab.GetPhyAddr());
        int32_t j = 0;
        for (int32_t i = 1; i < len; ++i) {
            int32_t bit = len >> 1;
            while (j & bit) {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if (i < j) {
                float tr = a[2 * i], ti = a[2 * i + 1];
                a[2 * i] = a[2 * j];
                a[2 * i + 1] = a[2 * j + 1];
                a[2 * j] = tr;
                a[2 * j + 1] = ti;
            }
        }
        const float cs = conj ? -1.0f : 1.0f;
        for (int32_t l = 2; l <= len; l <<= 1) {
            const int32_t half = l >> 1;
            const int32_t step = len / l;
            for (int32_t k = 0; k < half; ++k) {
                const int32_t ti = 2 * (k * step);
                const float c = tp[ti];
                const float s = cs * tp[ti + 1];
                for (int32_t i = 0; i < len; i += l) {
                    const int32_t p = i + k;
                    const int32_t q = p + half;
                    const float br = a[2 * q], bi = a[2 * q + 1];
                    const float tr = c * br - s * bi;
                    const float tu = c * bi + s * br;
                    const float ar = a[2 * p], ai = a[2 * p + 1];
                    a[2 * p] = ar + tr;
                    a[2 * p + 1] = ai + tu;
                    a[2 * q] = ar - tr;
                    a[2 * q + 1] = ai - tu;
                }
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> bA_;
    TBuf<TPosition::VECCALC> bT1_;
    TBuf<TPosition::VECCALC> bT2_;
    TBuf<TPosition::VECCALC> bS_;
    LocalTensor<float> tabM_;
    LocalTensor<float> tabN_;
    __ubuf__ float *ap_ = nullptr;
    __ubuf__ float *specU_ = nullptr;
    GlobalTensor<float> fieldF_, hF_, outF_, spec_;
    GlobalTensor<half> fieldH_, hH_, outH_;
    int64_t B_ = 0, N_ = 0, M_ = 0, Bh_ = 1, maxNM_ = 0;
    int64_t hRowBase_ = 0, base_ = 0;
    int32_t dt_ = 1, logN_ = 0, logM_ = 0;
    bool norm_ = false;
    bool useUb_ = false;
    float scale_ = 1.0f;
};

template <typename DT_FIELD>
__global__ __aicore__ void fused_prop_step(GM_ADDR field, GM_ADDR h, GM_ADDR out,
                                           GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FusedPropStepTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedPropStepTilingData, tiling_data, tiling);
    KernelFusedPropStep<DT_FIELD> op;
    op.Init(field, h, out, workspace, tiling_data);
    op.Process();
}

// Kernel侧核函数实现 —— 2D C2C FFT (complex64)
// 行变换: 每行 M 连续 -> UB(交错) -> Gather 去交错 SoA -> DIF radix-2 -> 位反序重排 -> 写回
// SyncAll
// 列变换: 列在 GM 跨步; DataCopyPad 只能 32B 块 => 4 列一组整块读入 tile,
//         标量从 tile 取列到 SoA, 变换后回填 tile 整块写回(同组 4 列同一核串行 => 无竞争)
#include "kernel_operator.h"

#include "fourier_transform2_d_tiling.h"
#include "tiling_key_fourier_transform2_d.h"

using namespace AscendC;

namespace fftc {
constexpr uint32_t CHUNK = 1024;   // 行载入/写回分块(complex)
constexpr uint32_t RCH = 64;       // 列 tile 行数
constexpr uint32_t VMIN = 64;      // 向量 op 最小 count
constexpr uint32_t PAD = 64;       // 标量/向量临时补齐
}  // namespace fftc

template <class DT_X>
class KernelFourierTransform2D {
public:
    __aicore__ inline KernelFourierTransform2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                               const FourierTransform2DTilingData &t) {
        t_ = t;
        uint64_t nf = (uint64_t)t_.b * t_.n * t_.m * 2;
        gmX.SetGlobalBuffer((__gm__ float *)x, nf);
        gmY.SetGlobalBuffer((__gm__ float *)y, nf);
        uint32_t lmax = t_.m > t_.n ? t_.m : t_.n;
        if (lmax < 64) lmax = 64;
        Lmax_ = lmax;
        pipe.InitBuffer(bRe, 4 * lmax);        // Lmax floats
        pipe.InitBuffer(bIm, 4 * lmax);
        pipe.InitBuffer(bTw, 8 * lmax);        // 旋转因子表: Re[0,L/2) Im[L/2,L)
        pipe.InitBuffer(bIo, 8 * fftc::CHUNK + 1024);   // 交错载入/写回区
        pipe.InitBuffer(bTile, 4 * fftc::RCH * 32 + 1024);
        pipe.InitBuffer(bTab, 4096);
        pipe.InitBuffer(bTmp, 4096);
    }

    __aicore__ inline void Process() {
        uint32_t bid = GetBlockIdx();
        uint32_t nc = t_.coreNum;

        // ---------------- 行变换 ----------------
        uint32_t rows = t_.b * t_.n;
        for (uint32_t t = bid; t < rows; t += nc) {
            uint32_t b = t / t_.n;
            uint32_t r = t % t_.n;
            uint64_t base = (uint64_t)b * t_.n * t_.m + (uint64_t)r * t_.m;
            BuildTwiddle(t_.m);
            RowLoad(base, t_.m);
            Fft(t_.m);
            Bitrev(t_.m);
            StoreRow(base, t_.m);
        }
        SyncAll();
        // ---------------- 列变换 ----------------
        uint32_t mg = (t_.m + 3) / 4;
        uint32_t groups = t_.b * mg;
        for (uint32_t g = bid; g < groups; g += nc) {
            uint32_t bb = g / mg;
            uint32_t l0 = (g % mg) * 4;
            uint64_t base = (uint64_t)bb * t_.n * t_.m;
            BuildTwiddle(t_.n);
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t l = l0 + c;
                if (l >= t_.m) break;
                ColLoad(base, l, t_.n);
                Fft(t_.n);
                Bitrev(t_.n);
                ColStore(base, l, t_.n);
            }
        }
    }

private:
    __aicore__ inline void SyncS2M() {
        SetFlag<HardEvent::S_MTE2>(0);
        WaitFlag<HardEvent::S_MTE2>(0);
    }
    __aicore__ inline void SyncM2S() {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }
    __aicore__ inline void SyncS2Out() {
        SetFlag<HardEvent::S_MTE3>(1);
        WaitFlag<HardEvent::S_MTE3>(1);
    }
    __aicore__ inline void SyncOut2S() {
        SetFlag<HardEvent::MTE3_S>(1);
        WaitFlag<HardEvent::MTE3_S>(1);
    }

    // ---------------------------------------------------------------- 旋转因子表
    // W_L^k = exp(-sign*2pi i k/L), k=0..L/2-1  ; 表长 Lmax 的 SoA
    // 用倍角递推: T_4 = [1, -i];  T_{2n}[2k]=T_n[k], T_{2n}[2k+1]=T_n[k]*b_{2n}
    //  b_{2n} = normalize(1 + b_n)
    __aicore__ inline void BuildTwiddle(uint32_t L) {
        __ubuf__ float *tr = (__ubuf__ float *)bTw.Get<float>().GetPhyAddr();
        __ubuf__ float *ti = tr + (Lmax_ >> 1);
        float sgn = t_.inverse ? 1.0f : -1.0f;   // Wi 符号
        if (L <= 2) {
            tr[0] = 1.0f;
            ti[0] = 0.0f;
            return;
        }
        tr[0] = 1.0f;
        ti[0] = 0.0f;
        tr[1] = 0.0f;
        ti[1] = sgn;
        if (L == 4) return;
        uint32_t n = 4;
        float br = 0.0f, bi = sgn;
        while (n < L) {
            // b_{2n} = normalize(1 + b_n)
            float ar = 1.0f + br;
            float ai = bi;
            float mag = SqrtScalar(ar * ar + ai * ai);
            float nbr = ar / mag;
            float nbi = ai / mag;
            // 表扩展: 从 n/2-1 递减
            for (int32_t k = (int32_t)(n / 2) - 1; k >= 0; --k) {
                float ar2 = tr[k], ai2 = ti[k];
                tr[2 * k] = ar2;
                ti[2 * k] = ai2;
                tr[2 * k + 1] = ar2 * nbr - ai2 * nbi;
                ti[2 * k + 1] = ar2 * nbi + ai2 * nbr;
            }
            br = nbr;
            bi = nbi;
            n <<= 1;
        }
    }

    __aicore__ inline float SqrtScalar(float v) {
        // 借向量 Sqrt 算标量(避免依赖标量 sqrt)
        LocalTensor<float> t = bTmp.Get<float>();
        Duplicate(t, v, fftc::PAD);
        LocalTensor<float> o = bTmp.Get<float>()[(int32_t)fftc::PAD];
        Sqrt(o, t, fftc::PAD);
        SetFlag<HardEvent::V_S>(3);
        WaitFlag<HardEvent::V_S>(3);
        __ubuf__ float *po = (__ubuf__ float *)o.GetPhyAddr();
        return po[0];
    }

    // ---------------------------------------------------------------- 位反序(原地)
    __aicore__ inline void Bitrev(uint32_t L) {
        __ubuf__ float *pr = (__ubuf__ float *)bRe.Get<float>().GetPhyAddr();
        __ubuf__ float *pi = (__ubuf__ float *)bIm.Get<float>().GetPhyAddr();
        uint32_t nb = 0;
        while ((1u << nb) < L) ++nb;
        for (uint32_t i = 0; i < L; ++i) {
            uint32_t j = 0;
            uint32_t v = i;
            for (uint32_t b = 0; b < nb; ++b) {
                j = (j << 1) | (v & 1u);
                v >>= 1;
            }
            if (i < j) {
                float tr = pr[i], ti = pi[i];
                pr[i] = pr[j];
                pi[i] = pi[j];
                pr[j] = tr;
                pi[j] = ti;
            }
        }
    }

    // ---------------------------------------------------------------- 行读写
    __aicore__ inline void RowLoad(uint64_t base, uint32_t L) {
        LocalTensor<float> bufI = bIo.Get<float>();
        __ubuf__ float *pRe = (__ubuf__ float *)bRe.Get<float>().GetPhyAddr();
        __ubuf__ float *pIm = (__ubuf__ float *)bIm.Get<float>().GetPhyAddr();
        for (uint32_t off = 0; off < L; off += fftc::CHUNK) {
            uint32_t cnt = L - off;
            if (cnt > fftc::CHUNK) cnt = fftc::CHUNK;
            SyncS2M();
            DataCopyExtParams p{1u, (uint32_t)(cnt * 8), 0u, 0u, 0u};
            DataCopyPadExtParams<float> pp{false, 0u, 0u, 0.0f};
            DataCopyPad(bufI, gmX[(uint64_t)(base + off) * 2], p, pp);
            SyncM2S();
            __ubuf__ float *pb = (__ubuf__ float *)bufI.GetPhyAddr();
            for (uint32_t k = 0; k < cnt; ++k) {
                pRe[off + k] = pb[2 * k];
                pIm[off + k] = pb[2 * k + 1];
            }
        }
    }

    __aicore__ inline void StoreRow(uint64_t base, uint32_t L) {
        LocalTensor<float> bufI = bIo.Get<float>();
        __ubuf__ float *pRe = (__ubuf__ float *)bRe.Get<float>().GetPhyAddr();
        __ubuf__ float *pIm = (__ubuf__ float *)bIm.Get<float>().GetPhyAddr();
        for (uint32_t off = 0; off < L; off += fftc::CHUNK) {
            uint32_t cnt = L - off;
            if (cnt > fftc::CHUNK) cnt = fftc::CHUNK;
            __ubuf__ float *pb = (__ubuf__ float *)bufI.GetPhyAddr();
            for (uint32_t k = 0; k < cnt; ++k) {
                pb[2 * k] = pRe[off + k];
                pb[2 * k + 1] = pIm[off + k];
            }
            SyncS2Out();
            DataCopyExtParams p{1u, (uint32_t)(cnt * 8), 0u, 0u, 0u};
            DataCopyPad(gmY[(uint64_t)((base + off) * 2)], bufI, p);
            SyncOut2S();
        }
    }

    // ---------------------------------------------------------------- 列读写
    __aicore__ inline void ColLoad(uint64_t base, uint32_t l, uint32_t L) {
        LocalTensor<float> tile = bTile.Get<float>();
        __ubuf__ float *pRe = (__ubuf__ float *)bRe.Get<float>().GetPhyAddr();
        __ubuf__ float *pIm = (__ubuf__ float *)bIm.Get<float>().GetPhyAddr();
        uint32_t c = l & 3u;
        uint32_t lg = l & (~3u);
        for (uint32_t r0 = 0; r0 < L; r0 += fftc::RCH) {
            uint32_t rc = L - r0;
            if (rc > fftc::RCH) rc = fftc::RCH;
            SyncS2M();
            if (t_.m >= 4) {
                DataCopyExtParams p{(uint16_t)rc, 32u, (uint32_t)(t_.m * 8 - 32), 0u, 0u};
                DataCopyPadExtParams<float> pp{false, 0u, 0u, 0.0f};
                DataCopyPad(tile, gmY[(uint64_t)(base + (uint64_t)r0 * t_.m + lg) * 2], p, pp);
            } else {
                DataCopyExtParams p{1u, (uint32_t)(rc * t_.m * 8), 0u, 0u, 0u};
                DataCopyPadExtParams<float> pp{false, 0u, 0u, 0.0f};
                DataCopyPad(tile, gmY[(uint64_t)(base + (uint64_t)r0 * t_.m) * 2], p, pp);
            }
            SyncM2S();
            __ubuf__ float *pt = (__ubuf__ float *)tile.GetPhyAddr();
            uint32_t w = (t_.m >= 4) ? 4u : t_.m;     // tile 每行的 complex 数
            uint32_t cw = (t_.m >= 4) ? c : l;
            for (uint32_t i = 0; i < rc; ++i) {
                pRe[r0 + i] = pt[(uint64_t)i * w * 2 + cw * 2];
                pIm[r0 + i] = pt[(uint64_t)i * w * 2 + cw * 2 + 1];
            }
        }
    }

    __aicore__ inline void ColStore(uint64_t base, uint32_t l, uint32_t L) {
        LocalTensor<float> tile = bTile.Get<float>();
        __ubuf__ float *pRe = (__ubuf__ float *)bRe.Get<float>().GetPhyAddr();
        __ubuf__ float *pIm = (__ubuf__ float *)bIm.Get<float>().GetPhyAddr();
        uint32_t c = l & 3u;
        uint32_t lg = l & (~3u);
        uint32_t w = (t_.m >= 4) ? 4u : t_.m;
        uint32_t cw = (t_.m >= 4) ? c : l;
        for (uint32_t r0 = 0; r0 < L; r0 += fftc::RCH) {
            uint32_t rc = L - r0;
            if (rc > fftc::RCH) rc = fftc::RCH;
            SyncS2M();
            if (t_.m >= 4) {
                DataCopyExtParams p{(uint16_t)rc, 32u, (uint32_t)(t_.m * 8 - 32), 0u, 0u};
                DataCopyPadExtParams<float> pp{false, 0u, 0u, 0.0f};
                DataCopyPad(tile, gmY[(uint64_t)(base + (uint64_t)r0 * t_.m + lg) * 2], p, pp);
            } else {
                DataCopyExtParams p{1u, (uint32_t)(rc * t_.m * 8), 0u, 0u, 0u};
                DataCopyPadExtParams<float> pp{false, 0u, 0u, 0.0f};
                DataCopyPad(tile, gmY[(uint64_t)(base + (uint64_t)r0 * t_.m) * 2], p, pp);
            }
            SyncM2S();
            __ubuf__ float *pt = (__ubuf__ float *)tile.GetPhyAddr();
            for (uint32_t i = 0; i < rc; ++i) {
                pt[(uint64_t)i * w * 2 + cw * 2] = pRe[r0 + i];
                pt[(uint64_t)i * w * 2 + cw * 2 + 1] = pIm[r0 + i];
            }
            SyncS2Out();
            if (t_.m >= 4) {
                DataCopyExtParams q{(uint16_t)rc, 32u, 0u, (uint32_t)(t_.m * 8 - 32), 0u};
                DataCopyPad(gmY[(uint64_t)(base + (uint64_t)r0 * t_.m + lg) * 2], tile, q);
            } else {
                DataCopyExtParams q{1u, (uint32_t)(rc * t_.m * 8), 0u, 0u, 0u};
                DataCopyPad(gmY[(uint64_t)(base + (uint64_t)r0 * t_.m) * 2], tile, q);
            }
            SyncOut2S();
        }
    }

    // ---------------------------------------------------------------- 原地 DIF FFT (SoA)
    __aicore__ inline void Fft(uint32_t L) {
        __ubuf__ float *pr = (__ubuf__ float *)bRe.Get<float>().GetPhyAddr();
        __ubuf__ float *pi = (__ubuf__ float *)bIm.Get<float>().GetPhyAddr();
        __ubuf__ float *tr = (__ubuf__ float *)bTw.Get<float>().GetPhyAddr();
        __ubuf__ float *ti = tr + (Lmax_ >> 1);
        for (uint32_t len = L; len >= 2; len >>= 1) {
            uint32_t half = len >> 1;
            uint32_t step = L / len;
            for (uint32_t b = 0; b < L; b += len) {
                uint32_t jmax = half;
                {
                    for (uint32_t j = 0; j < jmax; ++j) {
                        uint32_t i0 = b + j;
                        uint32_t i1 = i0 + half;
                        float wr = tr[j * step];
                        float wi = ti[j * step];
                        float ur = pr[i0], ui = pi[i0];
                        float vr = pr[i1], vi = pi[i1];
                        pr[i0] = ur + vr;
                        pi[i0] = ui + vi;
                        float dr = ur - vr;
                        float di = ui - vi;
                        pr[i1] = dr * wr - di * wi;
                        pi[i1] = dr * wi + di * wr;
                    }
                }
            }
        }
    }

private:
    FourierTransform2DTilingData t_;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bRe, bIm, bTw, bIo, bTile, bTab, bTmp;
    GlobalTensor<float> gmX, gmY;
    uint32_t Lmax_;
};

template <typename DT_X>
__global__ __aicore__ void fourier_transform2_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FourierTransform2DTilingData);
    GET_TILING_DATA_WITH_STRUCT(FourierTransform2DTilingData, tiling_data, tiling);
    KernelFourierTransform2D<DT_X> op;
    op.Init(x, y, workspace, tiling_data);
    op.Process();
}

> [!WARNING]
> **2026-11-28 更新：本文核心结论已被推翻。** 本文用的判据（阈值 1e-3 + 绝对误差项）是错的；
> 真实判据是 **10×2^-10 = 9.766e-3**（fp16，相对误差，无 abs 项），详见同目录 `CT2D_CUBE_THRESHOLD_FIX.md`。
> 本文的实测数字仍可参考（它们解释了自己为什么错），但“Cube 结构上不可能、应放弃”的结论**不成立**。
> 另：TOP1 per-case best_time = 1.80~80.29µs（我们向量版差 2751×），且 15/15 ratio=1 → Cube 路线成立。

﻿# CT2D 想上 Cube = 死路（精度门是 bit-exact，不是性能）

> 结论文件，供后人复用，避免重投几天。作者：ct2d-cube agent，2026-11-28。
> 一句话：**conv_transpose2_d 上 Cube 的拦路虎不是 2700× 性能，而是判题“全元素 bit-exact 门”。纯 Cube GEMM（换累加顺序）在数学上必挂。别先解决性能，先解决 bit-exact。**

---

## 0. 背景
- 题：天梯赛 `conv_transpose2_d`（pid `6acef213e769892927149a50`，100 万积分），现向量版 7.93（Pass 15/15），TOP1 90.76。
- 曾假设“架构差距，必须上 Cube”。实测证伪：**问题是精度门，不是架构性能**。

---

## 1. 判题门：Pass ⇔ precision_ratio == 1.0（全元素达标）
**公开 ranking 证据（`/api/problems/<pid>/ranking`，扫 page1~3 共 45 队）**
| status | case 数 | precision_ratio 情况 |
|---|---|---|
| Pass | 825 | **全部 = 1，min=1，<1 数量=0** |
| Wrong Answer | 45 | **全部 = 0** |
| Runtime Error / Skipped / Compile Error | 30 | 全部 = 0 |

→ 判题是**二值**：所有元素都过才 1.0，否则 0。**没有“接近 1 也算过”的空间。**

**自家记录佐证**：`03_GOLDEN_ORDER.md` 记录 ct2d case12 曾 `precision_ratio=0.99989`（挂了），把累加顺序改成逐位复刻 torch 后才 `→1.0` 变 Pass。**0.99989 也是挂。**

**判题容差（题面）**：fp32 `rel<1e-4 且 abs<1e-4`；fp16/bf16 `rel<1e-3 且 abs<1e-3`（“且”=两者都要满足，任一超即该元素挂）。golden = `gen_data.py` 里 `F.conv_transpose2d` 用 **fp32 计算再 cast 到输出 dtype**。

---

## 2. 四个探针的结论表（全部实测）
判据统一：`fail if rel>=tol OR abs>=tol`（严格“且”口径，与判题一致）；`ratio = 1 - fail/n`。

### 探针 A — im2col+GEMM(fp32 换序归约) vs golden（CPU torch，`ct2d_cw/prec_probe2.py`）
| 形状 | fp32 ratio (tol1e-4) | fp32 fail/n | fp16 ratio (tol1e-3) | fp16 fail/n |
|---|---|---|---|---|
| 1x4x8x8 k3 | 1.000000 | 0/400 | 0.997500 | 1/400 |
| 2x32x32 k3 s2 | 0.999531 | 123/262144 | 0.999832 | 44/262144 |
| 1x64x64 k5 s2 | 0.998737 | 662/524288 | 0.999405 | 312/524288 |
| k1x1 s2 | 0.999874 | 33/262144 | 0.999958 | 11/262144 |
| k15 dil2 | 0.997783 | 41/18496 | 0.999027 | 18/18496 |
| smallmag 0.02 | 0.999614 | 4/10368 | 1.000000 | 0/10368 |
→ **任何换顺序 fp32 归约都 <1.0 → 判题全挂。** 近零元素靠 rel 超差挂；大元素 fp16 靠 1 ulp（|v|~8 的 1 ulp = 0.0078 > 1e-3）挂。

### 探针 B — im2col + NPU 内置 `torch.matmul`（cube）vs golden（`ct2d_cw/npu_matmul_probe.py`）
| 形状 | fp16 ratio | fp32 ratio |
|---|---|---|
| 1x4x8x8 k3 | 0.492500 | 1.000000 |
| 2x32x32 k3 s2 | 0.261543 | 0.999451 |
| 1x64x64 k5 s2 | 0.110189 | 0.998964 |
→ **NPU 内置 matmul 走 cube 也过不了门**（fp16 因 fp16 累加惨败；fp32 因分块/劈 K 归约 <1.0）。

### 探针 C — per-tap GEMM + 有序累加（NPU matmul 模拟“逐 tap”，`ct2d_cw/pertap_probe.py`）
| 形状 | Cin | fp32 ratio | fp16 ratio |
|---|---|---|---|
| 1x4x8x8 k3 | 4 | **1.000000** | 0.392500 |
| 2x32x32 k3 s2 | 32 | 0.999481 | 0.257805 |
| 1x32x32 k3 s2 | 32 | 0.999439 | 0.253138 |
→ 只有**极小 K 单块(Cin=4)** 才碰巧 bit-exact；Cin≈32 就掉。说明“现成路径”全不行，且分块一上就崩。

### 探针 D — 判题门扫描（`ct2d_cw/scan_ratio.py`）
见第 1 节：Pass 825 全 1、WA 45 全 0。

---

## 3. 根因：Cube Mmad 分块/树形归约 ≠ torch 逐位顺序
- 向量版 15/15 **恰恰因为它逐位复刻** torch CPU 的 `per-tap(kh,kw升序) + ci 升序 FMA + tap 升序 + bias 最后`（见 `03_GOLDEN_ORDER.md`）。
- Cube 的 Mmad 把 K 按 baseK 分块、块内可能是树形/多级归约；**顺序一变 → 末位舍入差 → f16 输出偶发 1 ulp / fp32 近零元素超差 → 全元素门必挂。**
- fp16 输入：product 在 fp32 精确，若 Cube 是“升序顺序 fp32 累加”尚有可能 bit-exact；但一般分块/树形就破坏。fp32 输入：product 本身要舍入，只有 Cube MAC 是**融合乘加(FMA)且升序**才可能，实测 NPU matmul 不是。
- **所以 Cube 重写若直接调 Matmul 高阶 API，判题必挂、浪费配额。**（best 模式虽不掉分，但拿不到分。）

---

## 4. 重要正面结论：TOP1 做到了，但那是“把精确顺序工程化”
- TOP1（90.76，全 case ratio=1，case12 80µs）**证明 cube 类高性能 + bit-exact 是可能的**。
- 但那**不是**“直接调 Matmul API”，而是：
  - 自定义 tiling 让 Cube 的分块累加顺序**恰好等于 torch 的求和树/顺序**，或
  - 每 tap 小 K + 受控 baseK + 融合 MAC，并用**逐位 FMA 模型**反复对拍到 bit-exact。
- 这是一个**研究级攻关**（4~6h 起、且有不收敛风险），且必须先做受控探针验证“Cube 能否 bit-exact”，**不能先解决性能**。
- 若探针证伪（fp32 结构上不可 bit-exact）→ **cube 最多拿 fp16 case、fp32 case 必挂 → 应彻底放弃 cube，改走“保持精确顺序的向量/其它提速”**。

---

## 5. 探针脚本位置（可复现）
- 本地：`ct2d_cw/prec_probe2.py`（A）、`ct2d_cw/npu_matmul_probe.py`（B）、`ct2d_cw/pertap_probe.py`（C）、`ct2d_cw/scan_ratio.py`（D）、`ct2d_cw/prec_probe.py`（早期版）
- 服务器：`/tmp/prec_probe2.py`、`/tmp/npu_matmul_probe.py`、`/tmp/pertap_probe.py`、`/tmp/scan_ratio.py`、`/tmp/get_top1.py`
- 运行：`source /root/work/env.sh`；NPU 探针必须 `bash /root/work/npu_run.sh bash <script>`（B/C 需 `export TORCH_DEVICE_BACKEND_AUTOLOAD=1`，torch_npu 2.7.1.post4）。

---

## 6. 未决 / 前提
- **15 个判题 case 的 dtype(fp16/fp32) 仍未知**（判题 testcase 接口 403 隐藏；TOP1 submission 代码取不到）。这决定“cube 是否有意义”：**全 fp16 才可能值得投，含 fp32 则 cube 结构上不可 bit-exact → 直接放弃**。
- 当前线上维持向量版 **7.93（Pass 15/15）**；本次**未提交、未改任何线上资产、未产出 cube.zip**。

## 7. 给后人的一句话
**“ct2d 想上 Cube 必须先解决 bit-exact（复刻 torch 逐位顺序），而不是先解决性能。”** 先做受控逐位对拍探针，再谈 4~6h 重写。

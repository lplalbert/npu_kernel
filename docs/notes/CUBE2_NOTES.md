# CUBE2（QuantBatchMatMulV3 K-C 量化场景 **性能优化**）实现笔记

## 结论（2026-11-25）
- 工程：`/root/work/ladder/cube2`，7 个可编辑文件已实现，**真机 14 例全部 bit-exact**。
- 交付包：**`/root/work/ladder/cube2_7files.zip`**，md5 `3489e221c0f901802b56734520e44086`，
  7 条目 / `op_host/`+`op_kernel/` 前缀 / 无工程名前缀 / 与源码逐字节一致 / 禁词 0 / 非 ASCII 0 / CR 0。
- `bash build.sh -j8` 成功；`bash build.sh -e`（真机 aclnn example）成功。
- 官方 `gen_data.py` 数据 + 官方 `compare_data.py`：**PASSED! MERE=0.000000 MARE=0.000000**（bit-exact）。
- **未提交**，配额未动。

## 题面（`GET /api/problems/6ad9c7f83ff2fef7e81e894c`）
`out = bf16( int32(x1@x2) → fp32 → *scale[n] → *pertokenScale[m] )`，
int8 输入、bf16 输出、**仅 2D ND**、**无 transpose 属性**（这点与 cube1 不同，def 里不能带 transposeX1/X2，
否则判题机 examples 里的 aclnn 签名对不上）。
维度范围：`m<=256, n<=256, 32768<=k<=65535`；15 个 case 的 baseline 16.57~55.09 µs。
判题阈值：bf16 MERE<2^-7、MARE<10*2^-7（社区口径）；我们做到 **exact_diff=0**，同时满足更严的 rel/abs<1e-4。

## 与 cube1 的差异（复用结论）
| 项 | cube1 | cube2 |
|---|---|---|
| 属性 | transposeX1/X2 四种组合 | **无属性**（固定 x1=(m,k)、x2=(k,n)）|
| 目标 | 能力扩展（k>65535） | **纯性能**（k 最大 65535，反而更小）|
| golden impl | 同 | 同（只去掉 swapaxes 分支），累加顺序与舍入次数完全一致 |
- 直接沿用 cube1 已验证的结论：不用 `AscendC::Matmul` 高阶 API、按输出行块切核、无 workspace、
  UB 阻塞 GEMM（B tile 搬一次复用）、每 64/32 个 K 把 fp32 部分和折回 int32（保证 |part|<2^24 精确）、
  epilogue 复刻 golden 的两次 fp32 乘法 + bf16 RINT。

## 实现（`op_kernel/quant_batch_mat_mul_v3.h`）
- **按输出 tile 切核（2D：M×N）**：tiling 给出 `tileM / laneN / mTiles / nTiles`，核按
  `t = blockIdx; t < mTiles*nTiles; t += blockDim` 走网格，每个输出元素只被一个核写，
  **无跨核归约、完全不使用 workspace（ws=0）**。
- **laneN**：64/128/256（恒为 64 的倍数，满足"向量 count≥64 且 64 倍数"红线）。
  n≤64→64、n≤128→128、否则 256（尽量一个列 tile 覆盖全部 n）。
- **tileM**：`ceil(m / ceil(cores/nTiles))`，上限 `TILE_M_MAX=16`；m=256、40 核时 tileM=7（37 个 tile）。
- **UB 阻塞 GEMM**：`TILE_K=32` 的 B 块（32×laneN）用**一条** 2D `DataCopyPad` 搬入
  （`blockCount=kb, blockLen=cols, srcStride=n-cols 字节, dstStride=(laneN-cols)/32`，
  cols%32!=0 时退回逐行搬），只做一次 int8→half→fp32 转换后被 rows 行复用；
  `TQue<VECIN,2>` 双缓冲 + 提前一块预取，隐藏 MTE2 延迟。
- **A tile**：每 `A_TILE_K=256` 个 K 用一条 2D `DataCopyPad` 搬进 UB（blockCount=rows,
  blockLen=aLen, srcStride=k-aLen 字节），标量读前 `SetFlag/WaitFlag<HardEvent::MTE2_S>`；
  内层**一次 32bit 标量读喂 4 个连续 k**（`aW = aT.ReinterpretCast<int32_t>()` + 移位取字节），
  把标量读次数降到 1/4。aLen%32!=0 的尾巴退回 GM 标量读（正确性优先）。
- **精确性**：每个 TILE_K=32 的 fp32 块部分和上界 32·127·127=516128 < 2^24，是精确整数，
  `Cast(CAST_RINT)` 折 int32 无误差；k=65535 时总累加上界 1.06e9 < 2^31，int32 不溢出 ⇒ 与 numpy
  int32 matmul 完全一致。
- **Epilogue**（与 golden 同序）：`Cast(int32→fp32)` → `Mul(scale)`（scale 长度 1 时 `Muls` 广播）
  → `Muls(pertoken)`（pt 长度 1 时广播）→ `Cast(fp32→bf16, CAST_RINT)` → `DataCopyPad` 写回，
  每行一次。scale/pt 的 `(1,)` 广播都已支持。
- 所有向量 op count = `rows*laneN` / `laneN` / `TILE_K*laneN`，恒为 64 的倍数。

## tiling 侧的坑（实测得出，反直觉）
- 我先按"向量指令数"建模选了更窄的 lane（64/128），结果**大 case 慢 1.6 倍**（256×256×65535：9.36→14.67 ms）。
  逐 case 反推：**每次 Axpy 调用的实测成本 ≈ 17~20 ns（约 30 cycle），与 lane 宽度基本无关**
  ⇒ 时间 ∝ 每核 Axpy 调用数 = `rows * k`（每个列 tile）⇒ **lane 要尽量宽（一个列 tile 覆盖 n）**，
  行块要细到让 tile 数≈核数。已改回该策略。
- 把 `LocalTensor` 切片（accRow[TILE_M_MAX]、bRow[TILE_K]）提前建好、消除每次调用的
  `operator[]` 地址运算，**实测无收益**（4.53→4.54 ms，噪声内）⇒ 30 cycle 的成本在 Axpy API/向量流水本身，
  不在标量地址运算。

## 真机验证（自建驱动 `verify/npu/npu_main.cpp`，aclnn 两段式，无 transpose 参数）
`bash verify/check.sh`（gen → 跑 → 用官方阈值 cmp），**14/14 全部 `exact_diff=0`**：

| m | n | k | scaleLen | ptLen | 时间（avg of 5 / best）|
|---|---|---|---|---|---|
| 1 | 256 | 32768 | 256 | 1 | **1.20 / 1.00 ms**（= 官方 gen_data case）|
| 1 | 256 | 32768 | 1 | 1 | 1.20 / 1.00 ms |
| 16 | 64 | 512 | 64 | 16 | 0.20 / 0.06 ms |
| 37 | 17 | 513 | 17 | 37 | 0.31 / 0.18 ms |
| 40 | 200 | 1024 | 200 | 40 | 0.24 / 0.11 ms |
| 128 | 96 | 65535 | 96 | 128 | 4.54 / 4.34 ms |
| 256 | 128 | 32769 | 128 | 256 | 4.20 / 4.01 ms |
| 256 | 255 | 40000 | 255 | 256 | 5.82 / 5.63 ms |
| 3 | 100 | 65535 | 100 | 3 | 2.05 / 1.79 ms |
| 256 | 256 | 32768 | 256/1 | 256/1 | 4.72 / 4.52 ms |
| **256** | **256** | **65535** | 256 | 256 | **9.37 / 9.16 ms**（范围内最大规模）|
| 1 | 1 | 32768 | 1 | 1 | 1.53 / 1.38 ms |
| 32 | 64 | 512 | 64 | 1 | 0.19 / 0.06 ms |

- 全部远低于判题 3 s 看门狗（`aclrtSynchronizeStreamWithTimeout(3000)`），最大规模 9.4 ms 有 320 倍余量。
- 官方 `gen_data.py`（m=1,n=256,k=32768, 全 1 输入）+ 官方 `compare_data.py`：
  **PASSED! MERE=0.000000, MARE=0.000000**，golden[0]=32768 = 我们输出。

## 性能位置与后续空间（若还要冲分）
- 我们 1.2~9.4 ms vs baseline 16.6~55.1 µs ⇒ 慢 40~300 倍，**Pass 无风险但 score 很低**。
- 向量 GEMM 的地板：总 Axpy 调用数 = `m*k*ceil(n/lane)`，实测 17~20 ns/次，
  256×256×65535 时 16.8M 次调用 / 40 核 ≈ 8.5 ms ⇒ **已接近该路线的极限**。
- 唯一能进 30 倍以上的是 **Cube 单元**（`AscendC::Mmad` + `LoadData`/`SetFmatrix`/Fixpipe，
  需 `ASCEND_IS_AIC/AIV` 分流 + L1/L0A/L0B/CO1 手工管理 + int8 fractal 布局 + bf16 fixpipe 量化参数），
  工作量大、风险高，本轮未做（time box 内优先保证 Pass）。
- 次优杠杆：split-K（每核算一段 k，partial int32 落 workspace，尾部一次跨核归约）可把 m=1 这类
  小 M case 的并行度从 1~4 核提到 40 核，理论上 m=1 可再快 8~10 倍；代价是 workspace + 跨核同步。

## 纪律
未提交；配额未动；zip 已过"7 条目 / 前缀 / 逐字节 / 禁词 / 非 ASCII"检查。

## 复现
```bash
# 构建 + 安装 + 自建驱动对拍（14 例）
bash /root/work/ladder/cube2/verify/rebuild_check.sh
# 官方数据路径
bash /root/work/ladder/cube2/verify/official_check.sh   # 需要先按脚本内名称拷贝 bin
```

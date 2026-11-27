# Gf2Poly（gf2poly）工程 / 评审 / 修复记录

题面：`/root/work/ladder_specs/Gf2Poly.md`。工程：`/root/work/ladder/gf2poly`（7 个 editable 文件）。
交付包：`/root/work/ladder/gf2poly_7files.zip`（7 entries，无前缀）。**未提交**。本卡 CANN 9.2.0 / 910B。

## 0. 结论速览

| 项 | 结果 | 证据 |
|---|---|---|
| clean 重建（`rm -rf build build_out`） | 通过 | `/tmp/fixb_gf2poly_example.log`：`Build completed successfully`（最终源码） |
| 官方样例 `build.sh -e` | 通过 | `/tmp/fixb_gf2poly_example.log`：`PASS: y.bin, mismatched=0/1`（m=8,mul_reduce → 193） |
| 自建真机全量对拍（5 种 m × 3 种 mode） | **130/130 PASS** | `/tmp/fixb_gf2p_full.log`：`TOTAL=130 PASSED=130 FAILED=0`，`SELFCHECK OK` |
| 补充大/边界对拍（65535/262143/1048575 等） | **20/20 PASS** | `/tmp/fixb_gf2p_supp.log`：`SUPP TOTAL=20 PASSED=20 FAILED=0` |
| 7 文件 zip 刷新 | 是 | `gf2poly_7files.zip` 10041 B，7 entries（op_host×3 + op_kernel×4） |

对拍基准：题面 §3.1 reference（`samples/reference.py`）。**本题 `tests/ut/op_kernel/gf2_poly_data/` 下只有 README.md，
没有 `gen_data.py`/golden**，因此以 reference.py 为唯一权威；Codex 与 golden 冲突时按 reference.py 裁定。
自建对拍额外用向量化参考（clmul + 长除）并由 `SELFCHECK` 与 reference.py 对齐后才上真机。

## 1. Codex 评审与修复（2026-11-24）

评审输入：`_sec5_build/ascend_tasks/ladder_review/gf2poly.md`（`codex_review`, gpt-6-sol, effort=low）。
Codex 结论：「不通过」，top3 = ①形状/空张量/长度校验缺失（可构造越界写与错值）②m/mode 解析不一致与校验缺失 ③dtype 分支死代码。

### 1.1 判定（采信 / 部分采信 / 不采信）

| Codex 主张 | 判定 | 依据 |
|---|---|---|
| infershape 取 `N=dim(0)`，tiling 用 storage shape 各维乘积，rank>2 时输出形状与实际写入不一致（越界写） | **部分采信** | 题面契约固定 rank=2 [N,W]、连续存储，合法输入下二者等价；但确实需要统一：infershape 改为 `N=totalElems/W`（与 tiling 同一算法），并要求 rank>=1、`totalElems%W==0`；tiling 显式要求 a/b/t rank2 且形状精确匹配 |
| `m` 非法被静默改为 8，而 infershape 仍用原 m | **采信** | 两处都改为白名单校验，非法直接 `GRAPH_FAILED` |
| `totalElems/width` 截断余数；空张量强制 `totalRows=1` 导致 N=0 时仍写一行（越界写） | **采信** | 去掉截断；N=0 → `rowsPerCore=0`、blockDim=1、kernel 空跑，GM 长度按 ≥1 声明 |
| `mode` 两处解析规则不一致（infershape 把任何 `r*` 当 reduce） | **采信** | 两处统一为精确 `mul/reduce/mul_reduce`，非法即失败；`mode` 缺失按 def 默认 `mul_reduce` |
| `mod_poly` 长度未核对（kernel 固定读 W+1 字节） | **采信（只校验长度）** | tiling 校验 `mod_poly.shape == [W+1]`。首一性（`mod_poly[W]==1`）属数据检查，tiling 不能读输入数据，按题面 §3.4 由输入保证 |
| def 只声明 uint8，tiling 却按 fp16/bf16 选 MODE_0、kernel 实例化 half | **采信** | 删除 MODE_0：`gf2_poly_tiling_key.h` 只留 schMode=0、`gf2_poly.cpp` 只实例化 `uint8_t`、tiling 校验 4 输入 + 1 输出全 uint8。单 key 编译已验证 |
| t 必须 [N,2W]、a/b 必须同形 | **采信** | 显式校验 `t==[N,2W]`、`b==a` |
| 对齐 / `modBuf` / `LoadWords` 的 `ReinterpretCast` 不越界 | **不采信为缺陷**（与 Codex 结论一致） | 未改：`aligned8` 仅当 `width%8==0` 启用（tile 起点按整行偏移，天然 8B 对齐）；`modBuf` 按 32B 向上分配，最多访问第 W 字节 |
| GF(2) 位运算主干（clmul 4-bit 窗口、长除法、Barrett、mu 预处理）有错 | **不采信** | 静态推演与 Codex 一致未发现反例；实测 150 例（含 m=64/128/256 各模式，稀疏/稠密/单项式/零常数项 mod_poly）逐字节相等 |
| 性能：整行标量、表每行重建、逐位长除法、每 tile `PipeBarrier<PIPE_ALL>` | **采信但不改** | 与本轮修复无冲突；真正的吞吐提升需要向量化重写（超出「评审→修复→验证」范围）。如实记录在 §1.3，作为已知弱项 |

### 1.2 实际修改（文件 / 函数）

1. `op_host/gf2_poly_tiling.cpp::Gf2PolyTilingFunc`
   - 属性强校验：`m ∈ {8,32,64,128,256}`、`mode ∈ {mul,reduce,mul_reduce}`（缺省 mul_reduce）。
   - dtype 强校验：4 个输入 + 输出必须 `DT_UINT8`。
   - 形状契约：a/b rank2 且同形 `[N,W]`、t `[N,2W]`、mod_poly `[W+1]`、`0<=N<=2^20`；删除 `totalElems/width` 截断。
   - 空张量：`N=0` → blockDim=1、`rowsPerCore=0`、kernel 空跑。
   - `tileBytes` 圆整到 32B；`outWidth` 只由 mode 决定（mul → 2W）。
2. `op_host/gf2_poly_infershape.cpp::InferShapeGf2Poly`：与 tiling 共用 `ValidM` 与精确 mode 解析；`N=totalElems/W`（检查整除、rank>=1），非法直接失败。修掉 `r*` 前缀误判。
3. `op_kernel/gf2_poly_tiling_key.h`：只保留 `GF2POLY_TPL_SCH_MODE_0`（uint8）。
4. `op_kernel/gf2_poly.cpp`：只实例化 `Gf2Poly<uint8_t>`。
5. `op_kernel/gf2_poly.h`
   - `Init`：`rowsPerCore_` 允许 0；a/b/t/y 的 GM 长度按 `max(1, bytes)` 声明（空张量安全）。
   - `DoTile`：`rows<=0` 直接返回。
   - 位运算核心（`Gf2ClMul` / `Gf2Reduce` / `Gf2ComputeMu` / `Gf2ReduceBarrett`）不做改动。
6. `op_host/gf2_poly_def.cpp`：语义不变（uint8 四输入一输出、m int、mode 可选默认 mul_reduce，与题面一致），
   仅把中文注释改为英文（红线：注释勿中文）。改后再次 clean 重建 + 全量复验。

### 1.3 验收证据（含性能实测）

- 构建：`rm -rf build build_out && bash build.sh -j8` → `Build completed successfully`（单 tilingKey 编译通过）。
- 官方 golden：`bash build.sh -j8 -e` → `PASS: y.bin, mismatched=0/1`、`PASSED`。
- 全量对拍（`run_tests.py`，130 例，device 输出与 reference.py 逐字节比较）：
  - m ∈ {8,32,64,128,256} × mode ∈ {mul,reduce,mul_reduce} × {N=1,31,33}，N=4096（dense/sparse/monomial/lowconst0），N=65536，以及 m<=64 的 N=1048576 / m>=128 的 N=262144；覆盖 `T=P`、零常数项 mod_poly、已约减输入等边界。
- 补充对拍（`supp.py`，20 例）：N=65535、262143、1048575、2、3 等非对齐/超大点全 PASS。
- 性能（device 计时，含 ~0.6ms 主端/launch 底噪；本轮未改算法，与修复前同量级）：

  | case | ms |
  |---|---|
  | m=8 mul_reduce N=1048576 | 21.0 |
  | m=32 mul_reduce N=1048576 | 37.9 |
  | m=64 mul_reduce N=1048576 | 50.6 |
  | m=64 mul N=1048576 | 18.7 |
  | m=256 mul_reduce N=262144 | 89.3 |
  | m=256 mul N=262144 | 31.3 |

  已知弱项：全程标量 UB 访问、每行重建 4-bit 窗口表，有效带宽约 0.5–1.7 GB/s；若要冲榜需向量化（本轮未做）。

### 1.4 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/inner:$LD_LIBRARY_PATH
export PYTHONPATH=/model/envs/mull-npu/lib/python3.10/site-packages:$PYTHONPATH
export DEVICE_ID=0
cd /root/work/ladder/gf2poly && rm -rf build build_out && bash build.sh -j8   # 重建
cd /root/work/ladder/gf2poly && bash build.sh -j8 -e                        # 官方样例 golden
bash /root/work/gf2p_verify.sh                                              # 130 例全量 + 20 例补充
bash /root/work/pack.sh                                                     # 重打 7 文件 zip
```

日志：`/tmp/fixb_gf2poly_build.log`、`/tmp/fixb_gf2poly_example.log`、`/tmp/fixb_gf2p_verify.log`、
`/tmp/fixb_gf2p_full.log`、`/tmp/fixb_gf2p_supp.log`。
（`run_tests.py`、`supp.py`、`examples/test_gf2poly_io.cpp` 是验证脚手架，不属于 7 件套、不提交。）

---

## 2. 性能优化 v2（2026-11-27，本卡 910B2 / CANN 9.0.0）

交付：`/root/work/npu_kernel/ladder/zips/gf2_poly_7files_v2.zip`（7 entries，**不覆盖原包**）。
工程：`/root/work/ws/gf2_poly`（build_op.sh 从 v2 zip 重铺可复现）。

### 2.1 基线诊断（device 计时，--repeat 5 取 min，单核 43690 行/24 核）
| m | mode | N=1M 基线 µs | 每行每核 cycles | 结论 |
|---|---|---|---|---|
| 8 | mr | 19937 | ~820 | 每行固定开销主导（nw=1 却走 4bit 表） |
| 32 | mr | 36779 | ~1500 | 同上 |
| 64 | mr | 49922 | ~2000 | Barrett(3 次 clmul) 主导 |
| 128 | mr | 122440 | — | Barrett 主导 |
| 256 | mr | 351952 | ~14500 | Barrett 主导；nibble 循环 ~4cyc/op |
- 核数已用满（N=1..24 耗时持平，~24 AIV 核）；不是多核问题。
- 主因：**标量 4-bit 窗口 clmul 的 nibble 累加循环**（64 nibble × (nw+1) word × 带分支）+ **每行重建 3 次窗口表**（barrett 的 mu/P 表本可复用）。

### 2.2 本轮改动（op_kernel/gf2_poly.h + tiling）
1. `Gf2ClMul`：nw==1 且两操作数 <2^32（m=8/32）走 **ctz 置位位循环**（乘积 <2^63 单字），省掉 4bit 表；其余 nw 去掉 `if(val==0) continue` 与 `dst<total` 分支，`sft==0` 提到 word 循环外（**分支less**，z 补 1 word 余量）。
2. **Barrett 常量乘查表上提**：新增 `Gf2BuildWinTab`/`Gf2MulByTab`，在 `Init` 里对**全核共享**的 `muLow_`、`P_` 各建一次 16 项窗口表 `tabMu_/tabP_`，`Gf2ReduceBarrett` 的两处 clmul（q1·mu、q3·P）改用 `Gf2MulByTab`，**每行少建 2 张表**。
3. `tiling`：`TARGET_TILE_BYTES` 8KB→16KB（减少 per-tile `PipeBarrier<PIPE_ALL>` 次数；UB 约 96KB）。
4. 未改语义/契约/精度路径；注释全英文，无 cout/cin/非 ASCII。

### 2.3 验收
- 官方模板 `build.sh -j8 -e`：`PASS: y.bin, mismatched=0/1` `PASSED`。
- 自建对拍（device 输出 vs samples/reference.py，逐字节）：m∈{8,32,64,128,256} × mode∈{mul,reduce,mul_reduce} × N∈{1,4096} + N=16385（mr），共 35+ 例 **全 PASS**；`CORRECTNESS_ALL_OK`。
- 同进程多次 launch：`DETERMINISM_OK`（重复 launch 输出逐字节一致）。
- 日志：`/tmp/g2_finalall.log`、`/tmp/g2_final.log`、`/tmp/g2_scale.log`。

### 2.4 提速（N=1048576，µs，min/REP5）
| m | mode | 前 | 后 | 倍数 |
|---|---|---|---|---|
| 8 | mul_reduce | 19937 | 8458 | 2.36x |
| 8 | mul | 14760 | 3339 | **4.42x** |
| 8 | reduce | 6962 | 6804 | 1.02x |
| 32 | mul_reduce | 36779 | 27564 | 1.33x |
| 64 | mul_reduce | 49922 | 37045 | 1.35x |
| 128 | mul_reduce | 122440 | 78876 | 1.55x |
| 256 | mul_reduce | 351952 | 203075 | 1.73x |
| 256 | mul | 120802 | 76822 | 1.57x |
| 256 | reduce | 240386 | 129311 | 1.86x |

### 2.5 下一步（冲 TOP1 必须向量化，路径已定）
标量已接近极限；TOP1 单核 ~73 cycles/行（256bit clmul+reduce），需 **位并行/行并行向量化**：
- **lane=行、按字拆成独立 vector**（`W/4` 个 uint32 tensor，R 行/块，R 为 64 倍数，满足 op count≥64）：GM→UB 用 `DataCopyPad` 每字一次 2D 拷贝（blockLen=4B, srcStride=W-4B），**无需 UB 转置**。
- clmul(a,b)：逐 bit t of b，`mask = Muls((b>>t)&1, -1)` 对全部字 tensor 通用（lane=行，无广播）；`z_w ^= (a<<t)_w & mask`，跨字进位用 `a_{w-1}>>(32-t)`。估 ~140 指令/行（m=256）→ 约 20-35x。
- Barrett reduce：mu/P 为常量，同上逐位置位/窗口展开，或复用 tabMu/tabP。
- 余行（<R）走现有标量尾处理。m=8 可用 uint16 lane（1 字节/行，乘积 16bit）。
- 预计 m=256 mul_reduce 352ms → 10~20ms 量级；需逐位对拍（reference.py）保证精确。

---

## 3. 向量化 v3（2026-11-27）— 行并行 int16 lane

交付：`/root/work/npu_kernel/ladder/zips/gf2_poly_7files_v3.zip`（7 entries，不覆盖 v1/v2）。
工程：`/root/work/ws/gf2_poly`。

### 3.1 关键硬件约束（踩坑）
- **dav_c220 向量 `Xor/And/Or` 只支持 int16_t/uint16_t**（uint32 编译期 static_assert 拒绝）。
- **向量 `Adds/Muls` 只支持 half/float/int16_t/int32_t**（`unsigned short` 被拒！）→ 故全部用 **int16_t** 元素。

### 3.2 设计：lane=行、按 16bit 字拆独立 vector
- 一个 tile = R 行；对每个 16bit 字 w（`nw16 = W/2`），把该字在所有行上的值存成连续 vector（长度 R）→ "字列"。
- 位串 clmul：对 b 的每一位 t，`mask = Muls((b>>off)&1, -1)`（bit0 隔离用两次 Shift），对所有字 tensor 通用（lane=行 → 无需行内广播）；`z[bt+w] ^= (a<<off)_w & mask`；跨字进位用 `a[w-1]>>(16-off)`。
- Barrett：mu/P 为常量 → 预存 **置位位置表** `muBits_/pBits_`，逐置位 `z ^= q<<j`（`VMulConst`）。
- I/O：`DataCopyPad` 每字一次 2D 拷（GM 侧 `blockLen=2`,`srcStride=W-2` 字节；UB 侧 `dstStride=0`）。需 `GlobalTensor<int16_t>` 视图（DataCopyPad 要求同元素类型）。
- `R` 按 UB 预算自适应（`(9*nw16+4)*2` 字节/行，~150KB）：m=64→1920, m=128→960, m=256→512（须 64 倍数）。
- 余行（<R）走原标量 `DoTile`。m=8/32（W=1/4）仍走标量路径（vec_ 仅在 m≥64 且 W%8==0 打开）。

### 3.3 验收
- 官方模板 `build.sh -e`：`PASS` `PASSED`；`CORRECTNESS_ALL_OK`（m8/32/64/128/256 × 3 模式 × N=1/4096 + N=16385）；`DETERMINISM_OK`×5。
- 逐位对拍 `samples/reference.py` 全一致。日志 `/tmp/g2_finalall3.log`、`/tmp/g2_v4/v5/v6.log`。

### 3.4 提速（N=1048576，µs，min/REP5；对比 v1 原始）
| m | mode | v1 | v3 | 倍数 |
|---|---|---|---|---|
| 64 | mul_reduce | 49922 | 10370 | 4.8x |
| 128 | mul_reduce | 122440 | 20458 | 6.0x |
| 256 | mul_reduce | 351952 | 53269 | **6.6x** |
| 256 | reduce | 240386 | 28683 | 8.4x |
| 128 | reduce | 48680 | 13537 | 3.6x |
| 64 | reduce | 22650 | 8147 | 2.8x |
| 256 | mul | 120802 | 38004 | 3.2x |
| 128 | mul | 31765 | 18510 | 1.7x |

### 3.5 剩余瓶颈 / 下一步
- 每字 2 字节块 DMA（32B 扇区放大 ~16x）估占 m=256 reduce 约 20/28ms → **I/O 是当前最大瓶颈**。
- 修法：一次搬连续 [R,W]（1 次大 DMA）+ UB 内 **16x16 转置**（`Transpose` 的 `TRANSPOSE_ND2ND_B16`，int16）把 [R][nw16] 转成 [nw16][R]；m=128/64 补零到 16 字。预计 m=256 mr 53→~30ms。
- 计算侧仍是 O((8W)^2) 位串；进一步的 Karatsuba / 8-bit 窗口需按行 select，暂不划算。

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

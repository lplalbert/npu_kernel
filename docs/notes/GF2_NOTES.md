# Gf2kMulSquare（gf2）工程 / 评审 / 修复记录

题面：`/root/work/ladder_specs/Gf2kMulSquare.md`。工程：`/root/work/ladder/gf2`（7 个 editable 文件）。
交付包：`/root/work/ladder/gf2_7files.zip`（7 entries，无前缀）。**未提交**。本卡 CANN 9.2.0 / 910B。

## 0. 结论速览

| 项 | 结果 | 证据 |
|---|---|---|
| clean 重建（`rm -rf build build_out`） | 通过 | `/tmp/fixb_gf2_example.log`：`Build completed successfully`（最终源码） |
| 官方样例 `build.sh -e` | 通过 | `/tmp/fixb_gf2_example.log`：`PASS: c.bin, mismatched=0/1`（a=87,b=131,k=8,mul → 193） |
| 自建真机全量对拍 | **62/62 PASS** | `/tmp/fixb_gf2_full.log`：`TOTAL=62 PASSED=62 FAILED=0`，`SELFCHECK OK` |
| 判题规模补测（65535 / 百万行 × k=128） | 通过 | `/tmp/fixb_gf2_big.log` |
| 7 文件 zip 刷新 | 是 | `gf2_7files.zip` 8675 B，7 entries（op_host×3 + op_kernel×4） |

对拍基准：题面 §3.1 reference（`samples/reference.py`，逐位精确）。**本题 `tests/ut/op_kernel/gf2k_mul_square_data/`
下只有 README.md，没有 `gen_data.py`/golden**，因此以 reference.py 为唯一权威；Codex 与 golden 冲突时按 reference.py 裁定。
自建对拍用向量化参考（k=8 查表、k=128 clmul+移位约减）并由 `SELFCHECK` 逐例与 reference.py 对齐后才上真机。

## 1. Codex 评审与修复（2026-11-24）

评审输入：`_sec5_build/ascend_tasks/ladder_review/gf2.md`（`codex_review`, gpt-6-sol, effort=low）。
Codex 结论：「不通过」，top3 = ①属性/形状缺校验、静默算错 ②dtype 分支与 op def 矛盾（死代码）③缺真机验证 + 标量路径性能。

### 1.1 判定（采信 / 部分采信 / 不采信）

| Codex 主张 | 判定 | 依据 |
|---|---|---|
| `k∉{8,128}` 被静默改成 8（`tiling.cpp` 原 157–159 行），会静默算错 | **采信** | 题面 k∈{8,128}；非法属性不评测，但静默降级不可接受，改为 `GRAPH_FAILED` |
| `totalElems/width` 整除截断 + 用 storage shape 推行数，而 infershape 用 logical shape | **部分采信** | 契约固定 rank=2、形状 [N,k/8]、连续存储，storage==logical，二者本就一致；但截断确实掩盖非法形状 → 改为显式校验 rank/末维/b 同形，行数直接取 dim(0) |
| 反例 `[1,15], k=128` → totalRows=0 静默不写 | **采信** | 现在 infershape/tiling 都直接失败（`a.shape[1]!=16`） |
| 空张量路径 | **部分采信** | 题面 §3.4 明确「无空张量」测试；仍加固为 `totalRows=0` → blockDim=1、kernel 空跑、GM 长度按 ≥1 声明，不越界不崩 |
| def 只声明 uint8，但 tiling 按 fp16/bf16 选 MODE_0、kernel 实例化 `half`（死代码/矛盾） | **采信** | 删除 MODE_0：`tiling_key.h` 只保留 schMode=0、`gf2k_mul_square.cpp` 只实例化 `uint8_t`、tiling 校验 dtype 必须 uint8。单 key 编译已验证 |
| `DataCopyPad::blockLen` 是字节单位、`InitBuffer` 非 32B 自动补齐、k=128 `ReinterpretCast<uint64_t>` 不越界 | **不采信为缺陷**（与 Codex 结论一致） | 未改；另把 `tileBytes` 显式圆整到 32B，让 UB 缓冲长度确定对齐 |
| `Gf2MulWords` 的 `wl==1` 分支丢 `v3` | **不采信** | 数学上 `a<<4i < 2^255`，`t2<=7` 且 `s∈{0,4,..,60}` ⇒ `v3 = t2>>(64-s) ≡ 0`；已加注释说明 |
| `Gf2FoldTop` 的 `tov` / 二次折叠是否充分 | **不采信** | `th<2^7` ⇒ 二次折叠最高 13 次，无需第三次；实测逐位相等 |
| 对 GM 用 `SetValue`（红线） | **不采信** | 全文件只对 `LocalTensor` 用 Set/GetValue，GM 写回一律 `DataCopyPad` |
| 向量 op count 需 64 倍数 | **不适用** | 纯标量/位运算路径，无向量指令 |
| 逐行标量 + 每 tile `PipeBarrier<PIPE_ALL>` 是性能陷阱 | **部分采信** | k=8 改为每 8 行一次 `uint64` UB 读写（标量访问 ~8×↓）；`PipeBarrier<PIPE_ALL>` 保留（每 tile 16KB，摊薄后占比小，换成 MTE3→MTE2 事件链风险大于收益） |

### 1.2 实际修改（文件 / 函数）

1. `op_host/gf2k_mul_square_tiling.cpp::Gf2kMulSquareTilingFunc`
   - 属性强校验：`k ∈ {8,128}`、`operation ∈ {mul,square}`（原 `!="square"` 一律当 mul）；
     `operation` 缺省（attr 指针为 null/空串）按注册默认值 `mul` 处理，非法值才失败。
   - dtype 强校验：输入/输出必须 `DT_UINT8`。
   - 形状契约：a/b 必须 rank2、`[N, k/8]`、a/b 同形、`0<=N<=2^20`；删掉 `totalElems/width` 截断。
   - 空张量：`N=0` → blockDim=1、`rowsPerCore=0`、kernel 空跑。
   - `tileBytes` 圆整到 32B（保证 `InitBuffer` 与 `ReinterpretCast<uint64_t>` 对齐）。
2. `op_host/gf2k_mul_square_infershape.cpp::InferShapeGf2kMulSquare`：与 tiling 同一套 k/rank/shape 校验，非法即失败；合法时输出形状 = a 形状。
3. `op_kernel/gf2k_mul_square_tiling_key.h`：只保留 `GF2KMULSQUARE_TPL_SCH_MODE_0`（uint8），删除 fp16 key。
4. `op_kernel/gf2k_mul_square.cpp`：只实例化 `Gf2kMulSquare<uint8_t>`。
5. `op_kernel/gf2k_mul_square.h`
   - `DoTile`：k=8 路径改为每 8 行一次 `LocalTensor<uint64_t>` Get/SetValue（含尾行逐字节回退）；k=128 路径不变（每行 2 读 2 写）。
   - `Init`：GM 长度按 `max(1, N*W)` 声明（空张量安全）；`rowsPerCore_` 允许 0；`DoTile` 对 `rows<=0` 直接返回。
   - 删除未使用的 `Gf2ClMul64`；全部注释改为英文（红线：注释勿中文）。
6. `op_host/gf2k_mul_square_def.cpp` / `op_kernel/gf2k_mul_square_tiling_data.h`：语义不变（uint8、k int、operation 可选默认 "mul"；
   结构体字段同原），仅把中文注释改为英文（红线：注释勿中文）。改后再次 clean 重建 + 全量复验。

### 1.3 验收证据

- 构建：`rm -rf build && bash build.sh -j8` → `[INFO] Build completed successfully!`（单 tilingKey 编译通过）。
- 官方 golden：`bash build.sh -j8 -e` → `PASS: c.bin, mismatched=0/1`、`PASSED`。
- 自建全量（`gf2_run_tests.py`，62 例，全部与 reference.py 逐字节比较）：
  - k=8：N=1,2,7,8,9,15,31,32,33,64,255,256,257,1000,4096,16384,16385,16389,32768,65536,1048576；mul/square；全 0/全 FF/单位元/最高位单 bit。
  - k=128：N=1,2,3,7,8,9,31,32,33,255,1000,1024,1025,2049,4096,16384,16385,65536,262144；mul/square；全 0/全 FF/单位元。
  - 覆盖：tile 边界（k=8 tile=16384 行、k=128 tile=1024 行）、核数边界（N<coreNum）、非对齐尾块。
- 判题规模补测：`N=65535/1048575/1048576`，k=128 mul/square、k=8 mul，全部逐字节相等。
- 性能（device 计时，含 ~0.6ms 主端/launch 底噪）：
  | case | 数据量 | ms |
  |---|---|---|
  | k=128 mul N=1048576 | 16MB in / 16MB out | 14.34 |
  | k=128 square N=1048576 | 16MB in / 16MB out | 14.61 |
  | k=128 mul N=262144 | 4MB in / 4MB out | 4.46 |
  | k=128 mul N=65535 | 1MB/1MB | 1.50 |
  | k=8 mul N=1048576 | 1MB/1MB | 1.34 |
  | k=8 square N=262144 | 0.25MB/0.25MB | 0.99 |

### 1.4 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/inner:$LD_LIBRARY_PATH
export PYTHONPATH=/model/envs/mull-npu/lib/python3.10/site-packages:$PYTHONPATH
export DEVICE_ID=0
cd /root/work/ladder/gf2 && rm -rf build && bash build.sh -j8      # 重建
cd /root/work/ladder/gf2 && bash build.sh -j8 -e                   # 官方样例 golden
bash /root/work/gf2_full.sh                                        # 自建 62 例全量对拍
python3 /root/work/gf2_big.py                                      # 判题规模补测（需先 export 同上前提）
bash /root/work/pack.sh                                            # 重打 7 文件 zip
```

日志：`/tmp/fixb_gf2_build.log`、`/tmp/fixb_gf2_example.log`、`/tmp/fixb_gf2_full.log`、`/tmp/fixb_gf2_big.log`。
（`gf2_run_tests.py`、`examples/test_gf2k_io.cpp`、`gf2_big.py` 是验证脚手架，不属于 7 件套、不提交。）

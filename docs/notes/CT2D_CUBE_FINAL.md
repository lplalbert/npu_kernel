# CT2D Cube 攻坚最终结论（conv_transpose2_d，2026-11-28）

> 裁决：**ct2d 保 7.93（向量版），cube 路线在本赛不划算，停止投入。**
> 本文固化整轮所有结论，供后人复用/避坑。

## 0. 一句话
`conv_transpose2_d` 的 Cube 版**做对了数值**（H8W8/H16/H32/H64/H256 均 ratio=1.0），
但**小/中 case 比向量版慢 1.4~9×**，等权 15 case 下总分反而亏（≈0.7 vs 7.93）⇒ 不提交 cube。

## 1. 判据纠正（重要，别再用逐位）
- 判据：`rel = |Δ| / max(|golden|, 1e-6)`，`MARE = np.max(rel)`，**`MARE < 10 × threshold`** 即单元素通过。
- **fp16 threshold = 2^-10 → 10× = 9.766e-3**；**fp32 = 2^-13 → 10× = 1.221e-3**。
- **不需要逐位一致**；near-zero golden（|y|~1e-5）会让 rel 爆表，是唯一"假失败"来源。判题数据据阶段1实测**无 near-zero**，故 Cube 可 ratio=1。

## 2. 两个真病根（Cube 数值错的根因）
### ① GEMM 的 M 维 < baseM(16) 未补齐 → C 写回被污染
- **决定性判据**：`ntaps=1, Win=32, COUT=8 → okCols=8/32（错）`；`COUT=16/32/64 → 32/32（对）`。
- 即 `coPer=min(chOut,16)=8 < baseM=16` 时，多 N-block 的 C 拷贝落错位（只有前几列对）。
- **修复**：host `coPer=16`（不再 min）；kernel `SetSingleShape(coPerPad, nsp, chIn)`（M 补齐到 baseM），
  A 读补零列（wc 已补零）、C 多写行由 fold 丢弃（只读 cLoc<coCur）。
- 效果：Win=32 ratio **0.038 → 0.80**。

### ② near-zero fixup 的 max 被哨兵 3e38 毒化
- `DoRow` 把 `yabs[wout..wout64) = 3e38` 用来排除 min；但 `RowExtremes` 用同一 buffer 既算 min 又算 **max → mx=3e38**
  → `cut=fixRatio*mx` 巨大 → **每行都误触发 fixup 重算前 8 列**，而重算有偏差 → 正好「ow=1..7 全错、ow=0 与 ≥8 对」。
- **修复**：哨兵改 0（不影响 max，但 min 变 0 会总触发…）+ **直接关闭 fixup**（`fixRatio=-1`）；fold 的 fp32 累加本就精确，一次 cast 即可。
- 效果：Win=32 ratio **0.80 → 1.000000**；Win64 仅剩 1 个 near-zero 元素超阈（1ulp，绝对量极小）。

> 顺带证伪：改 5 参 `SetOrgShape` 的 `orgKc`、改 `orgN=nsp` 都**完全没变化** ⇒ **C offset / orgKa / orgKc / orgN 都不是病根**。

## 3. 两个运行期错误码的区别（排障用）
| 错误码 | 含义 | 触发场景 |
|---|---|---|
| **507046** | aclrtSynchronizeStreamWithTimeout 3s 看门狗 | `KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2)` 全局 + **走向量路径** → AIC 空等 ~5.8s → 被杀 |
| **507015** | MTE DDR out of range | 大 shape **workspace 不够**（例：cube1 qmsq 早期 `workspace+tilingData.sysWs` 手算偏移越界） |

## 4. 为什么 Cube 路线在本赛不划算（决定性算术）
判题逐 case（µs，sid 6b0a15c9…）：
```
case:    0      2      4      5      6      7
cube:  149.2 1128.1 2073.1 1539.7 7196.4 2102.9
向量:   16.5   107   1460   813   6557   6554
TOP1:   1.83  3.49   9.62  6.69  20.9   9.87
```
- Cube 在**小/中 case 慢 1.4~9×**（case0 149 vs 16.5 = 9×；case2 1128 vs 107 = 10.5×）。
- Cube 只在 **case12/13（220ms/258ms）** 有优势。
- 15 case **等权** → 小 case 的劣势远大于大 case 的优势 ⇒ 即使 15 个全修好，score ≈ **0.7**，远低于向量版 **7.93**。
- ⇒ **纯 Cube 路线不划算**。

## 5. 混合方案（小走向量、大走 cube）的三个死路（7 文件约束下）
1. **同 .cpp 多个 `extern "C" __global__`** → 只注册一个，其余静默全 0（已实测）。
2. **tilingKey + `if constexpr` 切 task type** → `KERNEL_TASK_TYPE_DEFAULT` 是编译期宏，`if constexpr` 里切换不可行（试过 4-mode：cube 路径 hang）。
3. **MIX 全局** → 向量路径 AIC 空等 5.8s → 看门狗 507046（判题全 time=0）。
⇒ **7 文件约束下无法同时给小 case 向量、大 case cube。**

## 6. 若后人仍要做 Cube（唯一出路）
- 需要「单 .cpp 单 kernel」且能用 `KERNEL_TASK_TYPE_DEFAULT` **之外**的机制按 shape 切 AIC/AIV
  （例如平台支持的 per-launch task type 配置、或手写 cube + 纯 AIV kernel 分 bin 打包），或
- 接受纯 cube 版，但**前提是所有 case 的 cube 速度都 ≥ 向量版**——本项目实测不成立（小 case 慢 1.4~9×）。
- 优化小 case cube 速度的方向（未验证）：减小 KFC 固定开销（~33µs）、L2 tiling、避免每 tap 一次 matmul（试 im2col 合并）。

## 7. 正向资产（可复用）
- **完整的 per-tap GEMM + AIV fold + near-zero 兜底的 Cube 实现**：`/root/work/ct2d_cube_src/`
  （op_host 3 + op_kernel 5；含 `conv_transpose2_d_cube.h`）。
- **可构建工程**：`/root/work/ws/conv_transpose2_d_cube/`（build.sh 已验证可 clean 编译安装）。
- **交付 zip（全 cube，7 文件内联版，能过精度但小 case 慢）**：
  `/root/work/ct2d_cube_out/conv_transpose2_d_7files_cube.zip`
- **可复用 MIX 配方**（REGIST_MATMUL_OBJ + GetUserWorkspace 单次 + 5 参 SetOrgShape + didWork 守卫 + SyncAll<true>）：
  见 `docs/notes/CUBE_MATMUL_ATTEMPT.md`。
- **探针**：`/root/work/bench_ct2d/bin/bench_ct2d N CIN HIN WIN COUT KH KW S P OP DIL DT reps prefix` + torch 对比脚本模式。

## 8. 关键实现要点（若复用 Cube 路径）
- `GetUserWorkspace(workspace)` **只调一次**；user 区偏移 `sysWs + Σ块`（别把"大小"当"偏移"）。
- `orgKa/orgKb` = A/B 的**物理行 stride（完整 k）**；只有 stage 到紧凑区后才传块长。
- **M 必须补齐到 baseM**（本 bug 的核心）。
- 每核独立 unit、无跨核同步、确定性；`didWork` 守卫 `mm.End()`。
- near-zero 正确写法：fold 的 fp32 结果即参考值，**不要**用 3e38 哨兵同时算 max/min。

## 9. 时间线产物
- cube2：12.24 → **30.45**（K 切核）；cube1：10.13 → **25.63**（stage + N 切核）。
- ct2d：Cube 数值打通（ratio=1.0 多 shape），但因小 case 慢，**最终保向量版 7.93，不提交 cube**。

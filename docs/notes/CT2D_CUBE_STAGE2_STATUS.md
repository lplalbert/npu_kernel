# CT2D Cube 阶段2 状态（2026-11-28，ct2d-cube-fix）

## 一句话
Cube 版**已实现并跑通链路，但在真实尺寸下数值错**：`Win ≤ 8` 的玩具形状 bit-exact，`Win = 32/64` 全错。
**阻塞点是 Matmul 高阶 API 的 C 写回（多 N-block / 多次 IterateAll）在本配置下不可用**，
不是我的 fold/兜底逻辑（fold 在 Win=8 时 bit-exact 验证过）。**未达交付标准，不提交。**

## 已交付
| 文件 | 说明 |
|---|---|
| `/root/work/ct2d_cube_out/conv_transpose2_d_7files_safe.zip` | **现网 7.93 版原包（15/15 Pass）**，与线上一致，可直接交（无变化） |
| `/root/work/ct2d_cube_out/conv_transpose2_d_7files_cube_experimental.zip` | Cube 实验版（fp16+fp32，7 文件已合并）—— **已知 Win≥32 数值错，不要提交** |
| `/root/work/ct2d_cube_src/` | 全部实验源码（op_host/op_kernel） |
| `/root/work/ws/conv_transpose2_d_cube/` | 可构建工程（build.sh 已验证可编译安装） |
| `/root/work/ct2d_cube_probes/` | 阶段1 精度探针（判据纠正证据，见 CT2D_CUBE_THRESHOLD_FIX.md） |

## Cube 实现（已跑通的部分）
- 算法：per-tap GEMM `P[t] = W_tap^T @ x[n][:, span]`（M=Co, K=Cin, N=HinWin，fp32 累加进 GM workspace）
  + AIV fold（相位/车道 Gather+Add，零元素 lane 补齐）+ near-zero 兜底（reference 顺序重算）。
- 无跨核同步（每核独立 unit），确定性；`GetUserWorkspace` 单次；5 参 `SetOrgShape`；`didWork` 守卫。
- fp16/fp32 都走 Cube，fp32/fp16 判据分别验证（阶段1）。
- **验证过的正确配置**：`Hin=Win=8, k=3`（任意 Co/S/stride/units）→ ratio=1.000000，**MARE=0（bit-exact）**；
  `N=4` 多核（4 units）也 bit-exact → 多核/多 unit 本身没问题。

## 阻塞点（精确事实）
| Win | N(nsp) | 结果 |
|---|---|---|
| 8 | 64 | **ratio=1.000000, MARE=0** |
| 32 | 1024 | ratio≈0.037, MARE≈1e5（全错） |
| 32 (N chunked to 64) | 64 per call | 仍全错 |
| 32 (4-arg SetOrgShape / 5-arg orgKc / 单 unit / 一核 / 归一 | — | 全部仍全错 |
- 错法：**只有 ow=0 一列正确**（`y[:, :, :, 0] == golden` ，ow≥1 全错）→ 指向 C 写回的列/行 stride 不对；
  P 平面的 row stride（nsp）与 API 实际写回 stride 不一致（多 N-block copy-out 落错位）。
- 已试无效：`orgN/orgKc/orgKb` 各种组合（4 参 / 5 参）、单 call-per-tap vs N 分块 64、一核 vs 多核、
  `GetMDLConfig` vs `GetNormalConfig`、M padding（coPad=baseM 上取整）。
- 结论：需要用「手写 cube（Nd2Nz / LoadData / Mmad / Fixpipe）」或在 kernel 侧改用 `mm.GetTensorC()`
  （L0C→UB 显式取结果）来绕开 copy-out，并按官方 conv2d 源码
  （`/usr/local/Ascend/cann-9.0.0/opp/built-in/.../cann_conv2d_v2/`）核对 C 布局。

## 另一个坑（重要，别踩）
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 会让**纯向量路径**每次 launch 多花 **~5.8s**
（AIC 空等/超时）。所以 cube/vector 双路径不能共用一个 MIX kernel：
必须用 tilingKey 分成 4 个实例（fp16-cube / fp16-vector / fp32-cube / fp32-vector），
在 `if constexpr` 里分别声明 task type。**给下一轮的第一条建议。**

## 阶段1（已完成，可复用）
判据、探针、数据分布敏感性、Cube 不可能 bit-exact 的实测：见 `docs/notes/CT2D_CUBE_THRESHOLD_FIX.md`。
核心：fp16 容差 = 10×2^-10 = 9.766e-3；fp32 = 1.221e-3；`rel=|Δ|/max(|g|,1e-6)`；
失败只发生在 near-zero golden（|y|~1e-5..1e-4）；TOP1/2/3 全 Cube 且 ratio=1（判题数据无 near-zero）。

## 给后人（按优先级）
1. 先用 **cpu 仿真（tikicpulib）或小形状矩阵**验证 Matmul C 写回：M=16,N=1024,K=32,fp16→fp32，
   与 torch 对比；把 `SetOrgShape/SetSingleShape/IterateAll` 的**最小可复现**降到 20 行。
2. 最小可用替代：**每 tap 只算 N=64 的一小段 P**，fold 也按 64 段做（放弃大 N 一次算完），
   用 tiling 的 `baseN` 对齐；这样复制 cube1 已验证的 N≤64 行为。
3. 或直接手写 cube（Mmad+Fixpipe）——conv2d 官方源码就是手写，控制力最强。

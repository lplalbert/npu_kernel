# 01 · 全部 38 题成绩表

> 数据来源：CANNJudge `/api/problems/<pid>/ranking`（本账号记录），抓取于 2026-11-25。
> 原始 JSON：`handoff_scores.json`。
> 「坏 case」= 非 `Pass` 的用例数（含 WA / RE / Skipped）。

## 一、天梯赛（10-11 月赛）— 18 题，Pass 14

赛事：`contest_id = 6aced8d2e769892927fdb17f`，`name = ladder_october`，截止 **2026-11-30 17:00**
计分：`score ≈ 100 × mean(global_best_time / my_time)`，**`best` 模式**（取历史最优，反复提交不掉分）
配额：**每日 50 次**，次日 00:00（北京）重置

| 难度 | 算子 | status | 排名 | 得分 | 坏 case | submissionId |
|---|---|---|---|---|---|---|
| 5 | flash_attention | **Pass** | 62 | 10.25 | 0 | `6b05c00985ad43bbe0653413` |
| 5 | conv_transpose2_d | **Pass** | 41 | 8.00 | 0 | `6b05dd9a85ad43bbe0ab47ae` |
| 5 | depthwise_conv2_d | **Pass** | 24 | 12.89 | 0 | `6b0619d285ad43bbe017f9b6` |
| 4 | gf2_poly | **Pass** | 30 | 11.35 | 0 | `6b05c8ae85ad43bbe07a78e6` |
| 4 | cube1 | **Pass** | 21 | 6.29 | 0 | `6b06272c85ad43bbe02e38d3` |
| 4 | cube2 | **Pass** | 22 | 7.34 | 0 | `6b062c7c85ad43bbe037751f` |
| 3 | gf2k_mul_square | **Pass** | 24 | 19.23 | 0 | `6b05c96985ad43bbe07c4cd5` |
| 3 | quant_matmul_sigmoid_mul_quant | **Pass** | 32 | 9.35 | 0 | `6b05d0d785ad43bbe08e2057` |
| 3 | mhc_sinkhorn_grad | WA | 56 | 0 | 8 | `6b05bd3b85ad43bbe05e264f` |
| 2 | bit_plane_transform | **Pass** | 26 | 23.30 | 0 | `6b05c5ee85ad43bbe073be10` |
| 2 | group_norm | **Pass** | 24 | 24.43 | 0 | `6b05bece85ad43bbe0624322` |
| 2 | mse_loss | **Pass** | 29 | 34.91 | 0 | `6b05c20785ad43bbe06a0343` |
| 2 | group_norm_silu_quant | **Pass** | **11** | 16.96 | 0 | `6b05f38985ad43bbe0d75fa9` |
| 2 | confusion_mul_grad | WA | 90 | 0 | 1 | `6b05be0585ad43bbe0605351` |
| 2 | multi_add_rms_norm_dynamic_quant | WA | 57 | 0 | 5 | `6b05c10585ad43bbe067a2a3` |
| 2 | index | RE | 33 | 0 | 9 | `6b05c30785ad43bbe06c71c3` |
| 1 | trace | **Pass** | 125 | 18.57 | 0 | `6b05caeb85ad43bbe07fecad` |
| 1 | bn_training_update | **Pass** | 41 | 19.60 | 0 | `6b06110985ad43bbe009c20d` |

> 未做：`GateGateUpAct`（Kirin 平台，本机 910B 无法评测，已跳过）

## 二、制造业算子赛（gcc2026_mfg）— 15 题，Pass 13

截止 2026-12-07。4 文件结构（`op_host/<op>.cpp` + `op_kernel/<op>.cpp` + `_tiling.h` + `tiling_key_<op>.h`）。

| 算子 | status | 排名 | 得分 | 坏 case | submissionId |
|---|---|---|---|---|---|
| complexexp | **Pass** | 30 | 45.63 | 0 | （历史提交，见平台） |
| randomwalk | **Pass** | 20 | 37.69 | 0 | 同上 |
| gathercoo | **Pass** | 35 | 35.06 | 0 | 同上 |
| complexabs | **Pass** | 62 | 32.07 | 0 | 同上 |
| gridcluster | **Pass** | 28 | 31.14 | 0 | 同上 |
| complexdiv | **Pass** | 44 | 30.81 | 0 | 同上 |
| cylinderquery | **Pass** | 34 | 25.35 | 0 | 同上 |
| ballquery | **Pass** | 64 | 23.97 | 0 | 同上 |
| furthestpointsample | **Pass** | 35 | 15.91 | 0 | 同上 |
| fourier_transform2d | **Pass** | 36 | 15.23 | 0 | `6b05f4fa85ad43bbe0d9ffec` |
| spmmsum | **Pass** | 46 | 10.49 | 0 | 同上 |
| segmentmincsr | **Pass** | 46 | 10.24 | 0 | 同上 |
| fused_prop_step | **Pass** | 55 | 5.72 | 0 | `6b0603b285ad43bbe0f3f836` |
| spspmm | WA | 52 | 0 | 8 | `6b053a0185ad43bbe0fc4b03` |
| radius | WA | 76 | 0 | 8 | （已放弃） |

## 三、昇腾算子挑战赛 S10 — 5 题，全部 Pass

| 算子 | status | 排名 | 得分 | submissionId |
|---|---|---|---|---|
| poisson_nll_loss | **Pass** | 108 | 37.01 | `6b04f99e85ad43bbe064a533` |
| mish | **Pass** | 178 | 32.83 | `6b043fcf85ad43bbe0ebe303` |
| bessel_i0e | **Pass** | 177 | 32.33 | `6b04f82385ad43bbe06149b7` |
| dot | **Pass** | 119 | 25.59 | `6b04fcd885ad43bbe06c155d` |
| reduce_log_sum_exp | **Pass** | 120 | 13.31 | `6b04e43785ad43bbe0341a6a` |

> S10 的 `bad=45` 是隐藏用例不返回状态的正常现象，`status=Pass` 即全过。

## 四、三条线的技术亮点（可直接复用的优化手法）

| 算子 | 手法 | 效果 |
|---|---|---|
| conv_transpose2_d | 复刻 torch 累加顺序 + 向量化 Gather + 100× 的 ow 对齐优化 | 10.7s → 107ms |
| cube1 | 向量 GEMM（M 切核 + UB 阻塞）+ transpose 走 strided DMA + UB 内重排 | transpose case 提速 64× |
| cube2 | 同 cube1 家族，纯性能题 | 最大 case 9.37ms，距 3s 看门狗 320× 余量 |
| group_norm_silu_quant | `volatile` 阻断 FMA + 双字正确舍入 exp + 修缓冲复用 | 67M 元素 int8 **逐位 0 差异** |
| quant_matmul_sigmoid_mul_quant | 单核 → 多核切 M + 两次 pass amax | 512³ 4.74s → 4ms |
| confusion_mul_grad | 17.6× | 6.02ms → 0.34ms |
| depthwise_conv2_d | `Muls`+`Add` 两步（禁用 FMA）+ 零车道建模 pad | 115/115 **bit-exact** |
| reduce_log_sum_exp (S10) | 修 depth-1 队列死锁 + P3 全工作单元切核 | 112ms → 5.74ms（19.5×） |

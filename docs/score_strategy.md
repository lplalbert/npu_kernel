# 算子赛刷分打法（计分公式破译 + 靶子清单）

> 2026-11-24 实测 API 破译。S10 与 mfg 同 API、同公式，仅题目/额度不同。

## 一、计分公式（已破译）

- 每题配置：`score_mode=1`、`use_baseline=false`、`baseline=null`、`ranking_submission_mode=best`（取历史最佳）。
- 榜单每行 `score` 是该次提交的最终得分，`result[].time` 是逐 case 耗时、`result[].score` 恒为 0（逐 case 分不进 result，只在榜单算）。
- **公式（拟合 + 对照 rank1 验证）**：
  ```
  单 case 得分率 = 全局best_time_i / 我的time_i
  该题 score   = mean_over_all_cases(得分率) × 100
  ```
  其中"全局best"是全站所有队伍在该 case 的最好成绩（所以 rank1 自己也不是每 case 都第一 → score 71~95 而非 100）。
- **权重**：S10 每题 50 case、可见仅 5（10%），隐藏 45 占 90%；mfg 每题 8 可见 / 15~24 总。→ **必须通用优化，禁止对可见 case 特化**。
- 精度：`precision_ratio` 必须 1.0（任一 case 不过 → 该次提交不作数/降级）。

## 二、关键结论：分数是"逐 case 等权均值"，不是加权总分

→ 小 case 与大 case 对总分贡献相同。**小 case 的固定开销（kernel 启动、InitBuffer、tilingKey 加载）和大 case 的吞吐一样值钱**。
→ 优化靶子 = **得分率最低的 case**，不是最大的 case。

## 三、靶子清单（2026-11-24 我 vs 全局best）

### S10（均分口径，50 case）
| 题 | 我的 time (ms) | 全局 best | 得分率 | 现均分 | 榜首 |
|---|---|---|---|---|---|
| dot | **181~235** | 3.4~4.4 | **≈0.02** | ≈2 | 91.14 |
| reduce | 4.75/8.6/35.9/7.7/97.1 | 3.53/4.07/4.20/3.93/4.62 | 0.74/0.47/0.12/0.51/0.05 | ≈38 | 71.89 |
| poisson | 4.1~4.4（大例 27） | ~3.5 | ~0.8 | ~80 | — |
| bessel | 5.3~6.3 | ~3.5? | ~0.6 | — | — |
| mish | 已知 Pass | — | — | — | — |

### mfg（8 可见 / 15~24 总）
| 题 | 我的 time (µs) | best | 得分率 | 榜首 |
|---|---|---|---|---|
| complexabs | 5.7/5.8/12.7/9.6/17.3/19.3/22.1/22.3 | 3.27/3.49/3.6/3.7/4.49/6.45/9.04/9.05 | 0.57/0.60/0.28/0.39/0.26/0.33/0.41/0.41 | 95.42 |
| ballquery | 7.8/5.6/47.7/23.7/**143**/**152**/5.7/13.9 | 3.56/3.24/4.60/4.18/**5.69**/**3.09**/3.04/3.86 | 最小 0.02 | 91.38 |
| 其余 4 题 | 见 `_mfg_notes.md` | — | — | — |

## 四、统一病根与治则

1. **循环内 `PipeBarrier<PIPE_ALL>` 把 MTE2→V→MTE3 全串行化** —— 为修正确性砸进去的，代价是大 case 慢 2~20×。治法：双缓冲 `TQue<VECIN,2>/<VECOUT,2>` 或 `TQueBind`，只留必要的 `SetFlag/WaitFlag`（MTE2→V、V→S、S→MTE3），去掉循环 PIPE_ALL。
2. **小 case 固定开销**：减 tilingKey 实例数、减每 call InitBuffer/重复 malloc。
3. **多核 + 自适应核数**：`GetCoreNumAiv()` 别硬编码（本机 24 核，判题机可能不同）。
4. **先测纯 `DataCopy` 的 GB/s 作 roofline 上限**，判断每题带宽受限还是计算受限。
5. 索引类（ballquery/radius/cylinderquery/fps）：块级/栅格预筛，只对候选块逐点精确扫描 + first-K。
6. 归约类（dot/poisson sum·mean）：SyncAll 在判题机必崩 507035 → 单核向量归约（lane 累加）已验证能到 0.5ms 级；若要多核只能绕开 SyncAll（如 AtomicAdd/多 kernel 两段式）。

## 五、判题机与版本红线

- 判题机 **CANN 9.0.0**，本机 9.2.0：跨版本安全重排只用 `Gather + CreateVecIndex`（offset 单位**字节**）；vector `Scatter` 不支持；`ReducePairElem` 编译失败。
- **507035 = VECTOR_CORE_EXCEPTION** 三大诱因：workspace 申请不足（要 `coreNum*8` 且 pad 512B）、16bit 标量 cast（须 UB 内 Cast）、**向量 op count=1**（chunk 被钳到 1 是最隐蔽的；chunk≥64、写回用真实 count）。
- **TBuf 无自动同步**：MTE3→下轮 MTE2、V→MTE2(WAR) 都要 SetFlag/WaitFlag。
- 提交用 curl（PowerShell Invoke-RestMethod 丢 Cookie）；文件禁 getenv/printf/DEBUG（HTTP 400）。

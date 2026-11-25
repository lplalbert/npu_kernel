# 算子赛参考资料与"别人的实现"索引

> 2026-11-24 调研整理。用于避免闭门造车：这些是官方公开的赛题解析与顶级选手源码。

## 一、官方赛题解析（PDF，已提取文本）

| 资料 | 本地文本 | 内容 |
|---|---|---|
| CANN 挑战赛赛题解析（2026-09-10） | `s10_赛题解析_20260910.txt` | 五大赛区赛题：RotaryPosEmb / SqrtSoftplusTopK / AddRmsNormBias / QuantMatmulReluQuant / BatchMatmulMaxSum；**评分规则**（15 测试点、全过才计分、逐点 T/t、均值排名、**排行榜仅针对最后一次提交**） |
| CANN 算子天梯赛 6/7 月赛题评析 | `ladder_2026_06_07_pdf.txt` | 6 月题目与通过率（vector 53~68%、Conv2D 仅 3.6%）、冠军代码做法；**赛制**：超历史最佳 3% 即月度冠军、禁止白盒、泛化性验收、每月最多获奖 3 题 |
| CANN 算子天梯赛 8-9 月赛题解析 | `ladder_8-9月赛题解析_20260915.txt` | 8 月冠军代码解析 + 9 月新题 |

原始 PDF 在远端 `/tmp/s10_solution.pdf`、`/tmp/ladder.pdf`、`/tmp/ladder_89.pdf`。

## 二、顶级选手源码（最有价值）

### 官方大仓 `cann-ops-competitions`（远端已 clone 到 `/tmp/cops`）
```
git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/cann/cann-ops-competitions.git
```

**关键发现**：S10 提交尚未开源（赛季进行中），但 **S9 有 5 题的完整多队提交**：
| S9 题 | 难度 | 与我们的关系 |
|---|---|---|
| Concat | ⭐ | 搬运类 |
| Greater | ⭐ | 逐元素比较 |
| IndexAdd | ⭐⭐ | 索引累加 |
| Transpose | ⭐⭐ | 重排（Gather 用法） |
| **SquareSumV1** | ⭐⭐ | **多轴+keepdims 的 reduce —— 与 ReduceLogSumExp 同族** |

### 标杆实现：`uuku/S9/SquareSumV1`（本地副本 `_sec5_build/ascend_tasks/refs/uuku_s9_squaresumv1/`）
路径：`01_official/operator-challenge/cann-ops-challenge-S9/submissions/SquareSumV1/uuku/`
- `op_kernel/square_sum_v1.cpp` — **3992 行 / 194KB**
- `op_host/square_sum_v1.cpp` — tiling，648 行
- README：**五例全 Pass，总用时 1398.72 µs**
- 注：`uuku` 也是 **S10 reduce_log_sum_exp 榜单第一名**

**他赢在 6 条（可直接迁移）**：
1. **全程向量归约，绝不标量累加**：`WholeReduceSum`（主力）、`BlockReduceSum`（注意 repeat 在 mask 之前）、`RepeatReduceSum`、`ReduceSum(dst, work, src, count)`（通用版需 work buffer）。
2. **`TQue<VECIN,2>` 双缓冲 + `EnQue/DeQue` 管同步**，不用 `PipeBarrier<PIPE_ALL>`（框架自动插 MTE2→V、V→MTE3、WAR）。
3. **按 shape 布局类分派 6 种模式**：`ModeSuffixReduce / ModeInnerAccum / ModePlaneReduce / ModeGenericFallback / ModeContiguousBlockReduce / ModeRank4Reduce02`（按"归约轴在后/在中间/跨 plane"分派，不是按用例白盒）。
4. **显式 UB 预算管理**（`kAivUbBytes=192KB`），每模式自算 tile + `AlignElems32<T>()` 对齐。
5. **非 bf16 内层归约用 `MulAddDst`** 直接累加（省一次 Mul+Add）；bf16 单独路径。
6. **batch rows 按模式调优**（64/128/255/256），一次 MTE2 命令喂满一趟向量归约。

### 性能工程方法论模板
`competition-performance` 仓（`gitcode.com/meipork/competition-performance`，38 个算子）：
- **`GeluV2/wangcai-jarvis/VALIDATION_REPORT.md`** — 500+ 行完整性能工程案例：official-vs-custom 多轮 sweep、`msprof` 分析、tile/core 矩阵调优、ACLNN smoke 测试
- **`Assign/未名湖人队/`** — 唯一带官方测评数据的提交：基准 1672us → 1665.98us（14 次重复最佳）；要点：8 核/32 核动态调度、192KiB UB 三级流水、DataCopyPad 处理尾块
- 多队同题对比：`Fills`/`GeluV2`/`Unpack`/`Atanh`/`Erfinv`/`Scale`/`TensorEqual`/`Assign`/`IsNan`（9~11 队各题）

## 三、其他官方资料
- 开源算子仓：`gitcode.com/cann/ops-math`、`ops-cv`、`ops-nn`、`ops-transformer`
- 样例仓：`gitcode.com/cann/cann-samples`
- Ascend C 算子开发指南（hiascend 文档中心）

## 四、三个赛事的规则差异（重要，别混）

| | S10 算子挑战赛 | 制造业赛 | CANN 挑战赛（赛区制） | 天梯赛 |
|---|---|---|---|---|
| 计分 | 逐 case `best/t`，**取历史最佳提交** | 15 题均分，best | 15 测试点，逐点 T/t 均值，**只认最后一次提交** | 超历史最佳 **3%** 即月度冠军 |
| 题量 | 5 题 × 50 点 | 15 题 | 每赛区 1 题 | 每周上新 |

> ⚠️ **CANN 挑战赛只认最后一次提交**（"排行榜仅针对最后一次提交呈现，请确保比赛结束前提交的是最优实现"），与 S10/mfg 的 `best` 模式完全不同——参加时要留好最后一次。

# gcc_2026_mfg 制造业算子赛 · 做题笔记

工作机: 910B2 单芯片 / CANN 9.2.0 / g++11 / py3.10 / `DEVICE_ID=0` 可真机跑自定义算子
判题机: CANN **9.0.0**（与本机差异是主要坑源）· 每题 4 个 editable 文件 · 50 次/天 · 冷却 80~190s

## 一、提交结果

| 题目 | problemId | submissionId | status | precision | 时延 我/best (µs) |
|---|---|---|---|---|---|
| **ComplexAbs** | 6af2d3706bd0f9be375466ed | 6b04f07585ad43bbe04ee54f | **Pass 8/8** | 1.0 | 5.7/3.27 · 5.8/3.49 · 12.7/3.6 · 9.6/3.7 · 17.3/4.49 · 19.3/6.45 · 22.1/9.04 · 22.3/9.05 |
| **ComplexExp** | 6af6cbd36bd0f9be37ba0aae | 6b04efb585ad43bbe04d05e5 | **Pass 8/8** | 1.0 | 3.96/2.79 · 4.10/3.45 · 16.9/4.46 · 17.7/6.41 · 27.9/14.2 · 27.9/14.2 · 4.04/2.84 · 4.13/3.43 |
| **BallQuery** | 6af1c92b6bd0f9be3798a712 | 6b04f28d85ad43bbe0540d51 | **Pass 8/8** | 1.0 | 7.84/3.56 · 5.60/3.24 · 47.7/4.60 · 23.7/4.18 · 143/5.69 · 152/3.09 · 5.72/3.04 · 13.9/3.86 |
| **GatherCoo** | 6af1cbe56bd0f9be379e1c23 | 6b04f61d85ad43bbe05caa43 | **Pass 8/8** | 1.0 | 11.2/3.37 · 5.82/3.61 · 28.8/10.97 · 22.6/6.01 · 120/49.95 · 54.0/39.5 · 12.0/3.49 · 5.69/3.50 |
| **GridCluster** | 6af1cab46bd0f9be379bc6c6 | 6b04fc2285ad43bbe06a6c3a | **Pass 8/8** | 1.0 | 5.78/3.87 · 5.46/3.47 · 17.96/5.29 · 17.02/4.21 · 33.9/13.6 · 22.3/7.47 · 17.5/4.71 · 18.0/5.30 |
| **ComplexDiv** | 6af2d43a6bd0f9be3756251e | 6b04fd9985ad43bbe06de843 | **Pass 8/8** | 1.0 | 6.85/3.27 · 5.40/3.48 · 29.0/4.52 · 47.5/5.15 · 98.5/15.0 · 198.6/47.7 · 6.20/3.30 · 5.10/3.45 |
| **CylinderQuery** | 6af1c67a6bd0f9be3792fb4a | 6b05130c85ad43bbe09d3ad0 | **Pass 8/8** | 1.0 | 8.41/3.71 · 5.95/3.28 · 39.4/4.97 · 18.1/4.42 · 98.4/6.54 · 152.8/3.78 · 6.10/3.55 · 14.2/4.14 |

已用提交 ~26/50。**7 题全部 Pass 8/8、precision_ratio 全 1.0。**

## 二、★★★★★ 最重要的一条：CANN 9.0.0 上「count 很小的向量 op」直接崩 507035

S10 线 A/B 实锤 + 本轮验证：**判题机 9.0.0 对 count 很小的向量指令（Add/Gather/Cast/Div/Muls…）
直接报 `ACL error 507035`（vector core exception）**，本机 9.2.0 不复现。

推论出的硬规则（已按此修好 GridCluster）：
1. **所有向量 op 的 count 一律 ≥ 64**；需要时把 buffer 按 64 补齐，多算的部分**不写回**。
2. **tiling 里的 `chunk`/`tile` 不许按 n/outM 等真实规模钳到 <64** —— 这是 GridCluster 的根因：
   原来 `chunk = min(chunk, n)`，遇到小 shape（N=1/7）chunk 变成 1/7 → 小 count → 崩。
   改成 `chunk = max(64, min(预算, 1024))`，写回仍用真实行/列数。
3. 内核里再加一层保险：`cntOp = alignUp(cnt,64)`（上限 = chunk），所有向量 op 用 cntOp，只有
   `DataCopy/DataCopyPad` 写回用真实 cnt。
4. 复制搬运用真实 count 没问题（MTE 侧小长度是安全的，ComplexAbs 的 1x1 案例一直过）。

## 三、其它环境/流程红线

1. **提交只认 curl**：PowerShell `Invoke-RestMethod` 会丢 `Cookie`（假 401）。`.ps1` 被执行策略挡，用 `.py`。
2. **提交代码不能有 debug 痕迹**：`getenv`/`printf`/`DEBUG` → HTTP 400「存在不合规内容」。
3. **判题机 9.0.0 ≠ 本机 9.2.0** 的其余差异：
   - `ReducePairElem<ReduceType::SUM,…>` 在 9.0.0 编译失败 → 禁用（改用 `Gather` 做去交错）。
   - 9.0.0 上 **vector `Scatter` 不支持**（dav_c220 `ASCENDC_REPORT_NOT_SUPPORT`，静默不写）。
   - 跨版本安全重排只有 **`Gather` + `CreateVecIndex`**，**offset 单位是字节**（f32 去交错 = 8j / 8j+4）。
   - 带 `Attr` 的算子在**本机** `GetWorkspaceSize` 直接 561002（plog `compile_info is nullptr`），**判题机正常** → 带 attr 的题只能编译过 + 提交验证。
4. **纯 GM→UB→GM 必须 `TQueBind<VECIN,VECOUT,2>`**；`TQue<VECIN,2>` 不插 MTE2→MTE3 依赖 → 随机错位。
5. **TBuf 没有自动同步**，两类必补：V 读过→下一轮 MTE2 覆盖前要 `SetFlag/WaitFlag<V_MTE2>`；
   输出 buffer 最好直接做成 `TQue<VECOUT,2>`（框架自动插 V→MTE3 与 MTE3→V）。
6. **`DataCopyPad` 的 UB 侧地址必须 32B 对齐**（GM 侧可任意）；行式数据按 `rowStride=alignUp(rowBytes,32)` 摆。
7. **UB 预算按真实分配算**（queue 深度翻倍、偏移/交织缓冲很占），估小→507035。
8. **偏移表/scratch 别重叠**：曾把 `scratch` 放在 `offBuf_[3*sz]` 正好压掉 `offI` 一半 → gather 用垃圾 offset → 507035。

## 四、关键实现套路

- **纯逐点**：多核按元素均分（blockFactor 对齐 64）+ tile 切块 + 双缓冲；`DataCopy` 走对齐、尾块 `DataCopyPad`。
- **complex64 去/重交织**：`offE[j]=8j`、`offO[j]=8j+4`；重交织 `offI[k]=4n*k-(8n-4)*floor(k/2)`
  （等价 `4*floor(k/2)+4n*(k&1)`），`W=[Re|Im]` 后一次 Gather。
  **注意**：offset 表按满 tile `nOp` 生成时，虚部必须放 `W[nOp]`（不是 `W[nProc]`）。
- **int64 输出**：不要用 `Cast(int32→int64)`（9.0.0 上可疑）；用 `W=[v|0]`（高半段用 **float 视图 Muls 0** 清零，
  再用 `Gather` 交织成 int32 流，int64 小端就是 [v,0]）。
- **索引类（BallQuery）**：升序扫描、`d²<r²` 严格小于、first-K 不排序、不足 nsample 尾部重复首个命中、
  0 命中全 0、f16 升 fp32 再比较；标量读 UB 用裸指针 `(const __ubuf__ float*)t.GetPhyAddr()`。
- **GatherCoo**：`TQueBind` + 按组搬运；行按 32B 对齐摆；`rowBytes%32==0` 整批写回，否则逐行 `DataCopyPad`。
- **GridCluster**：`Gather` 去交错各维 + `Adds(-start)` + `Div(size 向量)` + `Cast(TRUNC)` + `Muls/Add` 累加；
  int64 输出见上。
- **ComplexDiv**：广播逐行/逐列映射（`aN==1`→第 0 行，`aM==1`→标量广播用 `Duplicate`）；
  复数 `(ac+bd, bc-ad)/(c²+d²)`，A/B 各 2 次 Gather 去交错 + 1 次 Gather 重交织。

## 五、待办 / 已知风险

- **ComplexDiv f16 的临界元素**：判题机 8/8 precision=1.0，说明其 golden 口径与我的 f16 路径一致，本地那 1~2 个越界点是我测试参考（double→f32）与 f16 单次舍入的差异，非实现问题。

- **ComplexDiv f16**：本地有 1~2 个元素越过 `1e-3`（f16 双重舍入：精确值→f32→f16 vs numpy 单次舍入），
  取决于判题机 golden 口径；若判题 precision<1，可考虑 f16 直接做 f16 除法或改用相对误差更宽松的比较策略。
- **GridCluster / ComplexDiv 的 `pos/size` 边界**：本地人造 case `bigvox`（size=0.001、pos 量化到 1/1024）
  出现 ~1% 元素 trunc 差 1 —— 因为商正好落在整数上，vdiv 与 IEEE 除法的 1ulp 差被放大。
  判题机数据是连续随机值时不会命中，但说明 `Div` 不是严格正确舍入。
- BallQuery 大 case 慢 25~45x（逐元素标量扫描）；ComplexExp c64 约 roofline 1/3 —— 回头刷分第一优先级，先不深挖。
- 其余 8 题未开始：radius / furthestpointsample / cylinderquery / spmmsum / segmentmincsr /
  fusedpropstep / fouriertransform2d / spspmm / randomwalk（radius 模板已拉好）。

## 六、计分公式与分数审计（2026-11-24 实测）

**公式: score ≈ 100 * mean_over_cases(global_best_time / my_time)**（use_baseline=false，逐 case 等权）
校验: ComplexExp 榜首 8 个可见 case 算得 94.77，实际榜分 89.39；rank2 94.48→88.78；rank3 94.19→88.15。
差值 ~5-6 分来自隐藏用例（可见 8 / 总 15-24），所以**可见 case 得分率是可靠代理但不能特化**。

我的 6 题估计分（可见 case）:

| 题 | est | 最惨 case（我/best → 得分率） |
|---|---|---|
| BallQuery | 27.2 | 152/3.09→0.02 · 143/5.69→0.04 · 47.7/4.60→0.10 · 23.7/4.18→0.18 |
| ComplexDiv | 37.4 | 47.5/5.15→0.11 · 98.5/15.0→0.15 · 29.0/4.52→0.16 · 198.6/47.7→0.24 |
| GridCluster | 39.4 | 17.0/4.21→0.25 · 17.5/4.71→0.27 · 18.0/5.29→0.30 |
| ComplexAbs | 40.6 | 17.3/4.49→0.26 · 12.7/3.6→0.28 · 19.3/6.45→0.33 |
| GatherCoo | 45.2 | 22.7/6.01→0.27 · 12.0/3.49→0.29 · 11.2/3.37→0.30 |
| ComplexExp | 59.0 | 16.9/4.46→0.26 · 17.7/6.41→0.36 |

- **纯逐点 roofline 实测 ≈ 1.2 TB/s**（ComplexAbs 8192×8192 = 1210 GB/s，已到带宽上限）。
- GridCluster 407~680 GB/s（roofline 的 1/3~1/2，还有空间）。
- **BallQuery 0.5 GB/s → 完全不是带宽问题**：实测标量扫描 **≈45 cycles/点**（262K 次判距/152µs/43 核），
  这是它 49× 差距的直接原因；块级预筛（或 Compare+GetCmpMask 落 64-bit 字再 ctz）可把它降到 ~1 cycle/点量级。
- 所有题**小 case 稳定在 1.4~2.2×（5.1~5.8µs vs best 3.0~3.5µs）**，而 best 自己也付 ~3µs →
  这 ~2µs 主要是 op 加载/调度固有成本，不是可消除的写码开销。

## 七、Radius（第 7 题）卡点记录 —— 未 Pass，已按 40 分钟规则跳过

- 实现: 逐 batch 维护 min 距离不存在(直接逐点判距) + 标量顺序扫描 + 槽位 (i*K) 写入,
  未命中填 -1; int64 输出用 W=[v|0] 思路改成「int32 视图 Duplicate(-1) 打底 + 标量写命中槽」。
- **本地真机 8/8 全 PASS**（多 batch / N=1 / f16 / 全空槽 / 首 K 截断）。
- 判题机 **4 次提交 precision 恒为 0.9360416666666667**（一位不差），且对下列改动**完全不敏感**:
  1. row0 填充策略（全 K 槽 vs 仅命中槽）;
  2. 换成 bool/int 双读 ignore_same_index;
  3. 强制剔除对角点（j==i）。
- 反推分母: wrong=(1-0.936...)*2*M*K 在 M*K ∈ {12000,24000,48000,96000} 上是整数，
  其中 1535 ≈ 48×32 → 约 48 行**只错 row1 的 32 个槽**（row0 正确、命中数不变）→ 是「命中集合」
  不同（典型是一处提前/滞后的命中把整行平移），不是填充策略问题。
- 剩余最可能原因（按可能性）:
  (a) **d2 <= r*r 边界口径**：golden 可能用严格 <，而数据是格点使得恰好 d²=r² 的点常见（48/750≈6.4% 查询命中一处边界）；
  (b) **多 batch 配对**：ptr_x 与 ptr_y 的 batch 数/空 batch 与我的 y 推进不一致（会表现为整行 row1 错、row0 对）；
  (c) golden 的扫描 j 区间与「批内」定义不同。
- 下次动手建议：先用**单 batch、极小规模**（如 M=4,N=8,r 取格点边界值）在本地人造 case 上把 (a)/(b) 分离；
  或让题面方确认 <= 与 batch 配对。已把这 4 个 submissionId 记在上面。
### Radius 追加结论（第 5 次提交后定案，已跳过）

5 次提交、同一 case 的 precision 记录：

| 改动 | precision | 结论 |
|---|---|---|
| 基线（row1 = 全局下标 gj） | 0.9360416666666667 | — |
| row0 只填命中槽（不填满 K） | 0.9360416666666667 | 填充策略无关 |
| ignore_same_index 改 bool/int 双读 | 0.9360416666666667 | attr 读取无关 |
| 强制剔除对角点 j==i | Runtime Error（无 precision） | 该构建崩了，未定性 |
| **row1 改批内局部下标 (gj - xS)** | **0.9079166666666667（更差）** | **反证：row1 就是全局下标；且该 case 是多 batch** |

由 0.93604 → 0.90792 的下降量（0.0281）≈ 2.8% 槽位翻转，恰好等于「多 batch 的行从正确变错"的量级，
两边夹证：**全局下标正确、批内局部错误**。

剩余唯一嫌疑 = **多 batch 配对**（ptr_x/ptr_y 的 batch 数不同、或空 batch、或我的 `by` 推进在某个边界上错位），
表现完全吻合观测：某个 batch 的整段 row1 错、row0 对、命中数不变、任何填充/对角改动都无效。

已确认**不要再试**的假设：row0 填充策略、ignore_same_index 取值、剔除对角点、row1 批内局部下标。
下一步定位方法：造 `ptr_x=[0,N0,N0+N1]` + `ptr_y=[0,M0,M0+M1]`（N0≠0 且 M0≠0）的人造 case，
并额外覆盖「空 batch」「Bx≠By」两种边界，与本地参考对拍。

## 八、CylinderQuery（第 7 题）Pass 经验 —— 复用 BallQuery 套路

- 语义: `delta = xyz - new_xyz`; `[xl,yl,zl]^T = rot^T @ delta`（即 `xl = r00*dx + r10*dy + r20*dz`）;
  命中 = `yl²+zl² < radius²` 且 `hmin < xl < hmax`（**严格**）; 按输入索引序取前 nsample; 不足尾部重复首个命中; 0 命中全 0。
  f16 输入先无损转 f32 再算（与 FPS 相反！这里的题面明确要求 f32 语义）。
- 实现 = BallQuery 骨架 + 每查询多算 lx/ly/lz。**技巧：把三个判据合成一个 margin**：
  `margin = min(radius²-(yl²+zl²), xl-hmin, hmax-xl)`，标量扫描只需 `margin > 0` 一次比较，
  比逐点做 3 次标量比较快。每 (查询, chunk) 约 26 条向量指令。
- 结构: 每核一组查询(qpc≤8) → 每 batch → chunk 遍历 N（Gather 去交错成 f32 x/y/z）→ 每查询向量算 margin → 标量顺序收集。
  查询参数（new_xyz + rot）整批升 f32 放 UB，标量用 `qf_.GetValue()` 读。
- **坑（本次唯一卡点）**：`rot` 段搬入 UB 时起始地址要 **32B 对齐**，
  即 new_xyz 段要按 `32/sizeof(T)` **个元素**补齐（f16 是 16 元素、f32 是 8 元素）。
  只按 8 元素补齐时 M=8 的 f16 case（24 元素=48B）会崩，表现为该 case 静默不打印（不是精度错）。
- 本地 18/18 全 PASS（0.01~0.14ms）。提交一次即 Pass 8/8。
- 待优化（分数低，估 29）：慢 case 是「无命中」的 c6（B=2,N=2048,M=64,ns=32）152µs vs best 3.78µs ——
  因为对每个查询都把整个 N 扫完（无命中不能提前退出）。可做：向量级预筛（一次 Compare 出 mask，
  整块无命中就跳过标量扫描），或按查询并行度不足时给每个查询多分核。

## 九、本轮新增 Pass（第 8、9 题）

| 题目 | problemId | submissionId | status | precision |
|---|---|---|---|---|
| **SpMM Sum (spmmsum)** | 6af275c16bd0f9be37a3bbd5 | 6b051ab985ad43bbe0aea10e | **Pass 8/8** | 1.0 |
| **FurthestPointSample** | 6af1c8736bd0f9be37971169 | 6b051e6885ad43bbe0b6b94c | **Pass 8/8** | 1.0 |

时延（我/best，µs）：
- spmmsum: 43.2/4.55 · 65.3/5.83 · 29812/50.96 · 1031/34.2 · **671111/675.8** · 96495/135.4 · 45.6/4.43 · 62.2/6.23
  → 大 case 慢 ~1000x（按行切分不均衡 + 每 edge 一次 DataCopyPad/同步），估分仅 ~5。
- furthestpointsample: 34.7/4.36 · 46.3/3.64 · 144/11.2 · 41.7/6.25 · 315/24.3 · 49.1/7.98 · 35.0/4.68 · 38.2/5.46 → 估分 ~12。

### ★★★★★ 本轮最大的坑：本地 test harness 的 f16 转换死循环（浪费约 1.5 小时）
现象：FPS 的 f16 全部"挂死"（连"kernel 体改成空操作"都挂），误判为 f16 kernel/调度问题，做了 Cast/Gather/Duplicate/TBuf/offset-单位 等一堆 A/B。
真因：**测试程序里 `fromFloat`（f32→f16 手写转换）有 `while (av >= 2.0f) av /= 2.0f;` 循环，
遇到 inf 就永不退出**（rt() 在参考实现里被调用，某步产生 inf/异常值）。**与算子无关！**
修复：改成 **循环-free 的 frexp 版本**（`std::frexp` 取指数 + 舍入，含 inf/NaN/下溢分支），
之后 FPS 本地 12/12 全 PASS（f32 6 + f16 6），提交一次即 Pass 8/8。
**教训：以后凡是本地"挂死"，先用 DBGSTAGE 打印把挂点夹在 host 侧还是 device 侧，别急着改 kernel。**

### 其它重要环境事实（本轮实测）
1. **本机是共享的**：另一个 agent 在跑 `/root/work/dot`（S10 线），NPU 会被抢占、`/tmp/b_bin.log` 等**共用文件名会被覆盖**。
   → 已改用 `/root/work/mfg/mfg_build.sh` + `/tmp/mfgbuild_<tag>_*.log` 独立日志。
2. **提交账号也是共享的**：实测 `remaining=8`（used 42/50），且提交有 47s 冷却（HTTP 429 `retry_after_seconds`）。
3. 脚本自坑两条：`source set_env.sh` 会 `set -e`（`pkill` 返回 1 会把脚本直接干掉）；
   `pkill -f test_xxx` 会**匹配到运行脚本自己的命令行**把脚本杀掉 → 一律用 `pkill -x <binname>`。
4. 大 case 的判题耗时可能是"判题机共享带宽"问题（S10 线实测 ~48GB/s），不全是内核慢；**能不用 workspace 就不用**（本轮所有题 workspace=0）。

### spmmsum 实现复盘（可复用）
按行切分（rowsPerCore）+ 列分块（ck=1024）；每 edge 一次 `DataCopyPad`(mat 行块) → `Muls`(标量 value) → `Add` 进 f32 acc；
f16 时 acc 用 f32、写回前 `Cast` 回 f16（实测 f32→f16 的 `Cast` 完全正常）。
**已知弱点**：幂律图 hub 行会让单核独占（671ms 的 case），且每 edge 两次同步；
改进方向：按**前缀和（非零数）切分**、把同 col 的 edge 合并、或按 K 再并一次核。
硬规则照旧：向量 op 的 count 用 alignUp(ck,64)，写回只用真实 ck；空行必须写 0。

## 十、SegmentMinCsr（第 10 题）

| 题目 | problemId | submissionId | status |
|---|---|---|---|
| **SegmentMinCsr** | 6af1cdaa6bd0f9be37a1afe1 | 6b05205185ad43bbe0badb8b | 见下 |

实现（可复用套路）：
- 行切分（rowsPerCore）+ 列分块 ck；每 chunk 用 f32 `minV` + f32 `argV`（存全局行号 e，E≤5e6 < 2^24 在 f32 里精确）。
- 每 edge：`DataCopyPad` 载入 src 行块 → `Cast` 到 f32 → `Duplicate(eVec, e)` → `Compare(mask, cur, minV, CMPMODE::LT)`
  → `Select(minV, mask, cur, minV, SELMODE::VSEL_TENSOR_TENSOR_MODE)` → `Select(argV, mask, eVec, argV, ...)`。
  **严格 < 天然满足「首遇 tie-break」**。实测 Compare+Select 在本机可用。
- **int64 输出**（本次自己的写法，已验证）：`Cast(ai_int32, argV, CAST_RINT)` 写 ai 的 [0,cnOp)；
  用 **float 视图 `Duplicate(aiF[ck], 0.0f, ck)`** 把 [ck,2ck) 清零（0.0f 位模式=0，避免 int32 Duplicate 的不确定性）；
  再用**预建的字节偏移表** Gather 交织成 [v,0,v,0,...]（int64 小端 = [低32位,0]），最后按 int32 流 DataCopyPad（长度 2*cn*4）。
  → **不要**用 int32 的 `Duplicate/Adds` 做搬运，f32 视图 + Cast 更稳。
- 空段：out=0、arg=E（哨兵=E=src 行数）；E=0 全空图时哨兵为 0。
- 向量 op count 一律 alignUp(cn,64)（含 Gather 的 2*cnOp，否则 K=1 时 count=2 会崩）。
- 本地 12/12 PASS（含空段、E=0 全空、K=1、K=513 非对齐尾块、f16），用时 0.02~1.07ms。

## 十一、当前状态总览（交给下一班）

**9~10 题 Pass**：ComplexAbs / ComplexExp / BallQuery / GatherCoo / GridCluster / ComplexDiv /
CylinderQuery / SpMM Sum / FurthestPointSample / SegmentMinCsr(待确认)。
**未做**：fusedpropstep / fouriertransform2d / spspmm / randomwalk（难）；radius（已定案放弃）。

**剩余分数空间（按性价比）**：
1. **spmmsum 大 case 优化**（671ms/675µs，估分仅 5）：按前缀和非零数切分 + hub 行拆分 + 每 edge 的两次同步去掉（批量 load 多个 edge 的 mat 行）。
2. **FPS 性能**（估分 12）：目前每 batch 一核，B=1 时只用 1 核；可把 N 维也拆到多核（用 workspace 存每核的 minD 部分+全局 argmax 归约），但注意「判题机大 workspace 有 0.7~2.3s 固定开销」。
3. **CylinderQuery c6**（152µs/3.78µs）：无命中也扫全 N；加向量预筛（Compare 出 mask，整块无命中跳过标量扫描）。
4. 其它 6 题：小 case 都在 1.4~2.2x，大 case 的 roofline 差 3~50x，按 notes 第六节逐题刷。

**环境红线（本轮新增，务必遵守）**：
- 本机与提交账号都是**共享**的（另一 agent 跑 /root/work/dot）：用 `/tmp/mfgbuild_<tag>_*.log` 等独立文件名；
  跑测试前 `pkill -x <binname>`（**不要用 `-f`，会杀掉自己的脚本**）；`source env.sh` 会开 `set -e`，`pkill` 要 `|| true`。
- 本地"挂死"先分清 host 侧还是 device 侧：加 `fprintf(stderr,...)` 阶段标记。本轮 FPS 的 f16"挂死"其实是
  **测试程序里手写 f32→f16 转换的死循环**（`while (av >= 2.0f) av /= 2.0f` 遇 inf 不退出），与算子无关。
- `Cast` f32→f16 **正常**；`Compare/Select` **正常**；int32 的 `Duplicate/Adds` 不要用于搬运算。

## 十二、fusedpropstep / randomwalk 现场评估（下一班接手用）

### fusedpropstep（`6af2d5b76bd0f9be3759446c`，⭐⭐⭐⭐4）—— 建议单开一个专门的 agent/时段，别和别的题交替做
语义 = `out = IFFT2(FFT2(field) * h)`，**要在单 kernel 内自己实现 2D FFT/IFFT**：
- dtype: float16 / float32 / **complex64**；B∈[1,256]，N,M∈[2,8192] 且都是 2 的幂；h 的 batch 可为 1（广播）或 B。
- 精度: complex64 atol/rtol 1e-4；f32/f16 1e-3。
- 需要的东西（按顺序）：
  1. 2D FFT 分解为「按行 1D FFT × M」+「转置/按列 1D FFT × N」；radix-2 蝶形 + 位反序置换；
  2. twiddle 因子 `W_N^k = exp(-2πik/N)` 的片上生成（旋转移位/查表，注意 9.0.0 上不要用未验证的 sin/cos）；
  3. 逐点乘 h（Bh=1 时按 B 广播，取模索引复用）；
  4. 逆变换的 1/(N·M) 归一化只在 norm=True 时加；
  5. complex64 的交织/去交织沿用 ComplexAbs/ComplexDiv 那套 `Gather + CreateVecIndex`（offset 字节单位，offE=8j, offO=8j+4）。
- 可行的落地路径：先在**本机**用 numpy 对拍一个 1D FFT kernel（N=8/16/64），
  再做 2D（N,M≤256 起步），最后扩到 8192。**预估 2~4 小时**，不建议在剩余额度/时间内硬上。
- 注意：f16 路径可以在 f32 里做蝶形中间量（题面允许），最后再转回 f16。

### randomwalk（`6af277846bd0f9be37a6755a`，⭐⭐⭐⭐5）—— 比 FFT 现实得多，建议下一班优先做这道
语义 = 从 seed 出发沿 CSR 走 walk_length 步；out (B, L=walk_length+1) **int64**，首列 = seed。
- N≤1e7, E≤2e7, B≤1e5, walk_length≤64；rand (B, walk_length) **float32 必选**（确定性对拍的关键）。
- 均匀路径（p=q=1）：`k = floor(rand * (rowptr[v+1]-rowptr[v]))`，`v = col[start+k]`。
- 孤立点（出度 0）：**停留，v 与 prev 都不变，但仍消费 rand**。
- node2vec 偏置（p/q≠1）：候选 x 权重 = `x==prev ? 1/p : (x 与 prev 相邻 ? 1 : 1/q)`，按 CDF 选；
  第一步无 prev → 权重全 1。注意「x 与 prev 相邻」需要在 prev 的邻接区间里线性查找（O(deg²)，小度图没问题）。
- 实现建议（工作量约 40~60 分钟）：
  1. **按 walker 并行**（B 维天然负载均衡，每核拿一段 B），walker 内串行 walk；
  2. 每核以 64 个 walker 为一块：标量循环算轨迹，int64 输出用「**每个值写两个 int32 槽（低=id, 高=0）**」的标量 SetValue 写法，
     再一次性 `DataCopyPad` 写回（比 Gather 交织更省事）；
  3. rowptr/col/seed/rand 都用 `GlobalTensor::GetValue` 标量读（CSR 随机访问没法预取）；
  4. 参考实现必须严格按题面的 CDF 取「第一个 cum > r*total」，均匀分支用 `int(r*len)`。
- 主要风险：①golden 的 CDF 边界口径（`>` vs `>=`、searchsorted 的 side）；②p/q=1 时是否真的走均匀分支（题面说是）；
  ③大 B 时标量 GM 读可能很慢（B=1e5 × L=16 ≈ 1.6M 步，每步 ~10~30 次标量 GM 读，估计 1~10s 量级 —— 可能只是慢，不会错）。
- **精度测试就是靠 rand 逐位对拍**，所以本地造 case 一定要覆盖：孤立点、空图(E=0)、单点图、p≠q、首步无 prev、walk_length=1。

## 十三、SegmentMinCsr = Pass（第 10 题）+ RandomWalk 实现（第 11 题已交）

| 题目 | submissionId | status | precision | 时延我/best (µs) |
|---|---|---|---|---|
| **SegmentMinCsr** | 6b05205185ad43bbe0badb8b | **Pass 8/8** | 1.0 | 19.9/3.47 · 21.3/3.52 · **47135/39.6** · **55120/60.4** · **4119926/72.7** · **532283/112.6** · 138/5.44 · 58.4/5.19 |
| **RandomWalk** | 6b05238885ad43bbe0c2458c | 提交后 Running（8 case 均 prec=0 未出分） | 待定 | — |

SegmentMinCsr 结论：**判题跑了约 25 分钟才出 Pass**（不是 TLE，是巨大 case 太慢：最慢 4.1 秒）。
→ 说明「每 edge 一次 DataCopyPad + 3 次同步」的吞吐瓶颈是真的，但**只要不超时就是 Pass**；估分仅 ~5.9。
若后续要刷分：批量打包多个 edge 的 src 行（一次 DataCopyPad 搬 B 个 edge），同步次数从 O(E) 降到 O(E/B)。

### RandomWalk 实现（本轮新写，可复用）
- **按 walker 并行**（B 维），每核一段 walker；每核按 `RW_CHUNK_W=32` 个 walker 分块算轨迹。
- 每步：标量 `rowptr[GetValue]` → 均匀分支 `k = int(r*deg)`；node2vec 分支按 `w(x)=x==prev?1/p:(邻接?1:1/q)` 两次线性扫描算总权重和 CDF（首个 `cum > r*total`）。
- **int64 输出最省事的写法**：UB 里用 int32 缓冲，每写一个 int64 就 `SetValue(2k, id); SetValue(2k+1, 0)`（小端），
  然后一次性 `DataCopyPad` 写 2*n*L*4 字节 —— 不需要 Gather 交织。
- 孤立点：停留、**prev 与 v 都不变**，但仍消费 rand；第一步 prev=-1 → 权重全 1。
- rand 有 **float32 / float16 两种 config**（模板 tiling_key 是 DT_RAND），kernel 按 `T_RAND` 模板化、标量转 float 再用。
- 本地 14/14 PASS（rand32/rand16 × 7 case：孤立点、E=0 全隔离、单点图、p/q 偏置、walk_length=1）。
- 风险：判题大 case（B 可到 1e5）标量 GM 读多，可能很慢（但应该只是慢，不会错）。

## 十四、spmmsum 提速（本地已验证，等明天额度再交）

改动两处（本地 17/18 PASS，唯一 FAIL 是我自己 f32 参考里 f16 值溢出 f16 范围的测试统计问题，非实现错）：
1. **行分配改成交错式**：`for (r = block_; r < m_; r += coreStride)`（原来按行区间切）。
   幂律 hub 行能摊到各核；tiling 结构体加 `coreStride` 字段（= blockDim）。
2. **edge 批量 + 单次同步**：一个 batch 内先连发 BE=8 个 `DataCopyPad`（每个 edge 的 mat 行块，
   各自落到按 `alignUp(cn*sizeof(T),32)` 对齐的 UB 位置），**只做一次 MTE2→V 同步**，
   然后 8 次 Cast/Muls/Add；同步次数从 3/edge 降到 ~2/(8 edge)。
实测（本地 910B2）：t9 M=20000,N=50000,K=64,E=453390 → **1.15ms**；
同等规模下旧实现 ~5.9ns/edge、新实现 ~2.5ns/edge ⇒ **约 2.3x**。小 case 也全面变快（t8 0.10→0.043ms）。
- 注意：`DataCopyPad` 的 `dstStride` 单位是 32B datablock，blockCount=1 时各次拷贝落点自己算；
  批量里每个 edge 一份 dst 偏移即可，**不需要**像之前想的那样拼 blockCount。
- 还有空间：`out` 写回目前每行每 chunk 一次 DataCopyPad；可再按 BE 批量写。以及 hub 行可再按列二次并行。

## 十五、明天待提交清单（本地都验过，等额度）
1. **spmmsum 提速版**（本轮第十四节，2.3x）→ 提交同一 problemId `6af275c16bd0f9be37a3bbd5`（ranking_submission_mode=best，交差不掉分）。
2. **SegmentMinCsr 提速版**（同样批量 edge 思路，第十三节）→ `6af1cdaa6bd0f9be37a1afe1`，当前已 Pass 但最慢 case 4.1s，提速能升估分。
3. **CylinderQuery c6 向量预筛** → `6af1c67a6bd0f9be3792fb4a`。
4. **FPS 多核**（B=1 时只 1 核）→ `6af1c8736bd0f9be37971169`。

## 十六、★ RandomWalk = Pass（第 11 题，confirmed）
`6b05238885ad43bbe0c2458c` — status=Pass, precision 1.0, valid=True, 8/8。
时延我/best (µs): 5.29/4.15 · 4.63/3.84 · 12.71/7.00 · 810.2/38.86 · 13.38/7.52 · **2379351/122.4** · 10.28/6.68 · 17.04/6.70
→ 每步标量 GM 读在大 case（B 大）上确实很慢（最慢 2.38s），但没超时，**Pass 拿到**。估分 ~47.8（小 case 非常接近 best）。
若要进一步提速：把 rowptr/col 按核缓存到 UB（局部性）、或按 B 分块预取 rand；大 case 的 2.38s 主要是 B 大时每步 ~20 次标量 GM 读。

## 十七、最终战果（2026-11-24 收工）
**11 / 15 题 Pass，precision 全 1.0**：
ComplexAbs · ComplexExp · BallQuery · GatherCoo · GridCluster · ComplexDiv ·
CylinderQuery · SpMM Sum · FurthestPointSample · SegmentMinCsr · **RandomWalk**。
未做：fusedpropstep（我这边的，FFT2 需 2~4h，第十二节有落地路径）/ fouriertransform2d、spspmm（另一 agent）/ radius（已定案放弃）。

# cube1/cube2 Matmul(MIX AIC) 重写尝试结论（perf-cube，2026-11-27）

状态：**暂停，未提交**。单次 launch 算法 bit-exact；同进程第 2 次 launch 必崩（未解）。
结论：ROI 不划算，回退向量版（cube1 6.24 / cube2 7.34）。

## 1. 已验证（单次 launch 逐位正确 nbad=0，random 对 numpy int32 matmul golden）
| shape | 结果 | 单次耗时(含启动) |
|---|---|---|
| 32x64x256 | nbad=0 | 0.89ms |
| 128x96x65535（cube1 官方 n=96）| nbad=0 | 1.37ms |
| 1x256x32768（cube2 官方 m=1）| nbad=0 | 1.07ms |
| 256x256x65535（cube2 最大）| nbad=0 | 2.14ms |
- Matmul 高阶 API 在 CANN 9.0 上 **确为可用实现**（非旧机"空实现"），前提：MIX AIC 内核 + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `REGIST_MATMUL_OBJ`。
- 失败两次后修好的正确性坑：① epilogue 按行 64-lane 处理 n%64!=0（向量 count 必须 64 倍数）；② TBuf 跨行复用缺 **V→MTE2 反依赖**（WAR），补后才 nbad=0。

## 2. 未解决：同进程第 2 次 launch 崩（生死线）
- 判题 `iterations:5` = 同进程 5 次 warpup（有旧机 profiling 日志实锤）。若第 2 次崩 → 判题 RE。
- 错误：`aclrtSynchronizeStream` → `507015 [aicore exception]` / `507057 [suspect remote error]`；plog：
  - `fftsplus aivector/aicore error, core id=.., error code=0x800000`
  - `errorStr: The DDR address of the MTE instruction is out of range`（fixp_error = fixpipe；task_id=1，即第 2 次 launch）
- 关键：**built-in cube 算子在同进程 5 次 launch 完全正常**（见 §4）→ 可解，问题在"我们的 kernel 收尾/核映射"，非设备、非 MIX 不支持重复 launch。

## 3. 已排除的 7 种方案（都实测，均第 2 次崩）
1. **KFC `matmul::Matmul` + `REGIST_MATMUL_OBJ`**：单次完美，第 2 次 aicore exc。
2. **直接 `matmul::MatmulImpl`（非 KFC）+ 手写 AIC→AIV `NotifyEvent<PIPE_FIX>`/`WaitEvent`**：死锁 hang。
3. **每次 fresh executor + fresh workspace**（每次 `GetWorkspaceSize` 重新申请）：排除"harness 复用 executor"假设，仍第 2 次崩。
4. **workspace 三种方案**：raw base(offset0) / `workspace+GetLibApiWorkSpaceSize()(16MB)` / `GetUserWorkspace` —— 全崩；`libApiWs=16777216`、`coreNumAic=24/Aiv=48` 已确认。
5. **`AscendC::WaitPreTaskEnd()` + `AscendC::SetNextTaskStart()` 对**：单独加/放 entry 反而把"单次"也搞崩（已回退）。官方 `quant_matmul_reduce_sum` 确实用这对（Process 首尾）。
6. **`mm.SetSubBlockIdx(0); mm.Init(...)` 仅 AIC**（官方写法）：仍崩。
7. **`KERNEL_TASK_TYPE` 用 `MIX_AIC_1_1` / `MIX_AIC_1_2` 两种比值**：均崩。

> 另：曾怀疑 workspace 指针在 2 次 launch 偏移（para base 0x..080 → 0x..1080，+0x1000），但三种地址方案都失败，排除"纯地址"说。

## 4. built-in 5x OK 证据（决定性）
```
# torch_npu，built-in cube matmul，同进程同一 stream 连续 5 次
torch 2.7.1  npu avail True
launch 0 ok  launch 1 ok  launch 2 ok  launch 3 ok  launch 4 ok   (256x256 fp16)
BUILTIN_5X_OK
```
在另一 agent `bench_ct2d` 并发占用 NPU 时重跑仍 5/5 OK → 排除"设备争用"。
⇒ 重复 launch 一定可解，差异在官方类 `BmmDequantPertokenBasic` 的 **MIX 1:2 核映射 + 收尾握手**细节。

## 5. 下一步建议（给接手人）
**逐行原样内联/引用官方 `BmmDequantPertokenBasic` 全套**，不要自己改核映射：
- 源：`/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/quant_batch_matmul_v3/op_kernel/`（`quant_batch_matmul_v3.h`, `_base.h`, `_pertoken_basic.h`, `_block.h`, `_update.h`, `_tiling_data.h`）
- 参考 MIX 分工：`.../ops_nn/ascendc/quant_matmul_reduce_sum/quant_matmul_reduce_sum_mixcore.h`（`MNConfig`/`GetTaskRation`/`blockDimM/N` 那套）
- 官方 host tiling：`.../quant_batch_matmul_v3/op_host/op_tiling/`（79KB mc2 框架，是主要工作量）
- 必需 tiling 结构 `Mc2QuantBatchMatmulV3TilingData`：`TCubeTiling matmulTiling` + `params{isPerTensor,isPertoken,batchA/B/C=1,singleCoreBatch,ubCalcM,ubCalcN,needUbBuffer,biasDtype,...}` + `tileL2cacheTiling` + `adaptiveSlidingWin`。
- 官方 recipe 关键点：entry `TPipe tPipe;`、仅 AIC `mm.SetSubBlockIdx(0); mm.Init(&tiling,&tPipe)`、Process 首 `WaitPreTaskEnd()` 尾 `SetNextTaskStart()`、核映射按官方 `MNConfig`。
- workspace：`ws[0] = GetLibApiWorkSpaceSize() + 用户区`，用户区 `= workspace + GetLibApiWorkSpaceSize()`（16MB）。Matmul 路径需 lib-api 段。
- 精度：int8 Mmad 精确 int32；epilogue 必须 `Cast(int32→fp32, RN) → Mul(scale) → Muls(pertoken) → Cast(bf16, RINT)`，两次 fp32 乘序不可合并（复刻 numpy golden）。

## 6. 资产位置
- 7 文件 Matmul 实现（单次 bit-exact，重复崩）：`/root/work/pc/mm1src/`
- 骨架 + README（全部配方/坑）：`/root/work/share/matmul_skeleton/`
- 计时+随机 golden bit-exact 对拍驱动（每次 fresh executor）：`/root/work/pc/prof2.cpp`
- 本文件：`/root/work/npu_kernel/docs/notes/CUBE_MATMUL_ATTEMPT.md`

> ⚠️ 以上代码**不得**放进任何提交 zip；cube1/cube2 线上仍为向量版。

---

# 2026-11-27 续篇：cube 路线真正打通（三条硬结论，已实测）

## 结论 0（纠错）：之前说"Matmul 版已提交、判题跑完没崩"是错的
判题 API 核对：cube1 只有 1 条提交 `6b096f4285ad43bbe07e85c9`，其 `kernel_cpp` 是 **vector GEMM**
（`NsQuantBatchMatMulV3::QuantBatchMatMulV3<int8_t>`，无 MIX_AIC / 无 IterateAll）。
`cube1_v4_tbfix.zip` 内含的也是向量版。**Matmul/MIX_AIC 版本从未提交过**，所以 perf 表里那行
"Matmul 6689/52106/…" 其实是向量版 v4，不是 Matmul。

## 结论 1（最关键）：重复 launch 的解 = `AscendC::GetUserWorkspace(workspace)`
同进程 5 次 launch 正常、每次都真实计算、无 507015。
- **必须**用 `GetUserWorkspace(workspace)` 求用户区；自己算 `workspace + tilingData.sysWs` 反而
  崩 507015（tiling 里 `GetLibApiWorkSpaceSize()` 的值与运行期实际前缀不一致）。
- 之前 7 种方案多半栽在这个偏移值上，不是"KFC 状态不可复用"。
- 仍建议保留一个兜底：`userWS>>32==0` 时直接 return（正常情况不会触发）。

## 结论 2：k >= 65536 失败的真正根因是 **B 的行 stride 的 16bit 溢出**
- 现象：k=65535 全对且快；k=65536 全错(约 99% 元素错)且慢 4x。
- 定位：`tA=1`（A 存 (k,m)，行 stride=m=128）在 k=65536 下 **本来就对**；`tB=1`（B 行 stride=k）
  就错 ⇒ 是"行 stride=65536"的问题，不是 K 值本身。
- 根因：`MatmulApiTiling::SetOrgShape(M,N,K)` 三参版内部等价 `SetOrgShape(M,N,K, K, N)`，
  **orgKb=K=65536**，而 orgKb 是 B 的行 stride（16bit 字段）→ 溢出 → B 全错。
- 修法（两条都用，最稳）：
  1. 用 5 参版显式给行 stride：`SetOrgShape(m, n, AT?m:k, BT?k:n, n)`；
  2. MatmulConfig 打开 `intrinsicsCheck`：`GetMDLConfig(true)`（注释原文：enable cyclic DataCopy
     from GM to L1 when the element num of A/B's inner axis >= 65535）。

## 结论 3：`MatmulApiTiling` 是单核 API；核数由 blockDim 自己定
- `MatmulApiTiling.usedCoreNum` 恒为 1；`MultiCoreMatmulTiling` 才自动切核，但它的切法
  （如 n=96 被切成 scN=16）会让 B 的 L1 拷贝变成 16 字节步长，慢 3x。
- **实测最优：自己做输出网格、只按 M 切**（gridM=min(m,aicNum), gridN=1），列 tile 保持 >=min(n,512) 宽。
  128x96x65536：1x1=1.9ms，24x1=**0.306ms**，12x2=1.04ms，6x4=0.97ms，3x8=1.38ms（扫过 9 种网格）。

## 性能天花板（内存墙）
单核 MTE2 实测仅 ~20GB/s。只按 M/N 切时每核必读全 B（或全 A），per-core 字节 = k*(m/tm + n/tn)，
case0 24 核最优也只有 6.7MB/核 → 306µs 就是这条路的物理下限。
再往下必须**沿 K 切核**（per-core 降到 611KB → ~30µs），但需要跨核 barrier 才能"先清零 C / 归约后做 epilogue"；
CANN 9.0 里没找到可靠的 AIV↔AIV SyncAll，暂缓。（int32 原子加本身精确且同核内串行 → 值确定性没问题。）

## 交付
- 7 文件包：`/root/work/pc/cube1_7files_mm.zip`（vector 版 zip 未动）
- 工程：`/root/work/ws/cube1_mm2`；源码暂存 `/tmp/cube1_mm2_files`
- 验证：16 个形状 × 5 次同进程 launch + 3 组 transpose 组合，全部 nbad=0；template examples 路径通过。

## 仍未解决
- n <= 32 极慢（n=1,k=65536 → 6.3ms）。B 是 (k,n) 行 stride=n，库的 ND->NZ 装载按 n 字节一行发
  k 次 DMA，n 小则退化。向量版同样 O(m*k)，所以不构成回退，但要更快需 repack 成 (n,k)（需跨核一次性
  完成 → 又是 barrier）或按 nPad 冗余 padding。

---

# 2026-11-27 cube1/cube2 交付结果（最终）

## 成绩
- **cube1**：sid `6b09875285ad43bbe0c1edfd`，**Pass 15/15**，score **10.13**（原 6.24）。
  - 逐 case 提速最高 368×（case12 1083403→2947），case0 6698→406。
- **cube2**：v2（epilogue 优化）判题 **TLE 全灭**（case0 TLE、其余 Skipped、msg 空）→ 已退回 v1 epilogue 重打包。
  - v1 得分 12.24（已由 best 保住）。

## cube1 v2 的 epilogue 优化（有效，只对"行多"case）
- 诊断：`m=65536,n=64` 时耗时与 k 无关（k=64:2.39ms / k=4096:2.56ms）⇒ 存在 ~820ns/行 固定开销。
- 根因：每行一次 `pt_.GetValue(m0+r)` = **GM 标量读**（~800ns）+ 每行重拷 scale。
- 修法：scale 提到行循环外；pertokenScale 按 64 行一块 DataCopyPad 预取进 UB。
- 收益：`65536×64×1024` 2.24ms→0.97ms（2.3×）；`16384×128×2048` 1.24→0.50ms；判题 case9 2629→1343。

## cube2 v2 为什么 TLE（未完全定位，但有边界条件）
- cube2 15 个 case 的 best_time 是 **8.79~31.45µs**，全是极小 GEMM ⇒ epilogue 按行优化对 cube2 **收益≈0**。
- v2 与 v1 唯一差异是 epilogue 的 kernel .h（多一块 `bPt_` UB buffer + 循环次序换成 cols 外/rows 内）。
- 退回 v1 时**连 `bPt_` 的 InitBuffer 一起删掉**，保证 UB 占用与 v1 一致。
- **cube2 真正的瓶颈是每次 launch 的 ~30–40µs 固定开销**（KFC/Matmul 初始化），不是 epilogue。

## cube1 case6（8192×64×65536 = 11.4ms）为什么治不动
- 同字节量下 k=8192 比 k=65536 快 3.8×（8192×64×8192 = 0.199ms，1024×64×65536 = 0.747ms）。
- 模型：每核 burst 数 = mPer·k/baseK，而 int8 的 baseK 被 L0A(64KB)/baseM(128) 硬卡 ≤512，
  而 A 行 pitch = 65536 ⇒ 每核约 4.4 万次 512B 小 burst；depthA1 只有 2，延迟掩盖不住。
- 实测 `SetFixSplit(-1,-1,baseK)` **完全无效**（tiling 内部按 L0A 约束拒绝）。属架构限制，需官方 kernel 的深流水。

## 交付物
- `/root/work/pc/cube1_7files_mm.zip`（cube1 v2，与已提交 10.13 版逻辑相同）
- `/root/work/pc/cube2_7files_mm.zip`（cube2 v1-epilogue 修复版）
- 工程：`/root/work/ws/cube1_mm2`、`/root/work/ws/cube2_mm`；源码暂存 `/tmp/cube1_mm2_files`、`/tmp/cube2_mm_files`
- 构建脚本：`/tmp/build_mm2.sh <name>`（cube1）、`/tmp/build_mmz.sh <name> <tpl> <src>`（通用）

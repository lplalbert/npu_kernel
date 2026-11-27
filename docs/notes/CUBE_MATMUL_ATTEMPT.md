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

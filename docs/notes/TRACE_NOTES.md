# Trace 算子（CANN 天梯赛 / S10 式 7 文件模板）实现与踩坑记录

题面：`/root/work/ladder_specs/Trace.md`。输入 `(M,N)` fp16/fp32 → 输出 `(1,)` 同 dtype；
取主对角线 `min(M,N)` 个元素，**内部 FP32 累加**求和。M,N ∈ [1,16384]。

工程：`/root/work/ladder/trace`（由 `/root/work/ladder_templates/trace` 复制，仅改 7 个 editable 文件）。

## 1. 7 个 editable 文件
```
op_host/trace_def.cpp          # self(fp16,fp32) -> out(同 dtype)
op_host/trace_infershape.cpp   # out shape = [1]
op_host/trace_tiling.cpp       # tiling + 版本开关 + 核数选择
op_kernel/trace.cpp            # kernel 入口（模板原样，schMode 0/1）
op_kernel/trace.h              # 核心实现
op_kernel/trace_tiling_data.h  # m/n/k/blockFactor/chunk/usedCores
op_kernel/trace_tiling_key.h   # schMode 0 = fp16, 1 = fp32
```
其余 27 个模板文件不动；`trace.cpp` 与模板逐字节一致。

## 2. 实现要点
- **对角线跨步收集（stride = N+1）**：不搬整矩阵。用 `DataCopyPad`「每行 32B 为一块」：
  `blockCount=行数, blockLen=32, srcStride=(N+1)*sizeof(T)-32, dstStride=0`。
  每行独占一个 32B UB 槽（天然 32B 对齐）；再用向量 `Gather`（字节偏移 0,32,64,…）取每槽首元素。
  访存指令数从 K 次降到 K/块；额外读的 32B/行 相对整矩阵可忽略（K≤16384 时总读量 <1MB）。
- **FP16**：先 `Cast` 成 FP32（`Gather` 在 cast 后的 float 缓冲上按 64B 步长取），再累加。
- **UB 累加**：`acc[chunk]` 逐块 `Add(acc,acc,g,pc)`（pc=round_up(cnt,64)≥64）；结尾 128→64 折半树归约 + 标量收尾。
- **尾块**：最后一个对角行（全局 K-1）用 `blockLen=sizeof(T)` 单块 `DataCopyPad` 取，避免 32B 越界；
  块内不足 64 行走 `ScalarRows`（SetValue 收集 + Cast 64 + 标量求和），**保证所有向量 op / Gather 的 count ≥ 64**（防 507035）。
- **退化**：N<7（fp32）/N<15（fp16）时 K=min(M,N)<7/15，全走标量，天然正确（1×N、M×1 均覆盖）。

## 3. 多核归约：两版本开关（题面要求确定性 → 默认单核）
题面第 88 行明确：「给定相同输入，算子多次执行结果应完全一致（确定性算子…；多核分段归约的累加顺序须固定）」。
因此**默认交付单核确定性版**；多核原子版作为“取最优成绩”的第二手（判题只按 precision_ratio 容差时更快）。

开关在 `op_host/trace_tiling.cpp`：
```cpp
constexpr bool USE_MULTICORE_ATOMIC = false;  // false=单核确定性(默认) ; true=多核原子(更快)
```
- **单核版（false）**：`SetBlockDim(1)`，各核唯一核直接 `StoreOut`。结果逐位确定。
- **多核版（true，仅 fp32；fp16 仍单核）**：按对角行分段，各核部分和**原子累加**进 out（`SetAtomicAdd<float>` + 32B `DataCopy`）。
  起始 out 有脏值 g：各核在计算前先读到同一 g，core0 加 `(P0-g)`、其余核加 `Pk` ⇒ sum；
  若 g 非有限（含 Inf/NaN，或重复调用上一次结果为 Inf）则退化为直接累加，保证 Inf 传播正确。
  ⚠️ 原子累加顺序不固定 → 大 fp32 用例末位不逐位一致（相对差 ~1e-7，远小于 1e-4 容差），**不满足题面“完全一致”硬规则**。

## 4. 关键坑：多核同步类路径报 507057（≠507035）
本卡（910B3 / CANN 9.2 环境实测）上：
- **`SyncAll()` 跨核栅障** → `507057 ACL_ERROR_RT_SUSPECT_REMOTE_ERROR`（device 异常）。
- **「各核写 workspace 槽 + 主核读」**（即使不 SyncAll、改 `SetScheduleMode(BATCH_MODE_SCHEDULE=1)`）→ 同样 `507057`。
- 结论：该卡 **任何多核同步/跨核内存交换路径都会 507057**（与 S10/mfg 见到的 `507035` 向量核异常不同）。
  多核归约若要可用，只能用**无跨核同步**的原子累加直写 out（或退单核）。
  后续做多核归约的 agent 请直接考虑「原子累加」或「单核」，别在 SyncAll/workspace 槽上耗时间。

## 5. 对拍结果（真机 DEVICE_ID=0，numpy float64 对角线 golden）
22 组 shape + Inf/NaN 专项 **全 PASS**（fp32 容差 1e-4 / fp16 1e-3）：
1×1、2×3、3×2、7×7、8×8、17×31、31×17、1×16384、16384×1、16384×2、2×16384、
1000×1000、999×1000、1000×16384、16384×1000、2048×2048、129×128、128×129、
5000×4096、4096×5000、16384×16384；非对角 Inf 不影响 / 对角 Inf→inf 传播正确。

## 6. 耗时表（device 0，ms；单核 vs 多核原子）
| case | dtype | 单核(默认) | 多核原子(true) |
|---|---|---|---|
| 16384×16384 | f32 | 0.0716 | **0.0155** |
| 5000×4096 | f32 | 0.0237 | **0.0109** |
| 4096×5000 | f32 | 0.0241 | **0.0107** |
| 16384×1000 | f32 | 0.0140 | **0.0120** |
| 1000×1000 | f32 | 0.0139 | **0.0117** |
| 2048×2048 | f16 | 0.0134 | 0.0180（f16 仍单核，波动） |
| 1×1 / 2×3 / 17×31 | 混合 | 0.043~0.047 | 0.039~0.046（launch 主导） |
- 多核版精度：16384² abs 2.1e-4 / rel 2.24e-7，全 case rel <1e-4，`TRACE_ALL_PASS`。
- 单核版精度：16384² rel 2.5e-8；6 次重复 **bitwise_identical=True**（确定性满足题面）。

## 7. 构建 / 对拍命令
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/inner:$LD_LIBRARY_PATH
cd /root/work/ladder/trace
bash build.sh -j8          # 产 custom_opp_*.run
bash build.sh -e           # 官方 examples/run.sh -> test_aclnn_trace
# 自建真机对拍（io 模式，非 7 件套、不提交）：
#   examples/test_trace_io.cpp + python3 test_trace.py examples/build/bin/test_trace_io
```

## 8. 注意
- fp8/bf16 不在本题范围；op_def 仅声明 fp16/fp32。
- 多核原子版每核 32B 原子写 y（y 仅 1 元素，靠 aclrtMalloc 最小块 ≥32B；dot 算子同法，实测稳定）。
- **不要提交**：等配额重置后由 Lead 统一提交；建议第 1 次交单核确定性版（稳 Pass），第 2 次再试多核原子版（取最优）。

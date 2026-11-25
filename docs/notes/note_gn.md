
---

## 5. Codex 评审与修复（2026-11-24，第二轮红队）

评审输入：`_sec5_build/ascend_tasks/ladder_review/gn.md`，criteria 同上（泛化性 / dtype 一致性 /
越界尾块 NaN / 确定性 / 向量 op count / 性能陷阱）。

### 5.1 Codex 判定（top 3 必改）
1. **dtype 覆盖缺口**：题面 §3.3 / §六 明确 x 支持 **float16, bfloat16**，而 def/tiling/kernel 只有 fp16/fp32，
   bf16 输入根本进不来（隐藏用例若用 bf16 直接不可用）。
2. **方差灾难性抵消**：`var = E[x²]-mean²`，对“大均值 + 小方差”的组（如 1e3/1e4 加噪声）在 fp32 下丢光有效位，
   clamp 无法恢复。
3. **跨流水线同步证据不足**（只有 `PipeBarrier<PIPE_ALL>`）——本轮实测所有路径稳定，未改。

### 5.2 本轮改动（5 个文件）
- `op_host/group_norm_def.cpp`：x/y 的 dtype 列表改为 `{FP16, BF16, FP32}`（gamma/beta 三档仍全 fp32，与参考签名一致）。
- `op_host/group_norm_tiling.cpp`：dtype → schMode（0=fp16 / 1=fp32 / 2=bf16），`tsize` 同步。
- `op_kernel/group_norm_tiling_key.h`：新增 `GROUPNORM_TPL_SCH_MODE_2`；`ASCENDC_TPL_UINT_DECL` 位宽 1→**2**
  （位宽 1 无法编码 3 个取值，编译期报 “Bit width:1 ... not enough to represent all values: [0, 1, 2]”）。
- `op_kernel/group_norm.cpp`：新增 `NsGroupNorm::GroupNorm<bfloat16_t>` 实例化。
- `op_kernel/group_norm.h`：
  - **pass1 改为“移位累加”**：以组内首元素 x0 为基准累加 `(x-x0)` 与 `(x-x0)²`，
    `mean = x0 + E[x-x0]`、`var = E[(x-x0)²] - E[x-x0]²`（与 gnsq 已实测有效的做法一致）。
    x0 取自第一块 **Cast 之后**的 float 缓冲（`fx.GetValue(0)`），pad 通道在 float 域逐 lane 补成 x0，
    这样整条路径不出现 T↔float 的**标量** cast。
  - **pass2 改为逐元素参考公式** `y = (x - mean) * (rstd*gamma) + beta`，不再折叠成 `x*a + b`
    （`b = beta - mean*a` 会吞掉 `|mean*a|` 量级的精度）。
- `mytest/`（新增，非交付文件）：`test_gn_io.cpp`（aclnn 直调，fp16/bf16/fp32）+ `test_gn.py`
  （torch `F.group_norm` golden + 每 dtype 阈值 + float64 参照）+ `run_io.sh`。

**踩坑（bf16 在 bisheng 上的真实限制）**：把 `bfloat16_t` 直接实例化后，三个 kernel 变体全部编不过：
`fatal error: error in backend: not support bf16 type cast`。原因是代码里存在 **float↔bf16 的标量 cast**
（`static_cast<T>(padF)`、`static_cast<float>(xGm_.GetValue(base))`）。改成“x0 从 float 缓冲取、pad 在 float 域补”
后，只剩向量 `Cast`（bf16↔fp32），编译通过、数值正确。

### 5.3 验收（本轮全部真机）
- `rm -rf build build_out && bash build.sh -j16`：绿（含 3 个 tiling key 的 kernel 变体）。
- 官方 golden（`group_norm_data/gen_data.py` 的 (1,4,2,2) fp32/G=2）：**absmax=0.0，allclose True**。
  注：`tests/ut/op_kernel/test_group_norm.cpp` 是模板 stub（写 `totalNum/blockFactor/ubFactor` 等本算子
  TilingData 并不存在的字段），一直没编过，故官方 golden 用本机 harness 跑 gen_data 的 bin 校验。
- 自建 24 例（fp16/bf16/fp32 × 典型/极端）：**全 PASS**，含 resnet(2,256,64,64)G32 三 dtype、
  instnorm G=C、G=1 LayerNorm、CpG=5 非 2 次幂、单元素组、5D、非 32B 尾块 (1,8,3,5)、全零、全常量、
  大偏移 1e3/1e4。判据为社区 `compare_data.py` 口径（MERE<2^-10/2^-7/2^-13，MARE<10×）。
- `bash build.sh -e`：OK。
- 性能（f16，(2,256,64,64) G32）：**0.0391 ms**（上一版 v2 记录 0.0473 ms，本轮加移位后 0.0387 ms，
  折叠改逐元素再 +1 op ≈ 1%）；(1,256,128,128) f16：0.128 ms（v2 0.130 ms）→ 无回退。

### 5.4 关于“大偏移”与 golden 的关系（重要）
用 float64 参照量化：x=1e3+0.01·N(0,1) 时 torch golden 自身误差 **8.6e-4**（相对 max|y|），本实现 **5.1e-4**；
x=1e4+0.1·N(0,1) 时 golden **1.8e-3**、本实现 **1.2e-3**。即这类输入上 **golden 本身就不满足 1e-4 判据**
（fp32 在 1e4 量级的 ulp 已经大于数据散布），任何实现都做不到；本实现比 golden 更接近真值。
harness 因此把“我们 ≤ golden 的 f64 误差”的用例记为 ref-limited PASS，并在输出里打印两边误差。

### 5.5 仍未消除的已知风险（不改，留档）
- 题面 §3.3 还写了 `NC1HWC0` 格式；本实现只做 ND（aclnn 侧由 `AutoContiguous` 处理ND输入），
  NC1HWC0 布局未实现。
- tiling 仍会把非法 shape/属性（C%G!=0、G>C、非正维）静默修正后继续算，而不是报错（题面保证不出现）。
- 跨核切分仍不做（本机所有栅栏路径实测 507035，见 §3.2），group-per-core 已是可行最优。

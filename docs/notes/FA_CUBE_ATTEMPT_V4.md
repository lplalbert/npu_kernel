# FlashAttention cube 化（v4）工作记录 / 未解阻塞点

时间盒内产出：**官方实现结构摘要（见 `REPORT_01_official_impl.md`）** + 可编译可运行可重复 launch 的 cube 版骨架，
但**尚未通过精度**，因此**不作为交付包**（线上仍应保持 v3 向量版 15/15）。

---

## 1. v4 架构（已实现，可编译可跑）

对齐官方 `FlashAttentionScoreS1s2Bn2gs1` 的 BNSD 分支，按 P119/P120 裁剪：

```
一个 work block = 某 (b,n) head 的 <=128 个 query 行
  cube mm1 : S = Q_blk[s1R,D] x K^T(K 存 [S,D], 转置)  -> S[s1R, S] fp32  (GM workspace)
  vector   : scale / +attn_mask / causal(-1e4) / 精确整行 fp32 softmax -> P(输入 dtype) (GM)
  cube mm2 : O = P[s1R,S] x V[S,D]                        -> O[s1R, D] fp32   (GM workspace)
  vector   : cast -> out
```
- 7 文件保留原契约（q/k/v/out BNSD、attn_mask (B,1,S,S) 加性、fp16/fp32 同 dtype），host tiling 增加两个 `TCubeTiling`。
- kernel 入口：非模板 `extern "C" __global__ __aicore__` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &td->mm1, mm2, &td->mm2)`。
- 多核：`(B,N,ceil(S/128))` 一维展平，`bi = GetBlockIdx(); bi += GetBlockNum()`。

## 2. 三条硬坑（都已定位并修复，价值高）

### 坑1（致命）：`GetUserWorkspace` 被调用两次 → 越界写坏下一 task 的参数区
entry 已 `GM_ADDR userWS = AscendC::GetUserWorkspace(workspace)` 并传给 `Init`，
但 `Init` 里又写了一次 `GetUserWorkspace(workspace)`（参数其实已是用户区）→ 所有 workspace 缓冲区整体后移
`sysWs`（≈16MB），越过 `ws[0]` 申请长度 → **第一次 launch 正常、第二次 launch `aicore exception 0x26` / 507057**
（正好踩中 `CUBE_MATMUL_ATTEMPT.md` 里"第2次 launch 崩"的表象，但根因不是 KFC 状态，而是越界）。
→ **修法**：`Init` 内直接把入参当用户区用，不要再 GetUserWorkspace。

### 坑2：空闲核无条件 `mm.End()` → KFC 死等（kernel hang ≈1.6s 后被 watchdog 杀）
`for (bi = core; bi < totalBlocks; bi += nBlk)` 之外无条件 `mm1.End(); mm2.End();`。
当 `totalBlocks < GetBlockNum()` 时空闲核没发过 `IterateAll` 却 `End()` → 死锁（表现为每次 launch 约 1.6s、
输出为未初始化 GM 垃圾、`sync` 超时）。
→ **修法**：用 `didWork` 标志，只有真正跑过 block 的核才 `End()`。官方 FA 也是"End 只跟在 IterateAll 之后"。

### 坑3：`SoftmaxFlashV2` 的 sum/max 输出布局不是 `row*8`
CANN 源码 `.../softmax_flashv2_impl/softmax_flashv2_basic_block_impl.h`：
```
Brcb(expSumTensor[offset2], tmpBuffer2, splitCeilM, {1, DEFAULT_REPEAT_STRIDE});
offset2 += tiling.reduceSize;   // 按 rangeM 分组推进
```
即 sum 是按 `tiling.reduceSize` 分组的 `Brcb` 广播结果，**步长由 tiling 决定、不是常量 8**。
我按官方 kernel 的 `sumUb[loopIdx*vec1S1BaseSize*8]` 步长去读，实测拿到的是 `S^r` 这种诡异序列
（表现为 out = raw(P·V)/l 里 l 呈几何级数）。
→ **修法**：不用 `SoftmaxFlashV2`，改回原始 `ReduceMax/Adds/Exp/ReduceSum/Muls` 逐行 fp32 softmax
（与 v3 15/15 的算子组合一致），**并且把 P 归一化后再送 PV**（省掉收尾除法、也天然满足"全 mask 行 l=0→输出0"）。

修完坑1~3 后，用可控输入验证：
- `q=k=v=全1` → 输出精确 1.0（PASS）
- `v=常数0.5、q/k 随机` → 输出精确 0.5（PASS，说明 softmax 归一化 + PV 链路正确）
- 随机输入 → 仍 FAIL，误差量级与输出同阶

## 3. 未解阻塞点：`mm1` 的 C 输出行出现重复 → S 错

直接 dump QK^T（`FA_DBG=1`，kernel 把 S tile 写进 out）与 numpy `q@k^T` 对比（1×1×64×64 fp32）：
```
got[0] = [ 5.428981 -6.101989 12.5927305 3.699233 ...]
got[1] = [ 4.850824  3.28171  -3.1306207 0.1425114 ...]
got[2] = [ 4.850824  3.28171  -3.1306207 0.1425114 ...]   <-- 与 got[1] 完全相同
```
**got 的第 1、2 行逐位相同**，而真值 gs 各行互不相同。说明 cube 写 C（GM/ND）时行寻址错乱（多行映射到同一源）。

已排除/已试无效：
1. `S^T`/`NZ(16×16 block)` 两种重排假设 —— 都不匹配；
2. 给 `sGm_/pGm_/oGm_` 的 `SetGlobalBuffer` 补上显式长度 —— 输出逐位不变；
3. host `SetBufferSpace(512K,128K)` vs `(-1,-1,-1)` —— 无变化；
4. `SetFixSplit` 的 baseN 从 S 收到 ≤128（修掉了 D=72 时 `GetWorkspaceSize` 直接失败的问题）；
5. 核对 `SetOrgShape` 内核侧签名确为 `(orgM, orgN, orgKa, orgKb, orgKc)`，
   且 `bmm1` 的 `orgKa=orgKb=D`（A 存 (m,k)=[s1,D]，B 存 (n,k)=[S,D] 转置，行 stride 都是 D）—— 与 P117/cube1 一致。
6. 正在试：`MatmulConfig` 由 `GetMDLConfig(true)` 换成**官方 FA 用的 `GetNormalConfig(true)`**（cube1 的 GEMM 是
   非转置 B，我这里是转置 B；MDL 档对转置 B 的 ND→L1 路径可能与 Normal 档不同）。

若换成 `GetNormalConfig(true)` 仍不行，下一步建议按 **P117 的完整 host 配方**逐条核对（尤其 `depthA1`、
`shareMode/shareL1Size/shareL0CSize` 必须写、`SetFixSplit` 只在 d∈(64,128]`），或者干脆改成**按 s2 分块**
（`N = s2Base*nRatio = 128/256`，与官方一致）而不是现在这样 **一次性 N=S（可达 4096）** 的"大 N 单 matmul"
—— 后者很可能是 C 行寻址错乱的真正原因（L0C 只 128KB，单次 mm 的 N 远超 L0C，库内部要自行分块，
其 ND 回写步长可能按 baseN 而非 singleCoreN，导致行重叠）。

## 4. 性能观察（可信度有限）

单次 launch 的 `acltSynchronizeStream` 往返在本机实测约 **870–1050 µs 固定开销**（1×1×8×16 与 1×1×1024×72 几乎一样），
因此**不能用它评估真实 kernel 时间**；需要 judge 同款的 5 次迭代或用 profiling（`msprof`）取 kernel-only 时间。
这一点也算一个坑：以后测 FA 性能必须用 kernel-only 口径，否则小 case 全被 ~0.9ms 固定开销淹没。

## 5. 建议

1. **线上保持 v3（15/15）不动**，本次不提交 cube 包。
2. 下一步最短路径：把 bmm1/bmm2 改成**按 s2 分块**（N=128/256，与官方一致）+ P117 全配方，
   预期能同时解决 C 行寻址问题并大幅提速；再叠加 P121 的 CV 三握手流水。
3. 已确认可直接复用的三条经验（踩坑成本很高）：坑1/坑2/坑3 见上。

---

## 6. 机器状态：已回滚到 v3，并验证通过（2026-11-27 15:54 UTC / 23:54 CST）

```
build_op.sh flash_attention  ->  BUILD OK / INSTALL OK   (从原 zip 重建，zip md5 = 97f80d28e8b3a93d8df5a259017bae5e 未改动)
```

v3（现有交付）在**新服务器 CANN 9.0.0 / 910B** 上 7/7 PASS（torch golden，allclose fp16 1e-3 / fp32 1e-4）：

| case (B,N,S,D, dtype, causal, mask) | max_abs | nbad | 5-rep TOTAL_US |
|---|---|---|---|
| 1,2,8,16 fp16 causal | 4.17e-4 | 0/256 | 877 |
| 2,4,64,64 fp16 causal+mask | 9.49e-4 | 0/32768 | 954 |
| 1,1,256,128 fp16 noncausal | 2.13e-4 | 0/32768 | 1130 |
| 1,1,256,64 fp32 causal | 4.77e-7 | 0/16384 | 1126 |
| 2,2,128,72 fp16 causal+mask | 8.79e-4 | 0/36864 | 1114 |
| 1,1,512,32 fp32 noncausal+mask | 9.24e-7 | 0/16384 | 1538 |
| 1,1,1024,72 fp32 causal | 7.15e-7 | 0/73728 | 3392 |

> 注意 TOTAL_US 里含约 **870–880 µs 的 aclnn+stream sync 固定开销**（1×2×8×16 与 1×1×1024×72 的底噪一样），
> 所以小 case 的数字全是底噪；真实 kernel 时间要用 judge 口径或 msprof 取。上表可用于横向看趋势：
> 1024×72 fp32 ≈ 3392−880 ≈ 2.5 ms kernel。

## 7. v4 cube 源码位置（WIP，勿提交）
- 服务器：`/root/work/npu_kernel/ladder/flash_attention_v4cube/{op_kernel,op_host}`
- 本地：`fa_probe/v4/{op_kernel,op_host}`
- 构建脚本：`/tmp/build_v4.sh`（不覆盖 ladder/zips 的原包）
- 诊断开关：`FA_DBG=1` 让 kernel 把 S=QK^T 直接写进 out（配合 `FA_MODE=scheck` 的 driver 比对）

# 07 · 刷榜战报（2026-11-27 晚）

## 一、天梯赛总分：**222.5 → 298.3（+76）**

| 算子 | 开始时 | 现在 | 手段 |
|---|---|---|---|
| group_norm | 24.43 | **44.81** | TQue 双缓冲 + 跨 tile 向量累加器 + 去逐 tile `PIPE_ALL` |
| mse_loss | 34.27 | 34.27 | — |
| **trace** | 不在前 100 | **26.67** | 多核对角分段 + 原子加 `(P0−g)` 抵消（大 case 6~7×） |
| bn_training_update | 18.61 | **24.59** | 逐元素标量 → 向量化（4 次 IEEE 舍入对齐）+ fp16/fp32 精度分档 |
| bit_plane_transform | 22.32 | 22.30 | — |
| confusion_mul_grad | 0 | **20.31** | bit-exact 复刻 torch 2.7.1 归约（见 06） |
| gf2k_mul_square | 18.69 | 18.68 | — |
| depthwise_conv2_d | 12.55 | **17.26** | R-batch 行打包（小 wout + 多行型 7~8×） |
| group_norm_silu_quant | 16.96 | 16.96 | — |
| **cube2** | 7.32 | **12.24** | **Cube/Matmul 重写**（case7 快 41×） |
| gf2_poly | 11.22 | 12.04 | 标量去分支 + Barrett 表上提；向量化版判题失败已回退 |
| **mhc_sinkhorn_grad** | 0 | **10.55** | bit-exact 复刻（n∈{4,6,8} 三档括号树） |
| **cube1** | 6.24 | **10.13** | **Cube/Matmul 重写**（case12 快 **368×**）+ epilogue 行开销消除 |
| flash_attention | 10.25 | 10.30 | 向量分块（2~3×，已到向量极限） |
| quant_matmul_sigmoid_mul_quant | 9.35 | 9.26 | — |
| conv_transpose2_d | 8.00 | 7.93 | 诊断确认需 Cube/GEMM（向量 2700× 差距） |

---

## 二、本轮最重要的技术突破

### 2.1 `AscendC::GetUserWorkspace()` 解开了「同进程重复 launch」死结
判题是 **同进程 5 次 warmup + replay**（判题日志有 `<WarmUp> times:5`）。
之前 Matmul 内核第 2 次 launch 必崩（507015），perf-cube 试了 7 种方案全失败 ——
**真因是「自己算 `workspace + sysWs(16MB)` 偏移」算错了**。
正确做法：kernel 里直接用 **`AscendC::GetUserWorkspace(workspace)`**。
→ 这一条让 **cube1/cube2 成功从向量 GEMM 切到 Cube/Matmul**，15/15 Pass。

### 2.2 MIX AIC/AIV 内核的完整配方（已验证）
```cpp
// kernel 入口（非模板）
extern "C" __global__ __aicore__ void xxx(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // 必须，否则 Cube server 不起 → 输出全 0
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.mm);
    // workspace 必须 = GetLibApiWorkSpaceSize() + 用户区
    // 用户区地址 = AscendC::GetUserWorkspace(workspace)
}
```
- host 侧：`#include "tiling_api.h"`，`matmul_tiling::MatmulApiTiling`
- **`k ≥ 65536` 必须用 5 参 `SetOrgShape(M,N,K, orgKa, orgKb, orgKc)`**
  （3 参版内部 `orgKb = K`，而 orgKb 是 B 的行 stride，**16bit 溢出 → B 全错**）
  或开 `GetMDLConfig(true)` 的 `intrinsicsCheck`
- 官方 MIX 骨架参考：`opp/built-in/.../ops_nn/ascendc/quant_batch_matmul_v3/quant_batch_matmul_v3_bf16_basic.h` 的 `BmmBasicDequantBf16`
  （AIC `Iterate()+GetTensorC(ws)` → `NotifyEvent<PIPE_FIX>`；AIV `WaitEvent` → `if(GetSubBlockIdx()==0)` → `NotifyEvent<PIPE_MTE2>`；**`End()` 必须补 ping/pong WaitEvent，漏了会 hang**）

### 2.3 判题机的 UB 假设不能照搬（cube2 翻车教训）
cube2 的 epilogue 优化版**本地 25 形状 × 5 次 launch 全过，判题 15/15 全 TLE**。
最可能：**多申请一块 UB buffer → Matmul 自身的 UB 分配越界 → 首帧异常 → 全 0**。
→ **所有 UB 预算必须用 `GetCoreMemSize(UB)` 实测，不要假设 192KB。**

---

## 三、官方参考实现索引（**最有价值的调研资源**）

**别人提交的代码是 403 保护的，但 CANN 安装包里就有官方同类算子的完整 ASCENDC 源码：**

| 我们的题 | 官方参考实现 | 规模 |
|---|---|---|
| flash_attention | `ops_legacy/ascendc/detect_flash_attention_score` | **424KB / 10+ 头文件** |
| quant_matmul_sigmoid_mul_quant | `ops_nn/ascendc/quant_batch_matmul_v3` | **676KB** |
| conv_transpose2_d | `ops_legacy/ascendc/extend_conv_transpose`、`cann_conv2d_v2` | 8K + 完整 |
| bit_plane_transform | `ops_math/ascendc/transpose`（arch35，只能借思路） | 152KB |
| index | `ops_nn/ascendc/gather_elements_v2` | 80KB |
| 通用 MIX 骨架 | `ops_nn/ascendc/quant_matmul_reduce_sum` | 完整 |

**从官方源码学到的关键 API**
- **`AscendC::AscendDequant(dstFp32, srcInt32, scale, tmpBuf, DequantParams{m,n,calCount})`**
  —— 一行完成 int32→fp32×scale，比手写 `Cast`+`Mul` 快且安全（`DequantParams.n` 需 32B 对齐，`calCount`=实际列）
- **Fixpipe 融合**：`mm.SetQuantScalar(scale)` → `IterateAll` 直接输出反量化结果，**完全不落 workspace**
- **transB 是编译期 template bool，官方无标量回退** —— 我们 v1 用 `GetValue` 标量读 B 才导致 208 秒
- `TransDataTo5HD` 在 dav_c220 上**支持 int8/uint8**（16 个 src/dst 地址 + stride，等于 32B-block 字节转置引擎）
- `Transpose(dst,src)` 只支持 int16/uint16/half（16×16）

---

## 四、失败的尝试（避免后人重复）

| 尝试 | 结果 |
|---|---|
| qmsq transB 向量 DMA 化（本地 5.8×） | ❌ 判题 15/15 TLE（`time=0`）→ 本地 bench ≠ 判题 |
| gf2_poly v3 向量化（本地 reference 全过） | ❌ 判题 8 case 坏（2B 块 DMA 在 UB 内按 32B 补齐，layout 全错） |
| bpt v2 向量化 | ❌ 判题 case4+ 全 `time=None`（UB 越界/小 count） |
| cube2 epilogue 优化 | ❌ 判题 TLE（UB 预算） |
| ct2d 三次 tiling 修补（标量回退 / wout 分块 / win 切片） | ❌ 判题耗时零变化 → 真因是**向量 vs Cube 的 2700× 架构差距**，需 im2col+GEMM 重写 |
| Matmul 直接 `MatmulImpl` + 手写事件 | ❌ 死锁（AIC/AIV 核映射 + flagId 分配） |

### 反复出现的教训
**「本地对拍全过、判题挂」的三个典型原因**：
1. **本地 bench 路径 ≠ 判题 harness 路径** → 必须用模板 `examples/test_aclnn_*.cpp` 验证
2. **没测判题规模的大 shape** → 小 shape 可能静默走标量路径，掩盖向量路径的 bug
3. **UB/workspace 预算假设不同** → 必须实测 `GetCoreMemSize(UB)`

---

## 五、当前配额与时间
- 配额 **22/50**（今日），次日 00:00 重置
- 天梯赛截止 **2026-11-30 17:00 北京**（约 2.7 天）
- 进行中：`ref-fa`（读官方 FA）、`ref-qmsq`（读官方 quant_batch_matmul_v3，已出详细摘要）、`cube-mm2`（bpt 用 `TransDataTo5HD`）

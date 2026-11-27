# FlashAttention v4 (cube) — 2026-11-28 深夜续修记录（未通过，v3 仍为交付）

> 结论先说：**S=QK^T 的行错乱未修好**（本轮又排掉了 6 种假设，并把根因范围收窄到“cube 的 A 行 / M 维本身”），
> 按时间盒**停手**。线上/服务器已回滚为 **v3（15/15 Pass，score 10.30，zip md5 97f80d28… 未动）**。本轮**未提交任何改**。

## 0. 机器状态（已回滚，安全）
- `/root/work/npu_kernel/ladder/zips/flash_attention_7files.zip` md5 = `97f80d28e8b3a93d8df5a259017bae5e`（未改动）
- `bash /root/work/build_op.sh flash_attention` → BUILD OK / INSTALL OK（从原 zip 重建 v3 并安装到 custom vendor）
- v3 实测 3/3 PASS（1×2×8×16 f16 causal、1×1×256×64 f32 causal、1×1×1024×72 f32 causal）

## 1. 本轮新修的两个真 bug（保留，价值高）
### 1.1 workspace 子缓冲区把「大小」当「偏移」用 → S/P/O 完全重叠
`Init()` 里：
```
sGm_.SetGlobalBuffer(ws + core*perCore + t->blkS, ...)   // blkS 是 S 区大小
pGm_.SetGlobalBuffer(ws + core*perCore + t->blkP, ...)   // blkP 是 P 区大小
oGm_.SetGlobalBuffer(ws + core*perCore + t->blkO, ...)
```
而 host 里 `blkS=AlignUp(s1Base*S*4)`, `blkP=AlignUp(s1Base*S*tsize)`, `blkO=AlignUp(s1Base*D*4)`, `perCore=blkS+blkP+blkO`。
fp32 时 blkS==blkP==blkO → 三块**基址相同、完全重叠**，S/P/O 互相覆盖（这是“精度全错”的一个独立真 bug）。
修法（已改）：偏移改累计量——S 在 +0，P 在 +blkS，O 在 +blkS+blkP。
> 注意：这个 bug **不解释** S 的行重复（debug 只用到 sGm_ 仍错），但它必须修，否则连 S 之外也对不了。

### 1.2 空闲/半截核 End() 死等（dbg 分支）
原 `if (didWork){ mm1.End(); mm2.End(); }`：若某条路径只跑过 mm1 没跑 mm2（如只做 mm1 的调试分支），
`mm2.End()` 会 KFC 死等（实测把 910B2 打进 D 状态、`npu-smi` 仍显示 OK 但后续 launch 全挂）。
修法（已改）：`didM1/didM2` 分别守卫 `mm1.End()/mm2.End()`。

## 2. 对「S 行重复」的定论级新证据（关键，推翻上一轮猜测）
### 2.1 不是“大 N 单 matmul”的问题
- S=16/32/64 的单 tile（`baseM=baseN=baseK=64`，usedCore=1，**根本没有 N 分块**）照样错；
- 逐元素的 A/B 编码实验（q[i,:]=i 常数、k[j]=e0 ⇒ 真值 S[i][j]=i）：
  - **B(K) 列映射 = 完美恒等**（`got[0][j]=j` 全部 j）；
  - **A(Q) 行映射** = 错乱 `[9,10,10,12,12,14,14,16,16,...]`。
- 最终输出（过完 softmax+PV）逐行比对 golden：**out 行 i 恰等于 golden 行 P(i)，64/64 行都能精确对上，列全对**，
  P = `[9,10,10,12,12,14,14,16,16,18,18,20,20,22,22,24,...]` ≈ `8 + 2*ceil(i/2)`，越界后饱和到 S-1。
  ⇒ **输出矩阵是 golden 的“纯 M 行置换”（每两行取同一真值行、步进2）**，N/列完全正确。
  ⇒ 这是 **M 维（A 矩阵行 / L0A 装载）** 的错误，**bmm1 和 bmm2 共享同一行错乱**（所以最终 out 也是这个 P）。

### 2.2 这个行错乱对以下全部“免疫”（逐一实测，结果逐位相同/几乎相同）
| 变量 | 试过的取值 | 结果 |
|---|---|---|
| MatmulConfig | `GetNormalConfig(true)` / `GetNormalConfig()` / `GetMDLConfig()` / `GetNormalConfig(false,false,true)`(isVecND2NZ) | 全错，P 基本不变 |
| IterateAll | 同步 `IterateAll(gm,false)` / 异步 `IterateAll<false>+WaitIterateAll` / `enSequentialWrite=true` | 全错（**异步+Wait 会 hang，勿用**） |
| SetOrgShape | 5 参 `(s1Real,s2,D,D,s2)` / 官方式 `(s1Real,D,D,D,s2)` | 全错 |
| tail/shape | `SetSingleShape(s1,s2,D)` / `SetTail(s1,s2)` | 全错 |
| 分块/切分 | baseN=16、baseM=16、depthA1=2/depthB1=2 | 全错 |
| workspace 偏移修复 | 已修 | 行错乱形状不变（只影响 S 之外） |

### 2.3 tiling 本身很“干净”，不像 tiling 问题
`[FA_TIL mm1] M=64 N=64 Ka=64 Kb=64 scM=64 scN=64 scK=64 baseM=64 baseN=64 baseK=64 depthA1=1/3 depthB1=1 stepM=1 stepN=1 usedCore=1 transLen=0 AL[B=0 S=0 N=0 G=0 D=0] BL[全0]`
（单 tile、无切分、core=1）。仅 `transLen=0`、`ALayoutInfo*/BLayoutInfo*=0` 值得怀疑（转置 B 的 L1 转置工作区/布局信息可能没被正确设置），但未知其含义。

## 3. 下一轮最短路径（建议，按性价比）
1. **先做“最小独立 matmul 复现”**：把 `matmul_skeleton`（int8 已验证 bit-exact）原样改成 **A=fp16/fp32 ND、B 转置、C=fp32** 的 [64,64]=[64,64]×[64,64]ᵀ，看是否仍行错乱。
   - 若错 → 是「fp32/fp16 + transB 的 ND A 装载」在某 tiling 下的库/驱动行为，转去查 `ALayoutInfo*`/`transLength` 语义或换 `MatmulType` 的 LayoutMode。
   - 若对 → 说明是 FA kernel 的环境（MIX_AIC_1_2、blockDim、UB/TPipe、KFC 注册）干扰，逐个剥离。
2. **S^T 互换验证**：改算 `Sᵀ = K·Qᵀ`（A=K、B=Qᵀ），若错乱从“行”跑到“列/另一维”，即可定位是 A-load 还是别的。官方算的是 S（A=Q,B=Kᵀ），照官方最稳。
3. **对齐官方 host tiling**：官方 bmm1 `SetOrgShape(s1RealSize, mm1Kb=D, mm1Ka=D, mm1Kb=D, s2RealSize)` + `SetTail(s1RealSize,s2RealSize)`；关键是其 host tiling 的字段（transLength、layout info、depth 等）由完整 FA tiling 产出，我们的简版可能缺字段。
4. 之后才是「按 s2 分块（N=128/256）」的流式 + P121 CV 三握手流水（性能向）。
5. 仍建议：**不要碰 v3**；cube 版精度不过就不打包、不提交。

## 4. 其他踩过的坑（都实测）
- `IterateAll<false>(...) + WaitIterateAll()` 在本 kernel（MIX_AIC_1_2）会 hang（官方 FA 能用是因为它的同步/流水结构不同）。**本工程用同步 `IterateAll(gm,false)` 即可，勿加 WaitIterateAll。**
- 用「kernel 里 SetValue 把 tiling 写进 out」的调试路径（dbg=2）曾把设备打进 D 状态；**tiling 直接用 host 侧 `fprintf(stderr)`（`FA_DBG=1` 时）最安全**——注意 driver 只有 rc!=0 才回显 stderr，需自行打印。
- 设备无 `npu-smi set -t reset`（容器内被禁）；若打进 D 状态只能等/换机。

## 5. 文件
- v4 WIP 源码：`/tmp/fa_v4/{op_kernel,op_host}`、`/root/work/npu_kernel/ladder/flash_attention_v4cube/`、本地 `fa_probe/v4/`
- 构建：`/tmp/build_v4.sh`（装到 custom vendor，**不覆盖原 zip**）
- 测试台：`/tmp/fatest/*`；本轮新增诊断：`diag_s.py`(S dump)、`diag_encode.py`(A行/B列编码)、`diag_out_perm.py`(最终输出行置换)、`diag_tiling.py`
- 本轮诊断脚本存根：本地 `fa_probe/sh/*.sh|*.py`

---

# §8 2026-11-28 深夜续修（第二次接手）：仍未通，按裁决暂停；v3 安全

> 状态：**cube 版暂停（lead 裁决）**。线上/服务器保持 **v3（15/15 Pass，score 10.30，zip md5 `97f80d28e8b3a93d8df5a259017bae5e` 未动）**；服务器已 `build_op.sh flash_attention` 重装 v3，实测 3/3 PASS。**未提交、未打 cube 包。**

## §8.1 ★ S 行重复的「定论级」证据
用 A/B 编码实验直接解码 cube 的映射（`q[i,:]=i` 常数行、`k[j]=e0` ⇒ 真值 `S[i][j]=q[i][0]=i`）：
- **B(K) 列映射 = 完美恒等**（`got[0][j]=j`，所有 j）；
- **A(Q) 行映射 = 错乱** `[9,10,10,12,12,14,14,16,16,18,18,20,20,22,22,24,…]`（≈`8+2·ceil(i/2)`，越界饱和到 S-1）。
- 最终输出（过完 softmax+PV）逐行比 golden：**out 行 i 精确等于 golden 行 P(i)，64/64 行都能对上、列全对**，P 同上 ≈`8+2·ceil(i/2)`。
- ⇒ 输出是 golden 的**纯 M 行置换**（每两行取同一真值行、步进 2）；**N/列完全正确**。即 **M 维（A 行 / L0A 装载）错**，bmm1 与 bmm2 共享同一行错乱（所以最终 out 也是这个 P）。
- 关键点：该现象在 **S=16/32/64 的单 tile**（`baseM=baseN=baseK=64`，usedCore=1，**根本没有 N 分块**）上照样复现 → **推翻上一轮"大 N 单 matmul 回写步长错乱"的猜测**。

## §8.2 「已排除清单」（对以下全部免疫：逐一编译实测，结果逐位近同 / 同一 P）
| 变量 | 试过值 | 结果 |
|---|---|---|
| MatmulConfig（4 种） | `GetNormalConfig(true)` / `GetNormalConfig()` / `GetMDLConfig()` / `GetNormalConfig(false,false,true)`(isVecND2NZ) | 全错，P 不变 |
| IterateAll（同步/异步/顺序） | 同步 `IterateAll(gm,false)` / 异步 `IterateAll<false>+WaitIterateAll` / `enSequentialWrite=true` | 全错（**异步+Wait 会 hang**） |
| SetOrgShape（5 参两种） | `(s1Real,s2,D,D,s2)` / 官方式 `(s1Real,D,D,D,s2)` | 全错 |
| tail / shape | `SetSingleShape(s1,s2,D)` / `SetTail(s1,s2)` | 全错 |
| 分块 / 切分 | `baseN=16`、`baseM=16`、`depthA1=2&depthB1=2` | 全错 |

tiling 本身"干净"：`M=N=K=64, baseM=baseN=baseK=64, 单tile, usedCore=1, stepM=stepN=1`；仅 `transLen=0`、`ALayoutInfo*/BLayoutInfo*=0`（转置 B 的 L1 转置工作区/布局信息可能未正确设置）值得怀疑。

## §8.3 本轮新修的 2 个「真 bug」（保留在 v4 WIP 源码）
1. **workspace 子缓冲区把「大小」当「偏移」用 → S/P/O 三块完全重叠**：
   `Init()` 里 `sGm_/pGm_/oGm_` 的基址分别用了 `t->blkS / t->blkP / t->blkO`，而 host 里这三个是**区间大小**（`perCore=blkS+blkP+blkO`）。fp32 时 `blkS==blkP==blkO` → 三块基址相同、完全重叠、互相覆盖（这是"精度全错"的一个独立真 bug）。
   修法：偏移改累计量 —— **S@0，P@blkS，O@blkS+blkP**。
2. **`End()` 死等（KFC 死锁）**：原 `if (didWork){ mm1.End(); mm2.End(); }`，当某分支只跑过 mm1（如只做 mm1 的调试路径）仍调用 `mm2.End()` → KFC 死等，实测把 910B2 打进 **D 态**（`npu-smi` 仍显示 OK，但后续 launch 全挂）。
   修法：用 `didM1 / didM2` 分别守卫 `mm1.End() / mm2.End()`。

## §8.4 血泪坑（务必记住）
- **`IterateAll<false>(...) + WaitIterateAll()` 在本 kernel（`KERNEL_TYPE_MIX_AIC_1_2`）会 hang**，勿用；用同步 `IterateAll(gm,false)` 即可（官方 FA 能用异步是因为其同步/流水结构不同）。
- **调试别用 kernel 侧 `SetValue` 把 tiling 写进 out**（会把设备打进 D 态）；**tiling 用 host 侧 `fprintf(stderr)`（`FA_DBG=1` 时）最安全**。注意 driver 只在 `rc!=0` 才回显 stderr，需自行打印。
- 容器内无 `npu-smi set -t reset`（被禁）；打进 D 态只能等/换机。

## §8.5 下一轮最短路径（若将来再投入）
1. **最小独立 matmul 复现**：把已验证 bit-exact 的 int8 `matmul_skeleton` 改成 **A=fp16/fp32 ND、B 转置、C=fp32 的 64³**，隔离是「dtype/transB 的 ND A 装载」还是「FA kernel 环境（MIX_AIC_1_2/blockDim/TPipe/KFC 注册）」。
2. **S^T 互换验证**：改算 `Sᵀ = K·Qᵀ`（A=K、B=Qᵀ），看错乱是否从"行"跑到"列/另一维" → 即可定位是否 A-load。
3. **对齐官方 host tiling**：对比官方 `detect_flash_attention_score_s1s2_bn2gs1.h` 的 bmm1 `SetOrgShape(s1RealSize,mm1Kb,mm1Ka,mm1Kb,s2RealSize)`+`SetTail`，重点补 `transLength` / `ALayoutInfo*` / `depth` 等字段（我们的简版 host tiling 可能缺）。
4. 之后才是「按 s2 分块（N=128/256）流式 + P121 CV 三握手流水」（性能向）。
5. **不碰 v3**；cube 精度不过就不打包、不提交。

---
（本轮完整诊断脚本存根：本地 `fa_probe/sh/`；v4 WIP 源码：`/root/work/npu_kernel/ladder/flash_attention_v4cube/`、`/tmp/fa_v4/`、`fa_probe/v4/`。）

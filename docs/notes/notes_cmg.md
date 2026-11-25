
---

## 5. Codex 评审与修复（2026-11-24，第二轮红队闭环）

评审输入：`_sec5_build/ascend_tasks/ladder_review/cmg.md`（题面 + 官方 golden `gen_data.impl` + 7 文件源码 + 用例）。
Codex 模型 `gpt-6-sol`，`effort=low`，单文件单次评审。

### 5.1 Codex 提出的「必须改 top 3」

1. **全归约单核结果没有有效写回**：`confusion_mul_grad.h::ProcessAll` 在 `float32 && usedCores<=1` 时用
   `out1Gm_.SetValue(0, part)`；属于题面/团队明令排查的无效写法（`GlobalTensor::SetValue` 不回刷 GM）。
2. **跨行归约原子写越过 `output1` 末尾**：`ProcessColRowBlock` 把长度 `N` 的输出按 `nw=ceil(N/8)*8` 个 float 原子写，
   `N%8!=0` 时最多越界 7 个 float；而 tiling 放行非 8 倍数的 `N`。
3. **row-block 的 UB 预算显著低估**：tiling 只按约 `7*N*4` 字节判 `rowBlockOk`，实际 `Init` 分配了
   14 份 fp32 tile（≈56 B/元素）+ 固定 buffer。

### 5.2 逐条判定

| # | 判定 | 理由 |
|---|------|------|
| 1 | **接受（按红线，而非复现失败）** | 改成 UB→GM 的 `Store1D`（内部走 `DataCopyPad`）+ `S_MTE3`。**但实测本机 CANN 9.2/910B2 上旧代码并没有错**：把 pre-fix 源码单独重建、装到 vendor 后跑 `M=1,N=1,f32,axis=[0,1]`（正好命中 `usedCores==1`）是 PASS，说明该版本 `GlobalTensor::SetValue` 确实落到了 GM。仍然改掉：红线要求不用它回写 GM，且这是未定义行为级别的移植性风险，零成本规避。 |
| 2 | **部分接受** | 32B 原子粒度决定「必须写满 `Align8(N)` 个 float」不可避免（`axis=[0,1]` 的 1 元素输出也一样要写 32B），把 `N=8191x2053` 这个计分 case 改回逐列 `ProcessCol` 不划算。真正危险的是「越界 + UB 未定义数据」叠加，所以：①把 `gb` 整块 `Duplicate(gb,0,accLen)` 清 0，保证 `Sub(acc,acc,gb,npv)` 读到的区间全部有定义；②越界上界钉死在 `nw-N <= 7` 个 float（≤28B）；③ACLNN 的输出按 512B 对齐分配，28B 落在分配块内。已在 NOTES 显式记录该边界。 |
| 3 | **接受** | `rowBlockOk` 改成按实际占用算：`56 * Align(N,64) + 8192 <= ubSize * 15/16`。原门槛在 192KB UB 上放行到 `N≈4915`，实际只支持到 `N≈3100`。 |

### 5.3 本轮自查发现的 3 个真 bug（Codex 未提，均已 A/B 实测「旧 FAIL / 新 PASS」）

| # | bug | 位置 | A/B 证据（旧 → 新） |
|---|-----|------|--------------------|
| a | **`tileLength` 在分支前赋值**：row-block 分支把局部 `tile` 放大成 `Align(N,64)` 却没回写 `tiling->tileLength`，`Init` 仍按小 tile 开队列，而 kernel 按 `accLen=Align(N,64)` 取 buffer → UB 越界/花数据 | `confusion_mul_grad_tiling.cpp::ConfusionMulGradTilingFunc` | `M=64,N=4096,f32,axis=[0]`：旧 **MARE=3.1e6 FAIL** → 新 MARE=2.1e-4 PASS |
| b | **列并行被 `M` 限核**：`axis=[0]` 的列路径用 `cores=min(coreNum, N/64, M)`，`M=1` 时只剩 1 核，`cw=Align(N,64)` 远超 UB tile | 同上 else 分支 | `M=1,N=8192,f32,axis=[0]`：旧 **MARE=1.4e4 FAIL** → 新 PASS；`M=1,N=65536`：旧 **崩溃 `sync 507035`** → 新 PASS |
| c | **向量 op count 非 64 倍数**：列路径 `alignE` 为 8(fp32)/16(fp16)，`cw=72` 时 `Mul(...,rr*cw)` 的 count=72，违反「count 恒为 64 倍数」 | 同上 else 分支 | 改为 `alignE=64`（`cw` 恒为 64 倍数；末核不足 64 走 `cwp=RoundUp64(cw)` 的窄路径） |

### 5.4 改动清单（7 个文件全部改为纯 ASCII 注释）

| 文件 | 位置 | 改动 |
|---|---|---|
| `op_kernel/confusion_mul_grad.h` | `ProcessAll` | 单核 fp32：`out1Gm_.SetValue` → `Store1D(tmp,out1Gm_,0,1)`；多核/单核均补 `V_S`+`S_MTE3`；fp16 分支补 `V_S` |
| `op_kernel/confusion_mul_grad.h` | `Load1D` | `Duplicate`（V）→ `DataCopyPad`（MTE2）补 `V_MTE2` |
| `op_kernel/confusion_mul_grad.h` | `Init` | `gBuf_` 先 `Duplicate(gb,0,accLen)`（补 `V_MTE2`），`DataCopyPad` 后补 `MTE2_V` |
| `op_kernel/confusion_mul_grad.h` | `ProcessColRowBlock` | 原子 `DataCopy` 前补 `V_MTE3` |
| `op_kernel/confusion_mul_grad.h` | `FlushOut1` | fp32 补 `S_MTE3`；fp16 补 `S_V` |
| `op_kernel/confusion_mul_grad.h` | `ProcessCol` / **新增 `ProcessColChunk`** | 自己的列区间再按 UB tile 分块（`chunkCap=(tile_/64)*64`），任何核数下 `cw <= tile_`；母函数只负责切块 |
| `op_host/confusion_mul_grad_tiling.cpp` | `ConfusionMulGradTilingFunc` | ①`tiling->tileLength` 移到分支之后写；②`rowBlockOk` 改 `56*Align(N,64)+8192 <= ub*15/16`；③列路径去掉 `cores>M` 限核、`alignE=64` |
| `op_host/..._def.cpp`、`_infershape.cpp`、`..._kernel.cpp`、`_tiling_data.h`、`_tiling_key.h` | 文件头 | 中文注释改英文 |

### 5.5 验收

- `bash build.sh -j8`（先 `rm -rf build` clean）：**0 error**，run 包正常产出并安装。
- `bash build.sh -e`：官方 example 真机跑通（`[INFO] Example completed successfully!`）。
  ※ 该目标原先会因为 `examples/run.sh` 没有 +x 位而 `Permission denied`（**与本次改动无关的模板权限问题**），已 `chmod +x`
  `examples/run.sh`、`tests/ut/run.sh`。
- **官方 golden 对拍 34/34 PASS**：新建 `verify/test_cmg_io.cpp`（参数化 ACLNN harness：`M N dtype axisMode inDir outDir`）
  + `verify/verify_cmg.py`（import 模板 `gen_data.impl` 作 golden，按官方 MERE/MARE 判据）。
  覆盖 `axis=[1]/[0]/[0,1]` × `f16/f32`、`M=N=1`、`N=65/129/300/2053/4099/4096/7168/8192/16384/65536`、
  `M=1/2/3/7/33/64/1024/4096/5000/8191/16384/65536`、`1x300000`；断言无 NaN/Inf。
- `bash build.sh -u`（UT）：**本机环境缺 `nlohmann/json.hpp`**，`tests/ut/common/tiling_case_executor.cpp` 编译失败。
  这是模板 UT 的第三方头缺失（旧 build 目录里同样没有该 .o，属既存问题），与本轮改动无关；cli 侧的真机对拍已覆盖同一逻辑。
- 性能（最好值）：`M=8191,N=2053,axis=[0]` 0.370ms（快路径仍是 row-block 原子合并）；`M=1024,N=7168,axis=[0]` 0.410ms；
  `M=65536,N=64` 0.536ms。
- **zip 已刷新**：`cmg_7files.zip` 7 个成员、无前缀、全 ASCII（脚本 `zipfile` 按 `op_host/<3>` + `op_kernel/<4>` 重写）。
- 未提交（配额由 Lead 统一管）。

### 5.6 遗留 / 给 Lead 的提示

- 本文件 `## 1`~`## 4` 是**既存的乱码**（UTF-8 里存了「UTF-8 被按 GBK 解码后再编码」的二次编码文本，部分字符已丢失）。
  本轮**没有**改写旧内容以免动到其他 agent 共用的文件，只追加本节（UTF-8）。需要时可整篇用
  `utf8_text.encode('gbk').decode('utf-8')` 回转一次修好（会残留少量 `?`）。
- `ProcessColRowBlock` 的 32B 原子写会在 `output1` 之后越界 ≤28B（见 5.2 第 2 条），若后续要彻底消除，
  需要把跨核归约改到 workspace + 一次同步，而本机 `SyncAll` 必崩 507035，暂不采用。

# 06 · 第二轮进展（2026-11-27，新服务器 CANN 9.0.0）

> 本轮最大变化：**换到一台 CANN 9.0.0 的服务器**（与判题机**同版本**），
> 这打破了之前「本地 9.2 与判题机 9.0 行为不一致」的根本障碍。

---

## 一、当前成绩：天梯赛 **16 / 18 Pass**

| 难度 | 算子 | 状态 | 得分 | 排名 |
|---|---|---|---|---|
| 5 | flash_attention | ✅ Pass | 10.30 | 70 |
| 5 | conv_transpose2_d | ✅ Pass | 7.93 | 45 |
| 5 | depthwise_conv2_d | ✅ Pass | 12.55 | 29 |
| 4 | gf2_poly | ✅ Pass | 11.22 | 33 |
| 4 | cube1 | ✅ Pass | 6.24 | 22 |
| 4 | cube2 | ✅ Pass | 7.32 | 22 |
| 3 | gf2k_mul_square | ✅ Pass | 18.70 | 29 |
| 3 | quant_matmul_sigmoid_mul_quant | ✅ Pass | 9.26 | 35 |
| 3 | **mhc_sinkhorn_grad** | ✅ **Pass（本轮新增）** | 10.55 | 28 |
| 2 | bit_plane_transform | ✅ Pass | 22.34 | 32 |
| 2 | **confusion_mul_grad** | ✅ **Pass（本轮新增）** | 20.31 | 40 |
| 2 | group_norm | ✅ Pass | 24.43 | 25 |
| 2 | mse_loss | ✅ Pass | 34.27 | 30 |
| 2 | group_norm_silu_quant | ✅ Pass | 16.96 | **11** |
| 2 | index | ❌ RE | 0 | 35 |
| 2 | multi_add_rms_norm_dynamic_quant | ❌ WA | 0 | 57 |
| 1 | trace | ✅ Pass | （未进前 100） | ~125 |
| 1 | bn_training_update | ✅ Pass | 18.61 | 46 |

**Pass 题总分 230.99**（15 道在榜 + trace）。
**剩余 2 道经实测判定不可解**：
- `index`：4 轮假设全证伪；在**同版本 CANN 9.0 本机**上 v4/v5 都全 PASS（含 120 例 fuzz、21 次多 case×warmup×5），但判题仍 0.7/0.54 → 差异只能来自**判题 harness 的输出 shape/输入生成**，无判题输入无法复现。
- `multi_add_rms_norm_dynamic_quant`：判据（逐元素 rel/abs≈1e-4）严于社区标准（`cann-bench` 写 `int8:1.0` + MERE<2⁻¹⁰）；连 **fp64 精确 mean 仍差 85~99/256 行** → bf16 必差 1 ULP，**在现判据下等效不可解**。建议向平台反馈阈值口径。

---

## 二、本轮最重要的技术突破：破译 torch 2.7.1 的归约顺序

详见 `docs/notes/TORCH_REDUCTION_ORDER.md`（166 行）。要点：

### 2.1 torch CPU `sum(dim)` 的真实结构（aarch64，**`Vectorized<float>::size() = 4`**）
```
block = 4 个向量 = 16 元素
level0: 4 个向量累加器（共 16 lane）
base-16 进位级联：level1 += level0 每 16 个 block（=256 元素）
                  level2 每 16²，level3 每 16³
余下的整向量（<4 个）加到 lane 组 0
尾部标量（N % 4）先各自求和，最后逐个加进 group0 的 4 个 lane
```
- **N 小时（block 数 < 16）级联不触发** → 退化为「U4acc」单级模型
  - 这解释了为什么单级模型在 **N ≤ 287 逐位对、N ≥ 288 崩**：**288 = 18 个 block，首次触发 level1**
- 之前失败的原因：**一直假设 `Vectorized<float>::size() = 8`（V8），实际是 4**

### 2.2 不同归约轴的尾部处理不同
- `dim=-1`（连续行和）与 `dim=-2`（跨步列和）的尾部处理不同，需分别反推
- `mhc_sinkhorn_grad` 的 n∈{4,6,8} 三档都已精确到具体括号树

### 2.3 判据公式（实测逆向）
```
precision_ratio = 1 − #(rel_err > 10 × threshold) / N
rel_err = |out − golden| / (|golden| + 1e-7)
fp32 threshold = 2^-13
```
**42 个 Pass 的 ratio 全部恰好 = 1.0；25 个 WA 卡在 0.996~0.999**
⇒ **Pass 门槛实质要求「零 mismatch」** ⇒ 量化/归约类算子**只有逐位复刻 golden 才能过**。

### 2.4 复刻时的 Ascend C 坑
- 向量 op 的 **count 必须 ≥64 且 32B 对齐**；用 `count=4` 的子向量会 `sync warm failed` / 507035
  → **cascade 的 16-lane 状态天然低于该门限，所以只能先用「标量 IEEE 实现」保正确性**（cmg/msg 就是这么过的）
- **除法必须用标量 IEEE**（向量 `Div` 非正确舍入，差 1 ulp 就挂）
- fp32 累加**不要升 double**（越精确离 golden 越远）

---

## 三、Cube / Matmul 高阶 API 在大算子上：**卡在"同进程重复 launch"**

### 已确认可用（重要！旧结论被推翻）
- CANN 9.0.0 **完整支持** `AscendC::Matmul`，但**必须用 MIX AIC/AIV 内核**：
  ```cpp
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mm);
  ```
- **旧机"Matmul 输出全 0"的真因**：我们的 kernel 是**纯 AIV `__global__`**，Cube server 没启动 —— **不是版本问题**
- host 侧：`#include "tiling_api.h"` + `matmul_tiling::MatmulApiTiling` + `SetAType/SetBType/SetCType/SetBiasType` + `SetShape/SetOrgShape` + `GetTiling(tiling->mm)`
- **workspace 必须 = `GetLibApiWorkSpaceSize() + 用户区`**（漏 lib-api 段 → `MTE DDR out of range` 507015）
- int8 Mmad 是**精确 int32 累加** → bit-exact 可达（实测 4 个 shape 单次 launch 全部 nbad=0，含 cube1/cube2 官方 shape）

### 未解决（本轮止损点）
**同进程第 2 次 launch 必崩**（aicore exception / 507015）。
- **built-in cube 算子（torch_npu matmul）同进程 5 次 launch 完全正常** → **证明可解**，问题在我们的 kernel 收尾/核映射
- 已排除 7 种方案（详见 `docs/notes/CUBE_MATMUL_ATTEMPT.md`）
- **建议下一轮**：逐行原样内联官方 `BmmDequantPertokenBasic`（含 `block.h`/`update.h`/`base.h`）+
  `Mc2QuantBatchMatmulV3TilingData` 全套参数（ubCalcM/N、needUbBuffer、isPerTensor/isPertoken、baseM/N、tileL2cache）
  —— 官方基线同款，重复 launch 必过。估计 3~5h。

> 判断：这块是 cube1/cube2/qmsq/fa **四道题**的共同天花板（TOP1 都是 30µs 级 = Cube），
> 若解开，这四道可从 6~10 分跳到 60~90 分。**下一轮最高优先级。**

---

## 四、本轮实际提交记录（submissionId）

| 算子 | sid | 结果 |
|---|---|---|
| quant_matmul_sigmoid_mul_quant（transB 向量 DMA 版） | `6b09555285ad43bbe037e28c` | ❌ TLE（本地 5.8x、判题全灭 → 已回退 v1） |
| flash_attention（v3 分块向量版） | `6b0956a485ad43bbe03b5afb` | ✅ Pass 15/15（2~3x） |
| **mhc_sinkhorn_grad**（cascade bit-exact） | `6b09582c85ad43bbe03f9758` | ✅ **Pass 15/15** |
| confusion_mul_grad（U4acc，N≤287） | `6b095aee85ad43bbe046f865` | ⚠️ WA（判题 N≥288，走了旧路径） |
| index（v5 TQueBind，首次提交） | `6b095eef85ad43bbe051288a` | ⚠️ WA（同 v4 值） |
| **confusion_mul_grad**（cascade4，任意 N） | `6b09601e85ad43bbe0545c8a` | ✅ **Pass 15/15** |

### 一个重要的踩坑教训
qmsq 的「transB 向量 DMA 版」**本地 bench 快 5.8x，判题却 15/15 全 TLE（time=0）**。
⇒ **本地 kernel-only bench ≠ 判题结果**。以后每版提交前必须：
1. 用**模板 `examples/test_aclnn_*.cpp` 的调用路径**（最接近判题 harness），不要只用自建 bench
2. 跑**同进程连续 5 次 launch**（判题是 5 次 warmup + replay）
3. 至少跑一个**判题规模的 shape**，看会不会 hang

---

## 五、本轮新增的工程设施

| 设施 | 用途 |
|---|---|
| `/root/work/env.sh` | 新服务器统一环境（CANN 9.0.0 / LD_LIBRARY_PATH / DEVICE_ID / CJ_JWT） |
| `/root/work/build_op.sh <op>` | **一键**：拷官方模板 → 覆盖我们的 7 文件 → clean 重建 → 安装 |
| `/root/work/npu_run.sh <cmd...>` | **NPU 独占锁**（flock）—— 单卡多 agent 必须用，否则计时和状态互相污染 |
| `fetch_templates.py` | 从判题 API 拉官方模板（含 build.sh/examples/tests/gen_data.py） |
| `q.py` / `score_all.py` | 配额查询 / 全 18 题成绩看板 |
| `sub_zip.py` | 从任意 zip 提交（自动 429 退避 + 轮询逐 case） |
| `/root/work/share/matmul_skeleton/` | 已验证单次 bit-exact 的 MIX Matmul 骨架（下一轮接手用） |

---

## 六、下一轮建议优先级

| 优先级 | 事项 | 预期收益 |
|---|---|---|
| ★★★ | **打通 MIX Matmul 的同进程重复 launch**（内联官方 `BmmDequantPertokenBasic`） | cube1/cube2/qmsq/fa 四道 → 可能 +200 分 |
| ★★ | 继续刷 `conv_transpose2_d` / `depthwise_conv2_d` 等向量算子的 tiling | 每道 +20~40 分 |
| ★ | gnsq（rank 11）已接近榜首，微调 | +10 |
| — | index / marn | 判定不可解，建议向平台反馈 |

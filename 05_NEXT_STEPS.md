# 05 · 未过的 6 道题：卡点分析与建议

> 这 6 道都已攻到「只差最后一步」，但受限于 golden 不可复现或时间，暂时止损。
> **每一条都写清了「已排除什么」**，接手的人不用重复劳动。

---

## 天梯赛（4 道）

### 1. `confusion_mul_grad`（难度 2，40万）— WA，15 个 case 里错 3 个
- **现状**：`cascade V8` 版，坏元素 **1（case8/2047）+ 12（case11/12289）+ 2（case13/1129）= 15 个**
- **已排除（6 档模型）**：cascade V8 / V16、old-kernel V8（4 累加器+配对合并+尾部在后+无级联）、
  NPU 顺序（torch_npu 实测，2-D 时 NPU 换形状相关 tiling，`(1024,1024)` 差 720/1024 行）、
  rowblock-off（关原子加后精度**一模一样** → 该 case 不走原子路径）、Kahan
- **结论**：判题机 golden 既非本机 torch 也非 NPU 顺序，**固定顺序模型已穷举完**
- **若续做**：先读 `tests/ut/op_kernel/confusion_mul_grad_data/gen_data.py` 确认是 `np.sum` 还是 `torch.sum`
  - 若是 **numpy** → 用 `03_GOLDEN_ORDER.md` 第一节的 numpy 规则重写（这是**最可能一击命中**的方向，因为
    `group_norm_silu_quant` 就是靠它做到逐位的）
  - 若是 torch → 需要判题机同版本的 `SumKernel.cpp` 源码，黑盒猜不出来
- 现成 5 个变体包都在本地 `_it6/zips/`，可直接 A/B

### 2. `multi_add_rms_norm_dynamic_quant`（难度 2，40万）— WA，5 个 case 错
- **现状**：**可见 6 个 case 的 `precision_ratio` 全是 1.0**，错的全在**隐藏 case**（idx 10-14）
- **已证明上限**：把 `mean(x²)` 换成 fp64 精确值后再比，**仍差 42 个 bf16 元素**
  （rstd 只有 660/1024 行逐位相同）→ **任何实现都无法与 torch 的 fp32 行归约逐位一致**
- **关键旁证**：官方生态标准 `cann-bench/.../multi_add_rms_norm_dynamic_quant/proto.yaml`
  对同一算子写的是 `precision_thresholds: int8: 1.0`（允许 ±1 LSB）+ 浮点 MERE<2⁻¹⁰/2⁻⁷。
  **按那个口径我们是 15/15** → 判题对 bf16 用了比社区标准更严的逐元素阈值
- **建议**：**向平台反馈阈值口径**（附 `exact-mean 仍差 42 元素` 的复现脚本），不要继续投入

### 3. `index`（难度 2，40万，老题）— RE→WA，9 个 case 全错
- **进展**：`clamp` 后 **507035 已消除**（证明崩的原因是**索引越界读**）；canary 实验证明
  **写到达了被判的 y**（→ 排除管道问题）
- **已排除 4 轮假设**：数据通路（TQue / TQueBind / local copy 三版 prec **一字不差**）、
  语义（同下标 vs §3.3 扁平解码，两种都 ≈随机）、写不到 y（canary 排除）、
  `indexed_sizes` 是"尺寸"而非 0/1 掩码（v6 结果与 v4 一致 → **就是 0/1 掩码**）
- **当前最强线索**：case0/1 是**单被索引维**（两种语义等价）也错，且精度恒为 `0.7 / 0.54`
  —— 对 bool 随机数据 0.5 = 完全随机，说明 **输出整体错**（不是 30% 元素错）
- **建议下一步**：用「输出下标」技巧（`02_PITFALLS.md` F2）——把算出的源下标直接写进输出，
  一次提交就能读出**我们算的是哪个下标**，与 golden 期望的下标一对比就知道错在哪
- 现成包：`_it1_work/index_7files_v5.zip`（RE 已修）、`v6.zip`（mask 语义版）、`canary.zip`

### 4. `mhc_sinkhorn_grad`（难度 3，60万）— WA，8 个 case 错
- **现状**：已做到本机 15/15 逐位相等（n=4/6/8 各自的归约顺序都反推出来了，见 `03_GOLDEN_ORDER.md` 第四节）
- **判题仍 WA**，且已排除 8 族归约模型 + n=8 的 5×5 全顺序组合
- **决定性的矛盾指纹**：case12（n=8，68→7，比值 **0.103**）与 case14（n=8，841→523，**0.622**）
  **同一个 n 上差 6 倍** —— 任何固定顺序的 golden 都不可能，说明是**依赖规模的两级分块**
- **建议**：与 cmg 同源问题。若 cmg 的 numpy 假设成立，这道也应先查 `gen_data.py` 是否是 numpy

---

## mfg（2 道）

### 5. `spspmm`（稀疏×稀疏矩阵乘）— WA，`precision_ratio` 7 次提交**完全不变**
- **`0.857177734375`**，既是 `3511/4096` 也是 `7022/8192`（两种分母都精确 → 分不出是 valueC 还是 indexC）
- **两个 canary 都无效（决定性负面信号）**：
  - `CANARY_ROW0_PAD`（`indexC[0]` 整片写 −1）→ 比值**不变**（若生效，上限只有 0.5）
  - `CANARY_VALC_ZERO`（`valueC` 整片保持 0）→ 比值**仍不变**（若生效，应为 0.1428 或 0）
  - 两者 kernel md5 均已确认落地、case0 耗时也从 14.0ms 变到 20.8ms（新代码确实跑了）
  → **`precision_ratio` 对我们 kernel 的输出完全不敏感**
- **结论**：该 case 的比值来自我们从未触及的路径（tiling / 别的输出 / 框架侧比较），
  或统计口径与我们的模型都不符
- **有价值的旁证**：公开 ranking 里有队伍做到 **`10/8192`（几乎全对）** → 这件事**可达**，不是 golden 不可复现
- **若续做**：需要 `result[0].msg` 的真机日志或 case0 的输入数据，否则无法继续

### 6. `radius`（已放弃）
- 5 次提交，precision 卡在 `0.9360416666666667`
- 已证明全局索引正确（改本地索引反而更差 0.90792），官方 `gen_data.py` 无法获取（404/403）
- **建议**：不值得再投入

---

## 天梯赛还有一道**没做**的

### `GateGateUpAct` — **Kirin 平台，跳过**
题面要求 Kirin（手机 NPU）平台，本机 910B 无法评测，因此未做。若手上有 Kirin 环境可考虑。

---

## 如果只剩有限时间，优先级建议

| 优先级 | 题 | 理由 |
|---|---|---|
| ★★★ | `confusion_mul_grad` | 先查 `gen_data.py` 是否 numpy；若是，用已破译的 numpy 规则**很可能一击命中**（gnsq 已验证该路径可行） |
| ★★ | `index` | 用「输出下标」技巧 1 次提交就能定位；老题 40 万 |
| ★ | `mhc_sinkhorn_grad` | 与 cmg 同源；若 cmg 通了，这道大概率跟着通 |
| — | `multi_add_rms_norm_dynamic_quant` | 建议**向平台反馈阈值口径**，不投入 |
| — | `spspmm` / `radius` | 需要额外信息（真机日志 / 输入数据 / gen_data），不建议盲试 |

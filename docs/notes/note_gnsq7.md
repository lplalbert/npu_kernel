
## 7. exp 精度收尾（2026-11-25，目标 15/15）

**判题 11/15 的 4 个失败 case 各差 1~10 个 int8 元素**，本轮把剩余差异清零。

### 根因链（逐层隔离得到）
1. **统计量**：已按 numpy 的 8192 分块 + Parker pairwise 复刻（§6），但 `var` 仍差 1 ulp →
   原因是 C++ 里 `r[j] += d*d` 被编译器**融合成 FMA**（少一次舍入）。改为
   `volatile float sq = d*d; r[j] = r[j] + sq;`（三处）后 **mean/rstd 与 numpy 逐位相同（max|d|=0）**。
   这一条单独就消掉 4~9 个 flip。
2. **Exp**：设备向量 `Exp` 与 `np.exp` 约 10% 元素差 1~2 ulp → 大 case 剩 2~3 flip。
   改为「chunk 内筛出距半整数 < 1e-6·|q| 的极少数元素（约 0.0026%），这些元素用
   **双字(double-float, 两个 fp32) 正确舍入的 exp** 重算后覆盖」（aicore 禁 double，
   故用 TwoSum/TwoProd/Fast2Sum；host 上用 numpy 逐位验证：75000 个随机输入只差 52 个 = 0.069%，
   正好等于 numpy 自身的 1ulp 率）。
3. **一个隐蔽的缓冲复用 bug**：筛选用的临时缓冲最初复用了 `tb`（= 缓存的 quantScale），
   而 quantScale 每个 channel 只 Duplicate 一次、跨多个 chunk 复用 → 从第 2 个 chunk 起 qs 被污染，
   表现为 **67M 元素的大 case 有 72.3%（=3/4）元素错**。改为每 chunk 重新 Duplicate(qsb) 后归零。

### 现在的结果（本机 910B2 / CANN 9.2）
- `test_gnsq.py all`：**13/13 全 PASS**，两个 16.7M case `int8_bad=0`；
- `iso_flips.py`（独立数据，qs=0.01 更敏感）：16.7M 与 **67M** 两个 case 都是
  `device vs golden flips=0`、`mean/rstd max|d-np| = 0.000e+00`；
- 性能：16.7M 从 0.19ms → **10.0ms**（标量 pairwise + volatile 保护 + 边界重算的代价，
  判题 case 总时间被 harness 开销主导，仍远低于 3s 同步超时）。
- zip：`gnsq_7files.zip` 已重打包（7 文件、无前缀、md5 一致、11801B）。

### 已知坑（写进下一班）
- aicore **禁用 double**（`cast to/from double ... not allowed`）；`sqrtf` 不存在，只能用 `sqrt()`。
- `volatile float` 是防 FMA 融合/防编译器合并的有效手段（本次两处都靠它）。
- **临时缓冲不要复用仍在使用的缓冲**（本次是 tb=qsb），跨 chunk 复用的量务必每 chunk 重算。
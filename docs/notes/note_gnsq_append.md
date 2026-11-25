
## 6. 判题反馈与 numpy 归约顺序逆向（2026-11-25，pid 6ad9c36f3ff2fef7e81c55cc）

**判题结果**（submission 6b05c76285ad43bbe0773885）：Wrong Answer，pass=11/15，score=0（**必须 15/15 才有分**）。
4 个 WA case 的 precision_ratio = 0.99999237 / 0.99999449 / 0.99999046 / 0.99999970，
即每个 case 只有 **1~10 个 int8 元素差 1**（131072 / ~1M / 33.5M 元素量级）→ 判题要求 int8 **逐位相等**。

**关键发现（本次最大产出）：numpy ndarray.mean/var 的归约顺序（已在 host 上逐位验证）**
numpy 1.26 对 C 连续数组的多轴归约 = **按 8192 个元素分块（NPY_BUFSIZE），每块内部用 Parker pairwise
求和，块间把结果顺序累加（fp32 标量）**；块内 pairwise 规则：
- n < 8：顺序累加（初值 0）；
- 8 <= n <= 128（PW_BLOCKSIZE）：8 个交错累加器 r[j]（r[j] 初值取 a[j]，按 i+=8 累加），
  合并 `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`，再顺序加尾部；
- n > 128：`n2 = n/2; n2 -= n2%8;` 递归 `pw(a,n2)+pw(a+n2,n-n2)`。
验证：13 种 shape（105 / 8191 / 8192 / 8193 / 16385 / 32768 / 131072 …）× 8 seed，
mean 与 var（两趟：先 mean，再 sum((x-mean)^2)）**全部 bit-exact**；fp16 输入先 cast 到 fp32 同样成立。
（注意：**不能**用按行 pairwise / 全局 8 累加 / 顺序求和 / Kahan，全部会差 1ulp。）

**改法（只改 op_kernel/group_norm_silu_quant.h）**：
- 新增 `PwLeaf`（单叶，8 累加器，标量 UB 取值）+ `PwSum`（显式栈的迭代后序，深度上限 64）+
  `PwSumChunked`（8192 分块 + 块间顺序累加）；
- pass1：`sum = PwSumChunked(x)`，`mean = sum / gsz`（向量 Div，numpy 是除法不是乘倒数）；
- pass2：`var = PwSumChunked((x-mean)^2) / gsz`（两趟，元素内联在叶循环里算 (x-mean)^2，与 numpy 的
  「先整数组算平方再 pairwise」等价）；`rstd = Div(1, Sqrt(var+eps))` 全走向量 op；
- pass3 不变（仍严格照 golden 的逐 op 顺序：(x-mean)*rstd*gamma+beta、1+exp(-y)、y/(...)、/qs、rint）。

**效果（本机 910B2 / CANN 9.2）**：meanOut/rstdOut 与 numpy **逐位相同**；两个 16.7M 大 case 的 int8 差异
从 8 / 21 降到 **2 / 3**；其余 11 个自建 case 仍全 PASS。

**剩下的差距（尚未解决，必须解决才能 15/15）**：剩余 flips 全部来自 **`Exp` 的 1~2 ulp**
（隔离实验 iso_flips：device vs host(喂入 device 的 mean/rstd) 仍有 6/17 处差异，其余 11/17 已归零）。
做法建议：实现「正确舍入」的 fp32 exp（判题 golden 是 `np.exp`，其自身与真值差 ~0.06% 的 1ulp，
只要我们的 exp 正确舍入，残余 flips 期望 ~0.01/16.7M）：
`k=rint(x*log2e)`、`rh=x-k*ln2_hi`（Sterbenz 精确）、`rl=-k*ln2_lo`、
`exp(r)=1+u, u=r+r^2*P(r)`（P 用 minimax/泰勒到 r^8，**补偿求和**：TwoProduct 算 r^2 的低位、Horner 用双字、
最后 Fast2Sum 把 (1+u) 的进位余项 e 加回去），再乘 `2^k`（整数移位构造）。约 40~60 条向量 op/chunk。

**性能**：受标量逐元素求和影响，16.7M case 从 0.19ms → 3.9ms（判题 case kernel 实测只 4~5us、总时间被 harness
开销主导，故可接受；若要提速，把叶内 8 累加器改成 32B 向量 `Add(acc, acc, v[i], 8)` 即可回到 ~0.2ms 量级）。

**状态**：`gnsq_7files.zip` 已重打包（7 文件、无前缀、md5 一致），但 **13 个自建 case 仍有 2 个差 2~3 个 int8**
→ **暂不建议提交**（提交仍是 WA、且占额度）。
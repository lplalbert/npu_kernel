# torch 2.7.1 CPU `sum(dim)` 逐位归约顺序（aarch64, Vec::size()=4）— 破解与复刻

> 适用：CANN 天梯赛判题 golden 用 torch 2.7.1 CPU 生成，精度判据要求**逐元素**接近，
> 近零（相消）元素实际需 **bit-exact**。本文给出可 bit-exact 复刻 `sum(dim)` 的模型。
> 实战成果：`mhc_sinkhorn_grad`(msg) 与 `confusion_mul_grad`(cmg) 均借此 **Pass 15/15**。
> 环境实测：新机 `223.111.149.177`，aarch64，`torch 2.7.1+cpu`，CANN 9.0.0。

---

## 0. TL;DR

- **`Vectorized<float>::size() == 4`**（aarch64 NEON，128-bit），不是 AVX 的 8/16。
- `sum(dim)` 对「连续的被归约轴」用 **`cascade_sum`** 结构（`aten/src/ATen/native/cpu/SumKernel.cpp` 思路）：
  block = `4*Vec` = **16 元素**；4 个向量累加器（level0，16 lane）；**base-16 进位级联**；
  余整向量并到 lane 组0；**尾部标量先求和**；最后 group0 的四条 lane 逐个相加。
- **临界点 N=288**：当 block 数 < 16 时级联不触发 ⇒ 退化为「U4acc」单级模型；N=288 = 18 个
  block 首次触发 level1 进位 ⇒ 与 U4acc 分道扬镳。这是 N≤287 用 U4acc 对、N≥288 崩的根本原因。
- `dim=-1`（连续 last-dim）与 `dim=-2`（跨步）**尾部(<Vec)的处理不同**：前者左折叠、后者顺序累加。
- **除法**（若算子里有 `x/sum`）必须用**标量 IEEE 除法**；向量 `Div` 非正确舍入，害精度。
- 判据：`precision_ratio = 1 − #(rel_err > 10·2⁻¹³)/N`，`rel_err=|out−g|/(|g|+1e-7)`；
  42 个 Pass 队全部 ratio==1.0 ⇒ 门槛实质要求**零 mismatch**（近零元素必须 bit-exact）。

---

## 1. 单级模型 `U4acc`（对 N 较小、<16 个 16-block 时）

把一行/一列向量 x[0..N-1] 归约为标量（**逐元素用 fp32 舍入，不能用 double 累加**）：

```
V = 4 ; W = 4*V = 16          # block 大小
T = N / V                     # 向量(4元素块)个数
# 4 个 lane 累加器 A[0..3][0..V-1]（每个是一个 Vec4）
g = 0
while g + 4 <= T:             # 每 4 个“向量块”一组，分给 A0..A3
    for j in 0..3:
        for l in 0..V-1: A[j][l] += x[(g+j)*V + l]
    g += 4
for t in g..T-1:              # 剩余整向量块 -> 全部进 A0（lane-wise）
    for l in 0..V-1: A[0][l] += x[t*V + l]
# 每 lane 顺序合并 4 个累加器
s[l] = ((A[0][l]+A[1][l])+A[2][l])+A[3][l]
# 尾部不足 V 个元素 (r = N - T*V)
#   连续轴(dim=-1)：左折叠后加到 lane0
#   跨步轴(dim=-2)：逐元素顺序加到 lane0
r = N - T*V
if r > 0:
    if dim == -1:  t = fold(x[T*V .. N-1]); s[0] += t
    else:          for i in T*V..N-1: s[0] += x[i]
out = ((s[0]+s[1])+s[2])+s[3]
```

要点：
- **"剩余整向量块全进 A0"** 不是 round-robin（这是最初踩错的关键）。N 为 4 的倍数时等价于
  「元素 i → lane i%4，横向 seq」，即俗称 rr4；非倍数时靠尾部规则区分。
- 若把上面 4 个累加器写成「每 4 个向量一组」的 4-lane 累加，其实等价于单遍 `Vec4` 累加器。
- U4acc **只在 N ≤ 287 与 torch 一致**（实测 N=5..287 全 neq=0；N=4,<4 退化为纯 seq）。

## 2. 通用模型 `cascade_sum`（任意 N，含 N≥288）

把 N 切成 block=16 元素（4 个 Vec4）。四个层 `level0..level3` 各 16 lane（= 4 个 Vec4）：

```
V=4; kBlk=4*V=16; nvec=N/V; nblocks=nvec/4; tailStart=nblocks*kBlk
# 级联步长：lp = max(4, ceil_log2(nblocks)/4); step0 = 1<<lp
#   nblocks<16 -> lp=4 -> step0=16 (每 16 block 进位)
step0 = 1 << max(4, ceil_log2(nblocks)//4)
level0..level3 = 0 (各 16 lane)
i = 0
for b in 0..nblocks-1:                  # 逐 block
    for j in 0..3, l in 0..3: level0[j*4+l] += x[b*16 + j*4 + l]
    i += 1
    if i % step0 == 0:
        level1 += level0; level0 = 0
        if i % step0^2 == 0:
            level2 += level1; level1 = 0
            if i % step0^3 == 0:
                level3 += level2; level2 = 0
# 合并 4 层
L[k] = ((level0[k]+level1[k])+level2[k])+level3[k]     # k=0..15
# 余下的“整向量”(从 tailStart 到 nvec*V，步 V) 全部加到 lane 组0（前 V 条 lane）
for a in tailStart .. nvec*V step V:
    for l in 0..V-1: L[l] += x[a + l]
# group0 = 四组 lane 逐条顺序相加：grp[l] = ((L[l]+L[l+V])+L[l+2V])+L[l+3V]
for l in 0..V-1: grp[l] = ((L[l]+L[l+V])+L[l+2V])+L[l+3V]
# 尾部标量 (N%V 个) 先求和（升序），再把 group0 的 V 条 lane 逐个加进去
fin = fold(x[nvec*V .. N-1])            # 空则 0
for l in 0..V-1: fin += grp[l]
out = fin
```

- **N blocks < 16 时**：level1/2/3 分支从不触发 ⇒ 等价 U4acc（这就是 N≤287 两模型一致的原因）。
- **N=288**：nblocks=18，第 16 个 block 时首次 `level1 += level0` ⇒ 与 U4acc 产生差异。
- 已用 fp32 严格模拟验证：cascade(V=4) 对 **N=4…4096 全部 bit-exact**；U4acc 仅 ≤287。
- 参考实现：`/root/work/probe_c7.py`（cascade_row）、`probe_c3.py`/`probe_c5.py`（外围分块排除）、
  `probe14.py`/`probe16.py`（U4acc 与临界点）、`probe_msg2.py`（n=4/6/8 穷举树）。

## 3. 不同归约轴 / 不同维的差异（msg 实测，n∈{4,6,8}）

msg 是 (T,n,n)，对 `dim=-1`(last-dim 连续) 与 `dim=-2`(跨步) 做行/列和。穷举全部无序二叉归约树得：

| n | `sum(dim=-1)`（行和，连续） | `sum(dim=-2)`（列和，跨步） |
|---|---|---|
| 4 | 纯 seq `(((0+1)+2)+3)` | 纯 seq |
| 6 | `(((0+(4+5))+1)+2)+3`（尾部**折叠**到 lane0） | `((((0+4)+5)+1)+2)+3`（尾部**顺序**加到 lane0） |
| 8 | `(((0+4)+(1+5))+(2+6))+(3+7)`（=rr4） | 同 rr4 |

- 即：**同一 n，连续轴的尾部(<V)是「先折叠再并入 lane0」，跨步轴是「逐个顺序并入 lane0」**；
  4 的整数倍时两者都退化为 rr4；n=6 最能区分。
- msg 的 n 恒 ∈{4,6,8}，故以上 6 条即完整精确顺序，可 kernel 内按 n 分支硬编码。

## 4. 复刻到 Ascend C 时的关键坑

1. **向量 op 的 count 有下限/对齐约束**：`Add/Duplicate` 等若 count 很小（如 4/16 且偏移 16B）
   会触发 vector core exception（`sync warm failed`、507035）。把 cascade 直接按 4-lane 向量化会踩。
   - 安全做法 A：**全部用标量实现**（C++ `float` 累加即 IEEE fp32，与 torch 逐位一致）——cmg/msg
     最终都是标量版，正确性最高，代价是慢。
   - 安全做法 B（提速方向）：一次处理 **16 个 block（=256 元素=64 lane×4）**，用 **count=64/128/256**
     的整块向量 op，再在 lane 层做级联合并（务必逐位验证）。
2. **除法必须标量 IEEE**：msg 里 `(g-dot)/sum`，向量 `Div` 非正确舍入 → 改成标量除法后 neq 才归零。
   cmg 的 axis=[1] 只有乘+和，没有除法。
3. **fp32 累加**：模拟/验证脚本里若用 Python `float`（double）累加会得出错误模型；必须每步
   `np.float32(a+b)`。这是早期误判「找不到模型」的元凶之一。
4. **`Vec::size()` 一定是 4**（aarch64），别照搬 x86 的 8/16，也别信旧内核的 `kV=8`。
5. **线程无关**：同一 (M,N) 的 sum 结果与 `torch.set_num_threads` 无关（实测 1 vs 192 逐位相同），
   故可本地单机反推并移植。但注意 M=1 走另一条串行路径（与判题 M 大无关）。
6. cmg **axis=[0] / [0,1]**：输出量级大、非近零，任何合理顺序都在相对误差容差内 → 无需逐位；
   保持原实现即可，别去动。

## 5. 判据公式（cannjudge precision_ratio）

- 逐元素：`rel_err = |out − golden| / (|golden| + 1e-7)`；fp32 阈值 `T = 2⁻¹³ ≈ 1.22e-4`。
- `precision_ratio = 1 − count(rel_err > 10·T) / N`（实测与观测比值逐位吻合，如 250/251、
  12281/12289、1128/1129）。
- 93 个提交里 **42 个 Pass 全部 ratio==1.0**；WA 队卡在 0.996~0.999 ⇒ 门槛实质要求**零 mismatch**，
  即近零(相消)元素必须 bit-exact（这也是「提高精度/用 fp64」反而更容易挂的原因：fp64 有自己
  的舍入，近零处与 torch fp32 golden 仍差 1e-5 量级，rel 爆掉）。
- 另可参考开源 grader `cann-bench/src/kernel_eval/utils/compare.py`（MERE/MARE + small/cancel 兜底，
  但实测 cannjudge 更严，等价于要求 ratio==1.0）。

## 6. 复现脚本 / 入口（本机 /root/work）

反推：
- `probe_c7.py`  — cascade(V=4) vs U4acc 全区间对照（**最核心**，证明通用模型）
- `probe14.py`   — U4acc 大范围 + thread 无关性
- `probe16.py`   — U4acc 临界点（N=287→288）
- `probe_msg2.py`— msg n=4/6/8 穷举精确树
- `probe_c3.py`/`probe_c5.py` — 外围 chunk 模型排除（注意其中标量曾用 double，仅作排除参考）
- `probe_judge*.py`, `q7.py`/`q8.py`/`q9.py` — 判据分析与排行榜 precision_ratio 统计

真机验证（模板 aclnn 驱动 + torch golden，`--repeat 5` 同进程多次 launch）：
- cmg：`_ladder_cmg/out/test/test_aclnn_cmg.cpp` → 编译为 `ws/confusion_mul_grad/examples/test_cmg`；
  `cmg_verify.py <M> <N> <mode> <keep> <seed>`（mode 0=[1],1=[0],2=[0,1]）
- msg：`run_msg.cpp`(自建泛型驱动) → `ws/mhc_sinkhorn_grad/examples/run_msg`；
  `msg_verify.py <T> <n> <I> <mode> <seed>`
- 编译命令模板见 `/root/work/build_tcmg.sh`、`/root/work/build_run.sh`
- **NPU 独占锁**：`bash /root/work/npu_run.sh bash /tmp/your_test.sh`（flock，多 agent 共享必用）

构建/打包：`bash /root/work/build_op.sh <op>`（从 `zips/<op>_7files.zip` overlay 到 `tpl/<op>` 并
clean 构建安装）；打包 7 文件用 `/root/work/mkzip.py <src dir> <zip>`；提交前 `scan_badwords.py`
（禁词 cout/cin/cerr/printf…）。

## 7. 已产出的包

- msg：`/root/work/npu_kernel/ladder/zips/mhc_sinkhorn_grad_7files.zip` — **Pass 15/15**
- cmg：`/root/work/npu_kernel/ladder/zips/confusion_mul_grad_7files.zip` — **Pass 15/15**
  （核心改动：`ProcessRowU4`(N≤287) + `ProcessRowCascade4`(N>287, 标量级联)；axis=[0]/[0,1] 未动）

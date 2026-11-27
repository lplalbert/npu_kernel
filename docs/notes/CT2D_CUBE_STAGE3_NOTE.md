# CT2D Cube 阶段3（06:xx）：病根已定位——不是 C offset，是 **M<baseM 的 M 未补齐**

## 关键结论（推翻阶段2"Matmul C 写回不可用"的说法）
1. **C offset / orgKa / orgKc / orgN 都不是病根**（改 5 参 orgKc、改 orgN=nsp 均无变化）。
2. **真病根：GEMM 的 M 维 < baseM（16）时 C 写回被污染**。
   - 实测：ntaps=1、Win=32、CIN 任意：**COUT=8 → okCols=8/32（错）；COUT=16/32/64 → 32/32 全对**。
   - 即 `coPer(=min(chOut,16))=8 < baseM(16)` 时错，=16 时对。
3. **修复（已在本目录实施）**：
   - host: `coPer = 16`（不再 `min(chOut,16)`），coBlocks=ceil(chOut/16)。
   - kernel: `SetOrgShape(coutPad_, nsp, coutPad_, hinWin_, nsp)` + **`SetSingleShape(coPad_=coPerPad, nsp, chIn)`**（M 补齐到 baseM），A 读 padding 列（wc 已补零），C 多写的行被 fold 丢弃（只读 cLoc<coCur）。
   - 效果：`Win=32 3x3` ratio **0.038 → 0.796**；`Win=64` 0.022→0.895；fp32 同样 0.79。**这是最大的一块，已解决**。

## 残余 bug（未解决，ratio 0.80 未达 1.0）
- 逐列分析（Win=32,C=8,K=3x3,out 34×34）：**只有 ow=1..7 这 7 列错**，ow=0 与 ow≥8 全对（maxdiff~0.3–0.8，非 near-zero，是结构性）。
- 7 列 = 第一个 8-lane 块内的 1..7（lane0 对）。强烈指向 **fold 侧 8-lane 对齐 / Gather byte-offset 表**（idxT 的 `aStart=(sOff/8)*8; extra=sOff-aStart` 与 `(extra+(j-j0))` 在 j∈[1,7] 的偏移），或 DataCopyPad 从 P 行 `aStart` 起的 32B 对齐窗口首块。
- 下一步建议（按优先级）：
  1. 把 `aStart` 强制 32B（8 float）对齐后，**把 fp 预加载 8 个对齐前导元素的 P 值**并在 idxT 里用绝对 `(j-j0)+sOff-aStart`…逐 j 打印 fp 与期望 P 对比（最小复现：ntaps=1,Win=32,只 7 列错）。
  2. 或直接 `mm.GetTensorC()` 取 C 到 UB 自己写 P（绕开 copy-out，布局完全可控）。
  3. 或每 tap 只算 N=64 段（status 建议2）。

## 其他确认
- **Cube 路径只在 `win%16==0` 启用**；Win=8/24 走 vector（一直全对）→ 阶段2 的"Win8 对/Win24 对"其实是 vector 路，不是 cube 证据。
- fp16 判据 9.766e-3、fp32 1.221e-3；残余 7 列误差 0.3–0.8 远超阈值 → 必须先修这 7 列。

## 工程位置
- 修改落在 `/root/work/ct2d_cube_src/`（原始备份 `/root/work/ct2d_cube_src_bak/`）。
- 构建：inline（tpl/conv_transpose2_d + overlay op_host/op_kernel → build.sh → 装 vendor）。
- 探针：`/root/work/bench_ct2d/bin/bench_ct2d N CIN HIN WIN COUT KH KW S P OP DIL DT reps prefix` + torch 对比。
- **线上仍 7.93（未提交 cube，未改线上包）。**

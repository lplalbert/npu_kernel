
## 9. 判题反馈与 RE 修复（2026-11-25，pid 6ad9c4023ff2fef7e81ca27e）

**判题结果（submission 6b05c30785ad43bbe06c71c3）**：Runtime Error，pass=0/9。
逐 case：case0 prec=0.70、case1 prec=0.54（Wrong Answer，float32 输出）、case2 Runtime Error（其余 Skipped）。
case2 的原始日志显示 `aclrtSynchronizeStreamWithTimeout(..., 3000) failed, ret=507035`，
507035 = `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（vector core exception）；profiling 汇总里该 case BlockDim=40，
其余 case BlockDim ∈ {2,13} → 由 tiling 反推 outElems = 1~2K / 12~13K / >39936，**判题 9 个 case 都很小**，
kernel 实测 4~5us（上报的 4.69ms/4.12ms 是 harness 固定开销），所以不是性能问题。

**根因（Codex 红队评审定位，与本机现象一致）**：三条路径共用的 `DataCopy(outLocal, inLocal, count)`
（LocalTensor→LocalTensor，VECIN→VECOUT）在 CANN 9.0 不是受支持的本地拷贝路径；纯搬运算子应改用
`TQueBind<VECIN,VECOUT>` 复用同一块 UB 直接 GM→UB→GM。这同时解释了 RE（vector core exception）与
两个 case 的大比例错值（输出槽内容错）。本地 CANN 9.2 容忍该路径，所以 32 例全过、判题机挂。

**改法（只改 op_kernel/index.h）**：
- `TQue<VECIN,2> inQ + TQue<VECOUT,2> outQ` → `TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> ioQ`；
- 三条路径（ProcessFast / ProcessContig2D / CopyRow）全部删掉 local→local DataCopy，改成
  `AllocTensor → DataCopyPad(GM→UB) → EnQue/DeQue → DataCopyPad(UB→GM) → EnQue/DeQue → FreeTensor`；
- kernel 内现在**只有 DataCopyPad（MTE2/MTE3）**，没有任何向量指令；地址/分核/32B 扇区对齐逻辑未改。

**验收（本机 CANN 9.2 / 910B2）**：
- `rm -rf build build_out` clean 重建通过，install OK；
- `bench_index.py`：官方 gen_data golden + 32 例自建用例（14 组 dtype、空张量、高秩拒绝、非对齐多核、
  确定性双跑）**37 项全 PASS / ALL_PASS**；
- 随机差分 fuzz（`mytest/fuzz2.py`，rank1~6、7 种 dtype、随机 mask/indices、73 例有效）**0 fail**；
- 性能无回归（略好）：gather1d_f32 0.212→0.167ms，contig_lastdim_f32 4.21→3.28ms，gather_rows_f16 0.0127→0.0134ms。
- zip：`index_7files.zip` 已重打包（7 文件、无前缀、zip 内 md5 与源码逐一一致）。

**未验证**：判题机 CANN 9.0.0 上的实际效果（本 agent 未提交）。
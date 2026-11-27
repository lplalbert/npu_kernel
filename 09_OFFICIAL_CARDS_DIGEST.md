# 09 · 追加金矿（2026-11-27 深夜，来自 ref-fa / ref-qmsq / cube-mm2 的官方卡片研读）

## A. 会让人白干几小时的坑

### A1. **build 缓存/安装坑**（ref-qmsq 实证）
`rm -rf build` **不够**！旧装到 `$ASCEND_HOME_PATH/opp/vendors/<op>_custom` 的 kernel 不一定被替换，
改了代码跑出来还是旧 binary → 假诊断（我们曾因此误以为“46× 提速”）。
**可靠流程**：
```bash
rm -rf build build_out && bash build.sh -j8
rm -rf $ASCEND_HOME_PATH/opp/vendors/<op>_custom
bash build/custom_opp_*.run --quiet
# 用 ws 大小签名等验证装的是哪一版（v1 host 公式 ws = sysWs + M*N*4 + M*N*2）
```

### A2. **同一 `op_kernel/*.cpp` 里放多个 `extern "C" __global__`**（官方 failure_pattern `f2`）
→ **只会注册一个，其余静默输出全零**，无编译错、rc=0。多套 tiling 必须用 tilingKey 分派，**不要拆成多 kernel 同文件**。

### A3. **`f4` L0A/L0B 事件窗口冲突** —— 解释了「同进程第 2 次 launch 崩」
`MTE1_M / M_MTE1` 只保证**发射顺序**，不保证 L1/L0 的读写窗口。
blockDim=24 且大 shape 时，**0~2 次迭代内随机崩（errcode 0x8000004000），单次运行不现**。
**解**：`WaitFlag<M_MTE1>` 之后、`copy_l1_to_l0` 之前插 `PipeBarrier<PIPE_ALL>`（代价约 1.49×）；
或跨调用续接槽位 + 隔步 PIPE_ALL。

### A4. ★★ **`GetUserWorkspace` 在 `Init` 里调用两次 → 这是「第 2 次 launch 崩」的最终真因**
（ref-fa 实证）entry 已取过用户区，`Init` 里**再调一次** → 所有 workspace 缓冲区**整体后移 ~16MB**
→ 第 2 次 launch `aicore exception 0x26` / 507057。
**这不是 KFC 状态问题！之前 perf-cube 试的 7 种方案都白费，就是因为没意识到这条。**
**对策：`GetUserWorkspace` 全 kernel 只调用一次并缓存指针。**

### A6. **`aclnn + aclrtSynchronizeStream` 有 ~870~880 µs 固定底噪**（ref-fa 实测）
小 shape 的性能会被底噪完全淹没（`1×2×8×16` 与 `1×1×1024×72` 的 TOTAL_US 底噪一样）。
**测小算子性能必须用 kernel-only（msprof）或与判题同口径**，否则会误判"没提速"。

### A7. **`SoftmaxFlashV2` 的 sum/max 输出布局不是 `row*8`**
CANN 源码用 `Brcb(...)` 按 `tiling.reduceSize` 分组写。
按 `row*8` 读会拿到 `S^r` 这类怪值（表现为 out 呈几何级数）。
→ **稳妥做法：改用原始 `ReduceMax/Adds/Exp/ReduceSum/Muls` 做逐行 fp32 softmax + P 先归一化再送 PV**。

### A8. **Matmul 写 C（GM/ND）时行寻址错乱的一个高发原因**
把 bmm1 做成「一次性 N=S（可达 4096）的大 N 单 matmul」时，L0C 只有 128KB，
单次 N 远超 L0C 会导致库内部分块回写的 **ND 步长按 `baseN` 而非 `singleCoreN`** → 多行映射到同一源（值对、行错）。
→ **按 s2 分块（N=128/256，与官方一致），C 用 `orgKc=S` + 基址偏移写子块**。

### A5. 空闲核无条件 `mm.End()` → KFC 死等（hang）

---

## B. FlashAttention cube 化配方（官方 P120/P119/P121/P117）

### B1. P120 `fa_vec_dot_to_cube_mmad` 四条硬配方
1. **Q/K/V 保持 fp16/bf16 直接进 mmad，不要先 Cast fp32**（fp32 只出现在 L0C 的 S、softmax 状态、O 累加）
2. **K 用 ColumnMajor 视图 + `Nd2NzParams` 一次 DMA 成 nZ 进 L1**，L1→L0B `ifTranspose=false`，免显式 transpose
   （若用 `BType(..., isTranspose=true)`，`SetOrgShape` 的 **orgKb 必须是 D 而不是 K**）
3. **Q tile 常驻 L1**，KV 循环只搬 K/V
4. **P 回写输入 dtype 再喂第二个 mmad**：fp16 用 `CAST_NONE`、bf16 用 `CAST_RINT`（**不是 CAST_ROUND**）；O 在 L0C/GM 用 fp32 累加，收尾一次 Cast
- **判定线：矩阵维 < 64 不要上 cube，退回向量**（启动开销不划算）

### B2. P119 online softmax
两遍 softmax 会让 K 读两遍且两遍间无流水。online 化后 K/V 各读一遍，
`QK^T(cube) → softmax(vector) → PV(cube)` 在同一 S2 循环内首尾相接。
rescale 顺序必须 `acc = acc*alpha + p@V`（alpha = exp(m_old − m_new)，m_old 初值 -INF）；
全 mask 行 `l==0` → 输出 0。

### B3. P121 SplitFuse CV 流水（性能天花板）
- AIC 跑 `QK[i] → PV[i-PRE]`，AIV 跑 `softmax[i] → rescaleO[i]`
- **GM workspace 多槽环（SLOTS = PRE_LAUNCH+1，FIA 取 3）+ `CrossCoreSetFlag<0x2,PIPE_FIX>` 三握手 flag（qkReady/softmaxReady/pvReady）**
- **禁用 SyncAll**（会退化成串行，收益归零）
- **PRE_LAUNCH 预发射**：循环上界 `kvEnd + preKVNum`；初始化时把所有 HardEvent 预置 SetFlag 暖机
- 坑：槽数 < PRE_LAUNCH+1 → 数据竞争（偶发错）；**AIV 侧 coreIdx 必须除 `GetSubBlockNum()`**

### B4. P117 双 BMM 链 host 配方
- `bmm1/bmm2` 各建一个 `MatmulApiTiling`（**禁止共用一份 TCubeTiling**）
- `SetBufferSpace(L1, L0C)` 显式给容量（失败要 fallback 缩 tile）
- `d ∈ (64,128]` 时 `SetFixSplit(128,128, alignUp(d,16))`
- `baseK >= singleCoreK && depthA1 == 2` 时 `depthA1 → 3`
- tiling 里 `shareMode=0 / shareL1Size / shareL0CSize` 必填
- bmm1 的 B 必须 `transpose=true`（Q@K^T）

### B5. F5 性能陷阱
CV 分离 FA 的 ring 单迭代成本 ≈ **每核 GM 字节 / 55~80GB/s**（拥挤窗口；安静窗口可达 145GB/s）。
**加深 ring / 放大 KV tile / 前移搬运点都无效** —— 只有「减少 ws_s/ws_p 往返字节」或「跨 q-block 复用 K/V」才是真杠杆。
→ 我们「S/P 各过一次 GM」的设计，字节账 4B(S)+4B(P)+4B(K/V)=12B/score 已是这类结构下限。

### B6. `SoftmaxFlashV2` 的输出布局坑
其 sum/max 是按 `Brcb` + `tiling.reduceSize` 分组写，**不是简单 `row*8` 步长**。
→ 若读错，l/m 全错。**稳妥做法：弃用 SoftmaxFlashV2，用原始 `ReduceMax`/`ReduceSum` 做逐行 fp32 softmax**。

---

## C. Matmul 多核 / L2 优化（官方 P104/P105/P107）

- **P104 多核切分**：`MatmulApiTiling::GetTiling` 常给 `singleCoreM ≈ M` → `totalBlock = 1`（**单核！**）。
  → **host 侧手动缩小 `singleCoreM/N`**（≥ baseM/N、16B 对齐）使 `totalBlock ≥ aicCoreNum`，
  重算 `mTotalCnt/nTotalCnt/usedCoreNum` 与尾块；kernel 加 `if (GetBlockIdx() >= totalBlock) return;` + 尾块守卫。
- **P105 L2 tile**：大 matmul 加 L2 级 tiling（`mTileCntL2/nTileCntL2/mTileBlock/nTileBlock`，单 tile footprint ≤ L2/2）；
  `calOrder`：M≫N → `ROW_FIRST` 复用 A，N≫M → `COL_FIRST` 复用 B；配合 P107 蛇形遍历。
  解决 B 被反复挤出 L2 → MTE2 重读。
- **P117 host 配方**：`SetBufferSpace(L1,L0C)` 打满多缓冲；`SetFixSplit(baseM,baseN,baseK)`；
  `SetOrgShape` 的行 stride 必须按真实 layout（**k≥65536 的 orgKb 16bit 溢出**正是这条）。

---

## D. 量化 matmul（官方 P102/A8）

- **P102**：用标量/向量拼 matmul 是"原罪"，向量化收益封顶；
  唯一数量级收益 = **架构级 MIX CV 融合重写**（Cube 做 Mmad，Vector 做 elementwise，GM workspace 交接）。
  四阶段不可跳：AIC_ONLY 单 matmul 验证 → MIX 骨架 → 配对探针 → 逐 stage。
  **"hang/NaN ≠ 平台不支持"**，下结论前要满足四条证据标准。
- **多核向量版也能拿 43~130×**（qmsq 实测，非转置），是低成本方案；transB 需「一次性转置 x2 → 非转置快路」。
- **A8 量化精度**：`Maxs/Mins` 精确卡 `[quantMin, quantMax]`；
  `Cast` 用 `CAST_RINT`（四舍五入）/`CAST_ROUND`（最近偶）**分阶段选**。

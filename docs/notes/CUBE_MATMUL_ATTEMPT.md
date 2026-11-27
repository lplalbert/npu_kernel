# CUBE Matmul 攻坚笔记（QuantBatchMatMulV3 / cube1 / cube2）

> 2026-11-28，cube1 从 10.13 → **25.63**（rank 18→11），cube2 从 12.24 → **30.45**（rank 16→12）。
> 本笔记记录可直接复用的核心认知，供 ct2d/fa 等同类 GEMM 复用。

## 0. 结论速查（最重要）
1. **`MatmulApiTiling` 默认给单核** → 必须 host 手缩小 `singleCoreM/N` 或自建 grid，kernel 用 `GetBlockIdx()` 分块。
2. **`orgKa/orgKb` 是 A/B 的「物理行 stride（leading dim）」，不是 K 长度**：
   - A(m,k) 非转置 → orgKa 必须 = 完整 k；A(k,m) 转置 → orgKa = m。
   - 传 65536 会溢出 16bit → 0 → 结果全错（nbad 巨大）。
   - **K 分块时仍传「完整 k」**（除非你先 DMA 到紧凑区，此时才传块长）。
3. **`GetMDLConfig(true)` 的 intrinsicsCheck**：当 orgKa/orgKb **≥65535** 时强制走 **loop-based GM→L1 慢路径**（实测慢 ~10-20×）。k=65535 就触发。→ 默认用 **`GetMDLConfig()`**（无 loop）；只有「物理 leading dim >65535 且没 stage」时才需要 true 作为正确回退。
4. **K 切核（split-K）** 是小 M/N 大 K 的正解：每核一段 K，A/B 各读一遍（避免按 M 切时每核重读整块 B），int32 分片用 **`IterateAll(c, enAtomic=true)` 原子累加**（int8→int32 精确、可交换）。
5. **N 切核** 是对 n>65535 的正解：每核一段 N，B 的 N-tile stage 到 stride=nPer 紧凑区，C 写每核紧凑区（orgKc=nPer），epilogue 写回真实 out。
6. **跨核 barrier 用 `AscendC::SyncAll<true>()`（AIV 版，可用）**；`SyncAll<false>()`（含 AIC）会 hang。

## 1. 多核组织
- **MIX 骨架**：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`，`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td->mm)`。
- **workspace**：`AscendC::GetUserWorkspace(workspace)` 全 kernel **只调一次**；user 区偏移 `sysWs + Σ块`（**别把"大小"当"偏移"**）。
- **单核算子分块**：每个 block 自己算 (m0,n0)，`SetTensorA/B` 传对应 GM 偏移；`IterateAll(c_[m0*n_+n0], k0>0)`。
- **tiling 的 usedCoreNum=1**（每个 block 一个 matmul 实例），blockDim = gridM*gridN 或 gridK。
- **空闲核 `mm.End()` 要 didWork 守卫**，否则 KFC 死等（本实现所有核都 End，OK）。

## 2. 四种 transpose 组合
- **编译期 template bool**（4 套 Matmul 类型），绝不标量 `GetValue` 读 B。
- A 偏移：AT? `kc0*m_` : `kc0`；B 偏移：BT? `kc0` : `kc0*n_`。
- `SetOrgShape(m,n, AT?m:k, BT?k:n, n)`（5 参，Ka/Kb/Kc 分别 A/B/C 的行 stride）。

## 3. compact-workspace（k>65535 或 n>65535）
- 溢出侧的操作数 K/N 切片 **DMA 到 stride=块长 的紧凑区**（GM→UB→GM，`DataCopyPad` blockCount=rows, blockLen=kc, srcStride=kfull-kc, dstStride=(pitch-kc)/32 等）。
- stage 后 orgKa/orgKb = 块长（**这次真的是块长**，因为内存紧凑）。stage 只是「读一次+写一次」，对大 M 会 ~2× A 流量，但仍远快于 loop 路径。
- **N 切核 + 紧凑 C**：per-nBlock C 临时区 m*nPer*4，epilogue 读紧凑 C（pitch=nPer）写回 out（pitch=n）。orgKc = nPer。

## 4. IO 估算选路径（dimsFit 时）
```
M/N 网格:  m*k*gridN + k*n*gridM + m*n*4
K 切核:    m*k + k*n + (gridK+2)*m*n*4   (含 C 清零+原子+epilogue 读)
取小者。大 m*n 时 C 流量爆炸 → 自然回退 M/N；中 m*n 大 k → K 切核收益大。
```
经验：纯 M 切比 N 切快（N 切每核重读全 B）；M 太小才切 N。

## 5. epilogue 向量化
- **整块 strided 拷贝**（rows×cols 一次 DataCopyPad）+ 逐行向量乘 scale(n)/pt(m)，同步次数 O(tiles) 而非 O(rows)。
- 逐行标量 `GM.GetValue` 很慢（~800ns），必须批到 UB 再标量读。
- 精度：CPU golden 用 int64 累加 + volatile fp32 两次 scale，与判题口径一致；`nbad=0` 是硬要求。

## 6. 判题口径
- fp16: MARE < 10×2^-10 = 9.766e-3；fp32: 1.221e-3。
- cube1/cube2 是 int8→int32 精确，必须 nbad=0（ratio=1.0）。
- "末维上限消除" = k 或 n >65535 必须通过且不能掉速（cube1 考点）。

## 7. 分支决策（最终 cube1）
- `dimsFit(k,n<=65535)`：
  - IO 估算选 K 切核 或 M/N 网格
- `k>65535 & m,n<=65535`：**stage A/B + K 切核**（compact-workspace）
- `n>65535 & k,m<=65535`：**N 切核 + 紧凑 C**
- 其余（n>65535 且 k>65535、m>65535 等）：MDL(true) 回退（正确、慢）

## 8. 实测（910B2，本地 contention 大仅看相对）
cube2 K-split: 256x256x65535 1.06→0.43ms（judge 3605→216µs，score 12.24→30.45）
cube1 stage: 8192x64x65536 11.28→5.8ms（judge 11099→554µs）
cube1 N 切核: n=65536 案例 judge case10/14 降 4.2×
cube1 最终 25.63；剩余 case9/11/13 是大稠密 GEMM（M/N 路径，ROI 低）。

## 9. 坑
- `MatmulConfig` 是编译期模板参数；要 per-shape 换 config 需 template bool 分派（CfgPick<SLOW>）。
- `DataCopyPad` UB 侧 stride 单位是 **32B**，GM 侧是 **字节**；blockLen<32B 会按 32B 槽摆。
- 窄 N / 窄 K 的 ND→NZ 装载退化是独立性能问题（padding 或 NZ 预排）。
- 原子累加对 int32 精确无顺序问题；浮点归约则不可（ct2d 需注意）。

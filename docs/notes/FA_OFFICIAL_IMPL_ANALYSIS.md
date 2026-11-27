# 官方 CANN FlashAttention ASCENDC 实现剖析（CANN 9.0.0 / 910B dav_c220）

源目录：`/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/detect_flash_attention_score/`
（7655 行 / 10 头文件 / 424KB，另有同源副本在 `ops_oam/ascendc/detect_flash_attention_score/`）
本地镜像：`fa_probe/official/`

---

## 0. 文件分工与入口

| 文件 | 行数 | 作用 |
|---|---|---|
| `detect_flash_attention_score.cpp` | 228 | **kernel 入口**（非模板 `extern "C" __global__ __aicore__`），按 TILING_KEY 分派实现类 |
| `..._common.h` | 195 | 枚举（ImplMode/Layout/Sparse/AttenMask…）、极值常量、DataCopy2D helper |
| `..._s1s2_bn2gs1.h` | 2233 | **主实现**：SplitS1S2 + BN2GS1 排布（fp16/bf16/fp32 高性能都走它） |
| `..._s1_bn2gs1.h` | 1928 | SplitS1 变体（s2 不分块，长 S1） |
| `..._bn2gs1s2_b.h` | 1570 | SplitBB/DB 变体（bmm2 输出 NZ） |
| `..._pse.h / _util.h / _dropmask.h / _drop_mask_adapter.h / _empty_tensor.h / _var_len_score.h` | | PSE(alibi)、工具、dropout、空张量、变长(TND) |

入口要点（`detect_flash_attention_score.cpp:169-228`）：

```cpp
extern "C" __global__ __aicore__ void detect_flash_attention_score(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *qStartIdx,
    __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    set_mask_norm();                      // ★ 掩码归一化（Select 的 mask 位宽归一），我们必须照抄
    ... TILING_KEY==90/92/94 → 空张量快速路径（只写 max/sum/out 的常量）...
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(10000001010220230943UL)) {   // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true,
                                  CubeFormat::ND, true);   // ← 最后 true = enableL1Reuse
        return;
    } else if (TILING_KEY_IS(10000000001021230099UL)) {  // SplitBbDBHighPerf → bn2gs1s2_b (bmm2 出 NZ)
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, AA_HIGH_PRECISION, LAYOUT_SBH,
                                         false, true, false, half, float, true, LayoutMode::SBNGD);
        return;
    }
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(10000000010220220943UL)) { FlashAttentionScoreS1s2Bn2gs1<..., bfloat16_t, float, true>; }
#endif
#if (ORIG_DTYPE_QUERY == DT_FLOAT)
    if (TILING_KEY_IS(10000000000220110943UL)) { FlashAttentionScoreS1s2Bn2gs1<..., LAYOUT_BSH, ..., float, float, true>; }
#endif
}
```

- **非模板 kernel 入口**；一个 op 里按 tilingKey 静态展开多个变体。
- **fp16/bf16：INPUT_T=half/bfloat16_t，T(计算类型)=float** → P 和 O 用 fp16 存、fp32 累加。
- **fp32：INPUT_T=float，T=float** → 全 fp32。
- `ImplModeEnum::AA_HIGH_PRECISION` = 0（另外有 AA_HIGH_PERFORMANCE=1、AA_INVALID_LINE_HIGH_PRECISION=2）。
- `set_mask_norm()` 在首行调用——这是 `SelectWithBytesMask` 使用前必须的 mask 归一（否则 -1e4/极值掩码行为不定）。**我们现有掩码是纯 Add 不做 Select，可不引入；一旦照抄 Select 路径必须加。**

### Cube vs Vector 两条编译分支
`#ifdef __DAV_C220_CUBE__` 时宏里用 `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling)` 并直接 `op.Init(...)/op.Process()`；**否则（AIV 编译单元）走 `GetUserWorkspace(workspace)` + Init/Process**。也就是说：**类本身是 MIX（AIC 与 AIV 各编译一遍），matmul 对象由 REGIST_MATMUL_OBJ 注册（KFC 自动跨 AIC/AIV 握手），我们不需要手写 NotifyEvent/WaitEvent。**

---

## 1. 是否用 Cube 做 QK^T 和 PV —— 是，且是唯一算力来源

`..._s1s2_bn2gs1.h:94-110` 定义**两个 Matmul 对象**：

```cpp
using a1Type   = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;               // A=Q  (GM, ND)
using b1Type   = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, enableL1Reuse>; // B=K^T (转置位=true)
using bias1Type= MatmulType<TPosition::GM, CubeFormat::ND, float>;
using c1Type   = MatmulType<TPosition::GM, CubeFormat::ND, T>;                    // C=S (GM, ND)
matmul::Matmul<a1Type,b1Type,c1Type,bias1Type, GetMmCfg(enableL1Reuse)> bmm1;
using c1NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
matmul::Matmul<a1Type,b1Type,c1NzType,bias1Type, GetMmCfg(enableL1Reuse)> bmm1Nz; // 仅在 s2%64!=0 时出 NZ 再转 ND

// bmm2 (PV)
using a2Type   = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;   // A=P (GM, ND, 已 cast 成 INPUT_T)
using b2Type   = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, ...>; // B=V (GM, ND, 不转置)
using c2Type   = MatmulType<TPosition::GM, CubeFormat::ND, T>;         // C=O 片 (GM, ND)
matmul::Matmul<a2Type,b2Type,c2Type,bias2Type, GetMmCfg(enableL1Reuse)> bmm2;
```

- **MatmulConfig：`CFG_EXCEED = GetNormalConfig(true)`；L1Reuse 变体 `CFG_IBSHARE_EXCEED = GetIBShareNormConfig(true)`。`true` = intrinsicsCheck（大 K/大内轴走循环 GM→L1 拷贝）。** 这与我们 cube1 用 `GetMDLConfig(true)` 是同一类开关。
- **QK^T**：A=Q `[s1RealSize, D]`，B=K `[s2RealSize, D]`，`SetTensorB(keyGm[off], true)`（第二参=true 表示 B 转置）→ C `[s1RealSize, s2RealSize]` fp32。
- **PV**：A=P `[s1RealSize, s2AlignedSize]`（fp16/bf16 或 fp32），B=V `[s2RealSize, D]`，不转置 → C `[s1RealSize, D]`。
- **形状/尾块 API（逐字照抄）**：
  - bmm1：`SetOrgShape(s1RealSize, mm1Kb, mm1Ka, mm1Kb, s2RealSize)`；`SetTail(s1RealSize, s2RealSize)`；`IterateAll<false>(mm1Res[taskIdMod2], false, false, true)`。
  - bmm2：`SetOrgShape(s1Size, mm2Kb, s2AlignedSize, mm2Kb, dSize)`；`SetTail(-1, -1, s2RealSize)`；`IterateAll<false>(mm2Res[taskIdMod2], false, false, true)`。
  - `IterateAll` 的 `<false>` = 不等到结束就返回（异步），靠 `WaitIterateAll()`+`End()` 同步 —— **这是它做深度 3 流水线的关键**。
- **mm1Ka/mm1Kb/mm2Kb** 由 layout 决定归约长度与外层 stride（BSH/SBH/BNSD/TND 各一套，见 `ComputeConstexpr` 507-530 行）。**我们的布局是 BNSD（q/k/v = (B,N,S,D)），对应 `LAYOUT_BNSD` 分支：`mm1Ka=mm1Kb=mm2Kb=dSize`，A 的行 stride=D，天然连续，最好办。**
- **offset 计算**：`Bmm1SetTensorA` / `SetBmm1TensorB` / `IterateBmm2`（1025-1101、1927-1977 行）给了完整 GM 偏移公式，BNSD 分支最简单：`bOffset=boIdx*n2GS1D, n2Offset=n2oIdx*gS1D, gOffset=goIdx*s1D, s1Offset=s1oIdx*s1BaseD`（A）；`bOffset=boIdx*n2S2D, n2Offset=n2oIdx*s2D, s2Offset=s2StartIdx*dSize + s2LoopCount*s2BaseNratioD`（B）。

---

## 2. online softmax 怎么做 —— **不在寄存器累加，而是"分块 + UB 里 rescale 残差"，O 部分常驻 GM/UB**

核心是官方 AscendC 高阶 API **`SoftmaxFlashV2`**（`..._s1s2_bn2gs1.h:1834-1870`）：

```cpp
// s2LoopCount==0：初始化（isUpdate=false）
SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(
        vec1S1RealSize, s2AlignedSize, sizeof(T), sizeof(T), apiTmpBuffer.GetSize()/sizeof(T),
        /*isUpdate=*/false, /*isBasicBlock=*/IsBasicBlockInSoftMax(...));
SoftmaxFlashV2<T, /*isUpdate*/false, /*isBasicBlock*/true, /*isDataScalar*/true, /*isReuseSource*/false, SOFTMAX_DEFAULT_CFG>(
        srcTensor /*S片, in-place 变 P*/, sumUb /*l, in/out*/, maxUb /*m, in/out*/,
        srcTensor, expUb /*★输出 rescale 因子 exp(m_old-m_new)*/, sumUb, maxUb, apiTmpBuffer, newTiling);
// s2LoopCount>0：isUpdate=true，用上轮 sumUb/maxUb 增量更新
SoftmaxFlashV2<T, true, true, ...>(...同上...);
```

**这套 API 的语义（照抄即可，不要自己写 m/l 累积）**：
- 输入 `srcTensor` = 本 s2 分块的 S 片（已乘 scale、已加 mask、已置极值）；
- 输入输出 `sumUb`(=l 向量，每行一个)、`maxUb`(=m 向量，每行一个)；
- **`isUpdate=false`**：用本片初始化 m/l，`expUb` 输出 `exp(x - m_new)`（等价于 P，且此时 rescale 因子应为 1）；
- **`isUpdate=true`**：`m_new = max(m_old, rowmax(片))`，`l_new = l_old*exp(m_old-m_new) + rowsum(exp(x-m_new))`，**`expUb` 输出 `exp(m_old - m_new)`（每行一个校正因子）**；
- `srcTensor` 被就地写成 P = `exp(x - m_new)`。

**O 的累积（`ProcessVec2` 1979-2066 + `Bmm2ResultMul` 2068-2097）**：
```
s2LoopCount==0 : O_ub = bmm2_片
否则          : O_ub = O_prev * expUb[row]  +  bmm2_片     // Mul(broadcast 每行因子) + Add
s2LoopCount==last: O = O_ub / l ; Cast → INPUT_T ; DataCopyPad 到 attentionOut GM
```
- 这就是标准 flash-attn 的 **"O 乘 exp(m_old−m_new) 再累加"**，只是 O 存在 **UB `stage2TBuf`（dSizeAlign16≤64 时）或 GM `vec2Res`（dSizeAlign16>64 时）**，不是寄存器。
- **P 一定要先 cast 回 INPUT_T 并写回 GM（`stage1Res`）**，因为 bmm2 的 A 只能从 GM 读（`MatmulType<GM,ND,INPUT_T>`）。所以每块 s2 的流程是 **GM→cube→GM→UB→vec→GM**，四次 GM 往返；靠 double buffer + 深度 3 流水掩盖。
- **全屏蔽行**：靠 `AdjustSoftMaxRes<T,T>(srcTensor, maxUb, negativeIntScalar, 0.0, shape)`（`InvalidLineSplitS2Process` 1770-1804）在**最后一块**把整行全被屏蔽的行的 P 强制清 0；并把 l/m 写成极端值使 `sum` 最终为 0 → 输出 0。这条只在 `implMode==AA_INVALID_LINE_HIGH_PRECISION` 打开；**高性能档（AA_HIGH_PRECISION）不做**，此时全屏蔽行输出 = NaN 风险由调用方/掩码保证（官方高性能档要求掩码不是全 −inf 行）。**我们要求"全屏蔽行输出 0"，所以要保留我们自己的 l>0 保护，或移植 AdjustSoftMaxRes 逻辑。**

### 与我们 v3 的对照
- 我们 v3：整行 S 落 UB → 一次 ReduceMax/Exp/ReduceSum（**单 pass、无 rescale**），天然 bit-stable；慢是因为 S² 次 `ReduceSum(D)` + S² 次 `Muls/Add(D)` 全在 Vector。
- 官方：S 分块 → 块内 `SoftmaxFlashV2`（Vector，一次处理 [vec1S1, s2Aligned]），累加靠 `expUb` rescale。**数值上多了一次 O×(m_old−m_new) 的重标定 → 与整行一次性 softmax 不完全 bit-identical，但数学等价、误差 ~fp32 eps 级。判题 allclose（fp16 1e-3 / fp32 1e-4）足够。** 若要更稳，可把 exp 因子在 fp32、O 全程 fp32。

---

## 3. UB / L1 分块参数怎么定（从 `InitBuffer` 反推，`..._s1s2_bn2gs1.h:440-470`）

```cpp
uint64_t stage1Size   = 8*1024;      // 8K 元素
uint64_t stage1AttenSize = 9*1024;   // attenmask 9K
uint64_t stage1PongSize  = 35*1024;  // NZ 场景 35K
uint64_t stage2Size   = 64*128;      // = 8192 元素 → T=fp32 时 32KB
uint64_t maskTBufPongSize = 16*1024;
TPipe::InitBuffer(maskTBufPing, 9K);  // atten mask
TPipe::InitBuffer(maskTBufPong, 16K); // dropout mask
TPipe::InitBuffer(pseTBuf, 16K);      // pse
TPipe::InitBuffer(stage1PingBuf, stage2Size*sizeof(T)); // t.a 32KB  ← P 片
TPipe::InitBuffer(stage2TBuf,   stage2Size*sizeof(T)); // t.c 32KB  ← bmm2 结果 / O
TPipe::InitBuffer(commonTBuf,   stage2Size*sizeof(T)); // t.b 32KB  ← 临时/API 缓存
TPipe::InitBuffer(softmaxSumBuf[0/1], s1BaseSize*blockBytes); // 4KB → s1BaseSize = 4K/32B = **128**
TPipe::InitBuffer(softmaxMaxBuf,      s1BaseSize*blockBytes); // 4KB
TPipe::InitBuffer(softmaxExpBuf[0/1], s1BaseSize*blockBytes); // 4KB  ← expUb
TPipe::InitBuffer(stage1PongBuf, ...); // i.a 32KB
```

**反推出的 tiling（host 侧，具体值在编译好的 tiling 库里，源码不公开）**：
- **`s1BaseSize = 128`**（4KB / 32B blockBytes = 128，铁证）。
- **`s2BaseSize = 128`**（通用 910B FA 默认；`s2BaseNratioSize = s2BaseSize * nRatio`，nRatio 是 N:1 复用比 1..8）。
- `s2BaseSize` 上限 `SPLIT_S2_SIZE_LIMIT = 1024`（仅 TND 变长时按 1024 切）。
- **`vec1S1BaseSize = min(s2BaseNratioSize / s2AlignedSize * 8, s1RealSize)`** → vec1 每次只处理 8 行一组（因为 `s2AlignedSize` 大时 [8 × s2Aligned] 就是 32KB P 片的量级），`realSplitN = ceil(s1RealSize / vec1S1BaseSize)` → s1 方向再分 `realSplitN` 次。
- **`vec2S1BaseSize = (dSizeAlign16 > 64) ? 64*128/dSizeAlign16 : s1RealSize`** → dSize≤64 时 O 整块 [s1Real, D] 32KB 存 UB；dSize>64 时分 `64*128/dSizeAlign16` 行一批，O 溢到 GM `vec2Res`。
- **UB 预算 ≈ 9K+16K+16K+32K+32K+32K+4×4K+32K ≈ 185KB**，贴近 910B AIV 的 192KB UB（极限压榨，所以官方能跑满带宽）。
- `dSizeAlign16 = ceil(D/16)*16`（D=72→80，D=256→256）。

---

## 4. 多核切分 —— 按 **(B, N2, G, S1_out)** 一维展平，S2 在核内串行

`Process()`（680-881）+ `ComputeAxisIdx`（894-907）+ `GetBNIdx`（883-892）：

```cpp
int64_t multiCoreInnerOffset = blockIdx * tilingData->multiCoreParams.splitFactorSize;
//   （enableL1Reuse 时：blockIdx/2，两个 AIV 共享一个 AIC 的 L1，成对处理同一 bn）
int64_t multiCoreInnerLimit  = min(totalSize, multiCoreInnerOffset + splitFactorSize);
// totalSize = bSize * n2Size * gSize * s1OuterSize   （s1OuterSize = ceil(S1/s1BaseSize)）
// 每个 multiCoreInnerIdx 解出 4 个轴：
boIdx   = idx / n2GS1o;
n2oIdx  = idx % n2GS1o / gS1o;
goIdx   = idx % gS1o  / s1OuterSize;
s1oIdx  = idx % s1OuterSize;
```

- **核切分维度 = B×N×G×ceil(S1/128)，一维连续切给各核（splitFactorSize）**；**S2 方向不做核切分，在核内 for 循环流式走完**。→ 天然无跨核归约。
- 负载不均时靠 host 的 `splitFactorSize`/`totalSize`（若 totalSize 不是核数整数倍，尾部核少做）。
- `enableL1Reuse`（fp16 高性能档）：**两个 AIV 绑一个 AIC**，对同一个 (b,n,g) 的相邻两个 s1o 块处理，K/V 的 L1 拷贝可被两核共用（IBShare）→ 省一半 L1 带宽。这就是 `l1ReuseBlockMod2 = blockIdx % 2`、`multiCoreInnerOffset = blockIdx/2*splitFactorSize` 的来源。

---

## 5. 深度 3 软件流水（性能的真正来源，`Process` 774-812）

```cpp
for (s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
    if (taskId>0 && notLast) WaitBmm1Result(extraInfo[(taskId+2)%3]);   // 收回 2 步前 bmm1
    if (notLastTwoLoop)      IterateBmm1(extraInfo[taskId%3]);          // 发起本步 bmm1（异步）
    if (taskId>0 && notLast) ProcessVec1(extraInfo[(taskId+2)%3]);      // 处理 2 步前 bmm1 结果(softmax)
    if (taskId>1)            WaitBmm2Result();                          // 收回上一步 bmm2
    if (taskId>0 && notLast) IterateBmm2(extraInfo[(taskId+2)%3]);      // 发起 1 步前 P 的 PV
    if (taskId>1)            ProcessVec2(extraInfo[(taskId+1)%3]);      // 处理上一步 bmm2 结果(累加 O)
    taskId++;
}
```
- `extraInfo[3]` 三份状态轮转：**同一步内同时有 bmm1(i)、vec1(i-2)、bmm2(i-2)、vec2(i-1) 在飞**，把 Cube 和 Vector 完全重叠。`lastNotPair`/`notSecondLast`/`notLast` 只是尾块配对处理（L1Reuse 下 AIV0/AIV1 配对）。
- 我们首版可先**串行不流水**（正确性优先），再按这个 3 段骨架加 2 层 double buffer 提性能。

---

## 6. 移植到我们 7 文件的工作量评估

我们的契约（`fa_probe/ours/`，v3 = 15/15 Pass）：
- q/k/v/out 均 `(B, N, S, D)` **BNSD ND**，dtype fp16 或 fp32（三件同 dtype）；attn_mask 可选 `(B,1,S,S)` **加性** fp32/fp16；causal = 加性 −1e4；softmax_scale 属性（默认 1/√D）。
- 输出 `out[b,n,i,:] = softmax_j(scale·q·k + mask) @ v`，内部 fp32，末尾单次 cast。

**映射到官方结构**：几乎完美对齐到 `LAYOUT_BNSD` 分支（A 行 stride=D 连续，最省事）：

| 官方 | 我们 | 差异/工作量 |
|---|---|---|
| `LayoutTypeEnum::LAYOUT_BNSD`, noPse, hasAtten=causal/mask, noDrop | ✅ 直接 | 小 |
| `s1BaseSize=128`, `s2BaseSize=128` | 可沿用 | 无 |
| bmm1/bmm2 GM Matmul + `SoftmaxFlashV2` | ✅ 照抄 | 中：要接 `REGIST_MATMUL_OBJ` + TCubeTiling |
| scale 用 `Muls(scaleValue)` 在 vec1 | ✅ 一致 | 无 |
| mask 用 `SelectWithBytesMask` + `set_mask_norm` | 我们现用 `Add` | **建议保留 Add（更简单、已是 15/15）**，只在 causal 尾块用 Duplicate+Add |
| `expUb` rescale 累积 O | 新写 | 中 |
| 全屏蔽行 → `AdjustSoftMaxRes` 清 0 | 我们 v3 靠 `l>0 保护` | **保留我们的保护即可**（避免引入 AdjustSoftMaxRes 复杂度） |
| softmaxMax/softmaxSum 输出 | 我们**不输出** max/sum | 删掉这些 DataCopy 即可 |
| PSE/dropout/prefix/var-len/TND | 无 | 全删，省大量代码 |
| 多核切分 B×N×G×S1o | 相同（我们 G=1） | 无 |

**结论：可以移，但不要整份照搬（2233 行里 ~60% 是 PSE/dropout/var-len/TND/sparse/l1Reuse/尾块配对，我们没有）。**
**建议裁剪版 ~600-800 行**：`FlashAttentionScoreS1s2Bn2gs1` 的骨架
= `Init/SetTiling/ComputeConstexpr(BNSD)/InitBuffer/Process(串行或深度3)/IterateBmm1/Bmm1SetTensorA/SetBmm1TensorB/ComputeBmm1Tail/ProcessVec1(scale+mask+SoftMaxCompute)/SoftMaxCompute(SoftmaxFlashV2)/IterateBmm2/ProcessVec2/Bmm2ResultMul/Bmm2ResultDiv/Bmm2DataCopyOut`
，丢掉 PSE/Drop/TND/var-len/sparse/AdjustSoftMaxRes/l1Reuse。

**关键可复用配方（我们已在 cube1/cube2 打通 5 次 launch 不崩）**：
1. 非模板 `extern "C" __global__ __aicore__` 入口，首行 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);`
2. `REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm1, &td->mm1, op.mm2, &td->mm2);`（全局函数，不加 AscendC::）
3. workspace 用 `AscendC::GetUserWorkspace(workspace)`；host `ws[0] = GetLibApiWorkSpaceSize() + 用户区`
4. host `#include "tiling_api.h"` + `MatmulApiTiling` + **5 参 `SetOrgShape(M,N,K,orgKa,orgKb,orgKc)`**（k≥65536 时 3 参会 orgKb 溢出）；`MatmulConfig` 用 `GetMDLConfig(true)`（intrinsicsCheck）
5. `mm.SetSingleShape(curM,curN,K)`、`IterateAll(gmC, false)`、`mm.End()`

**风险点**：
- **Tiling 结构体要放两个 `TCubeTiling`（mm1/mm2）** —— 两个 matmul 形状不同（bmm1: M=s1Real, N=s2Real, K=D；bmm2: M=s1Real, N=D, K=s2Real）。`s2Real` 会随尾块变（≤128），所以要像官方那样**每块 `SetOrgShape` 动态给尾块**，而 TCubeTiling 按满块 128 配。
- **精度**：P cast 到 INPUT_T 再进 cube（fp16 输入时 P 也是 fp16）——官方同样如此，fp16 allclose 1e-3 OK；fp32 时 P 保持 fp32 无损。O rescale 引入的差异 fp32 下 ~1e-6 相对，1e-4 达标。
- **全屏蔽行**：保留 `l>0 ? 1/l : 0` 与"整行 −1e4 → softmax 均匀"的现状语义（**不要**引入 −inf，否则 rescale 会 NaN）。
- **小规模 case（S=1..8）**：cube 启动/尾块开销可能让小 case 比向量还慢；官方用 `TILING_KEY 90/92/94 空张量路径` 只写常量。我们 15 例里有 S 很小的时间（1.66µs 级）——**建议保留一条"S 很小或 D 很小 → 走向量快路径"的分支**（我们 v3 已够快），大 S 走 cube。这是两条路，不是替换。

---

## 7. 一句话结论

官方 = **BNSD/SBH 上两个 GM-resident Matmul（QK^T 转置 B、PV 不转置）+ `SoftmaxFlashV2` 增量 softmax（输出 expUb 校正因子）+ UB/GM 里 `O=O*expUb+ΔO` 残差累加 + 按 B×N×G×S1o 一维切核 + 深度 3 Cube/Vector 流水**。我们不缺算法，缺的是**把 S² 次 Vector `ReduceSum/Muls` 换成 cube matmul**；核心可移植骨架约 600–800 行，其余官方代码是 PSE/dropout/var-len/TND 等我们不需要的分支。

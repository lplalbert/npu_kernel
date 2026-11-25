# 算子赛调优调研速查（官方 runbook / best-practice 提炼）

> 2026-11-24 调研。来源：CANN 官方 `cannbot-knowledge` runbook、`cannbot-skills` 性能最佳实践、昇腾社区 API 文档、ops-nn MR、CSDN 算子挑战赛拆解。
> 本文件只放「能直接改代码」的结论，详细出处见文末链接。

## 一、`SyncAll` 跨核归约的硬性配对（dot/poisson reduce 必读）

**规则（官方 TOPK-13）**：kernel 用了 `AscendC::SyncAll()`，tiling **必须**调
`context->SetScheduleMode(BATCH_MODE_SCHEDULE)`，否则默认调度模式 0 下同一次算子的核不保证同批调度，
SyncAll 的全核栅障永远等不齐 → **卡死 / 507035**。

```cpp
// op_host/xxx_tiling.cpp，放在 SetBlockDim 之前
// BATCH_MODE_SCHEDULE 在 CANN 9.2.0 头文件里未导出，必须自定义（官方取值 1）
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
context->SetScheduleMode(BATCH_MODE_SCHEDULE);
context->SetBlockDim(static_cast<uint32_t>(blockNum));
```

**配套规则（TOPK-14）**：调 SyncAll 时**禁止部分核早退 / 禁止 BlockDim 限制**。
所有启动的核都必须能走到 SyncAll；`if (start >= total) return;` 这类早退会让人等在栅障上 →
死锁。没数据的核也要参与（partial 记 0），SyncAll 之后再决定谁写输出。
且 `blockDim ≤ 物理核数`（本机 24）。

**证据**：昇腾官方算子库 ops-nn MR **#1112 标题即 "set batchmode for syncall — 算子使用了SyncAll，需要设置SetScheduleMode为batchmode"**（同一坑）。
另有一份 ascend950/arch35 的 runbook 记录"四种同步方案全失败→回退单核"，但它**没试 SetScheduleMode**，且平台不同（950 vs 我们的 910B2），不可照抄"回退单核"结论。

## 二、双缓冲（memory-bound 逐元素算子的第一优化手段）

**收益**：MTE2/VECTOR 重叠从 0% → 80~90%，吞吐 +20~50%。

**关键：只改 `InitBuffer(num=2)` 没用！必须把循环重构成预取模式：**

```cpp
TQue<QuePosition::VECIN,  1> inQueue;   // depth=1（非原地场景推荐值，与双缓冲无关）
TQue<QuePosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueue,  2, tileSize * sizeof(T));  // num=2 才是双缓冲
pipe.InitBuffer(outQueue, 2, tileSize * sizeof(T));

CopyIn(0);                                  // 预取第一个
for (uint32_t i = 0; i < tileNum - 1; i++) {
    CopyIn(i + 1);      // MTE2 预取，与下两行并行
    Compute(i);         // Vector 计算当前
    if (i > 0) CopyOut(i - 1);   // MTE3 写回上一个
}
Compute(tileNum - 1);                       // 收尾
for (uint32_t i = (tileNum > 1) ? tileNum - 2 : 0; i < tileNum; i++) CopyOut(i);
```

**硬约束 / 自检清单**：
- [ ] **num 翻倍 ⇒ `ub_factor`（tile 大小）必须减半**，否则 UB 溢出。
- [ ] UB 预算：`tile_size × sizeof(dtype) × buffer_num × num_tensors < UB容量`（910B 约 192KB）。
- [ ] 尾 tile 数量随 tile 减半而翻倍，尾块逻辑要同步更新。
- [ ] 读-改-写 in-place（如 reduce 的累加缓冲）用 **`TQueBind<VECIN,VECOUT>`** 共享物理 buffer，省一半。
- [ ] 同一 TPosition 的 QUE buffer：910B1 ≤ 8 块（`in(2)+out(2)=4` 安全）。
- [ ] 自定义 pingpong 时每对 SetFlag/WaitFlag 用独立 event ID。
- [ ] **自适应**：UB 大到能一次装下全部数据时，不要开双缓冲（同步开销反而亏）。

**事件模型**：`MTE2_V`（搬入完成→可算）、`V_MTE3`（算完→可写回）、`MTE3_MTE2`（写回完成→可覆盖，**WAR 必须加**）。

## 三、调优顺序（CSDN 算子挑战赛复盘，与官方一致）

**先对齐 → 再块大小 → 再 nBlock/kBlock → 最后开双缓冲。**
1. **地址对齐优先**：GM 起始地址 32B 对齐（建议 64B）；`DataCopyPad` 的 UB 侧地址必须 32B 对齐。
   对齐没做对，后面所有优化都是沙地盖楼。
2. **块大小**：太小→并发不足/同步事件过密；太大→UB 放不下、运行期分配失败。用二分找临界值再留 10% 余量。
3. **双缓冲放最后**：它掩盖的是前面调完后仍存在的搬运空泡。

## 四、小 case 固定开销（我们的分数大头，因为逐 case 等权）

- 减少 tilingKey 实例数（每个 key 一份 kernel 二进制，判题侧加载有成本）。
- 减少每次调用的 `InitBuffer` 次数、避免重复分配。
- 小 N 时可能不需要双缓冲/多核（UB 一次装下）。
- **dot 实测：空 kernel 判题 7ms 地板** → 这部分是 op 加载/下发成本，改不动；但真 kernel 181ms 说明有 174ms 是可优化空间。

## 五、其它已知结论

- 精度：累加一律用 float（int32 用 int64 保精确），只在输入输出边界保 half/bf16 转换。
- 多核结果"时对时错/每次位置不同" ⇒ 基本是**尾块/边界切分越界读**，打印每核 offset+长度核对。
- CPU 仿真不暴露搬运延迟，真机调优必须以 profiling 为准。
- 判题机 CANN 9.0.0：跨版本安全重排只用 `Gather + CreateVecIndex`（offset **字节**单位）；vector `Scatter` 不支持。
- `507035` 另一常见诱因（另一份 runbook）：每中间量各开全尺寸 TBuf 导致总 UB 远超上限。

## 六、出处

- SyncAll 须配批调度：[runbook](https://raw.gitcode.com/cann/cannbot-knowledge/raw/a5c5132938f02ef273b17fc9e3fbefda6d2a4fd5/knowledge/ops/ascendc/runbooks/runtime/syncall_requires_batch_schedule_mode.md)
- 950 上多核 reduce 507035（四方案失败，未试 SetScheduleMode）：[runbook](https://raw.gitcode.com/cann/cannbot-knowledge/raw/4deebd93648dad3b70da9a003954605c558c1e4a/knowledge/ops/ascendc/runbooks/runtime/507035_syncall_multicore_reduce.md)
- 官方双缓冲设计：[double_buffer_design.md](https://raw.gitcode.com/cann/cannbot-skills/raw/master/ops/ascendc-performance-best-practices/references/elementwise/double_buffer_design.md)
- SyncAll API 文档（建议 batchmode，否则多流死锁）：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md>
- 算子挑战赛源码拆解（调优顺序/五个翻车点）：<https://blog.csdn.net/weixin_33624741/article/details/166651608>

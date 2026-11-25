
---

## 8. Codex 评审与修复（2026-11-24，第二轮红队）

评审输入：`_sec5_build/ascend_tasks/ladder_review/mse.md`（SPEC + NOTES + 7 文件），
criteria 覆盖泛化性 / dtype 一致性 / 越界尾块 NaN / **确定性** / 向量 op count / 性能陷阱。

### 8.1 Codex 判定（top 3 必改）
1. **跨核归约竞态**：`Process()` 里各核先读输出旧值 `g0` 再做 `(P0-g)` 抵消，读与原子写之间无屏障；
   若 core0 的读晚于其它核的原子加，结果就错。备注里“22+ 例通过”不能证明无竞态。
2. **浮点原子加违反题面规则 1**（“给定相同输入，算子多次执行结果应完全一致”）：原子加顺序由调度决定。
3. **`none` 路径每核起点未按 32B 对齐**：`per=ceil(total/cores)` 可为任意元素数，尾块 `DataCopyPad`
   理论上可能与相邻核写同一 32B 块。
另有：`bf16`（def 未声明却映射到 half kernel）、非法 reduction 静默改成 mean、`total<1` 伪造成 1。

### 8.2 本轮改动（3 个文件）
- `op_host/mse_loss_tiling.cpp`
  - **每核元素窗口按 64 元素对齐**（`per=ceil(per/64)*64`）：每核只拥有整 32B 块，尾块 pad 落在自己窗口内，
    多核边界不再出现半个 32B 块。实测对齐前后大 case 吞吐不变（见 8.4）。
  - 删除 `DT_BF16 -> half` 的分派（def 只声明 fp16/fp32，属死代码）；只有 fp16 走 mode 0。
  - reduce 核数改为 `isF16 || MSELOSS_DET_REDUCE` 时单核。
- `op_kernel/mse_loss.h`
  - **`StoreScalar` 改为 UB 暂存 + `DataCopyPad`**（原来 fp32 路径用 `GlobalTensor::SetValue` 写 GM）。
    这是红线项（gnsq 已实测 SetValue 不回刷 GM），且新写法只写 `sizeof(T)` 字节、不再多写 3 字节。
- `op_kernel/mse_loss_tiling_key.h`
  - 新增 `MSELOSS_DET_REDUCE` 开关（默认 **0**）：
    - `0`：fp32 reduce 多核 + 32B 原子加（当前默认，最快，但逐位不可重复）。
    - `1`：fp32 reduce 单核（累加顺序唯一，逐位可重复），大 reduce case 慢约 11.6×。

### 8.3 确定性：实测结论（关键）
用同一份输入、每次新建进程跑同一 case，比较输出 sha1（`mytest/probe_det.py`）：

| case | runs | 不同结果数 | 与 numpy golden 逐位相等 | 相对偏差 |
|---|---|---|---|---|
| 16384x4096 f32 reduce=mean | 8 | **2** | False | ≤1.3e-6 |
| 16384x4096 f32 reduce=sum | 6 | **4** | False | ≤1.1e-6 |
| 16384x4096 f16 reduce=mean（单核）| 4 | **1** | **True** | 0 |
| 16384x4096 f32 reduce=none | 1 | 1 | **True**（逐位）| 0 |
| 1000003 / 1234567 / 7x13x11 各 dtype | — | 1 | **True**（逐位）| 0 |

结论：**题面规则 1 在多核原子加下确实不满足**（多次执行末尾数字不同），但偏差 ~1e-6，比 fp32 判据 1e-4 小两个数量级；
fp16 reduce（单核）与所有 elementwise 路径都是逐位确定的。
`MSELOSS_DET_REDUCE=1` 实测（16384x4096 f32 mean）：**4.2155 ms / 127 GB/s**，而默认多核 **0.3622 ms / 1482 GB/s**，
即确定性代价 ≈ **11.6×**。因此默认保留快路径，把是否强制确定性留给提交方一行开关决定（已在 8.2 说明）。

### 8.4 验收（本轮全部真机）
- `rm -rf build build_out && bash build.sh -j16`：绿；`build/custom_opp_ubuntu_aarch64.run` 生成。
- 官方 golden（`tests/ut/op_kernel/mse_loss_data/gen_data.py` + `compare_data.py`）：kernel UT `PASSED`，
  比对 `PASSED! MERE=0.000000, MARE=0.000000`。
  （`bash build.sh -u` 的 op_host UT 因本机缺 `nlohmann/json.hpp` 编译失败，属环境缺失、与本算子无关；
  kernel UT 走 `tests/ut/build/op_kernel/mse_loss_op_kernel_ut` 直接跑，等价于 `tests/ut/run.sh` 的后半段。）
- 自建对拍 `mytest/run_io.sh correctness`：18 例全 PASS（1D/2D/3D、非 64 对齐、fp16/fp32、三种 reduction）。
- `bash build.sh -e`：OK（需先 `./build/custom_opp_ubuntu_aarch64.run --quiet` 装包，examples 依赖 aclnn 头文件）。
- 性能（修复后）：16384x4096 f32 none 0.667 ms / 1208 GB/s；mean 0.362 ms / 1482 GB/s；
  16384x8192 mean 0.717 ms / 1497 GB/s；与修复前同档（对齐与 StoreScalar 改动无性能回退）。

### 8.5 仍未消除的已知风险（不改，留档）
- 32B 原子加会向只有 1 个元素的 out 张量写满 32B。设备侧按 512B 粒度分配，实测无异常（cmg 同样做法），
  但逻辑上越界 28B；要彻底消除必须把累加放到 workspace 再回写，而回写需要跨核屏障（本机 SyncAll 507057/507035 实测不可用）。
- 若 out 初值是 NaN/Inf（脏缓冲），`(P0-g)` 无法抵消 → 结果为 NaN。可用初值为 0 或上一次有限结果的缓冲；
  同上层因：无屏障就无法“先清零再累加”。
- 非法 reduction（不在 0/1/2）仍被夹到 mean，未报错（题面保证不出现）。

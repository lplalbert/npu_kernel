
## 十九、spspmm（第 12 题）判题 WA 根因 + 修复（待 Lead 提交）

- **判题 WA 定位**：submission `6b053a0185ad43bbe0fc4b03`（2026-11-24 07:17，status=Wrong Answer），
  case0（可见第一例，best_time≈9.94µs）`precision_ratio = 0.857177734375 = 3511/4096`（U=4096，即 64×64 一类），
  其余 7 例全 Skipped；即首个小/中 case 就 WA，不是大 case TLE、也不是个别数值越界。
- **根因 = f16 分支缺 `MTE2→S` 同步的竞态**（f32 无此问题）：
  - f32 路径：`DataCopyPad` 把列索引（int64）搬进 UB 后，用 `SetFlag/WaitFlag<MTE2_S>` 保证标量能读到；
  - f16 路径为了做 `Cast`，只用了 `MTE2_V` + `V_S`，标量随后读 `cols(bRI64_)` / `ai64` 时**没有显式等 MTE2 完成**
    （只靠 MTE2→V→V_S 传递，在本机实测不保证）→ 偶发读到旧/垃圾列索引 → `hit` 错位、行错乱。
  - **本机复现**：同一结构（64×64×64）**f32 全对、f16 偶发 badIdx≈970/8192（缺 275、多 203 个 (i,n) 对），
    同一输入重跑 6 次又全 0** —— 典型竞态。判题机（CANN 9.0.0）指令调度不同 → 必现 → 约 14% 错。
  - 之前本地 f32/f16 对拍"全过"是因为 f16 竞态概率性未命中。
- **修法（只改 `op_kernel/spspmm.cpp` 的 f16 分支，共 4 处，最小改动）**：
  1. `LookupRuns` f16：`DataCopyPad` 后 **先 `MTE2_S`**（保证标量读 `ai64` 列索引）**再 `MTE2_V`**（Cast 读 `av`）→ Cast → `V_S`；
     函数尾补 `V_MTE2`（Cast 读过 `av`，下一轮 DataCopyPad 覆盖前等 V 完成）。
  2. `ConsumeRun` f16：**先 `MTE2_S`**（保证标量读 `cols`）**再 `MTE2_V`**（Cast 读 `rv`）→ Cast → `V_S`；
     迭代尾在 `S_V`/`V_MTE2` 后再补 `S_MTE2`（标量读过 `cols`，下一轮 DataCopyPad 前等 S 完成）。
  - f32 路径完全未动；其余算法/emit/填充/二分不变。
- **修复后本地验证（910B2 / CANN9.2.0）**：
  - `torch_check.py`：f32 12/13、f16 12/13 —— 唯一 FAIL 仍是老 harness bug（CASE `200x50x1` 把未 clamp 的 `nb=500`
    传给二进制导致 fread 失败，非算子问题；正确 e2=50 单跑 PASS）。
  - **原先偶发失败的 64×64 f16 case 连跑 30/30 PASS**（修复前约 20~30% 概率错）；f32 5/5 PASS。
  - 新造 f16 幂律图（9 个 shape ×3 density ×2 alpha = 54 例）**54/54 PASS**；f16 多 chunk / multi-hub 另测通过。
  - f32 性能无回归（时延与修复前一致）。
- **未提交**（额度紧张，按 Lead 要求不自行提交）。4 个 editable 文件可直接交：
  `op_host/spspmm.cpp`、`op_kernel/spspmm.cpp`、`op_kernel/spspmm_tiling.h`、`op_kernel/tiling_key_spspmm.h`。
- 附：`/tmp/spspmm.cpp.bak_before_ffix` 为修复前备份（已从 `op_kernel/` 移出，避免被打进包）。

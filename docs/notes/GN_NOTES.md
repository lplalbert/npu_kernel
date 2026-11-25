# GroupNorm (problemId 6ad9c2a83ff2fef7e81bf865) — 结论备注

## 0. 交付状态
- **可交付版本 = v2**（= v1 两趟 + 一个零风险微优化：对齐满块时跳过冗余整块 `Duplicate`，大 case ~3–4%）。
  - 目录：`/root/work/ladder/gn`（7 个 editable 文件）。
  - 7 文件整理：`/root/work/ladder/gn/final7/`；打包：`/root/work/ladder/gn/gn_7files.zip`。
- **v1 备份**（原始两趟，无微优化）：`/root/work/ladder/gn_backup_v1/{gn_kernel_v1.h, gn_tiling_v1.cpp, gn_tiling_data_v1.h}`。
  - 回退：`cp gn_backup_v1/gn_kernel_v1.h gn/op_kernel/group_norm.h && cp gn_backup_v1/gn_tiling_v1.cpp gn/op_host/group_norm_tiling.cpp && cp gn_backup_v1/gn_tiling_data_v1.h gn/op_kernel/group_norm_tiling_data.h && cd gn && rm -rf build build_out && bash build.sh -j8`。
- 未提交。

## 1. 实现要点（v2）
- 输入 x (N,C,H,W) 或 5D；按 **group 分核、一组全程单核**：group 在 NCHW 下是连续块 `CpG*H*W`（CpG=C/G），**无跨核归约、天然确定性**。
- 内部全程 **FP32**：fp16 先 `Cast`；`ReduceSum` 求 sum/sumsq（**向量 op count 均为 64 的倍数**，尾块 `DataCopyPad` + 显式 32B rightPadding 补零）。
- 归一化 `y = x*a_c + b_c`（`a=rstd*γ_c, b=β_c-mean*a`，per-channel 向量 `Muls/Adds`）。
- 方差用 `E[x²]-mean²`（负值 clamp 0）；`rstd` 用向量 `Sqrt`（aicore 禁 double/sqrtf）。
- 中文注释：无；红线：无跨核归约、count 64 倍数、fp32 累加、fp16 Cast。

## 2. 正确性
- 官方 `gen_data.py` golden（(1,4,2,2) fp32/G=2）：**absmax=0.0，allclose True**。
- 自建 15 组 torch `F.group_norm` golden：**15/15 PASS**（含 base/InstanceNorm/LayerNorm(G=1)/ResNet/ViT/非对齐/单元素组/5D/全零/全一/大数值；fp32 absmax≤1e-6，MERE≈1e-7，无 NaN/Inf）。
- `rm -rf build && build.sh -j8` + `examples/run.sh`（build.sh -e）均 ✅。

## 3. 性能与结论
| shape | v1 | v2(交付) | 备注 |
|---|---|---|---|
| ResNet (2,256,64,64) G32 fp16 | 0.0495 ms | **0.0473 ms (~4.5%)** | 大 group 两趟 |
| big (1,256,128,128) G16 fp16 | 0.1343 ms | 0.1300 ms (~3.2%) | |
| ViT (1,512,14,14) G32 fp16 | 0.0204 | 0.0200 | 基本持平 |
| (1,32,128,128) G32 | 0.0194 | 0.0192 | 持平 |
| InstNorm (1,16,32,32) G16 | 0.0077 | 0.0075 | 持平 |

### 3.1 单趟（UB 可驻留 group，3×→2× 访存）：无收益，已弃
- 只对 gsz≲2000–4000 的组可驻留 UB；该区间数据小、**同步/启动延迟主导**，省一次读无收益。
- 实测 InstNorm(gsz=1024) 单趟 **0.0076→0.0167ms（2.2× 变慢）**（标量 a/b 广播 + 额外同步）；ResNet/ViT(gsz 超 UB/超 oneCap) 仍两趟、无变化。→ 回退。

### 3.2 跨核切分 + 归约：被 507035 栅栏封死，**不做**
- 题面无「累加顺序须固定」条款（原子加不违规）；cmg 的「32B 原子加 +(P0−g)」在本机**可用**——但那是“累加进输出、核间不回读”场景。
- GroupNorm 算出 sum/sumsq 后**必须回读全局统计量再算 rstd 做归一化 → 需要栅栏**；本机 aclnn 所有栅栏/回读路径（SyncAll、32B workspace DataCopy+int32/float 原子计数器、自旋）**均实证 507035**；单 kernel 模板无法两段启动规避。
- 负载均衡：`N*G=64` 组 vs AIV 核数 <64，group-per-core 的 makespan 已是 `ceil(64/核数)`（约 2 组/核）的最优；再降必须切单组→多核→需上述栅栏→不可行。**v1/v2 已是本机 group-per-core 最优。**

### 3.3 TQue 双缓冲：评估后未采用
- 两趟是 **3× 访存、带宽近饱和**（ResNet 有效 ~254 GB/s；去掉一次整块向量写只换来 ~4%）。TQue 双缓冲只**掩盖延迟、不减访存**，上限 <3% 且改动有跌破可交付版的风险 → 时间盒内保留零风险 ~4% 微优化，未做 TQue 重写。若判题环境带宽更高，可再评估。

## 4. 预估能否超榜主 3%
- 跨核（唯一能压 makespan 的大杠杆）被栅栏封死；group-per-core 已最优。
- v2 的 ~3–4%（大 case）能否整体 >3% 取决于榜主口径与用例分布；**不保证**。若判题环境支持多核归约/SyncAll，跨核切分是唯一有量级收益的路。

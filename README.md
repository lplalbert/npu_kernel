# Ascend C 算子竞赛 · 交接包

> 三条线并行：**CANN 算子天梯赛（10-11月赛）** / **制造业算子赛（gcc2026_mfg）** / **昇腾算子挑战赛 S10**
> 状态快照时间：**2026-11-25**（天梯赛截止 11-30 17:00，mfg 截止 12-07）

---

## 一、成绩总览（截至 2026-11-25，全部为平台真实判题结果）

| 赛道 | 结果 | 说明 |
|---|---|---|
| **天梯赛** | **14 / 18 题 Pass** | 三道 100 万积分档全部拿下（FlashAttention / ConvTranspose2D / DepthwiseConv2D） |
| **制造业赛 mfg** | **13 / 15 题 Pass** | 剩余 2 道：spspmm（WA）、radius（WA，已放弃） |
| **S10 挑战赛** | **5 / 5 题 Pass** | 全部 Pass，reduce 另做了 19.5× 提速 |

天梯赛 Pass 的 14 道覆盖**全部 5 个难度档**：

| 难度 | 算子 | 平台排名 | 得分 | submissionId |
|---|---|---|---|---|
| 5 | flash_attention | 62 | 10.25 | `6b05c00985ad43bbe0653413` |
| 5 | conv_transpose2_d | 41 | 8.00 | `6b05dd9a85ad43bbe0ab47ae` |
| 5 | depthwise_conv2_d | 24 | 12.89 | `6b0619d285ad43bbe017f9b6` |
| 4 | gf2_poly | 30 | 11.35 | `6b05c8ae85ad43bbe07a78e6` |
| 4 | cube1 | 21 | 6.29 | `6b06272c85ad43bbe02e38d3` |
| 4 | cube2 | 22 | 7.34 | `6b062c7c85ad43bbe037751f` |
| 3 | gf2k_mul_square | 24 | 19.23 | `6b05c96985ad43bbe07c4cd5` |
| 3 | quant_matmul_sigmoid_mul_quant | 32 | 9.35 | `6b05d0d785ad43bbe08e2057` |
| 3 | mhc_sinkhorn_grad | 56 | 0.00 (WA) | `6b05bd3b85ad43bbe05e264f` |
| 2 | bit_plane_transform | 26 | 23.30 | `6b05c5ee85ad43bbe073be10` |
| 2 | group_norm | 24 | 24.43 | `6b05bece85ad43bbe0624322` |
| 2 | mse_loss | 29 | 34.91 | `6b05c20785ad43bbe06a0343` |
| 2 | group_norm_silu_quant | **11** | 16.96 | `6b05f38985ad43bbe0d75fa9` |
| 2 | confusion_mul_grad | 90 | 0.00 (WA) | `6b05be0585ad43bbe0605351` |
| 2 | multi_add_rms_norm_dynamic_quant | 57 | 0.00 (WA) | `6b05c10585ad43bbe067a2a3` |
| 2 | index | 33 | 0.00 (RE) | `6b05c30785ad43bbe06c71c3` |
| 1 | trace | 125 | 18.57 | `6b05caeb85ad43bbe07fecad` |
| 1 | bn_training_update | 41 | 19.60 | `6b06110985ad43bbe009c20d` |

> 完整 38 题数据（含 mfg / S10）见 `01_STATUS.md` 与 `handoff_scores.json`。

---

## 二、这个包有什么用

**别人接手时最需要的三件事，这里都有：**

1. **能直接提交的产物** —— `*/zips/<op>_7files.zip` 就是判题平台要的 7 文件包（mfg 为 4 文件包），已校验「条目数 / 路径前缀 / 与源码 md5 一致 / 无违规标识符」。
2. **源码** —— `*/<op>/op_host|op_kernel/` 下的可编辑文件，可直接改。
3. **踩过的坑** —— `02_PITFALLS.md`（软硬件 + 判题机行为）和 `03_GOLDEN_ORDER.md`（**golden 归约顺序的逆向结论，这是本赛最贵的东西**）。

---

## 三、目录结构

```
.
├── README.md                 ← 你在这里
├── 01_STATUS.md              ← 38 题完整成绩表（排名/得分/status/坏 case 数）
├── 02_PITFALLS.md            ← 硬件 & 判题机坑（必读）
├── 03_GOLDEN_ORDER.md        ← golden 归约顺序逆向（必读）
├── 04_TOOLS.md               ← 工具脚本说明
├── 05_NEXT_STEPS.md          ← 未过 6 道题的卡点分析与建议
├── handoff_scores.json       ← 平台抓取的成绩原始数据
├── _manifest.json            ← 每个算子的来源目录 / 文件清单 / 完整性
├── ladder/                   ← 天梯赛 18 个算子
│   ├── <op>/op_host/*.cpp, op_kernel/*.h|*.cpp
│   └── zips/<op>_7files.zip  ← 直接可提交
├── mfg/                      ← 制造业赛 15 个算子
│   ├── <op>/...
│   └── zips/<op>_7files.zip
├── s10/                      ← S10 五个算子
│   ├── <op>/...
│   └── zips/<op>_7files.zip
├── tools/                    ← 提交 / 校验 / 诊断脚本
└── docs/                     ← 赛题解析、作战手册、各题 NOTES、排行榜快照
```

---

## 四、5 分钟上手

```bash
# 1) 校验所有 zip 是否完好（条目数、md5、违规标识符）
python tools/verify_handoff.py

# 2) 提交某道题（自动处理 429 冷却、轮询逐 case 结果）
python tools/subpoll.py trace
#   或从指定 zip 提交（A/B 变体）
python tools/sub_zip.py <zip路径> <problemId> <tag>

# 3) 看某次提交的逐 case 详情（含真机 profiling msg）
python tools/one_detail.py <submissionId>

# 4) 提交前必跑的三件套
python tools/scan_badwords.py    # 违规标识符（cout/cin/printf/getenv…）
python tools/verify_zips.py      # zip 与源码一致
python tools/verify_handoff.py   # 本包结构校验
```

**远端开发机**（原环境，CANN 9.2 / 910B2）：
`plink -ssh -P 40012 -pw <pw> root@<DEV_HOST>`，工程在 `/root/work/ladder/<op>`、`/root/work/mfg/<op>`。
> 注意：交接时该机 SSH 端口（40012）已拒连，若需重连请先确认端口转发是否恢复。

---

## 五、必须知道的四件事（先读这几条，能省几天）

1. **判据是「相对误差 < 1e-4」** 且 **「绝对误差 < 1e-4」**（不是 numpy allclose 的 `atol+rtol·|b|` 或组合）。
   → 近零输出元素只要差 1 ulp 就会挂 → 高精度量化类算子基本只有**逐位复刻 golden** 才能过。

2. **「提高精度」是错的，「复刻 golden 的归约顺序」才是对的。**
   实测：fp64 真值算出来的结果比纯升序 fp32 错得更多（`n4_T32768` 用例：fp64 错 148 个元素，升序错 0 个）。
   因为 golden 本身就是某个特定舍入顺序的产物。详见 `03_GOLDEN_ORDER.md`。

3. **提交前必查违规标识符**：`cout` / `cin` / `cerr` / `printf` / `getenv` / `system` / `popen`
   被判「提交代码中存在不合规内容」→ **HTTP 400 且照样扣配额**。把 `cout`（channel-out）改名为 `chOut` 即可。

4. **判题机有 3 秒硬看门狗**（`aclrtSynchronizeStreamWithTimeout(stream, 3000)` → `ret=507046`）用于 cube 类算子；
   但其它算子的 harness 宽容得多（实测有 case 跑 272 秒仍 Pass）。**不要把性能优化凌驾于正确性之上**。

---

## 六、交接时需要向用户确认的事

- CANNJudge 的 JWT / userId 是否仍有效（本包 `tools/` 里写死的是 2026-11-25 的快照，`exp` 到 2026-11-26 前后，**很可能已过期，需要重新登录取 Cookie**）。
- 原开发机 `<DEV_HOST>:40012` 的端口转发是否需要重新开通。
- 天梯赛是 **`best` 模式计分**（取历史最优），所以反复提交 A/B 不会掉分；mfg / S10 请自行确认计分口径。

---

## 七、公开仓库说明

本仓库为**公开**发布，因此已做以下处理：

- `tools/` 与 `docs/` 中的 **CANNJudge JWT、开发机 SSH 密码 / hostkey / 内网 IP** 已全部替换为占位符
  （`<CANNJUDGE_JWT_REDACTED>` / `<SSH_PASSWORD_REDACTED>` 等），**运行脚本前请按 `tools/CREDENTIALS.md` 填回**。
- 提交产物 zip（`*/zips/*.zip`）与源码不含任何凭据，可直接使用。

---

## 八、目录里各 zip 是什么

`ladder/zips/<op>_7files.zip` 就是判题平台「打包上传工程」要的东西：
7 个条目，路径形如 `op_host/<base>_def.cpp`、`op_kernel/<base>.h`，**无工程名前缀**。

mfg 通常是 4 个条目：`op_host/<base>.cpp` + `op_kernel/<base>.cpp` + `op_kernel/<base>_tiling.h` + `op_kernel/tiling_key_<base>.h`。

自检：`python tools/verify_handoff.py`（校验条目数 / md5 / 违规标识符）。

---

## 九、关于 4 个 mfg 算子的 3 文件包

`random_walk` / `cylinder_query` / `spmm_sum` / `segment_min_csr` 这 4 道**本来就没有** `tiling_key_<base>.h`
（单 tiling key，无需模板参数），因此它们的 zip 是 **3 个条目** —— 这是它们当年提交时的真实形态，不是缺失。
其余 mfg 算子为 4 个条目。

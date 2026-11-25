# Jev 模型调优赛道 · 作战手册（交接文档）

> 供新会话直接开工。最后更新：2026-11-24（通宵冲刺后交接）。
> 目标：**调优赛道冲一等奖**，截止 **2026-11-26**。主线：GEV-26B-Decide；保险：Jev-Omni。

## 0. 任务一页纸

- 赛事：昇腾 Jev 模型适配调优大赛 · 模型调优赛道（10 赛题 × 1-3 名 = 30 奖，一等奖=各题第1名 HUAWEI Pura X Max）
- **核心规则**：①同一队伍多榜获奖只发最高奖 → **单点聚焦**（主攻 1 题冲第 1 + 1 题保险）②评分 = 2PP 精度达标下的性能提升（S1 P95 + S2 吞吐）③最终以**主办方复审复现**为准（统一镜像 `quay.io/ascend/vllm-ascend:v0.30.0-a3`，CANN 9.2.1 / torch_npu 2.10.0.post7 / vllm 0.30.0 / Python 3.12.16）④评测硬件 **A3 单物理卡 2 逻辑 NPU（910C）**、BF16 原始权重禁量化、thinking=off、S1 允许 warmup 多轮平均⑤允许改 vLLM 调度器/ACLGraph、自定义 Ascend C 算子（放 src/）⑥提交 = **私仓** +「模型仓」「NPU」标签 + **PR 到 Ascend/model-agent**
- 交付九件套：start.sh / inference.py / README.md / src/(可选) / metrics.csv / results.csv / result.png / npu_device_call.png / logs/ / optimization.md / model_agent_pr.md(可选加分)

## 1. 已完成资产（全部可复用）

### 1.1 协议全破译（核心竞争力——2PP 护栏的依据）
对照官方 serve_decide.py（模型仓自带）+ 官方元数据逐项对齐：
- 24 槽位 verbaliser（noul[0,2) / score[2,8) / choice[8,24)），**稀疏读出** logprob_token_ids（CHUNK=128 上限内）
- **READ 契约**：SamplingParams temperature=1.0, top_p=1.0, top_k=0, min_p=0（全分布，禁截断 logprobs）——temperature=0.0 是错的
- choice 行格式 `{label} {opt}`（label=单 token 标签 A/B/C）；prompt 模板 `[kind] {kind}\n[state] ...\n[decision]:`（gemma 前缀空串，**不加 bos**）
- 决策公式 `p = softmax((logprob(verbalizer)+bias)/T[kind])`，缺失槽位 -1e9 截底
- bias 为**非零真值**（官方 decision_head.json），T 取 calibration.json per_kind（noul 1.00275 / choice 1.01720 / score 0.99929），softcap=30 由 LM head 施加不二次
- decision_head.json 字段：bias / labels(None→fallback A,B,C) / ranges(None→默认) / softcap / prefix_token(2) / verbalizer_ids

### 1.2 SystemOne 服务 v2.1（jev_work/tuning_track/inference.py）
- vLLM-Ascend 引擎 + **DecisionBatcher 决策批处理**（8ms 窗口聚合→一次 llm.generate 整批前向）：验证床实测 S2 吞吐 17→66 dec/s（并发 8）；规避单进程调度器 reorder KeyError（vllm attention/backends/utils.py:902）
- **torch.ops._C_ascend 算子桩**（_NS 类，文件头部）：只劫持 npu_add_rms_norm_bias（vllm-ascend patch_parallel/rms_norm.py 硬调该缺失算子）其余透传——全量打桩会吃掉 device_print 等 op（踩过）
- 4 接口：GET /health、GET /v1/decide/info、POST /v1/systemone（含 questions{} 批量形态）、POST /v1/decide
- **OneJev1.6-0.8B 验证床全链路跑通**：health+systemone 双 200，决策+完整概率正常

### 1.3 工具链（同目录）
- bench.py：S1（P50/P95/P99 多轮）+ S2（并发扫描）→ logs/bench_*.json
- accuracy_panel.py：**预注册式精度护栏**（40 题 panel seed=20261123，阈值 run 前固定：一致率≥95%、max_abs_delta≤0.02、分 kind≥90%）——每项优化合入前必须 PASS（实测 1.000/0.0000）

### 1.4 私仓 lplalbert/gev-systemone（gitcode main 分支 = 提交仓）
九件套已同步（inference.py v2.1、README、bench、panel、optimization.md 附录A协议不变量+附录B批处理、metrics.csv 十列版含基线/优化/阈值/结论、results.csv 12 列、model_agent_pr.md 贡献计划）。
**还缺**：CSV 实数据、result.png、npu_device_call.png、logs/ 实证据、src/ 补丁、最终 PR 到 Ascend/model-agent。

### 1.5 参考材料
- decision_head.json（bias 真值）、calibration.json（温度+fit 统计）、tokenizer、chat_template.jinja、config.json、example.py、head.pt
- 官方 serve_decide.py（仅对照勿抄）；竞品 skyline117/GEV-26B-Decide（apache-2.0）——其 fast_moe 提速 1.44-1.57x 但 7-9/40 题精度翻转被自弃，**此坑勿踩**

### 1.6 REA 浏览器自动化（jev_work/scenario_*.json 约 150 场景）
- 玩法：node ../../rea/scripts/rea.mjs capture-browser-scenario <scenario.json> --json（rea 仓 code/rea）
- 姿势：gitcode cookie 种子进 ttyd、base64 分块传文件（3KB/块）、ttyd fill/press、容器重启后 ttyd URL 不变
- 坑：gitcode 页 Service Worker 要 environment.service_workers=allow；heredoc 多行 fill 会截断→base64 单行管道

### 1.7 Codex 裁决（handoff/consult/）
- codex-20261122-230843.md：LLM additional_config 注入姿势（ascend_compilation_config）
- codex-20261123-021939.md：大文件下载（modelscope SDK > 校验流式下载器 > hf-xet；勿 curl -C -/wget -c 续传镜像 LFS）

## 2. 环境资产

| 环境 | 规格 | 用途 |
|---|---|---|
| **CANNLab**（gitcode.com/org/cann/cannlab） | 910B/910C、1/8 卡；196G /mnt/workspace 持久；SSH 免密直连（见 CANNLab_昇腾卡使用指南.md）；A2/A3 共享 100 卡时；**限制 only one A2/A3** | **主战场**。Jev 建 **A3/910C/1卡**（镜像优先 CANN 9.2.x） |
| Jev pod（Model Agent 容器） | 50G 配额（装不下 GEV 66.9G）、4h 会话回收、shm 64M、sudo 禁用 | 仅应急 |
| 本地 13×NVIDIA | 不能跑 NPU 路径 | 文档/脚本开发 |

**三赛卡型**：S10=910B3、制造业=910B、Jev=910C。错峰：11-26 前 A3 打 Jev → 删 A3 建 A2 打制造业（12-20）→ S10（2027-02）。mish 工程本地已完整备份（s10_work/mish/），旧 A2 实例可删。

## 3. 当前状态与卡点

| 项 | 状态 |
|---|---|
| CANNLab A3 实例 | **未创建**——创建页卡片 API 在自动化浏览器不渲染，需用户手点：gitcode.com/org/cann/cannlab/environment/create → NPU 卡 → 910C/1卡 → 镜像 CANN 9.2.x；槽位被占先删 A2（备份齐） |
| GEV 权重（66.9G = shard1 49.9G + shard2 17G，LFS 指针权威） | pod 放不下；A3 的 196G 可放。下载铁律：**一次性下载勿续传**（hf-mirror 假长度、modelscope CDN 末段死过、LFS 直链被网络策略挡）；**modelscope SDK 最稳**（25MB/s）：`modelscope download --model autotrust/GEV-26B-Decide --local_dir <dir>` |
| 真基线（S1/S2+panel） | 待权重+910C |
| 赛事群三问（用户办） | ①权重挂载/--model-dir 开发期机制（Q&A 26 官方"正在寻求云存储方案"）②租 910C 自建开发是否合规③910C 租用渠道 |
| Jev-Omni 保险线 | 协议材料齐（decision_config/eval_config/head.pt/example.py），权重 23.9G 未下 |

## 4. 新会话开工序列

1. **确认 A3 实例就绪**（用户手点创建或已建好）→ SSH 连上（新实例主机名会变：devenvc_<实例名>.<实例id>.atomgit.0，查 C:/Users/24976/.atomgitdevenv/.ssh/config）
2. **拉 GEV 权重**：modelscope SDK 到 /mnt/workspace/gev（校验 shard 49.9G/17G；decision_head.json 在 adapter_vllm/ 子目录）
3. **部署 v2.1**：私仓 clone 或 tar 管道传 tuning_track/ → `bash start.sh --model-dir /mnt/workspace/gev --output-dir ./logs`
4. **真基线**：`python bench.py --s1-repeats 30 --s2-concurrencies 1,2,4,8,16,32 --s2-duration 60` + `python accuracy_panel.py --tag baseline`
5. **优化冲刺**（每步 panel PASS）：L1 批处理调参（window/max_batch/max_num_seqs/max_num_batched_tokens 矩阵）→ L2 图模式 A/B → L3 稀疏读出/前缀缓存/DP2；kernel 级改动必过 2PP
6. **交付收尾**：CSV 填实测（基线+优化+阈值+结论）、result.png + npu_device_call.png（**须显示 910C**）、logs/ 归档、src/ 放补丁、model_agent_pr.md（候选 PR：DecisionBatcher skill、算子打桩 skill）
7. **提交**：私仓 +「模型仓」「NPU」标签 + **PR 到 gitcode.com/Ascend/model-agent** + 大赛页「提交作品」

## 5. 大坑速查

| 坑 | 解法 |
|---|---|
| vLLM 多进程引擎死（Broken pipe） | VLLM_USE_V1=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 单进程；shm 64M 根因 |
| aclnnAddRmsNormBias 缺符号 | inference.py 头部 _NS 算子桩（只劫持该 op） |
| 并发 500（reorder KeyError） | DecisionBatcher 批处理（v2 内置） |
| logprobs 截断毁校准 | READ 契约 temperature=1.0/top_p=1.0/top_k=0 |
| LFS 指针当权重（135B） | modelscope SDK / hf-mirror resolve 拉真身 |
| wget -c / curl -C - 拼接损坏 | 一次性下载勿续传 |
| /tmp 被容器重启清空 | 脚本/断点放 ~/ 持久卷 |
| pod 时间戳 UTC | +8h 为本地 |
| 提交件缺一即无效 | 目录/文件名严按规范；调优赛道必须**私仓** |
| choice 行 "A) opt" 写法 | 官方 `{label} {opt}`（空格） |

## 6. 文档索引

- handoff/CANNLab_昇腾卡使用指南.md —— CANNLab SSH/构建/提交姿势
- _sec5_build/ascend_tasks/jev_work/ —— 场景库、截图证据、pod 痕迹
- _sec5_build/ascend_tasks/jev_work/tuning_track/ —— 九件套源件
- handoff/consult/codex-2026112*.md —— 技术裁决
- 规范原文：jev_readme.md、jev_submit.md、registered_detail.txt（ascend_tasks/ 下）

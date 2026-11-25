# gcc_2026_mfg 制造业算子赛 · 会话交接（2026-11-23）

> 给接手会话的完整工作包。现状一句话：**机制分析完毕、策略已定、尚未写任何题的代码**。
> 第一件事 = ComplexAbs 打通「写码→CANNLab 构建→对拍→提交判题」全链路（详单见文末）。

## 1. 赛事快照

- 赛题：**昇腾AI行业创新大赛——2026制造行业算子专场（预赛）**，CANNJudge 赛事 id `6aed4926a5348dcef5dd8d0c`，页面 `https://cannjudge.cn/public/gcc_2026_mfg`
- 赛程：2026-11-09 ~ **2026-12-07 12:00 北京时间（UTC 04:00）截止**（还剩 ~14 天）；报名已由用户完成（gitcode 组队，1-3 人），**算力券已申请**
- 形态：15 道 **Ascend C 算子**题（点云 5 / 图计算 5 / 仿真 5），云端判正确性 + 按算子执行时延给分，无需本地有昇腾卡即可提交
- 奖金池 ¥250,000（gitcode 侧）；**评奖细则（奖项数/是否晋级决赛）尚未拿到**——gitcode 大赛页是客户端渲染+CloudWAF，见「未决事项」

## 2. 15 题清单（按难度分层，建议做题顺序 = 难度升序）

| 题目 slug | 标题 | 难度 | problemId | 榜首分（/100） | 参赛队数 |
|---|---|---|---|---|---|
| complexabs | 仿真：ComplexAbs 复数模长 | ⭐1 | `6af2d3706bd0f9be375466ed` | 95.5 | 70 |
| complexexp__v2 | 仿真：ComplexExp 复指数 | ⭐1 | `6af6cbd36bd0f9be37ba0aae` | 88.4 | 44 |
| ballquery | 点云：BallQuery 球查询 | ⭐1 | `6af1c92b6bd0f9be3798a712` | 91.4 | 76 |
| gridcluster | 点云：GridCluster 体素聚类 | ⭐1 | `6af1cab46bd0f9be379bc6c6` | 91.8 | 47 |
| gathercoo | 图：GatherCoo | ⭐1 | `6af1cbe56bd0f9be379e1c23` | 93.8 | 56 |
| complexdiv | 仿真：ComplexDiv 复数除法 | ⭐⭐3 | `6af2d43a6bd0f9be3756251e` | 95.1 | 46 |
| radius | 点云：Radius 半径搜索 | ⭐⭐3 | `6af1ca066bd0f9be379a6c84` | 96.3 | 71 |
| furthestpointsample | 点云：FPS 最远点采样 | ⭐⭐3 | `6af1c8736bd0f9be37971169` | 97.8 | 39 |
| cylinderquery | 抓取：CylinderQuery 圆柱查询 | ⭐⭐3 | `6af1c67a6bd0f9be3792fb4a` | 96.1 | 38 |
| spmmsum | 图：SpmmSum CSR 稀疏×稠密 | ⭐⭐3 | `6af275c16bd0f9be37a3bbd5` | 85.8 | 62 |
| segmentmincsr | 图：SegmentMinCsr 分段最小 | ⭐⭐⭐4 | `6af1cdaa6bd0f9be37a1afe1` | 89.9 | 69 |
| fusedpropstep | 仿真：FusedPropStep 融合频域传播步 | ⭐⭐⭐4 | `6af2d5b76bd0f9be3759446c` | 82.0 | 56 |
| fouriertransform2d | 仿真：2D C2C FFT | ⭐⭐⭐4 | `6af2d6646bd0f9be375ab041` | 91.7 | 49 |
| spspmm | 图：Spspmm 稀疏×稀疏 | ⭐⭐⭐⭐5 | `6af276736bd0f9be37a4cb08` | 84.8 | 48 |
| randomwalk | 图：RandomWalk（含 node2vec p/q） | ⭐⭐⭐⭐5 | `6af277846bd0f9be37a6755a` | 91.6 | 48 |

## 3. 计分与评奖机制（CANNJudge API 实锤）

| 机制项 | 值 | 推论 |
|---|---|---|
| 终榜计分 `final_ranking_score_mode` | **`average`** | 总成绩 = 15 题得分**平均**，未做题按 0 计 |
| 用例权重 `final_ranking_public_weight` | **0.1** | 可见用例只占 1 成，**隐藏用例 9 成**——别对着可见用例过拟合 |
| 每题得分 | 性能时延折算 0~100（`score_mode: 1`），`precision_ratio` 必须 1.0 | 正确性是门槛，分随速度走 |
| 计分用例 | 每题 `ranking_testcase_count: 8`（可见）/ 总 15~24 例（含隐藏） | 硬编码可见用例必死 |
| 提交 `ranking_submission_mode` | **`best`**（历史最佳计分） | 随便刷无惩罚 |
| 额度 `effective_daily_limit` | **50 次/天** | 充裕；判题约 8~12 分钟/次（S10 实测） |
| 单题迭代 `iterations` | 5 | 多半是计时取样次数 |

分数公式的精确形式**未完全反推**：拿 76 行榜单拟合，`score ≈ mean(100×best_i/t_i)` 量级对（榜首预测 96.8 vs 实际 91.4，偏差 ~5 分），大概率混入隐藏用例或基线归一项——知道「更快=更高分、精度一票否决」就够用，不必再逆向。

## 4. 已定策略：**全部完成为主，针对性刷榜为辅**

平均分口径下的数学：开新题 0→70 = 总均分 **+4.67**；已做题 85→95 = 总均分 **+0.67**（7 倍差距）。竞争面佐证：15 题前五名中**无一队覆盖超过 4 题**（榜首集团「尼古」4 题点云系、「元莲仙尊」4 题图系）——广度即护城河。

排期（截止 12-07）：
1. **第 1 天**：ComplexAbs 打通全链路（写码→构建→对拍→提交→看榜），固化本地开发循环
2. **本周**：⭐1 五题 + complexdiv、radius 全过（目标：每题 70+）
3. **下周**：⭐⭐3~⭐⭐⭐4 收割（FPS/CylinderQuery/SpmmSum/SegmentMinCsr/FusedPropStep/FourierTransform2D）；Spspmm/RandomWalk 量力
4. **最后 2~3 天**：对「已过但分低」的题按 ROI 回头刷性能（逐点/点云类优化空间大；注意隐藏用例 9 成权重 → 要通用优化不要 case 特化）

## 5. 开发基础设施

### CANNLab 昇腾开发卡（长期开发机）
- 完整使用指南：**`handoff/CANNLab_昇腾卡使用指南.md`**（连接/构建/测试/大坑速查全在里面，先读它）
- 一行连通性测试：`ssh -o BatchMode=yes devenvc_cz0xe.9c45ee71438b4c518bbc4c1af378d1ed.atomgit.0 "echo ok"`
- 规格：1× **Ascend 910B3**（64GB HBM，编译目标 `ascend910b`）、CANN 9.0.0 可写、16 vCPU、`/mnt/workspace` 196G 关机持久
- 卡上 `/mnt/workspace/mish/` = **S10 mish 题已判题 Pass 的完整参考工程**（op_host/op_kernel/tests/examples/build.sh 全套）——本赛题模板同属 `npu_op_package` 家族，可把它的 `build.sh`（含 `-u` CPU UT / `-e` NPU 实测）+ `tests/ut` + `examples` 改造成 mfg 各题的本地验证脚手架
- 额度：A2/A3 共享 100 卡时 + CPU 720 核时，另有 260 CANN 积分可兑换；**不用时关机**（关机不烧卡时）

### CANNJudge API（全部已打通）
- 认证：`Cookie: cannjudge_auth=<JWT>`，**JWT 见 CANNLab 指南「CANNJudge 提交」节（有效期至 2026-11-29，早于比赛截止，届时让用户 DevTools 重取）**；userId `<MY_USER_ID>`
- 题单（含完整题面/规格/禁止事项）：`GET /api/problems/contest/{contestId}`
- 单题详情 / 模板：`GET /api/problems/{id}` / `GET /api/problems/{id}/template`（返回 `files[]`，含 `editable` 标志）
- 提交：`POST /api/submissions/submit`，payload `{problemId, userId, files:[{path, content}]}`；**只能交 editable 文件**，交别的报「文件不可编辑」
- 查分：`GET /api/submissions/{id}?userId=...`（轮询到 status 非 Running/Pending；`result[]` 含每例 time/precision_ratio/testcase_status，Hidden 例不外显）
- 榜单：`GET /api/problems/{id}/ranking`
- 注意：S10 每日 10 次提交，**本赛是 50 次/天**，别混

### 题目工程模板（mfg 版，与 mish 略有差异）
- 结构：`CMakeLists.txt`（不可编辑，`find_package(ASC)` + `npu_op_package`）+ `op_host/{op}.cpp`（tiling+OpDef，**可编辑**）+ `op_kernel/{op}.cpp`（kernel 主体，**可编辑**）+ `op_kernel/{op}_tiling.h`（**可编辑**）+ `op_kernel/tiling_key_{op}.h`（**可编辑**）——**每题恰好 4 个可编辑文件**（mish 是 7 个，结构别照搬）
- 模板构建（README 官方口径）：`cmake -S . -B build -DCANNJUDGE_COMPILE_JOBS=4 && cmake --build build --parallel 4 --target binary package`
- 模板里 Init/Process 是**空壳**，tiling 只给了示例取参——全部要自己写
- ComplexAbs 的完整题面+模板已抓全，存档：`_sec5_build/ascend_tasks/mfg_work/_cj_cabs_all.txt`（4 个可编辑文件全文，可直接起手）

### 通用红线（题面「禁止事项」逐题核对，以下跨题共性）
- 禁回退 CPU、禁改语义走捷径（改比较符/跳查询/固定输出/漏 query）
- 索引类（ballquery/radius/cylinderquery 等）：int32 索引与 CPU Reference **逐元素一致（atol=0）**；严格小于、first-K 顺序、不足 nsample 时尾部重复首个命中、0 命中全 0——这些边界语义**全在判分范围**
- **float16 输入必须升 float32 再做距离/阈值比较**（低精度比较产生索引偏差=错）
- 数值类：atol/rtol 1e-3~1e-4（complexabs 1e-4）；FFT 类禁调现成 FFT 库，逆变换 1/(N·M) 归一化约定不能动
- 本地开发坑（S10 血泪，见 CANNLab 指南）：朴素 `DataCopy` 要求 count×sizeof(T) 32B 对齐，**真 NPU 静默丢尾部数据**（CPU 模拟器反而 abort）→ 变长一律 `DataCopyPad`，性能写法=tiling 块对齐 32B 只尾块走 Pad

## 6. 已有数据资产（本次分析产物，根目录/`_sec5_build/ascend_tasks/mfg_work/`）

- `_cj_mfg_contest.json` / `_cj_mfg_digest.txt` —— 赛事记录+计分配置（第 3 节的出处）
- `_cj_mfg_problems.json` / `_cj_mfg_probs_detail.txt` —— 15 题元数据（难度/用例数/题面摘要）
- `_cj_rank_all.txt` —— 15 题各自 top5 榜单+分数+团队覆盖分析
- `_cj_score_fit.txt` —— 分数公式拟合尝试（见第 3 节尾注）
- `_sec5_build/ascend_tasks/mfg_work/_cj_cabs_all.txt` —— ComplexAbs 完整题面+4 可编辑文件模板全文
- 上届参考：`_sec5_build/ascend_tasks/s10_work/mish/`（本地）与卡上 `/mnt/workspace/mish/`（判题 Pass 的工程）
- 提交 API 细节与判题字段：`memory/s10-mish-campaign.md`（mfg 与 S10 同 API，仅题号/额度不同）

## 7. 未决事项（接手后按需推进，勿被带偏）

1. **评奖细则**：几等奖/几名/预赛是否进决赛——gitcode 大赛页（`competition.gitcode.com/competition/2118537324714082306`）客户端渲染抓不到正文，最省事是**让用户登录后复制「评奖规则」段落**；拿到前按「总分（15 题均分）排名定名次」执行即可，不影响打法
2. **算力券**：用户已申请，具体兑换/挂载方式未确认；但 CANNLab 卡本身就是开发主力，券大概率用于补充卡时
3. **JWT 续期**：11-29 过期（早于 12-07 截止），提醒用户重取
4. 注意聚焦：本包只管 **gcc_2026_mfg 制造业算子赛**；S10/Jev/天梯赛等是别的线，别混做

## 8. 接手第一步（建议照抄）

1. 读 `handoff/CANNLab_昇腾卡使用指南.md`，SSH 连卡确认 `echo ok`（若实例重建主机名会变，指南里有查法）
2. 从 `_cj_cabs_all.txt` 起手写 **ComplexAbs**：kernel = 向量化 `sqrt(re²+im²)` 多核均分 + double buffer；tiling 按 UB 容量切块；对拍用 torch（卡上有 cpu torch）
3. 卡上构建 + 本地对拍 → 提交（只交 4 个 editable 文件）→ 轮询判题 → 看 precision_ratio=1.0 与 score
4. 跑通后把这条「本地开发循环」固化（建议在 mfg_work 下建 per-problem 目录），按第 2 节表逐题推进

## 9. 环境实况更新（2026-11-24 深夜，S10 会话实测，**与第 5 节有出入，接手先看这里**）

- **开发卡已不是 910B3/CANN9.0.0**：当前 CANNLab 实例是 **910C 系 / CANN 9.1.0 / Ubuntu24.04 / g++13 / py3.12**（SOC `Ascend910_9362/9363`，NpuArch 2201）。第 5 节的 910B3/CANN9.0.0 是旧快照。
- **python 坏，构建必先修**：`/opt/buildtools/python-3.12.9` 是 3.12.3 二进制 + 3.12.9 lib-dynload，`_ctypes` 报 `_PyErr_SetLocaleString` 未定义，**kernel 编译（TBE 用 python）必失败**。修复：
  ```bash
  export PYTHONPATH=/usr/lib/python3.12/lib-dynload:$PYTHONPATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0   # 让 torch(2.10.0+cpu) 可用
  ```
  所有 build/UT/提交脚本都要带这两条。`/opt/buildtools` 是 root，改不了，只能 PYTHONPATH 优先系统配套 `_ctypes`。
- **本地 NPU 跑不了 `ascend910b` 自定义算子**：设备是 NpuArch 2201/arch4，`ascend910b` 是 NpuArch 1001/arch3，example 必报 **ACL 361001**；且 ASC 构建只认 `ascend910b`（`ascend910b3` 被 CMake 拒）。=> mfg 同样只能「编译 + CPU 模拟器 UT + 判题机」验证，**别指望本地 NPU 对拍**。
- **判题 API 同 S10**（JWT/接口见 CANNLab 指南），mfg 是 50 次/天、冷却 ~80s；提交前先自查逻辑（用户要求：别轻易提交，失败找 Codex 评审方向）。
- **实例隧道会不定时失效**（`forward-bootstrap` 报 404 / timeout），需用户在 VS Code 重连或重启实例；连接主机名随实例变，查 `~/.ssh/config` 的 atomgit Include 区。
- 参考工程：S10 的 mish（`_sec5_build/ascend_tasks/s10_work/mish/`，本地）已判题 Pass，但它是 7 editable 结构；mfg 是 4 editable，**别照搬文件结构**，只借鉴 build.sh/tests/examples 脚手架与「TQue 自动同步」写法。


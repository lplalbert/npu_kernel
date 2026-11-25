# CANN 算子天梯赛（10-11月赛）作战手册

> 更新：2026-11-24。目标：11/30 17:00（北京）前拿月度奖。

## 一、赛事坐标（已实测确认）

| 项 | 值 |
|---|---|
| 赛事名 | CANN算子天梯赛（10-11月赛） |
| **contest `_id`** | `6aced8d2e769892927fdb17f` |
| contest **name(key)** | `ladder_october` |
| group | `public`（`_id=69c8ffcea2bcbfa3591f481e`） |
| 起止 | 2026-10-14 04:00Z ~ **2026-11-30 09:00Z**（= 北京 11/30 17:00） |
| 页面 | `https://cannjudge.cn/public/ladder_october/<problem_name>` |

## 二、API（全部实测可用，Cookie `cannjudge_auth=<JWT>`）

```
GET  /api/contests/name/ladder_october        # 按 name 取 contest → 拿 _id
GET  /api/contests/group/<groupObjId>         # 按组取所有 contest
GET  /api/problems/contest/<contestId>        # 该赛全部题目（含 desc/testcases/难度）
GET  /api/problems/<problemId>                # 单题详情（desc 是完整题面 markdown）
GET  /api/contests/<contestId>/my-submission-quota   # 配额！
POST /api/submissions/submit                  # 提交 {problemId,userId,files:[{path,content}]}
GET  /api/submissions/<id>?userId=<uid>
```
关键前端模块（找接口时用）：`/js/services/shared.js`（含 `/api/contests/name/…`、`/api/problems/name/…`）；
`/js/pages/open.js`（`/public/...` 页面）、`/js/pages/group.js`。

## 三、配额（硬约束）

`GET /api/contests/<cid>/my-submission-quota` → `{"used":49,"limit":50,"reset_at":"2026-11-24T16:00:00Z"}`
- **每日 50 次，次日 00:00 北京时间重置**（16:00Z）。
- 2026-11-24 当天已用 49/50 → **只剩 1 次**。
- 策略：本地验证到位再交；跨天集中交。

## 四、20 道题清单（新题 = `use_baseline=False`，无历史最佳，首月前三都获奖）

**新题（优先）**
| name | 标题 | 难度 | 类型 | 用例 | problemId |
|---|---|---|---|---|---|
| trace | Trace | 1 | vector | 15 | `6aceefd6e769892927123166` |
| confusionmulgrad | ConfusionMulGrad | 2 | vector | 15 | `6acef021e769892927127ff1` |
| multiaddrmsnormdynamicquant | MultiAddRmsNormDynamicQuant | 2 | vector | 15 | `6acef047e76989292712aaae` |
| bit_plane_transform | BitPlaneTransform | 2 | vector/密码学 | 15 | `6aec3d2e1dac5d235b9ba60d` |
| gategateupact | GateGateUpAct | 2 | vector/Kirin(CANN6.1.1) | 14 | `6aedc7dc6bd0f9be37baf2c2` |
| mhcsinkhorngrad | MhcSinkhornGrad | 3 | vector | 15 | `6acef064e76989292712c8e3` |
| quantmatmulsigmoidmulquant | QuantMatmulSigmoidMulQuant | 3 | fusion | 15 | `6acef084e76989292712e8ef` |
| gf2k_mul_square | Gf2kMulSquare | 3 | vector | 15 | `6aec3a341dac5d235b9509a4` |
| gf2_poly | Gf2Poly | 4 | vector | 20 | `6aec3c8d1dac5d235b9a3c50` |
| flashattention | FlashAttention | 5 | fusion | 15 | `6acef0f7e76989292713697f` |
| convtranspose2d | ConvTranspose2D | 5 | fusion | 15 | `6acef213e769892927149a50` |

**老题（`use_baseline=True`，要超榜主性能 3%）**：batch_norm_3d(2)、bntrainingupdate(1)、depthwiseconv2d(5)、group_norm(2)、groupnormsiluquant(2)、index(2)、mseloss(2)、cube1(4)、cube2(4)

## 五、Trace 题面要点（已实现中）

- 输入 `self (M,N)` fp16/fp32 → 输出 `out (1,)` 同 dtype
- 取主对角线 `min(M,N)` 个元素，**FP32 内部累加**求和
- M,N ∈ [1,16384]；支持非方阵
- **fp16 用例仅覆盖对角线长度 ≤2048 且取值 [-1,1]**（长对角线用 fp32）
- 难点：对角线 stride = N+1 的跨步收集（Gather/DataCopyPad 按行取）+ 多核分段归约汇总

## 六、判题机红线（与 S10/mfg 同源，全部实测）

- **507035 = VECTOR_CORE_EXCEPTION**：workspace 申请不足、16bit 标量 cast、**向量 op count=1**（最隐蔽）→ 一律 chunk≥64、向量 op/Gather count≥64、写回用真实 count
- **TBuf 无自动同步**：MTE3→下轮 MTE2、V→MTE2(WAR) 都要 SetFlag/WaitFlag
- **`DataCopyPad` 的 UB 侧地址必须 32B 对齐**
- **大 workspace 申请有 0.7~2.3s 固定开销**、**`SyncAll` 有 ~1.15s** → 能不用就不用；`SyncAll` 必须配 `SetScheduleMode(BATCH_MODE_SCHEDULE=1)` 且禁止部分核早退
- 判题机 **CANN 9.0.0**（本机 9.2.0）：跨版本重排只用 `Gather + CreateVecIndex`（offset **字节**单位）
- **禁止白盒**：不得按用例硬编码；封榜后有泛化性验收测试

## 七、当前进度（2026-11-24）

- `ladder-trace`：实现 Trace 中
- `ladder-marn`：实现 MultiAddRmsNormDynamicQuant 中
- `ladder-cmg`：实现 ConfusionMulGrad 中
- 题面已落盘：`/root/work/ladder_specs/{Trace,MultiAddRmsNormDynamicQuant,ConfusionMulGrad,BitPlaneTransform,GateGateUpAct,BatchNorm3D,BNTrainingUpdate,MseLoss,index,GroupNorm}.md`
- 本地副本：`_sec5_build/ascend_tasks/ladder_work/`

## 七点五、最终交付状态（2026-11-24 23:00 收尾）

### 14 道可提交（全部 zip 已刷新、7 文件无前缀、`ladder_submit.py --list` 全 OK）
| 难度 | 题目 | 关键成果 | Codex 红队 |
|---|---|---|---|
| 5 | flashattention | 26→30 组全过，2.67×（35.4ms @4096） | ✅ 已修（严格掩码 `-FLT_MAX` / 全屏蔽行输出 0 / dtype 契约） |
| 5 | convtranspose2d | **100× 提速**（10.7s→107ms），golden 逐位一致 | ✅ 已修（dtype 契约 + `w.shape[0]==C_in` 校验，非法输入 REJECTED） |
| 4 | gf2poly | 216 组全 PASS，误差 0 | 待审 |
| 3 | gf2k_mul_square | 29→13.9ms | 待审 |
| 3 | mhcsinkhorngrad | 确定性逐字节通过 | 待审 |
| 3 | quantmatmulsigmoidmulquant | 11/11 全绿（单核向量 GEMM，大 shape 会 TLE） | 待审 |
| 2 | multiaddrmsnormdynamicquant | 1.67× | 待审 |
| 2 | confusionmulgrad | **17.6×**（6.02→0.34ms） | 待审 |
| 2 | bit_plane_transform | 51/51 全过 + 64B 哨兵无越界 | ✅ 已修（OOB 拦截 / 方向完整匹配 / 维度校验） |
| 2 | mseloss（老） | 1.4TB/s（HBM 86%） | 待审（注意原子加的确定性条款） |
| 2 | group_norm（老） | ResNet 0.0495ms | 待审 |
| 2 | groupnormsiluquant（老） | 781GB/s | 待审 |
| 2 | index（老） | 600-1300GB/s；榜主 47.84ms，目标 <46.40ms | 待审（**def 只声明 3 种 dtype，需扩**） |
| 1 | trace | 单核确定性 0.0637ms | 待审 |
| 1 | （BatchNorm3D 老题） | 未做 | — |

### 提交脚本
`/tmp/ladder_submit.py`（本地 `_sec5_build/ascend_tasks/s10_work/remote_scripts/ladder_submit.py`）
```bash
python3 /tmp/ladder_submit.py --list      # 校验
python3 /tmp/ladder_submit.py --only fa,ct2d   # 部分提交
python3 /tmp/ladder_submit.py             # 全量（按难度降序）
```

### mfg 侧（12/07 截止）
11 题 Pass + spspmm（f16 `MTE2_S` 竞态修复）+ fouriertransform2d（batch 并行 32×）+ fusedpropstep（25/29，大 real 尺寸 iterations=5 会 trap）；radius 放弃。

### S10 侧
5/5 Pass；reduce 修掉 **`inQ_` depth-1 死锁**（严重 bug）+ P3 路径 19.5×。

## 七点六、Codex 红队评审方法论（重要，下一轮继续用）
- 调用：`codex_review(effort="low", targets=[...])`；**`gpt-6-sol` 不支持 `effort="minimal"`**（会 400）。
- 包大小：单题 ~20-60KB 较稳；一次 3-5 题（160KB+）易 180s 超时 → **一次 1-2 题**。
- **实测价值**：3/3 审过的题都被判「不通过」，且问题**本地对拍 100% 发现不了**：
  1. 因果掩码用 `-1e4` 加法 → 大分数下屏蔽失效（改 `-FLT_MAX`）
  2. q/k/v/mask、x/w **各自独立声明 dtype** 但 kernel 只按一个实例化 → 混合 dtype 越界读
  3. tiling 缺 `w.shape[0]==C_in`、`0≤n≤32G` 等校验 → 越界读写
- **结论：提交前必须逐题过 Codex**，这是抓「判题机随机 WA」最有效的手段。

## 九、实测判题反馈与铁律（2026-11-25 首批 14 道提交）

### 9.1 提交前必须跑的 3 个自检（缺一不可，全是踩出来的）
```bash
python3 /tmp/scan_badwords.py     # ① 禁词：cout/cin/cerr/printf/getenv/system/popen
python3 /tmp/scan_nonascii.py     # ② 非 ASCII（中文注释；保守起见清零）
python3 /tmp/verify_zips.py       # ③ zip 与源码逐文件 md5 一致（防止交了旧包）
```
```bash
python3 /tmp/subpoll.py <name>            # 提交+轮询+逐 case 详情
python3 /tmp/subpoll.py --poll <sid>      # 只查已有提交
python3 /tmp/subpoll.py --detail <sid>
```
- **`cout`/`cin` 变量名会被判「不合规内容」HTTP 400**（连 `int64_t cout = ws.GetDim(1)` 这种 channel-out 也命中）。只有 ct2d 踩了，改 `chOut`/`chIn` 后立刻通过。
- **400 拒绝也消耗配额**（一次 400 = 扣 1 次）。所以**提交前必须先跑上面 3 个自检**，别拿配额试错。
- 429 冷却：`retry_after_seconds` 120~170s，必须按返回值退避；提交间隔 <90s 必 429。

### 9.2 判题机时间上限极宽松
ct2d 有 case 跑 **272 秒仍判 Pass**（case13）、219 秒的 case 只因精度 0.99989 判 WA。→ **不要为了压时间牺牲正确性/精度**，本赛更看重 precision_ratio。

### 9.3 首批结果（13 道提交，1 道被 400 挡住）
| 题 | 结果 | 备注 |
|---|---|---|
| FlashAttention | ✅ Pass 15/15 | 慢 case 1220/1527ms |
| BitPlaneTransform | ✅ Pass 15/15 | |
| GroupNorm | ✅ Pass 15/15 | 老题 |
| MseLoss | ✅ Pass 15/15 | 老题；原子加方案被接受 |
| ConvTranspose2D | ⚠️ 14/15 | **case12 prec=0.99989**（差 0.0001） |
| ConfusionMulGrad | ⚠️ 14/15 | 差 1 个 case |
| GroupNormSiluQuant | ⚠️ 11/15 | int8 1ulp 疑云 |
| MultiAddRmsNormDynamicQuant | ⚠️ 10/15 | **可见 6 case prec 全 1 → 错在隐藏 case** |
| MhcSinkhornGrad | ⚠️ 7/15 | 287ms case |
| QuantMatmulSigmoidMulQuant | ❌ RE 8/15 | **36.5s → 超时**，需多核切 M |
| Index | ❌ RE 0/9 | dtype 扩展改动引入回归，需回退基线 |

### 9.4 Trace（fix-e）
3 轮 Codex 评审修了 **9 个真 bug**，含：
- 32B 窗口越过张量末尾（fp32 N≤5 / fp16 N≤13 方阵）
- `srcStride = (N+1)*sizeof(T)-32` 在行跨度 <32B 时下溢成 4G 级
- `Duplicate(g[rows],0,64)` 的 dst 偏移 24/156/252B **非 32B 对齐 → 507035 且整卡进错误态**
验收：官方 golden MERE=MARE=0，69 个检查点全过，16384² = 0.0635ms

## 十、当前状态（2026-11-25 01:35）

- 配额：`used≈18/50`（含 400 浪费）
- **4 道 Pass**（FlashAttention / BitPlaneTransform / GroupNorm / MseLoss）
- 7 道待修（均有明确失败定位）
- 3 个 agent（`it1`/`it2`/`it3`）在按判题反馈修
- Trace 审完待提交



- `code_template = "gitcode_template_noopapi"` → 工程模板来自 GitCode 该模板仓
- 提交 = `files:[{path,content}]`（同 S10 的 7 文件结构：`op_host/{def,infershape,tiling}.cpp` + `op_kernel/{x.cpp,x.h,x_tiling_data.h,x_tiling_key.h}`）
- 用户确认：「在 cannjudge 页面打包上传工程」

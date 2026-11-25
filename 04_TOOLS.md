# 04 · 工具脚本说明

全部在 `tools/`。**在远端开发机（Linux）或本地 Windows 都能跑**，只需 Python 3（无第三方依赖）。

## 一、提交前必跑的三件套

```bash
python tools/scan_badwords.py     # ① 违规标识符（cout/cin/printf/getenv/system/popen）
python tools/verify_zips.py       # ② zip 与源码逐文件 md5 一致
python tools/verify_handoff.py    # ③ 本包全部 zip 结构校验（条目数/前缀/md5/违规）
```
> `verify_zips.py` 里写死了它认识的 14 个 ladder zip 路径，**换目录用时改里面的 `SETS` 表**。
> `verify_handoff.py` 是通用版（扫 `_handoff_repo/*/zips/*.zip`），建议直接用这个。

## 二、提交与轮询

| 脚本 | 用途 |
|---|---|
| `subpoll.py <name>` | 提交 + 轮询 + 打印逐 case 结果（**最常用**） |
| `subpoll.py --poll <sid>` | 只轮询某次提交 |
| `subpoll.py --detail <sid>` | 只打印逐 case 详情 |
| `sub_zip.py <zip> <problemId> <tag>` | **从任意 zip 提交**（A/B 变体专用），自动处理 429 |
| `one_detail.py <sid>` | 打印完整 `result`（含 `msg` 真机 profiling 日志） |
| `exact_prec.py <sid>` | 打印**精确** `precision_ratio`（`repr()` 全精度）+ time + best_time |
| `fetch_all_scores.py` | 从 ranking 接口拉全部 38 题成绩 → `handoff_scores.json` |

### 怎么用 `subpoll.py`
先改脚本里的 `MAP`（name → problemId, zip 路径），然后：
```bash
python tools/subpoll.py trace          # 提交并等结果
python tools/subpoll.py --poll 6b05caeb85ad43bbe07fecad
```

## 三、打包 / 修复类

| 脚本 | 用途 |
|---|---|
| `repack_zip.py` | 按源码目录重打包 7 文件 zip（无前缀 → 带 `op_host/` `op_kernel/` 前缀） |
| `strip_nonascii.py` | 清掉全部 zip 里的非 ASCII 字符并重打包（**实测非必需**，中文注释不影响判题） |
| `build_review_bundle.py` | 把 7 文件 + NOTES 拼成给 Codex 评审用的 md |
| `verify_zips.py` | 14 个 ladder zip 与源码一致性 |

## 四、Codex 红队评审（强烈建议沿用）

本赛最有效的质量手段：**提交前让 Codex 逐题审一遍**。调用方式：

```
codex_review(effort="low", targets=["<某题的 md 评审包>"], criteria="...")
```

**实测价值：审过的题 100% 都被判「不通过」，且问题本地对拍全都发现不了**：

| 题 | Codex 抓到的真问题 |
|---|---|
| ConvTranspose2D | x/w dtype 各自声明但 kernel 只按 x 实例化 → 混合 dtype 越界读；缺 `w.shape[0]==C_in` 校验 |
| BitPlaneTransform | inverse 的 `n` 未校验 → 越界读（`[1,8,4]` + `n=64`） |
| FlashAttention | 因果掩码用 `-1e4` **加法** → 大分数下屏蔽失效；全屏蔽行 NaN；q/k/v/mask dtype 契约未落实 |
| Index | def 只声明 3 种 dtype，题面允许 7 种 → 隐藏用例进不来 |
| MhcSinkhornGrad | `numIters` 被 sumOut 覆盖且无校验 → 轮数静默缩小；kernel 把非法 tiling 改写成"最小合法用例" |
| Gf2kMulSquare / Gf2Poly | `k∉{8,128}` 被静默改成 8；fp16 死分支 |

**使用注意**：
- `gpt-6-sol` **不支持 `effort="minimal"`**（会 400），用 `low`
- **一次只审 1 个文件**；一次 2 个以上（>60KB）容易 180s 超时
- Codex 与「官方 golden 的实际行为」冲突时，**以 golden 为准**，并在 NOTES 写明理由
  （例：`index` 的索引语义，Codex 认为是扁平下标，但官方 `gen_data.py` 证实是"同下标"）

## 五、建议的工作流

```
1. 读题面 + 拉官方模板（含 gen_data.py）
2. 读 gen_data.py 确认 golden 框架（numpy? torch?）→ 决定归零顺序策略
3. 写 kernel；clean 重建；真机对拍（官方 golden 逐位）
4. 打包 7 文件 zip（带前缀）
5. Codex 红队评审 → 修 → 重验
6. 跑三件套自检
7. 提交 → 拉 result[i].msg → 按失败 case 定位 → 回到 3
```

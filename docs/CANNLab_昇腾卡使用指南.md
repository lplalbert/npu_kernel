# CANNLab 昇腾卡（910B3）使用指南

> 供其他会话直接使用。卡片归属：gitcode CANNLab（AtomGit 账号 lplalbert 的开发环境）。
> 最后核验：2026-11-23，实例 DevEnvC_cz0xe 全链路实测通过。

## 30 秒上手

```bash
# 1. 直连（SSH 免密，通道已就绪）
ssh -o BatchMode=yes devenvc_cz0xe.9c45ee71438b4c518bbc4c1af378d1ed.atomgit.0 "echo ok"

# 2. 传文件（tar 管道，比 scp 目录快）
cd /你的工程目录 && tar czf - 工程名 | ssh devenvc_cz0xe.9c45ee71438b4c518bbc4c1af378d1ed.atomgit.0 "tar xzf - -C /mnt/workspace"

# 3. 带环境变量跑任务（一行模板）
ssh devenvc_cz0xe.9c45ee71438b4c518bbc4c1af378d1ed.atomgit.0 \
  "source /home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1; \
   export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:\$LD_LIBRARY_PATH; \
   cd /mnt/workspace/工程名 && <你的命令>"
```

## 连接方式（按推荐排序）

| 方式 | 入口 | 说明 |
|---|---|---|
| **SSH 直连**（推荐） | `ssh devenvc_cz0xe.9c45ee71438b4c518bbc4c1af378d1ed.atomgit.0` | AtomGit Devenv 插件写入 `~/.ssh/config` 的 ATOMGIT Include 区（**勿动该区域**），走本地代理端口转发；`BatchMode=yes` 直接免密通；scp/tar 随便用 |
| WebIDE（备用） | `https://dev.gitcode.com/devdesktop/webide/instance/<实例id>?folder=/mnt/workspace` | 浏览器用；若要程序化驱动（如 REA）注意 **gitcode 页有 Service Worker，场景须 `environment.service_workers: "allow"`，否则捕获直接被拒** |
| VS Code Remote-SSH | 主机名同 SSH | 能连（日志 `Exec server created` 就是成功），但 aarch64 是预览版、UI 慢，日常别用 |

**注意**：实例重建/换名后 SSH 主机名会变（`devenvc_<实例名>.<新实例id>.atomgit.0`）——届时向用户要新连接串，或看 `~/.ssh/config` Include 的 `C:\Users\24976\.atomgitdevenv\.ssh\config`。

## 环境规格与布局

| 项 | 值 |
|---|---|
| NPU | **1× Ascend 910B3**（64GB HBM）——注意是 **910B3 不是 910C**，算子编译目标 `ascend910b` |
| CPU/内存 | 16 vCPU / 32GiB，aarch64 Ubuntu 20.04 |
| CANN | **9.0.0**，`/home/developer/Ascend/ascend-toolkit`（`source .../set_env.sh`；**用户可写**，无 opp/vendors 权限坑） |
| Driver | `/usr/local/Ascend/driver`（见下方 npu-smi 坑） |
| Python | 3.11.4，numpy 1.26.4、torch 2.7.1+cpu（跑参考实现够用）、ml_dtypes |
| 构建 | cmake 3.20.5、g++ 9.4（C++17 OK） |
| 磁盘 | `/mnt/workspace` 196G（**关机持久化**）、`/home` 196G |
| 设备号 | `DEVICE_ID=0`（实测可用；npu-smi 显示 "NPU 2" 是物理编号，别被带偏） |
| UT 依赖 | googletest 1.14 + nlohmann json.hpp **已在 `~/deps/` 下载好**（`GTEST_SOURCE_DIR=$HOME/deps/googletest-1.14.0 NLOHMANN_JSON_INCLUDE_DIR=$HOME/deps`） |

## 算子工程（S10 题型）构建/测试

```bash
cd /mnt/workspace/<op工程>
export GTEST_SOURCE_DIR=$HOME/deps/googletest-1.14.0
export NLOHMANN_JSON_INCLUDE_DIR=$HOME/deps
bash build.sh -j8        # 编译（op_host+op_kernel，产出 custom_opp_*.run）
bash build.sh -j8 -u     # CPU UT（tikicpulib 模拟器，不需要卡）
bash build.sh -j8 -e     # NPU example（真卡跑 sample.py generate→run→verify）
```

工程模板固定 38 个文件；**只有 7 个是可编辑/可提交的**（见下）。工程整体解法示范见 `_sec5_build/ascend_tasks/s10_work/mish/`（已判题 Pass）。

## CANNJudge 提交（算子挑战赛判题）

```bash
# 提交（只能交 editable 文件！build.sh/tests 等交了会报「文件不可编辑」）
curl -s -X POST "https://cannjudge.cn/api/submissions/submit" \
  --proxy http://127.0.0.1:7892 \
  -H "Content-Type: application/json" \
  -H "Cookie: cannjudge_auth=<JWT>" \
  -d '{"problemId":"<题目objectId>","userId":"<MY_USER_ID>","files":[{"path":"op_kernel/xx.h","content":"..."}]}'

# 查分（轮询到 status 不是 Running/Pending 为止）
curl -s "https://cannjudge.cn/api/submissions/<id>?userId=<MY_USER_ID>" \
  --proxy http://127.0.0.1:7892 -H "Cookie: cannjudge_auth=<JWT>"
```

- **JWT**（cannjudge_auth cookie 值）：`<CANNJUDGE_JWT_REDACTED>` —— **有效期至 2026-11-29**，过期让用户在浏览器 DevTools→Application→Cookies→cannjudge.cn 里重取
- editable 7 件套：`op_kernel/{mish.cpp,mish.h,mish_tiling_data.h,mish_tiling_key.h}` + `op_host/{mish_def.cpp,mish_infershape.cpp,mish_tiling.cpp}`（换题同构，文件名随算子名变）
- 判题结果字段：`status`（Pass / Time Limit Exceeded / …）、`result[]` 每例 `testcase_status`（Pass/Hidden）、`time`、`precision_ratio`（要 1.0）、`best_time`（榜首参考）；**Hidden 用例 45 个不外显细节，但决定最终 status**
- 提交额度 10 次/天；每次判题约 8~12 分钟；题目列表/题面拉法见 `memory/s10-mish-campaign.md`

## 额度与生命周期

- 额度：**A2/A3 共享 100 卡时** + CPU 720 核时（页面 `gitcode.com/org/cann/cannlab/environment` 可看用量）；另有 260 CANN 积分可兑换
- **不用时在列表页点「关机」**（关机不烧卡时；/mnt/workspace 与 /home 数据保留）
- 更多菜单：重启 / CANN包升级 / 删除。创建时选 **NPU A2**（910B3）——A3 是 910C，与 ascend910b 编译目标不符

## 大坑速查

| 坑 | 解法 |
|---|---|
| 朴素 `DataCopy(local,global,count)` 要求 count×sizeof(T) **32B 对齐**：CPU 模拟器 abort，**真 NPU 静默向下取整丢数据** | 变长一律 `DataCopyPad`；性能写法=tiling 把块对齐 32B + 尾块才走 Pad |
| `npu-smi: libc_sec.so cannot open` | `export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH` |
| `DataCopy(..., DataCopyExtParams)` 编译不过 | CANN 9.x 只有 `DataCopyPad`：GM→UB 4 参（带 `DataCopyPadExtParams<T>{false,0,0,0}`），UB→GM 3 参 |
| `ASCENDC_TPL_UINT_DECL` 报位宽不足 | 第 2 个参数是 bitWidth，须 ≥ 模式数的 2^（如 3 模式要 2） |
| tiling 里取输入 shape | `context->GetInputShape(0)` 是 `StorageShape*`，用 `->GetStorageShape()` 取 `const Shape&` |
| Jev pod（Model Agent）**时长制会回收** | 只当应急；开发一律用本卡 |

## 与本卡相关的其他情报

- 算子题完整解法示范 + 判题 API 细节：`memory/s10-mish-campaign.md`
- Jev pod 通道（已过期资源，历史参考）：`_sec5_build/ascend_tasks/jev_work/`（含 REA 浏览器自动化场景库）

# 08 · Ascend C 算子开发经验手册（可复用）

> 本手册是三条比赛线（天梯赛 18 题 / mfg 15 题 / S10 5 题，共 38 个算子）的实战沉淀。
> 配套：`kb/` 下是**官方 Ascend C 优化知识库**（718 篇 md，含 176 张编号优化卡片 + 9 篇失败模式），来自 `cannbot` 仓库。

---

## 零、开工前必做的事

1. **拿官方模板**：`GET /api/problems/<pid>/template` → 含 `build.sh`/`CMakeLists.txt`/`examples/`/`tests/`（**`tests/ut/op_kernel/<base>_data/gen_data.py` 就是判题 golden**）
2. **读 `gen_data.py`**：确认 golden 用什么框架（numpy? torch?）、dtype、累加/舍入方式 —— **这一步决定成败**
3. **在服务器上先 clone 这份知识库**：`git clone https://gitcode.com/cann/cannbot-knowledge.git`
   查优化卡片：`knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/`（176 张）
4. **查 CANN 内置同类算子源码**：`/usr/local/Ascend/cann-*/opp/built-in/op_impl/ai_core/tbe/impl/`
   —— 别人提交的代码是 403，但**官方实现就在盘上**（`quant_batch_matmul_v3` 676KB、`detect_flash_attention_score` 424KB …）

---

## 一、提交合规（会 400 且扣配额）

| 坑 | 事实 |
|---|---|
| 违规标识符 | `cout`/`cin` 作**变量名**就会命中「不合规内容」→ 改 `chOut`/`chIn`；`printf`/`getenv`/`system`/`popen` 不用 |
| 注释里的英文单词 | `system` 等出现在**注释**里不触发（只针对代码标识符） |
| 400 也扣配额 | **提交前必跑 `scan_badwords.py`** |
| zip 格式 | 7 条目、带 `op_host/` `op_kernel/` 前缀、**无工程名前缀**；mfg 是 4 条目 |
| zip vs 源码 | 提交前跑 `verify_zips.py`（逐文件 md5）；**文件带 BOM 会导致 md5 不一致** |
| 429 冷却 | 间隔 <90s 必 429，按 `retry_after_seconds`（120~170s）退避 |
| 返回体 | `submissionId` 在 **`data.submissionId`**（不是 `_id`/顶层） |

---

## 二、精度：本赛最大的坑

### 2.1 判据（实测逆向）
```
precision_ratio = 1 − #(rel_err > 10 × threshold) / N
rel_err = |out − golden| / (|golden| + 1e-7)
fp32 threshold = 2^-13 ; fp16 阈值 ≈ 10·2^-10 ; bf16 更宽
```
**42 个 Pass 队伍的 ratio 全部恰好 = 1.0；WA 队伍卡在 0.996~0.999**
⇒ **Pass 门槛实质要求「零 mismatch」** ⇒ 量化/归约类算子**只有逐位复刻 golden 才能过**。

### 2.2 核心原则：**「提高精度」是错的，「复刻 golden 的归约顺序」才是对的**
实测：fp64 真值算出来比纯升序 fp32 **错得更多**（`n4_T32768`：fp64 错 148 个元素，升序错 0 个）。
因为 golden 本身就是某个特定舍入顺序的产物。

### 2.3 已破译的 golden 顺序（可直接复用）
见 `03_GOLDEN_ORDER.md` 与 `docs/notes/TORCH_REDUCTION_ORDER.md`：
- **numpy 1.26**：8192 分块 + 块内 Parker pairwise（8 交错累加器）
- **numpy 2.4.6**：同上（`group_norm_silu_quant` 靠它做到逐位）
- **torch 2.7.1 CPU `sum(dim)`**：`Vectorized<float>::size()=4`；block=16 元素；4 个向量累加器；
  **base-16 进位级联**（每 16/256/4096 block）；N<288 退化为单级 U4acc（288=18 block 首次触发级联）
- **torch `conv_transpose2d`**：per-tap（kh,kw 升序）ci 升序点积**带 FMA** → tap 升序 → **bias 最后**
- **depthwise conv**：每 tap **`Muls`+`Add` 两步（禁 FMA）** ← 与 conv_transpose 相反！**必须逐题实测**

### 2.4 舍入相关的硬件坑
| 坑 | 对策 |
|---|---|
| 编译器把 `r[j] += d*d` 融合成 FMA（少一次舍入） | `volatile float sq = d*d; r[j] = r[j] + sq;` |
| 向量 `Div` **不是 IEEE 正确舍入**（偶发 1 ulp） | 精度敏感处用**标量 IEEE 除法** |
| `Axpy`/`MulAddDst` **都是两次舍入**（非融合 MAC） | 若 golden 是两次舍入 → 用 `Muls`+`Add`，**禁用 `Axpy`** |
| `Cast(int16→int8)` 是**空实现** | 换路径 |
| `Cast(int32→half)` 走 vconv_deq，roundMode 被忽略 | 需 `SetDeqScale` |
| `GlobalTensor::SetValue` 写 GM **不回刷** | 先写 UB 再 `DataCopyPad` |
| `__aicore__` 内禁 double 强转 | 用 `1.0f/(float)(n)` |
| `__aicore__` 内无 `sqrtf` | 只有 `sqrt()` |

---

## 三、硬件/同步坑（按踩坑代价排序）

### 3.1 会直接崩的
| 坑 | 错误码 | 对策 |
|---|---|---|
| 向量 op `count=1` | 507035 | count 必须 **≥64 且 64 的倍数** |
| 非 64 倍数 count | 慢 20~28× | 同上 |
| `Duplicate`/`Select` 的 **UB 偏移非 32B 对齐** | 507035 + **整卡进错误态** | 偏移必须 32B 对齐 |
| `DataCopy(Local→Local)` 在 **CANN 9.0 不支持** | 507035 | 用 `TQueBind` 直通 GM↔UB |
| **同进程第 2 次 launch 的 workspace 假地址** | 507035 | **用 `AscendC::GetUserWorkspace(workspace)`**（别自算 sysWs 偏移！） |
| `DataCopyPad` 的 UB 侧地址非 32B 对齐 | 507035 | 对齐；`rightPadding ≤ 32B` |

### 3.2 同步事件（最容易漏，本机概率过、判题必现）
```cpp
DataCopyPad(...);                                   // MTE2
SetFlag<HardEvent::MTE2_S>(); WaitFlag<MTE2_S>();   // 标量读前（必须！）
SetFlag<HardEvent::MTE2_V>(); WaitFlag<MTE2_V>();   // Cast/向量读前（必须！）
Cast(...);
SetFlag<HardEvent::V_MTE3>(); WaitFlag<V_MTE3>();
DataCopyPad(gm, ...);
```
- **`DataCopyPad` 后同时有标量读和 Cast 读 → 两个事件都要给**（只靠 `MTE2_V→V_S` 传递不保证）
- 标量写 UB 后给向量/MTE3 用 → 要 `S_V` / `S_MTE3`
- **写后写冲突（WAW/WAR）**：`Duplicate`（向量写）与 `SetValue`（标量写）同地址时，`PipeBarrier` 放错位置会失效
- **`TBuf` 没有自动同步**

### 3.3 多核
- **`SyncAll` / 跨核栅障在本机不可靠**（507035/507057）；`SetScheduleMode(BATCH_MODE_SCHEDULE=1)` A/B 实测无改善
- **安全范式**：行/样本级切分，每单元一个核完整负责，**无跨核归约** → 天然确定性
- 若必须归约：fp32/int32 用 `SetAtomicAdd` + `(P0−g)` 抵消；或 GM 计数器轮询（dot/trace 验证过可行）

### 3.4 `DataCopyPad` 的单位（最容易搞错）
- **GM 侧 stride = 字节**
- **UB 侧 `dstStride` = 32B 单位**
- `blockLen < 32B` 时，UB 内**按 32B 槽摆放**（会导致 layout 错位！）
- `LocalTensor[i]` 返回**切片不是标量** → 标量必须 `GetValue(i)`

### 3.5 UB 预算
- **必须用 `GetCoreMemSize(UB)` 实测**，不要假设 192KB（cube2 因多申请一块 UB → 判题 15/15 全 TLE）
- 预算按 0.7~0.9 留余量

---

## 四、Cube / Matmul 高阶 API

### 4.1 关键认知（推翻了早期结论）
- CANN 9.0 **完整支持** `AscendC::Matmul`，**必须用 MIX AIC/AIV 内核**
- 早期"输出全 0"的真因：kernel 是**纯 AIV `__global__`**，Cube server 没启动 —— **不是版本问题**

### 4.2 最小可用配方（已验证）
```cpp
extern "C" __global__ __aicore__ void op(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);      // ★ 必须
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.mm);  // 全局函数，不加 AscendC::
    ...
}
```
- host：`#include "tiling_api.h"`；`matmul_tiling::MatmulApiTiling t(pi);`
  `SetAType/SetBType/SetCType/SetBiasType` + **5 参 `SetOrgShape(M,N,K, orgKa, orgKb, orgKc)`**
  （**3 参版内部 orgKb=K，而 orgKb 是 B 行 stride，16bit 溢出 → k≥65536 时 B 全错**）
  或开 `GetMDLConfig(true)` 的 `intrinsicsCheck`
- workspace = `GetLibApiWorkSpaceSize() + 用户区`；用户区用 **`AscendC::GetUserWorkspace(workspace)`**
- **MIX 1:2 标准握手**（照官方 `BmmBasicDequantBf16`）：
  AIC `Iterate() + GetTensorC(ws)` → `NotifyEvent<PIPE_FIX>`；
  AIV `WaitEvent` → `if(GetSubBlockIdx()==0)` 计算 → `NotifyEvent<PIPE_MTE2>`；
  **`End()` 必须补 ping/pong WaitEvent（漏了会 hang）**
- int8 Mmad 是**精确 int32 累加** → bit-exact 可达

### 4.3 性能
- `MatmulApiTiling` 是**单核** API；要多吃核用 `MultiCoreMatmulTiling`
- 内存墙：单核 MTE2 ~20GB/s。按 M/N 切分时每核读全 B 或全 A
- L2 复用：`p105_matmul_l2_cache_tile_sizing`、`p107_matmul_serpentine_traversal`（见 `kb/`）
- epilogue 的**每行固定开销**很致命（一次 GM 标量读 ~800ns/行）→ 按 64 行一块用 `DataCopyPad` 预取到 UB 再读

---

## 五、调试与诊断手法

| 手法 | 用途 |
|---|---|
| **`one_detail.py <sid>`** | 拉判题 `result[i].msg`，含真机 profiling（Task Duration / Block Dim / 错误码） |
| **错误码速查** | `507015`=MTE DDR out of range（UB/workspace 越界）· `507035`=vector core exception · `507046`=3 秒看门狗超时 · `507057`=跨核栅障 |
| **`precision_ratio` 反推形状** | `1−r = k/E`（k=坏元素数，E=该输出元素数）。精确分数可解出 E 与 k |
| **Canary 实验** | `Process()` 开头直接往输出写常量 → 判别「值错」vs「写不到」 |
| **「输出下标」技巧** | 把算出的源下标写进输出 → 一次提交读出地址映射对不对 |
| **逐层隔离** | 把设备自己算出的中间量喂回 host 复算 → 判断是自己 bug 还是 golden 的锅 |
| **逐 case 耗时并排对比** | 定位"哪里没优化到"最快的方式（如 transpose case 零提速） |
| **官方知识库** | `kb/` 下 176 张优化卡片 + 9 篇失败模式，先搜再写 |

### 错误码 / 现象 → 最可能原因
| 现象 | 最可能原因 |
|---|---|
| 全部 case `time=0` | 首次 launch 就崩（不是慢）：UB 越界 / op 加载失败 / 看门狗 |
| 单个 case `time=0` + 后续 Skipped | 该 case hang（死循环/死锁） |
| prec 0.99x 恒不过 | 归约顺序不对（近零元素 1 ulp） |
| prec 0.5 左右 | 输出整体错（地址/布局错，不是精度） |
| 本地过判题挂 | 见下 |

### 「本地对拍全过、判题挂」的三大原因
1. **本地 bench 路径 ≠ 判题 harness** → 必须用模板 `examples/test_aclnn_*.cpp`
2. **没测判题规模的大 shape** → 小 shape 静默走标量路径，掩盖向量路径 bug
3. **UB/workspace 预算假设不同** → 实测 `GetCoreMemSize(UB)`

---

## 六、性能优化套路（按算子类型）

| 类型 | 套路 | 实例 |
|---|---|---|
| **elementwise / norm** | ①去逐 tile `PipeBarrier<PIPE_ALL>`；②`TQue<VECIN,2>/<VECOUT,2>` 双缓冲；③逐 tile `ReduceSum` → 跨 tile 累加器 + 一次 ReduceSum；④chunk 加大 | group_norm 24→**44.8**；bn 19→**24.6** |
| **reduction** | 多核分段 + 原子加 `(P0−g)` 抵消；`WholeReduceSum` | trace →**26.7**（6~7×） |
| **conv (depthwise)** | 一次 R 个输出行打包成一条向量（摊薄每行 ~22ns 固定开销） | depthwise 12.6→**17.3** |
| **conv (transpose/普通)** | **必须 im2col + Cube GEMM**；向量路线差 100~2700× | ct2d 三次 tiling 修补无效，确诊架构差距 |
| **matmul / GEMM** | MIX Matmul（见第四节） | cube1 case12 **368×**、cube2 case7 **41×** |
| **量化 matmul** | 官方 `AscendDequant()`；transB 用 template bool（**不要标量回退**）；Fixpipe 融合可省 workspace 往返 | cube1/cube2 15/15 |
| **bit 操作** | `And/Or/Xor` 只支持 **int16/uint16**（uint32 需 2× count）；`Muls/Adds` 支持 half/float/int16/int32；`TransDataTo5HD` 支持 int8/uint8 | bpt 最大 case 2× |
| **attention** | 读官方 `detect_flash_attention_score` + `p120_fa_vec_dot_to_cube_mmad` | 进行中 |

---

## 七、判题机 vs 本地环境差异（最贵的坑来源）

| 项 | 本地 910B2 (CANN 9.2) | 判题机 (CANN 9.0) |
|---|---|---|
| `DataCopy(Local→Local)` | 容忍 | **不支持**（507035） |
| UB 大小 | 192KB | 可能更小 → **必须实测** |
| torch / numpy 版本 | 2.5.1 / 1.26 | **2.7.1 / 2.4.6**（归约顺序不同！） |
| 看门狗 | 无 | **3 秒**（cube 类 harness）；但其它算子有 272s 仍 Pass 的 |

> **结论：尽量用与判题机同版本的 CANN 环境做验证**（本项目最终换到一台 CANN 9.0.0 的机器后，很多"本地过判题挂"的问题才得以定位）。

---

## 八、知识库索引（`kb/` 下）

```
kb/contrib/ascendc/
├── index.md
├── optimizations/evolution_strategies/
│   ├── cards/                176 张编号优化卡片
│   │   ├── a1_..a8_, d1_..d5_   精度/舍入/混合精度/bf16
│   │   ├── p10_..p124_          性能优化（matmul/reduction/fa/transpose/...）
│   │   └── ...
│   └── failure_patterns/     9 篇已知失败模式（f1~f9）
└── runbooks/performance/     同一套卡片的 runbook 版
```

**最常用的几篇**
| 编号 | 主题 |
|---|---|
| `p104` / `p105` / `p107` | Matmul 多核 tiling 扩展 / L2 cache tile / 蛇形遍历 |
| `p117` | Matmul API tiling host 配方 |
| `p115` / `p119` / `p120` / `p121` | FlashAttention tiling / online softmax / **向量点积换 Cube Mmad** / CV 流水 |
| `p114` | Transpose ND tiling + 对齐 |
| `p100` / `p101` | GatherMask 替代 Gather |
| `p109` / `p110` | Reduction 多轴 tiling / UB 累加器复用 |
| `p11` | 尾块处理 |
| `a1` / `a3` / `a5` / `a8` | fp32 中间量 / 舍入模式 / softmax 稳定性 / 量化精度 |
| `f2` / `f3` / `f4` | Matmul 多 kernel 注册破坏 / 手写 Mmad 16×16 陷阱 / L0A-L0B 事件窗口冲突 |

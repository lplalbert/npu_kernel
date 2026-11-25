# 02 · 硬件与判题机坑（全部实测，按踩坑代价排序）

> 环境对照：
> - **本地开发机**：910B2 / 910B3，单芯片 **24 AI 核**，64GB HBM，CANN **9.2.0**，`DEVICE_ID=0`
> - **判题机**：Ascend 910B（**Block Dim = 40 AIV 核**），CANN **9.0.0**
> **两者行为差异是本赛最大的坑来源。** 凡是「本机能过、判题机挂」的，多半在这张表里。

---

## A. 会直接导致判题挂掉的（最高优先级）

### A1. 判据是 `rel<1e-4` **且** `abs<1e-4`
不是 numpy `allclose` 的 `atol + rtol*|b|`，也不是 `或`。→ **近零输出元素差 1 ulp 就挂**。
后果：`precision_ratio` 常年卡在 0.999x 上不去，看着"快过了"其实结构上过不了。
**正确做法**：逐位复刻 golden 的归约/舍入顺序（见 `03_GOLDEN_ORDER.md`）。

### A2. 提交内容含违规标识符 → HTTP 400 且**照样扣配额**
被拒的返回：`{"message":"提交代码中存在不合规内容，请检查并删除后提交。如需Debug请在本地进行。"}`
**实测命中**：把 `cout`（channel-out）/ `cin`（channel-in）当**变量名**就会触发。
> `int64_t cout = ws.GetDim(1);` ← 这一行就会让整个提交被拒。
**对策**：统一改名 `chOut` / `chIn`；`printf` / `getenv` / `system` / `popen` 一律不用。
> 注：注释里出现英文单词 `system`（如 "reserved system area"）不会触发——只针对代码标识符/调用。

### A3. `DataCopy(LocalTensor → LocalTensor)` 在 **CANN 9.0 不受支持**
本机 9.2 容忍，判题机 9.0 直接 `507035 ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`。
**对策**：纯搬运算子用 `TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2>` 复用同一块 UB，
路径改为 `AllocTensor → DataCopyPad(GM→UB) → EnQue/DeQue → DataCopyPad(UB→GM)`，**删掉所有 local→local 拷贝**。

### A4. **同进程内 `workspace` 只能安全访问一次**
`AscendC::GetUserWorkspace(ws)` = `__get_kfc_workspace_addr() + RESERVED_WORKSPACE(16MiB)`，
而 **kfc 基址只在同一 executor 的第一次 launch 写入**。实测把指针编码写进输出再解码：

| 同进程第几次 launch | kernel 收到的 workspace |
|---|---|
| #1 | `0x12c042200000`（真实有效） |
| #2、#3 | `0x01000000`（= 0 + 16MiB，**假地址**） |

→ 第 2 次起任何读写都命中假地址 → **507035**。判题 `iterations:5` **必踩**。
**对策**（二选一）：
1. **完全不用 workspace**（首选；行/样本级切分天然不需要）；
2. kernel 里先校验 `if ((p >> 32) == 0) return;` —— 假地址直接 no-op，**绝不写假地址**。
**另一条被封死的路**：host 侧 `aclrtMalloc` 一块 device 暂存再把指针经 tiling 传给 kernel ——
**`opbuild` 对 tiling 源码里任何 `aclrtMalloc` 引用一律 `opbuild ops error`**（直接调 / `#include <acl/acl.h>` / 函数指针 / 藏进头文件都不行）。

### A5. 向量 op 的 `count` 必须 **≥ 64 且为 64 的倍数**
- `count = 1` → **必崩 507035**
- 非 64 倍数 → **慢 20~28×**（不是慢一点）
- `Compare` / `Select` 这类 mask op **不处理 `count % 64` 的尾巴**，尾部要单独处理

### A6. `Duplicate` / `Select` 的 **UB 偏移必须 32B 对齐**
`Duplicate(g[rows], 0, 64)` 当 `rows = 6/39/63` 时 dst 偏移是 24/156/252 字节 → **立刻 507035，
而且整卡进入错误态，后续 launch 全挂**（不只是这一次失败）。
**对策**：改成「清零源槽位 + 覆盖式 `DataCopyPad`」。

### A7. `DataCopyPad` 的两个对齐/单位坑
- **UB 侧地址必须 32B 对齐**；`rightPadding ≤ 32B`
- **stride 单位不统一**：**GM 侧 stride = 字节**，**UB 侧 `dstStride` = 32B 单位**
- 尾块处理：`DataCopyPad` 的 `blockLen` 按字节，`blockCount`/`srcStride` 按行

### A8. `LocalTensor[i]` 返回的是**切片不是标量**
标量读取必须 `GetValue(i)`。`t[i] * w` 这类写法编译期不报错但语义错。

---

## B. 同步 / 事件类（最容易漏，且难复现）

### B1. `DataCopyPad` 之后，**同时**有标量读和 Cast 读时，必须两个事件都给
```cpp
DataCopyPad(...);          // MTE2
SetFlag<HardEvent::MTE2_S>(); WaitFlag<HardEvent::MTE2_S>();   // 标量读 ai64
SetFlag<HardEvent::MTE2_V>(); WaitFlag<HardEvent::MTE2_V>();   // Cast 读 av
Cast(...);
```
只给 `MTE2_V → V_S` 传递**不保证**标量读安全 → 本机概率性通过、判题机必现（实测 spspmm 因此 WA 14%）。

### B2. 标量写 UB 后给向量/MTE3 用，要 `S_V` / `S_MTE3`
```cpp
ev.SetValue(t, ap[j]);                                   // 标量写 UB
SetFlag<HardEvent::S_V>(); WaitFlag<HardEvent::S_V>();    // 向量读前
... Muls(dst, ev, 1.0f, n) ...                           // V 读 ev
SetFlag<HardEvent::V_MTE3>(); WaitFlag<HardEvent::V_MTE3>();
DataCopyPad(gm, dst, ...);                               // MTE3 读 V 产物
```
**反面教材**：f32 分支只做了 `V_MTE3`，`ev` 仍是标量产物 → 9.2 能过、9.0 掉值（每行末尾丢几个值）。

### B3. 写后写冲突（WAR/WAW）
`Duplicate(sc, 0, 64)`（向量写）与 `sc.SetValue(0, sum)`（标量写）**写同一地址**时，
`PipeBarrier<PIPE_ALL>` 只放在 `SetValue` **之后**是不够的 —— 必须覆盖 V→S 的写后写冲突，
否则偶发读到旧值（实测导致 rstd 偶发算成 `1/sqrt(eps)`）。

### B4. `TBuf` **没有自动同步**，全部要手写 `SetFlag/WaitFlag`

---

## C. CPU 侧数值坑（精度类算子必看）

### C1. 编译器会把 `r[j] += d*d` **融合成 FMA**（少一次舍入）
**对策**：`volatile float sq = d * d; r[j] = r[j] + sq;`
> 这一条单独就消掉了 group_norm_silu_quant 的 4~9 个 int8 翻转。

### C2. 向量 `Div` **不是 IEEE 正确舍入**（偶发差 1 ulp）
精度敏感处（尤其商接近半整数时）改用**标量 IEEE 除法**。
> 实例：`u/yScale = 63.5` 正好踩 RNE 边界，向量 Div 差 1 ulp 把 64 翻成 63。

### C3. AscendC 的 `Axpy` / `MulAddDst` **都是两次舍入（非融合 MAC）**
→ 无法逐位复刻 x86/aarch64 torch 的 **FMA** 单次舍入结果（残差 1~2 ulp）。
**反过来**：如果 golden 本身就是两次舍入（`round(x*w)` 再 `round(acc+p)`），
那就必须用 `Muls` + `Add` 两步、**禁用 `Axpy`**（depthwise_conv2_d 115/115 bit-exact 的关键）。

### C4. `Cast` 的两个静默失败
- `Cast(int16 → int8)` 是**空实现**（`ReportNotSupport`，不报错但也不转）
- `Cast(int32 → half)` 走 `vconv_deq`，**roundMode 被忽略**，需要 `SetDeqScale`

### C5. `GlobalTensor::SetValue` 写 GM **不回刷**（输出随机为 0）
**对策**：先写 UB，再 `DataCopyPad` 到 GM。

### C6. `__aicore__` 内**禁止 double 强转**
`static_cast<float>(static_cast<double>(N) * M)` 直接**编译失败**。用 `1.0f / (float)(N_ * M_)`。

### C7. `__aicore__` 内**没有 `sqrtf`**，只有 `sqrt()`

### C8. 临时缓冲**绝不能复用仍在使用的缓冲**
实例：筛选用的临时缓冲复用了 `tb`，而 `tb` 缓存 quantScale 且**每 channel 只 `Duplicate` 一次、
跨多个 chunk 复用** → 从第 2 个 chunk 起 qs 被污染 → **67M 的大 case 72.3%（=3/4）元素错**。

---

## D. 多核 / 性能类

### D1. `SyncAll` 与跨核栅障在本机**不可靠**
尝试过的跨核归约方案全部失败：`SyncAll` → 507035 / 507057；
`SetScheduleMode(BATCH_MODE_SCHEDULE=1)` A/B 实测**无改善**。
**安全范式**：**行 / 样本级切分，每个单元由一个核完整负责，无跨核归约** → 天然确定性，也避开 A4/A1。
> 唯一例外：cube1/cube2 用 `blockDim=min(coreNumAiv, M)` 按 M 切核，各核独立写自己的输出行块。

### D2. 大 `workspace` 有固定开销
判题机上 `GetUserWorkspace` 请求大块（如 16MB）会带来 **0.7~2.3s** 的固定开销（S10 dot 实测）。
→ 能不用 workspace 就不用。

### D3. 判题 **3 秒硬看门狗**（仅 cube 类 harness）
`aclrtSynchronizeStreamWithTimeout(stream, 3000)` 失败 → `ret=507046`。
cube1 v1 的 case12 实测 `Task Duration = 6303ms` → 超时，后续 case 全 Skipped。
**但**其它算子的 harness 宽容得多（ct2d 有 case 跑 **272 秒仍 Pass**，gf2poly 有 349 秒 Pass）
→ **先确认你这道题是否受 3s 限制，不要一概而论**。

### D4. 本机 `Matmul` 高阶 API 是**空实现**
`Matmul<...>` 模板退化为 `MatmulServiceAux` 的空实现 → **matmul 静默输出全 0**；
`MatmulApiTiling::GetTiling` 不设 A/B/C/Bias dtype 会**段错误**。
**对策**：cube 类算子必须自己写**向量 GEMM**（见 `cube1` / `quant_matmul_sigmoid_mul_quant` 的实现）。

### D5. `DataCopyPad` 批量搬运比 GM 标量读快 **~300×**
cube1 v1 的瓶颈就是「A 的 GM 标量读」：每核 131K 次 × ~167ns ≈ 21.9ms，
改成一条 2D `DataCopyPad`（`blockCount=rows, blockLen=kb, srcStride=K-kb`）+ `MTE2_S`
→ 21.93ms → 4.55ms。

### D6. transpose 用例要**单独优化**
只优化 `tA=0,tB=0` 会让所有 transpose case **零提速**（实测 4 个 case 耗时一分未变）。
**定位手法**：把 v1/v2 的**逐 case 耗时表**并排列出来，一眼就能看出哪些 case 没被优化到。

---

## E. 工程 / 流程坑

| 坑 | 事实 | 对策 |
|---|---|---|
| 增量构建掩盖错误 | `build.sh -j8` 复用旧 `.o`，错误被藏 | **必须 `rm -rf build build_out` clean 重建** |
| zip 路径前缀 | 提交要求条目形如 `op_host/xxx.cpp`、`op_kernel/xxx.h` | 打包时**带前缀、无工程名前缀、7 条目** |
| zip 与源码不一致 | 曾出现「改了但交的是旧包」 | 提交前跑 `verify_zips.py`（逐文件 md5） |
| 文件 BOM | 头文件带 BOM 会让 zip 与源码 md5 不一致 | 存文件时用 UTF-8 **无 BOM** |
| 中文注释 | **实测不影响判题**（bpt / fa / gn 带中文注释都 Pass） | 但保守可全英文；注释里的 `system` 等不算违规 |
| `429` 冷却 | 提交间隔 <90s 必 429，`retry_after_seconds` 通常 120~170s | 按返回值退避重试 |
| 提交返回体 | `submissionId` 在 **`data.submissionId`**，不是 `_id`/顶层 | 解析别写错（写错会丢 sid） |
| 模板 kernel UT | 官方规模下**跑不完**（CPU 仿真每个向量 API 调用 1~3.5ms，官方 case 约 20000s） | 判题走真机路径，**别等仿真 UT** |
| `tests/ut/op_host/test_*_tiling.cpp` | 写死 `expectTilingData="0 1 0 "` 且不可改，本机还缺 `nlohmann/json.hpp` | 判题不走 `build.sh -u`，**不用试图满足它** |
| `examples/run.sh` 无可执行位 | `build.sh -e` 末尾报 `Permission denied` | `chmod +x examples/run.sh`；example 本身可跑 |

---

## F. 诊断手法（本赛最有用的三个）

### F1. Canary 实验 —— 一刀切开「值错」vs「写不到」
在 `Process()` 最前面直接往输出写固定常量（不做任何计算）并 `return`：
- 所有 case 的 `precision_ratio` 变 ≈0 → **写到达了被判的输出** ⇒ 真错在值/地址计算
- 仍 ≈ 原分布 → **根本没写到被判的输出** ⇒ aclnn 出参/管道问题
> 实例：index 用它排除了「管道问题」，把排查范围收敛到索引语义。

### F2. 「输出下标」技巧 —— 一次提交读出地址映射
不算 `x[src]`，直接把算出的**源下标**写进输出 → 与 golden 的下标一比对，就知道地址映射错在哪。

### F3. `precision_ratio` 反推形状
`1 - ratio = k / E`（k = 坏元素数，E = 该输出的元素数）。
只要 `1-r` 是精确分数就能解出 E 与 k：
> 例：`0.857177734375 = 3511/4096 = 7022/8192`（两种分母都精确 → 需要 canary 才能区分）
> 例：`0.9998333333333334 → 1/6000`；`0.99951171875 → 1/2048`；`0.9998931884765625 → 7/65536`

### F4. 拉真机错误信息
`GET /api/submissions/<sid>?userId=<uid>` → `result[i].msg` 里有 profiling 明细：
`Task Duration` / `Block Dim` / 具体错误码（507035 / 507046 / 507057）。
> 这是判断「超时」还是「UB 越界」的唯一可靠依据。

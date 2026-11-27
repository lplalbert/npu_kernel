# BitPlaneTransform（bpt）实现备注 / 交接文档

> 状态：线上 22.28 分（原向量版）。本文件记录到 2026-11-27 为止的全部结论，供后续接手。

## 1. 题目语义（务必按此，别记反）
- **transpose**：`x[N,C] uint8 -> y[ceil(N/32), 8C, 4] uint8`
  `y[g, 8j+r, d] = Σ_{s=0..7} bit_r( x[32g + 8d + s, j] ) << s`（lane 越界按 0）
- **inverse**：`x[G,8C,4] uint8 -> y[n,C] uint8`（`G=ceil(n/32)`）
  `y[i,j] = Σ_{r=0..7} bit_{i%8}( x[i/32, 8j+r, (i%32)/8] ) << r`
- 属性：`direction` 字符串（`"transpose"` / `"inverse"`，**必须全串精确匹配**，其它报错）、`n`（int，inverse 必填>0）。
- 非法输入校验不动作：transpose 要 rank==2；inverse 要 rank==3、last dim==4、middle dim %8==0、`n>0`、`ceil(n/32) <= G`。
- 每个 32-lane group 的字节数 = `32*C`；输出同大小（`8C*4 = 32C`）。

## 2. 标量 SWAR 核心（正确性的基准）
```cpp
uint64_t Transpose8x8(uint64_t x){            // 8x8 bit-matrix transpose，byte i = matrix row i
  uint64_t t;
  t=(x^(x>>7)) &0x00AA00AA00AA00AAULL; x^=t^(t<<7);
  t=(x^(x>>14))&0x0000CCCC0000CCCCULL; x^=t^(t<<14);
  t=(x^(x>>28))&0x00000000F0F0F0F0ULL; x^=t^(t<<28);
  return x;
}
```
- transpose：对每列 j，把 4 组 8 个 lane 字节（d=0..3）各拼成一个 uint64 → 4 次 `Transpose8x8`
  → 8 个 plane 字节（每 plane 一个 uint32，含 4 个 d）→ 写 `out[base+ (8j+r)*4 + 0..3]`。
- inverse：反向（读 8 个 plane word → 拆 4 个 d → `Transpose8x8` → 写回 32 个 lane 字节）。
- **这就是唯一天然 bit-exact 的形态**，任何快路都必须用它做等价校验。

## 3. 已验证的平台事实（省得再试）
- `Transpose(dst,src)`（基础 API）**只支持 int16/uint16/half**，16×16 元素转置 → **uint8 用不了**。
- **`TransDataTo5HD` 在 dav_c220 支持 `int8_t/uint8_t`**（`basic_api/impl/dav_c220/kernel_operator_vec_transpose_impl.h`
  的 assert 明列 `int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t`），8-bit 走
  `scatter_vnchwconv_b8`，参数结构 `TransDataTo5HDParams{ dstHighHalf, srcHighHalf, repeatTimes, dstRepStride, srcRepStride }`，
  入口 `TransDataTo5HD(dstList[16], srcList[16], params)`（16 个 UB 地址、需 32B 对齐）→ **这是唯一可用的"字节转置"硬件**。
- **`And/Or/Xor` 在 c220 只支持 int16/uint16**（uint8 无向量位运算）→ 位网络必须走 uint16（2 列打包，掩码 `0x0101<<r`、移位≤7、累加≤255，不跨字节串扰）。
- `WholeReduceSum` 在 c220 有（老签名那支），可做"8 元素一组求和"，可用来把 8 个 bit 打包成 1 字节。
- **UB 预算必须 `GetCoreMemSize(CoreMemType::UB)` 实测**（本机 192KB 量级；别假设）。
- 判题同一进程 `iterations:5` 次 launch（同进程重复发射本身没问题）。

## 4. 已踩过的 3 个真 bug（任何重写都会再遇到）
1. **跨"工作单元"缺 `PipeBarrier<PIPE_ALL>()`** → 一个核处理第 2 个 unit 时 **507035**（vector core exception）。
   原实现（每核串多个 group）是有这句的。
2. **`DataCopyPad` UB 侧按 `blockLen` 打包（32B 对齐），不是按你传的逻辑 pitch**。
   当列块 `cols < pitch`（只有最后一个短块会这样）时，UB 里的行距是 `cols` 而不是 `pitch`，
   用 `pitch` 去索引就会整块算错，且**只有末块错**，极难发现。
   → 要么 UB 行距直接用 `cols`，要么 `dstStride=(pitch-cols)/32` 显式补齐。
3. **尾部 partial lane-group（`rows%32!=0`）按列切分会算错**（实测 N=33,C=1 时第 2 组全 0）。
   → 把 partial group **交给 block 0 整宽标量处理**（与老实现一致）最稳。

## 5. 当前交付版（v3）的 tiling 模型
- 工作单元 = `(32-lane group) × (列块 chunkCols)`，`nUnits = fullGroups * nChunks`，`blockDim = min(coreNum, nUnits)`。
- `chunkCols`：默认 `= C`（一单元一 group，同老实现）；**仅当 `totalGroups < coreNum`** 才切列以填满核：
  `wantChunks = coreNum/totalGroups`（向下取整），`chunkCols = ceil(C/wantChunks)` **向上取整到 32**（保证单元数不超核数）。
  另：`chunkCols <= (ubSize*4/5)/64`（两个 UB buffer，每列 32B in + 32B out）。
- 快路条件：`lanes==32 && cols%32==0`，否则**自动回退标量**（这就是"形状守卫"）。
- 实测收益（bench，单次 µs）：中等形状 1.4~2×（64×64 53→37，128×128 62→34，256×256 97→49，512×512 161→89，
  inverse 512×512 131→73），大形状持平（1024²/4096×2048 ≈1.0×），个别 inverse 大形状 ~0.93×。
- **注意**：`N%32!=0 且 C 很大` 时 partial group 由单核整宽标量处理，会有一个串行尾巴
  （例：N=100,C=4096 → 尾巴约 0.9ms）。老实现同样如此，不是回归；想治可把 partial group 也按列并行（但要注意上面 bug 3）。

## 6. v3 判题失败（case3 prec=0.502、case4+ time=None）—— 教训
**根因：判题 15 个 case 的真实 shape 拿不到**（`/api/problems/<pid>/testcases` 只给 id；`/api/testcase/<id>` 403/404；
成功提交 `result[i].msg` 为空）。我们所有验证都是在**自己猜的形状**上做的。
⇒ **不要再用"我猜几个形状跑通"来验收**。必须：
1. **形状扫描覆盖极端**：`C ∈ {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192}` ×
   `N ∈ {1,2,3,31,32,33,63,64,65,127,129,1000,4096,65536,262144,1048576}` 抽样 ≥100 组合，双向都测；
2. 重点：**非 32 对齐 / 小 C / 大 C / N 奇数**；
3. **每个形状都必须 bit-exact**（`bad=0`），不是"大部分对"；
4. **形状守卫**：快路前提不满足必须自动回退标量，宁慢勿错；
5. 不要有"只对某种 C 生效"的隐式快路，否则别的 C 要么慢要么错。

## 7. 下一步（向量化，P114 + P100 官方范式）★这是唯一能接近 TOP1 的路
现状瓶颈：**逐列标量 SWAR**（每列 32 次 `GetValue` + 8 次 `SetValue`，约 1.6 标量操作/字节）。
TOP1 比我们大 case 快 47~128× ⇒ 必须把标量循环整体消掉。

### 7.1 目标形态（两步，宽操作，无跨 lane 标量）
目标：`out[j*32 + 4r + d] = Σ_s bit_r( in[(8d+s)*C + j] ) << s`
- **第 1 步（elementwise，可全向量化）**：以 **j 为向量维**，对 32 个 `(r,d)` 各算一条长度 C 的向量：
  `acc = Σ_{s=0..7} ((row_{8d+s} >> r) & 1) << s`。输入 `row_L` 是 `in[L*C .. L*C+C-1]`（**连续 C 字节**）。
  `And/ShiftLeft/ShiftRight/Add`，按 **uint16（2 个相邻 j）** 处理：掩码 `0x0101<<r`、移位≤7、累加≤255 → 不串字节、等效 2 列并行。
  优化：先 `B_r[L] = And(row_L, 0x0101<<r)`（32×8 次 And），再对每个 (r,d) 做 8 次 `Shift(Left/Right by |s-r|)` + `Add`
  → 约 768 次向量操作 / group（每次覆盖 C 字节）⇒ 理论 ~0.1 cycle/byte（比现在 ~36 cycle/byte 快两个数量级）。
- **第 2 步（一次硬件字节转置）**：第 1 步天然得到 `z[(r,d)][j]`（32 行 × C），GM 要 `y[j][(r,d)]`（C 行 × 32）
  = **32×C ↔ C×32 的 uint8 字节转置** → 用 **`TransDataTo5HD`（uint8，C0=32）** 一次完成。
  几何：b8 一次是 `16 行 × 32 字节 ↔ 32 行 × 16 字节`，即一次覆盖半个 lane 组（16 lanes）× 32 列；
  32 lanes × C 列需 `(32/16) × (C/32)` 次调用，16 个 src/dst 地址按 32B 对齐给。
  P114 的对齐范式：`tileNA = AlignUp(tileN,32)`、`repeats = tileNA/16 ≤ 255`、offset table 按 `tileNA` 建。
  → **务必先用一个 32×32 的独立小用例把 TransDataTo5HD 的 b8 语义（dstHighHalf/srcHighHalf、地址列表含义）试准**，再上大形状。
- 备选收尾：`GatherMask`（P100/P101）的 `deinterleave` 可做等距抽取，用来替代"按 stride C 聚一列"的 gather
  （`GatherMask(dst, src, deinterleave, isReverse, count, {1,1,8,0}, rsvdCnt)`，之后 `PipeBarrier<PIPE_V>()`）。

### 7.2 实施顺序建议
1. 先只做 **inverse 方向**的向量化：它的读/写都是连续行（`x[rd]` 连续、`y[lane]` 连续），**不需要任何字节转置**，
   最容易落地（每个输出行 = 8 个连续输入行的 elementwise 组合）。
2. transpose 方向再引入 `TransDataTo5HD`（有转置，风险高）。
3. 每个方向都保留标量实现做**形状守卫回退**，并用第 6 节的 100+ 形状扫描做 bit-exact 验收。

## 8. 本地工具
- `bench_bpt <dir> <N> <C> <repeat> <fullcheck> [dumpfile]`（`/root/work/bench_bpt.cpp`）：
  自带 spec 金标准 bit-exact 对拍 + 计时；`dir=0` transpose / `dir=1` inverse。
- 大形状扫描脚本模板：见 `/tmp/sweep_bpt.py`（笛卡尔积 + N*C 上限 + 双向 + 逐形状 PASS/FAIL 汇总）。
- 模板 examples 路径：`/root/work/ws/<name>/examples/run.sh`（官方样例 N=33,C=1，**必须 PASSED**）。

---

## 9. v3 的实测失败形状（重要：这就是判题失败的原因，别再交）
用 `/tmp/sweep_bpt2.py` 跑 242 组合 × 双向（N*C ≤ 256KB），**非 32 对齐的 C 大量失败**：
```
d=0 n=33 c=63/127/255        d=1 n=32 c=63/127/255
d=0 n=64 c=1/3/31/63/127/255 d=1 n=33 c=63/127/255
d=0 n=65 c=1/3              d=1 n=63 c=63/127/255
                            d=1 n=64 c=3/31/63/127/255
                            d=1 n=65 c=1/3
```
规律：**失败全部落在 `C % 32 != 0`（C∈{1,3,31,63,127,255}）且/或 group 数 >1** 的形状上。
典型现象：`bad` 恰好是"某一个 32 字节输出行"（firstbad 落在某个 32 的整数倍），说明
**某个 group / 某个列块整个没写或整块写错**，而不是零星精度问题。
⇒ 根因在"**非 32 对齐 C 时，vector 列块与 scalar 尾列块/多 group 单元的分工边界**"，
   而不是 SWAR 公式本身（32 对齐的形状全部正确）。
**在把这条边界彻底修好（并用 100+ 形状双向全 bit-exact 验证）之前，不要提交 bpt。线上保持 22.28。**

## 10. 给接手人的最短路径
1. 先把工程**只保留标量实现**（去掉切列/vector 快路），确认 242 组合全 `bad=0` → 得到可信基线。
2. 再**逐档**打开：先只对 `C%32==0 && lanes==32` 用 vector 路（其余不变），跑全扫描；
   全绿后再推广到 `C%32!=0` 的尾块（尾块单独按 `cols` 处理，注意第 4 节 bug 2）。
3. 每加一档都必须**全扫描 bit-exact**，禁止"我猜几个形状通过"。
4. `n%32!=0` 的 partial group 继续交给 block 0 整宽标量（已验证安全）。

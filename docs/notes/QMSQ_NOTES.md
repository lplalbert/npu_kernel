# QMSQ（QuantMatmulSigmoidMulQuant）调试笔记 / 环境预警

面向后续 fusion 题（ConvTranspose2D / Gf2Poly 等）的重要预警，尤其是需要 Matrix/Cube 的题。

## 结论速览
- 本机 CANN 9.2 / 910B2（dav_c220），用官方 ladder 模板做「带 Matmul 高阶 API」的自定义算子时，
  运行期 `Matmul` 模板会退化为 **`MatmulServiceAux`（KFC server 空实现）**：
  其 `SetOrgShape/SetTensorA/IterateAll` 等方法在 `MatmulServiceAuxBase` 内部带
  `if constexpr (ToMatmulConfig(MM_CFG).enableMixDualMaster)` 保护，默认配置下**全是空实现**，
  于是矩阵乘不执行、输出全 0（不报错）。→ 真机上「Matmul 高阶 API 实际不可用」（至少按当前模板/编译方式）。
- 因此本机做 Matrix 类算子，**优先用「向量/标量实现」保正确**，不要指望 Matmul 高阶 API 在真机跑出结果。

## 三个高价值坑（都实测踩过）
1. **`matmul_tiling::MatmulApiTiling::GetTiling` 未设置 A/B/C/Bias dtype 会踩内存 → 进程 segfault（139）**。
   必须在使用前调用：
   ```
   mmTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, transA);
   mmTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, transB);
   mmTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
   mmTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
   ```
   否则 host 侧堆被写坏，在 execute 阶段才 segfault，极难定位（表面看是 kernel 崩）。
   （定位法：把 tiling_data 换成纯 struct、去掉 MatmulApiTiling 调用，空 kernel 就能跑通 RC=0。）
2. **增量 `bash build.sh` 不会重编 kernel**（生成的 .o 哈希与 stub 相同，`Up-to-date` 骗过）。
   改了 op_kernel/*.{h,cpp,_tiling_data.h} 后**必须 `bash build.sh --make_clean` 再 build**，否则装进去的是旧 kernel。
3. **op_host tiling 里 `context->GetInputShape(i)` 返回的是 `gert::StorageShape*` 不是 `gert::Shape*`**。
   要 `->GetStorageShape()` 取 `const gert::Shape&`。直接赋给 `const gert::Shape*` 编译报错。
4. **kernel 里 `workspace` 参数带“系统前缀”**：真机上 `workspace` 指针指向的区域前面有一段 system workspace
   （即使这题不用 Matmul，`GetLibApiWorkSpaceSize()` 真机约 16MB 非 0）。若直接把自定义中间 buffer 放到
   `workspace[0]`，会 MTE 访问到系统区/越界 → **507035 vector core exception（MTE invalid GM address）**。
   正确做法：host tiling `currentWorkspace[0] = GetLibApiWorkSpaceSize() + 用户buffer`，kernel 里
   `GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);` 再在 `userWs` 上放自己的 buffer。

## 向量 GEMM 实现（route B）要点 —— 已全绿
- 单核：`for m in M: for nb chunk in N: accF(fp32)=0; for k in K: a=A[m,k]; load B[k,chunk];
  Muls/Cast; Add(accF); 每 64 个 K：Cast→int32 累加(accI)并清 accF; 末尾 flush; 写 gC(int32)`。
- 精度：int8×int8 乘积 ≤16129，64 项分块和 ≤1.03e6 < 2^24 → fp32 精确；再 int32 累加无损失。
- **官方 golden 对拍：11 个几何全部逐元素 int8 精确 + yScale 精确一致（rel=0）**：
  1×16×1、32³、8×64×8、33³、50×40×30(四种转置组合)、128³(K>64 多块)、K=130(奇数尾块)、K=130+双转置。
- 模板自带 aclnn 例子 `bash build.sh -e` 也已 EXRC=0。
- 性能（单核向量乘，O(M·N·K)）：**512³ ≈ 5.8s**；8192³ 约 4000 倍 → 数小时，判定**大规模必超时**（不是本机 bug，是向量实现固有）。
  如需性能需多核并行 + B 驻留分块，本题时间盒内未做。

### 尾行 bug 根因（已修，重要复现经验）
- 现象：某些形状（如 50×40×30，total=1500=23*64+28）**仅最后 28 个元素**（正好是 count%64 的尾巴）错，y≈gold/2。
- 定位法：让 driver 从 workspace 里 dump `gC`(int32) 和 `gU`(half) 中间值，直接与 golden 的 int32 矩阵乘/中间 U 对比 →
  先证 `gC` 与 golden 完全一致，再证 `gU` 尾巴错 → 定位到 **sigmoid 的分支选择**。
- 根因：**`Compare`/`Select` 是掩码类算子，不覆盖 count%64 的尾部**（掩码寄存器按 64 粒度）。尾部落选位默认 0 → 对 z<0 的元素
  错选到 sigP 分支（≈0.5·z），z≥0 的元素碰巧选对（所以 33³ 那种尾块只有 1 个元素且 z≥0 时看不出）。
- 修法：把 Compare/Select 的 count 向上取整到 64 的倍数，并对尾部补 0；**补零必须从 32B 对齐的元素偏移开始**
  （`padStart=((uc+7)/8)*8`），否则 `Duplicate(f2[uc],…)` 在非 32B 对齐 UB 偏移上直接 vector core 异常。
  例：`uc64=((uc+63)/64)*64; padStart=((uc+7)/8)*8; if(uc64>padStart){Duplicate(f2[padStart],0,uc64-padStart);…}
  Duplicate(f5,0,uc64); Compare(...,uc64); Select(...,uc64);` 之后只用前 uc 个。
- 教训：任何 `Compare/Select`（以及可能其它掩码类算子）在 count 非 64 倍数时都可能有尾部问题，务必做 64 对齐处理。

## 有用事实
- 910B2 = Atlas A2，`__NPU_ARCH__` 属 dav_c220；kernel 走 `_mix_aic_/_mix_aiv_` 分离编译。
- 自定义算子 `TilingData` 里可以安全内嵌 `struct TCubeTiling`（clean 后 `opParaSize` 会变成真实大小，例如 272），
  内嵌本身不崩；崩的是上面的 `MatmulApiTiling` 未设 dtype（见坑 1）。
- 算子包安装：`build/custom_opp_ubuntu_aarch64.run --quiet`，装到 `$ASCEND_HOME_PATH/opp/vendors/`。
- 真机自建验证：写 aclnn driver（读 bin→aclnn 调用→写 bin）+ 官方 `gen_data.impl` 生成 golden + numpy 比对，
  比模板自带的 1×16 例子彻底（模板例子不校验结果）。
- `getattr` 类：`GetSysWorkSpacePtr()` 是全局函数（非 AscendC 命名空间）；`AscendC::GetUserWorkspace(ws)` 在 AscendC 命名空间。

## 本题（QMSQ）路线
- 官方题面：`1 <= M,N,K <= 8192`，用例「覆盖 M=8192、N=8192、K=8192 上限」，精度对 int8 要求逐元素精确、
  yScale 相对误差 < 1e-4。golden 舍入点：h=fp16(fp32(acc)*s1*s2)、g=fp16(sigmoid(fp32(h)))、u=fp16(fp32(g)*fp32(h))、
  y=clip(rint(u/yScale),-128,127)，yScale=amax/127（amax==0 取 1）。
- 跨核归约红线：本机 SyncAll / workspace 跨核归约必崩 507057，只能原子加或单核。
- 因 Matmul 高阶 API 真机不可用，改走**向量 GEMM**（fp32 分块累加保 int32 精确：int8×int8 乘积 ≤16129，
  每 64 个 K 折叠一次 → 分块和 ≤1.03e6 < 2^24 精确，再 Cast int32 累加）。
- 若 8192^3 量级导致向量 GEMM 必然超时，则本题按「环境限制：Matmul API 不可用 + 大规模超时」记录放弃。

（纪律：未提交，配额未动。）

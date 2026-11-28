# 配方：同一算子按 tilingKey 选不同 task type（AIV_ONLY vs MIX_AIC）——CANON 9.0 已验证

> 状态：**机制在 CANN 9.0 / 910B2 上实测成立**（2026-11-28, agent ct2d-key）。
> 推翻了旧结论「含 matmul 的 kernel 无法 AIV-only」——那是**宏写法/放置错误**，不是接口限制。

## 0. 一句话
在 **`__global__` kernel entry 函数体**里，用**整数字面量 key** 写 `KERNEL_TASK_TYPE(<key>, <TYPE>)`，
即可让同一 kernel 的不同 tilingKey 拥有不同 task type；`KERNEL_TASK_TYPE_DEFAULT` 兜底未覆盖的 key。

## 1. 正确写法（三种都行，本质相同）
### (a) conv2d_v2 风格：函数体末尾无条件列出所有 key（最稳）
```cpp
template <uint32_t schMode>
__global__ __aicore__ void my_op(GM_ADDR ..., GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // 兜底
    REGISTER_TILING_DEFAULT(MyTilingData);
    GET_TILING_DATA_WITH_STRUCT(MyTilingData, td, tiling);
    if constexpr (schMode == 0) { /* 纯向量 body，绝无 matmul */ }
    if constexpr (schMode == 1) { /* cube body（含 matmul） */ }
    // per-key task type，必须是整数字面量 key：
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
}
```
### (b) upsample_linear1d 风格：放在运行时 `if (TILING_KEY_IS(1))` 分支内
### (c) 官方 conv2d_v2 里 `SET_KERNEL_TASK_TYPE(2058); SET_KERNEL_TASK_TYPE(2106); ...` 全是字面量

**要点（踩坑清单）**
1. **key 必须是整数字面量**：宏做 `##key` 生成变量名 `__enable_feature_for_compile_<key>`；写 `enum`/`static_cast<...>` 不会被 precompile 采集 → 全部退回默认。
2. **宏必须出现在 `__global__` entry 函数体里**（顶层语句或运行时 if 内均可）；放进 `__aicore__ inline` helper 函数里大概率不被采集。
3. **task type 由源码里的宏决定，不由"是否含 matmul"决定**：含 matmul 的 symbol 一样可以被标成 AIV_ONLY（但**运行期该分支就绝不能真的调用 matmul**，否则 AIC 不启动会出错/挂）。
4. dtype 若需 fp16/fp32 两种，可把 dtype 放在**运行时** `td.isF16` 分支里选 `<half>/<float>`，这样只需「每个 task type 一个 template 实例」，不必 4 个。
5. `KERNEL_TASK_TYPE_DEFAULT` 保留作兜底（官方要求未覆盖 key 有默认）。

## 2. 为什么之前失败
- 旧结论依据：扩 4 个 key + 写 `KERNEL_TASK_TYPE(k, AIV_ONLY)` → JSON 4 个 key 全 `MIX_AIC`。
- 复核 build 链：CANN 用 `device_precompile_project`（`-E` 预处理 + 定义 `__CHECK_FEATURE_AT_PRECOMPILE`）
  把 `KERNEL_TASK_TYPE(key,val)` 展开成 `auto __enable_feature_for_compile_<key> = val;`，
  再由 `bisheng --generate-ktype-section` 采集进 ktype section，`ascendc_compile_gen_json.py` 写进 JSON。
- ⇒ 只要 key 是字面量、宏在 entry 里，本机 cmake 链（`npu_op_kernel_library`）**能**采到。实测已成功。

## 3. 验证方法（看编译产物 JSON，不用上板）
产物路径：`build/op_kernel/ascendc_kernels/binary/<soc>/<op>/<Op>_<hash>.json` 的 `kernelList`。
映射来自 `ascendc_compile_gen_json.py::_get_kernel_type_dict`（kernel_type.value → JSON 字段）：

| kernel_type | JSON kernelType | taskRation | crossCoreSync |
|---|---|---|---|
| 0 AIV_ONLY | `MIX_AIC` | **0:1** | **0** |
| 1 AIC_ONLY | `MIX_AIC` | 1:0 | 0 |
| 4/5 MIX_AIV_1_0 / MIX_AIC_1_0 | `MIX_AIC` | 0:1 / 1:0 | 1 |
| 6 MIX_AIC_1_1 | `MIX_AIC` | 1:1 | 1 |
| 7 MIX_AIC_1_2 | `MIX_AIC` | **1:2** | **1** |
| 8 AiCore / 9 VectorCore | `AiCore`/`VectorCore` | 1:0 / 0:1 | — |

⚠️ **AIV_ONLY 不是字面 "AIV_ONLY"**，而是 `kernelType:"MIX_AIC" + taskRation:"0:1" + crossCoreSync:0`。
（对照：纯自动推断的无 matmul kernel 干脆**没有** `kernelType/taskRation` 字段；`MIX_AIV_1_0` 是 0:1 但 crossCoreSync=1。）

**判定**：目标 key 的 `taskRation` 是 `0:1` 且 `crossCoreSync` 为 `0` ⇒ 该 key 实际是 AIV_ONLY。

## 4. 官方旁证（CANN 自带 built-in JSON 里就有 per-key 混用）
- `BatchMatMulV3_*_high_performance.json`：kernelList 里 `('MIX_AIC','1:0')` 与 `('MIX_AIC','1:2')` 混排（AIC 重 / MIX）。
- `Conv3DBackpropFilterV2_*`：`1:1` 与 `1:2` 混排。
- 源码 `cann_conv2d_v2.cpp`：`SET_KERNEL_TASK_TYPE(key)=KERNEL_TASK_TYPE(key, KERNEL_TYPE_AIC_ONLY|MIX_AIC_1_2)`。
- 源码 `upsample_linear1d.cpp`：`KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_1)` + `if(TILING_KEY_IS(1)) KERNEL_TASK_TYPE(1, MIX_AIC_1_2)`。

## 5. ct2d 实测（本配方落地）
- ws `/root/work/ws/ct2d_hybrid2`：key0=AIV_ONLY 纯向量（删净 matmul 握手），key1=MIX cube（matmul）。
- JSON（fp16/fp32 两 binary 一致）：key0 → 0:1/ccs0；key1 → 1:2/ccs1。✅
- 运行期（bench 同进程 iters=5）：
  - 1x16x16x16 fp16 向量：min 0.117ms med 0.118ms —— **无 5.8s 卡顿**（旧 MIX 向量是 5877µs）。
  - 1x8x8x8 fp16：0.062/0.063ms；1x16x8x8 fp32：0.087/0.090ms。
- 结论：**per-key AIV_ONLY 彻底解决"向量被迫 MIX → AIC 空等 → 507046 看门狗"**。

## 6. 注意事项 / 未解决
- cube(MIX key) 在「同一 binary 同时含 AIV_ONLY key + MIX key」下，重复 launch 仍见 ~2s 卡顿（排查中；standalone 全 MIX cube 无此现象）。
- 运行期 AIV_ONLY 分支必须真正无 matmul 调用；否则语义/硬件不匹配。
- 7 文件约束下，cube 路径需保证精度（fp16 MARE<9.7656e-3 / fp32<1.2207e-3）。

/*!
 * \file trace_tiling.cpp
 * \brief Trace 算子 Tiling（v2：单核确定性 + 大 chunk 向量归约）
 *
 * 输入 self (M,N) -> 输出 out (1,)。对角线长度 K=min(M,N)。单核处理，避免跨核同步。
 * chunk 依 UB 大小与 dtype 选取，使 DataCopyPad 一次搬运尽量多的对角行，减少搬运/归约指令数。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/trace_tiling_data.h"
#include "../op_kernel/trace_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TraceTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    TraceTilingData* tiling = context->GetTilingData<TraceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& shape = inputShapePtr->GetStorageShape();
    int64_t m = 0;
    int64_t n = 0;
    if (shape.GetDimNum() == 2) {
        m = shape.GetDim(0);
        n = shape.GetDim(1);
    } else if (shape.GetDimNum() == 1) {
        m = 1;
        n = shape.GetDim(0);
    } else if (shape.GetDimNum() == 0) {
        m = 1;
        n = 1;
    } else {
        OP_LOGE(context, "Trace expects 2-D input");
        return ge::GRAPH_FAILED;
    }
    if (m < 1) m = 1;
    if (n < 1) n = 1;
    const int64_t k = (m < n) ? m : n;

    uint64_t tilingKey;
    bool isF16 = false;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_0);
        isF16 = true;
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_1);
    }

    // chunk: 单次 DataCopyPad 处理的对角行数（64 的整数倍）
    // 每行 UB 占用：inQ 32B 槽；fp16 额外一份 fp32 cast 中间(16*4=64B)。
    // chunk = 单次处理的对角行数：取 min(cap, roundup64(K))，小 K 时缓冲/折半规模随之变小（压低固定开销）。
    // cap 受向量指令 repeat 上限(255)与 UB 约束。
    const int64_t bytesPerRow = isF16 ? (32 + 64) : 32;
    int64_t ubChunk = static_cast<int64_t>((ubSize * 7 / 10) / static_cast<uint64_t>(bytesPerRow));
    const int64_t capChunk = isF16 ? 1024 : 2048;
    int64_t chunk = k;
    chunk = CeilAlign(chunk, static_cast<int64_t>(64));
    if (chunk > capChunk) chunk = capChunk;
    if (chunk > ubChunk) chunk = ubChunk;
    chunk = CeilAlign(chunk, static_cast<int64_t>(64));
    if (chunk < 64) chunk = 64;

    tiling->totalNum = m * n;
    tiling->ubFactor = chunk;
    tiling->m = m;
    tiling->n = n;
    tiling->k = k;
    tiling->chunk = chunk;

    // 核数：fp32 大 K 沿对角分段多核并行（fp16 K<=2048 单核即可）
    // 安全开关：kMultiCore=false 时退化为单核确定性版（慢~6x，但结果逐位确定）。
    constexpr bool kMultiCore = true;
    constexpr int64_t kMinRowsPerCore = 128;
    int64_t cores = 1;
    if (kMultiCore && !isF16) {
        cores = coreNum;
        int64_t need = (k + kMinRowsPerCore - 1) / kMinRowsPerCore;
        if (need < cores) cores = need;
        if (cores < 1) cores = 1;
    }
    int64_t per = (k + cores - 1) / cores;
    if (per < 1) per = 1;
    tiling->blockFactor = per;
    tiling->usedCores = static_cast<int32_t>(cores);

    context->SetBlockDim(static_cast<uint32_t>(cores));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TraceCompileInfo {};

IMPL_OP_OPTILING(Trace).Tiling(TraceTilingFunc).TilingParse<TraceCompileInfo>(TilingParseForTrace);

} // namespace optiling

/*!
 * \file reduce_log_sum_exp_tiling.cpp
 * \brief ReduceLogSumExp 算子 Tiling 实现
 *
 * 仅依赖输入 shape（安全）；axes 取值在 opapi 流程是设备侧内存，
 * 归约映射由 kernel 在设备侧解析。blockNum 给足核数，kernel 内按 outNum 自行切分。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/reduce_log_sum_exp_tiling_data.h"
#include "../op_kernel/reduce_log_sum_exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 512;
constexpr int64_t MAX_RANK = 8;

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
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReduceLogSumExpTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReduceLogSumExpTilingData* tiling = context->GetTilingData<ReduceLogSumExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShape = context->GetInputShape(0); // StorageShape*
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = inputShape->GetStorageShape();
    int64_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank > MAX_RANK, OP_LOGE(context, "rank too large"), return ge::GRAPH_FAILED);

    tiling->rank = rank;
    int64_t totalNum = 1;
    for (int64_t i = 0; i < rank; ++i) {
        tiling->xDims[i] = shape.GetDim(i);
        totalNum *= shape.GetDim(i);
    }
    if (rank == 0) {
        totalNum = 1;
    }
    tiling->totalNum = totalNum;
    int64_t acc = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        tiling->xStrides[i] = acc;
        acc *= tiling->xDims[i];
    }

    // axes 的长度（shape 安全；取值在设备侧由 kernel 自读）
    tiling->axesNum = 0;
    auto axesShapePtr = context->GetInputShape(1);
    if (axesShapePtr != nullptr) {
        const gert::Shape& axesShape = axesShapePtr->GetStorageShape();
        int64_t an = 1;
        for (int64_t i = 0; i < axesShape.GetDimNum(); ++i) {
            an *= axesShape.GetDim(i);
        }
        if (axesShape.GetDimNum() == 0) {
            an = 0;
        }
        tiling->axesNum = an > 8 ? 8 : an;
    }
    if (rank == 0) {
        tiling->axesNum = 0;
    }

    // ---- dtype -> tilingKey / 位宽 ----
    uint64_t tilingKey;
    int64_t typeSize = TYPE_SIZE;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(REDUCELOGSUMEXP_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(REDUCELOGSUMEXP_TPL_SCH_MODE_2);
        typeSize = 2;
    } else {
        tilingKey = GET_TPL_TILING_KEY(REDUCELOGSUMEXP_TPL_SCH_MODE_1);
    }

    // ---- UB 分块长度：float 中间缓存 + T 拷贝缓存 ----
    // 注意：必须给 TPipe/框架内部留 UB 余量，否则 fBuf/tBuf 会因超额分配而重叠，
    //       16bit 路径的 Cast 会读到脏数据（fp32 小例不触发，大例/16bit 半错）。
    int64_t bytesPerElem = TYPE_SIZE + typeSize;
    int64_t ubLen = static_cast<int64_t>(ubSize / static_cast<uint64_t>(2 * bytesPerElem));
    if (ubLen > 16384) {
        ubLen = 16384;
    }
    if (ubLen < 16) {
        ubLen = 16;
    }
    tiling->ubLen = ubLen;

    // ---- 核数：给足，kernel 内按 outNum 切分 ----
    int64_t blockNum = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
    if (blockNum > coreNum) {
        blockNum = coreNum;
    }
    if (blockNum < 1) {
        blockNum = 1;
    }
    context->SetBlockDim(blockNum);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceLogSumExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceLogSumExpCompileInfo {};

IMPL_OP_OPTILING(ReduceLogSumExp).Tiling(ReduceLogSumExpTilingFunc).TilingParse<ReduceLogSumExpCompileInfo>(TilingParseForReduceLogSumExp);

} // namespace optiling

/*!
 * \file mish_tiling.cpp
 * \brief Mish 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mish_tiling_data.h"
#include "../op_kernel/mish_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 与 op_kernel/mish.h 的 BUFFER_NUM 保持一致
constexpr int64_t QUEUE_BUF_NUM = 2;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

static ge::graphStatus MishTilingFunc(gert::TilingContext* context)
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

    MishTilingData* tiling = context->GetTilingData<MishTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 输入 shape -> 总元素数（0 维标量按 1 个元素处理） ----
    auto inputShape = context->GetInputShape(0); // gert::StorageShape*
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = 1;
    for (int64_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid input shape"), return ge::GRAPH_FAILED);

    // ---- dtype -> tilingKey / 位宽 ----
    uint64_t tilingKey;
    int64_t typeSize;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(MISH_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(MISH_TPL_SCH_MODE_2);
        typeSize = 2;
    } else {
        tilingKey = GET_TPL_TILING_KEY(MISH_TPL_SCH_MODE_1);
        typeSize = TYPE_SIZE;
    }

    // ---- 多核切分：每核至少 MIN_SPLIT_THRESHOLD 个元素（不足则降核数） ----
    // blockFactor / ubFactor 均对齐 32B 边界（alignElem 个元素），使主循环的
    // DataCopy 走对齐快速路径，仅每核最后一块走 DataCopyPad 尾巴。
    const int64_t alignElem = 32 / typeSize;  // 32B 对齐所需元素数：fp32=8，16bit=16
    int64_t blockNum = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
    if (blockNum > coreNum) {
        blockNum = coreNum;
    }
    if (blockNum < 1) {
        blockNum = 1;
    }
    int64_t blockFactor = (totalNum + blockNum - 1) / blockNum;
    blockFactor = ((blockFactor + alignElem - 1) / alignElem) * alignElem;

    // ---- UB 预算：in/out 双缓冲队列 + float 中间缓存 ----
    // float dtype:   队列 2*QUEUE_BUF_NUM*4 + sF/tF 8
    // 16bit dtype:   队列 2*QUEUE_BUF_NUM*2 + xF/sF/tF 12
    int64_t bytesPerElem = 2 * QUEUE_BUF_NUM * typeSize + 2 * TYPE_SIZE;
    if (typeSize != TYPE_SIZE) {
        bytesPerElem += TYPE_SIZE;
    }
    int64_t ubFactor = static_cast<int64_t>(ubSize / static_cast<uint64_t>(bytesPerElem));
    ubFactor = (ubFactor / alignElem) * alignElem;
    if (totalNum > 0 && ubFactor > totalNum) {
        ubFactor = ((totalNum + alignElem - 1) / alignElem) * alignElem;
    }
    if (ubFactor < 1) {
        ubFactor = alignElem;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor > 0 ? blockFactor : 1;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockNum);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MishCompileInfo {};

IMPL_OP_OPTILING(Mish).Tiling(MishTilingFunc).TilingParse<MishCompileInfo>(TilingParseForMish);

} // namespace optiling

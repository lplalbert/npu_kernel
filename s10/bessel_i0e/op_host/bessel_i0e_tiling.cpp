/*!
 * \file bessel_i0e_tiling.cpp
 * \brief BesselI0e 算子 Tiling 实现
 *
 * 逐元素算子，单核保证写边界不发生 cache line 竞争（blockDim=1）。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/bessel_i0e_tiling_data.h"
#include "../op_kernel/bessel_i0e_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE_F32 = 4;
constexpr int64_t MAX_UB_LEN = 4096;

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

static ge::graphStatus BesselI0eTilingFunc(gert::TilingContext* context)
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

    BesselI0eTilingData* tiling = context->GetTilingData<BesselI0eTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 元素总数（支持 0-8 维；0 维按 1 处理）----
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& shape = inputShapePtr->GetStorageShape();
    int64_t totalNum = 1;
    for (int64_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    if (shape.GetDimNum() == 0) {
        totalNum = 1;
    }
    if (totalNum < 1) {
        totalNum = 1;
    }
    tiling->totalNum = totalNum;

    // ---- dtype -> tilingKey / typeSize ----
    uint64_t tilingKey;
    int64_t typeSize = TYPE_SIZE_F32;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(BESSELI0E_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(BESSELI0E_TPL_SCH_MODE_2);
        typeSize = 2;
    } else {
        tilingKey = GET_TPL_TILING_KEY(BESSELI0E_TPL_SCH_MODE_1);
        typeSize = 4;
    }

    // ---- 多核：逐元素算子，各核处理不重叠的元素区间，无需跨核归约 ----
    // 每核输出区间按 128 元素（fp32 512B / fp16 256B）对齐，避免相邻核共享 cache line 的读写竞争
    constexpr int64_t ALIGN_ELEM = 128;
    constexpr int64_t MIN_PER_CORE = 2048;
    int64_t blockNum = (totalNum + MIN_PER_CORE - 1) / MIN_PER_CORE;
    if (blockNum > coreNum) {
        blockNum = coreNum;
    }
    if (blockNum < 1) {
        blockNum = 1;
    }
    context->SetBlockDim(blockNum);
    int64_t blockFactor = (totalNum + blockNum - 1) / blockNum;
    blockFactor = ((blockFactor + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
    if (blockFactor < ALIGN_ELEM) {
        blockFactor = ALIGN_ELEM;
    }
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = totalNum;

    // ---- UB 分块长度：6 个 float 缓冲 + 16bit 收发缓存 ----
    int64_t bytesPerElem = 6 * TYPE_SIZE_F32 + ((typeSize < 4) ? 2 * typeSize : 0);
    int64_t ubLen = static_cast<int64_t>(ubSize) / bytesPerElem;
    if (ubLen > MAX_UB_LEN) {
        ubLen = MAX_UB_LEN;
    }
    ubLen = (ubLen / 64) * 64;
    if (ubLen < 64) {
        ubLen = 64;
    }
    tiling->ubLen = ubLen;

    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBesselI0e([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BesselI0eCompileInfo {};

IMPL_OP_OPTILING(BesselI0e).Tiling(BesselI0eTilingFunc).TilingParse<BesselI0eCompileInfo>(TilingParseForBesselI0e);

} // namespace optiling

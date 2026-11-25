/*!
 * \file group_norm_silu_quant_tiling.cpp
 * \brief GroupNormSiluQuant 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/group_norm_silu_quant_tiling_data.h"
#include "../op_kernel/group_norm_silu_quant_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static ge::graphStatus GroupNormSiluQuantTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    GroupNormSiluQuantTilingData* tiling = context->GetTilingData<GroupNormSiluQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& xs = xShape->GetStorageShape();
    const size_t rank = xs.GetDimNum();
    OP_CHECK_IF(rank < 2, OP_LOGE(context, "self rank must be >= 2"), return ge::GRAPH_FAILED);

    int64_t N = xs.GetDim(0);
    int64_t C = xs.GetDim(1);
    int64_t HW = 1;
    for (size_t i = 2; i < rank; ++i) {
        HW *= xs.GetDim(i);
    }
    if (N < 1) N = 1;
    if (C < 1) C = 1;
    if (HW < 1) HW = 1;

    int64_t G = 1;
    float epsilon = 1e-5f;
    int64_t activateSilu = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* gp = attrs->GetAttrPointer<int64_t>(0);
        if (gp != nullptr && *gp > 0) {
            G = *gp;
        }
        const float* ep = attrs->GetAttrPointer<float>(1);
        if (ep != nullptr) {
            epsilon = *ep;
        }
        const bool* ap = attrs->GetAttrPointer<bool>(2);
        if (ap != nullptr) {
            activateSilu = (*ap) ? 1 : 0;
        }
    }
    if (G < 1) G = 1;
    if (G > C) G = C;
    while (C % G != 0 && G > 1) {
        --G;
    }
    const int64_t CpG = C / G;
    const int64_t gsz = CpG * HW;
    const int64_t totalGroups = N * G;

    // quantScale: scalar (numel == 1 or rank 0) or per-channel (C)
    int64_t qsPerChannel = 0;
    const gert::StorageShape* qsShape = context->GetInputShape(3);
    if (qsShape != nullptr) {
        const gert::Shape& qss = qsShape->GetStorageShape();
        int64_t qn = 1;
        for (size_t i = 0; i < qss.GetDimNum(); ++i) {
            qn *= qss.GetDim(i);
        }
        if (qn != 1) {
            qsPerChannel = 1;
        }
    }

    int64_t tsize = 4;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tsize = 2;
    }

    // UB budget per element: fx(4)+ta(4)+tb(4)+tc(4)+i8(1) + raw(tsize) + slack(4)
    int64_t perElem = 4 * 4 + 1 + tsize + 4;
    int64_t chunkMax = static_cast<int64_t>((ubSize * 55ULL / 100ULL) / static_cast<uint64_t>(perElem));
    chunkMax = (chunkMax / 64) * 64;
    if (chunkMax < 64) chunkMax = 64;
    if (chunkMax > 16384) chunkMax = 16384;
    int64_t chunk = (chunkMax < gsz) ? chunkMax : gsz;
    chunk = ((chunk + 63) / 64) * 64;
    if (chunk < 64) chunk = 64;

    int64_t blockDim = coreNum;
    if (blockDim > totalGroups) blockDim = totalGroups;
    if (blockDim < 1) blockDim = 1;

    tiling->N = N;
    tiling->C = C;
    tiling->G = G;
    tiling->CpG = CpG;
    tiling->HW = HW;
    tiling->gsz = gsz;
    tiling->totalGroups = totalGroups;
    tiling->chunk = chunk;
    tiling->qsPerChannel = qsPerChannel;
    tiling->activateSilu = activateSilu;
    tiling->epsilon = epsilon;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    uint64_t tilingKey = (tsize == 2) ? GET_TPL_TILING_KEY(GROUPNORMSILUQUANT_TPL_SCH_MODE_0)
                                      : GET_TPL_TILING_KEY(GROUPNORMSILUQUANT_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupNormSiluQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GroupNormSiluQuantCompileInfo {};

IMPL_OP_OPTILING(GroupNormSiluQuant)
    .Tiling(GroupNormSiluQuantTilingFunc)
    .TilingParse<GroupNormSiluQuantCompileInfo>(TilingParseForGroupNormSiluQuant);

} // namespace optiling

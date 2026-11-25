/*!
 * \file group_norm_tiling.cpp
 * \brief GroupNorm 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <algorithm>
#include "../op_kernel/group_norm_tiling_data.h"
#include "../op_kernel/group_norm_tiling_key.h"

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

static ge::graphStatus GroupNormTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    GroupNormTilingData* tiling = context->GetTilingData<GroupNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& xs = xShape->GetStorageShape();
    const size_t rank = xs.GetDimNum();
    OP_CHECK_IF(rank < 2, OP_LOGE(context, "x rank must be >= 2"), return ge::GRAPH_FAILED);

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
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* gp = attrs->GetAttrPointer<int64_t>(0);
        if (gp != nullptr) {
            G = *gp;
        }
        const float* ep = attrs->GetAttrPointer<float>(1);
        if (ep != nullptr) {
            epsilon = *ep;
        }
    }
    if (G < 1) G = 1;
    if (G > C) G = C;
    while (C % G != 0 && G > 1) {
        --G;
    }
    int64_t CpG = C / G;
    int64_t gsz = CpG * HW;
    int64_t totalGroups = N * G;

    int64_t tsize = 4;
    uint32_t schMode = GROUPNORM_TPL_SCH_MODE_1;   /* fp32 */
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tsize = 2;
        schMode = GROUPNORM_TPL_SCH_MODE_0;
    } else if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_BF16) {
        tsize = 2;
        schMode = GROUPNORM_TPL_SCH_MODE_2;
    }
    // UB budget: raw(T) + fbuf(4) + sqb(4) + oraw(T) + work(4) per element
    int64_t perElem = 2 * tsize + 4 + 4 + 4 + 32;  // + slack
    int64_t chunkMax = static_cast<int64_t>((ubSize * 55ULL / 100ULL) / static_cast<uint64_t>(perElem));
    chunkMax = (chunkMax / 64) * 64;
    if (chunkMax < 64) chunkMax = 64;
    if (chunkMax > 8192) chunkMax = 8192;
    int64_t chunk = (chunkMax < gsz) ? chunkMax : gsz;
    chunk = ((chunk + 63) / 64) * 64;   // multiple of 64
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
    tiling->epsilon = epsilon;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GroupNormCompileInfo {};

IMPL_OP_OPTILING(GroupNorm)
    .Tiling(GroupNormTilingFunc)
    .TilingParse<GroupNormCompileInfo>(TilingParseForGroupNorm);

} // namespace optiling

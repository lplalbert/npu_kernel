/*!
 * \file flash_attention_tiling.cpp
 * \brief FlashAttention 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <cmath>
#include <algorithm>
#include "../op_kernel/flash_attention_tiling_data.h"
#include "../op_kernel/flash_attention_tiling_key.h"

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

static ge::graphStatus FlashAttentionTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    FlashAttentionTilingData* tiling = context->GetTilingData<FlashAttentionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* qShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape& qs = qShape->GetStorageShape();
    OP_CHECK_IF(qs.GetDimNum() != 4, OP_LOGE(context, "q must be rank 4"), return ge::GRAPH_FAILED);
    int64_t B = qs.GetDim(0);
    int64_t N = qs.GetDim(1);
    int64_t S = qs.GetDim(2);
    int64_t D = qs.GetDim(3);
    if (B < 1) B = 1;
    if (N < 1) N = 1;
    if (S < 1) S = 1;
    if (D < 1) D = 1;

    // attributes
    float softmaxScale = 0.0f;
    int64_t isCausal = 0;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* sc = attrs->GetAttrPointer<float>(0);
        if (sc != nullptr) {
            softmaxScale = *sc;
        }
        const bool* ca = attrs->GetAttrPointer<bool>(1);
        if (ca != nullptr) {
            isCausal = (*ca) ? 1 : 0;
        }
    }
    if (softmaxScale <= 0.0f) {
        softmaxScale = 1.0f / static_cast<float>(std::sqrt(static_cast<double>(D)));
    }

    // optional mask (B,1,S,S)
    int64_t hasMask = 0;
    const gert::StorageShape* mShape = context->GetInputShape(3);
    if (mShape != nullptr && mShape->GetStorageShape().GetDimNum() == 4) {
        hasMask = 1;
    }

    // dtype size
    int64_t tsize = 4;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tsize = 2;
    }

    // key block size to fit UB: kraw(BK*D*tsize) + kf(BK*D*4) + fixed buffers
    int64_t fixed = 3 * D * 4 + std::max(D, S) * 4 + 128 + S * 4;
    if (hasMask) {
        fixed += S * tsize + S * 4;
    }
    if (isCausal) {
        fixed += S * 4;
    }
    int64_t avail = static_cast<int64_t>(ubSize * 8 / 10) - fixed;
    if (avail < static_cast<int64_t>(D) * (tsize + 4)) {
        avail = static_cast<int64_t>(D) * (tsize + 4);
    }
    int64_t BK = avail / (D * (tsize + 4));
    if (BK < 4) BK = 4;
    if (BK > 256) BK = 256;
    if (BK > S) BK = S;

    const int64_t totalRows = B * N * S;
    int64_t blockDim = coreNum;
    if (blockDim > totalRows) {
        blockDim = totalRows;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    tiling->B = B;
    tiling->N = N;
    tiling->S = S;
    tiling->D = D;
    tiling->isCausal = isCausal;
    tiling->hasMask = hasMask;
    tiling->totalRows = totalRows;
    tiling->keyBlock = BK;
    tiling->softmaxScale = softmaxScale;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    uint64_t tilingKey = (tsize == 2) ? GET_TPL_TILING_KEY(FLASHATTENTION_TPL_SCH_MODE_0)
                                      : GET_TPL_TILING_KEY(FLASHATTENTION_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFlashAttention([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FlashAttentionCompileInfo {};

IMPL_OP_OPTILING(FlashAttention)
    .Tiling(FlashAttentionTilingFunc)
    .TilingParse<FlashAttentionCompileInfo>(TilingParseForFlashAttention);

} // namespace optiling


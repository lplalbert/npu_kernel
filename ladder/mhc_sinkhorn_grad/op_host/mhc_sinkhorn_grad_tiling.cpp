/*!
 * \file mhc_sinkhorn_grad_tiling.cpp
 * \brief MhcSinkhornGrad operator tiling implementation
 *
 * Inputs:
 *   gradOutput: (B,S,n,n) or (T,n,n) fp32, upstream gradient, T = B*S
 *   normOut:    (2*numIters, T, n, n) forward normalization buffer
 *   sumOut:     (2*numIters, T, n)    forward reduction buffer
 * Output:
 *   gradInput:  same shape as gradOutput
 *
 * numIters is derived from normOut only (numIters = dim0 / 2), which is exactly
 * what the reference implementation does. sumOut is validated against it and a
 * mismatch is rejected: silently shrinking the iteration count would compute a
 * different problem and still report success.
 *
 * Samples are the parallel dimension: each core owns one contiguous, disjoint
 * range of samples, so no cross core reduction is needed and the result stays
 * deterministic.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mhc_sinkhorn_grad_tiling_data.h"
#include "../op_kernel/mhc_sinkhorn_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MAX_N = 8;

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

// Product of the leading dims, rejects empty or overflowing shapes.
static ge::graphStatus LeadingProd(const gert::Shape& shape, size_t dimNum, int64_t& prod)
{
    prod = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        int64_t d = shape.GetDim(i);
        if (d <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (prod > (INT64_MAX / d)) {
            return ge::GRAPH_FAILED;
        }
        prod *= d;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MhcSinkhornGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    MhcSinkhornGradTilingData* tiling = context->GetTilingData<MhcSinkhornGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto goShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, goShapePtr);
    const gert::Shape& goShape = goShapePtr->GetStorageShape();
    const size_t goDim = goShape.GetDimNum();
    // gradOutput is (T, n, n) or (B, S, n, n): at least two matrix dims are required.
    OP_CHECK_IF(goDim < 3, OP_LOGE(context, "gradOutput must be (T,n,n) or (B,S,n,n)"), return ge::GRAPH_FAILED);
    int64_t n = goShape.GetDim(goDim - 1);
    OP_CHECK_IF(n < 1 || n > MAX_N, OP_LOGE(context, "n must be in [1,%ld]", MAX_N), return ge::GRAPH_FAILED);
    OP_CHECK_IF(goShape.GetDim(goDim - 2) != n, OP_LOGE(context, "last two dims of gradOutput must both be n"),
                return ge::GRAPH_FAILED);
    int64_t T = 0;
    OP_CHECK_IF(LeadingProd(goShape, goDim - 2, T) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "bad gradOutput shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(T > (INT64_MAX / (n * n)), OP_LOGE(context, "gradOutput numel overflow"), return ge::GRAPH_FAILED);
    int64_t goNumel = T * n * n;

    auto normShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, normShapePtr);
    const gert::Shape& normShape = normShapePtr->GetStorageShape();
    const size_t normDim = normShape.GetDimNum();
    int64_t numIters = 0;
    if (normDim == 4) {
        OP_CHECK_IF(normShape.GetDim(1) != T || normShape.GetDim(2) != n || normShape.GetDim(3) != n,
                    OP_LOGE(context, "normOut must be (2*numIters,T,n,n)"), return ge::GRAPH_FAILED);
        OP_CHECK_IF((normShape.GetDim(0) % 2) != 0, OP_LOGE(context, "normOut dim0 must be even"),
                    return ge::GRAPH_FAILED);
        numIters = normShape.GetDim(0) / 2;
    } else {
        // rank check is not available in every caller, fall back to the element count
        int64_t normNumel = 0;
        OP_CHECK_IF(LeadingProd(normShape, normDim, normNumel) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "bad normOut shape"), return ge::GRAPH_FAILED);
        int64_t denom = 2 * T * n * n;
        OP_CHECK_IF(denom <= 0 || (normNumel % denom) != 0, OP_LOGE(context, "normOut numel mismatch"),
                    return ge::GRAPH_FAILED);
        numIters = normNumel / denom;
    }
    OP_CHECK_IF(numIters < 1, OP_LOGE(context, "numIters must be >= 1"), return ge::GRAPH_FAILED);

    // sumOut must describe exactly the same iteration count; a disagreement means the
    // inputs do not belong together, so fail instead of picking the smaller value.
    auto sumShapePtr = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShapePtr);
    const gert::Shape& sumShape = sumShapePtr->GetStorageShape();
    const size_t sumDim = sumShape.GetDimNum();
    if (sumDim == 3) {
        OP_CHECK_IF(sumShape.GetDim(0) != 2 * numIters || sumShape.GetDim(1) != T || sumShape.GetDim(2) != n,
                    OP_LOGE(context, "sumOut must be (2*numIters,T,n)"), return ge::GRAPH_FAILED);
    } else {
        int64_t sumNumel = 0;
        OP_CHECK_IF(LeadingProd(sumShape, sumDim, sumNumel) != ge::GRAPH_SUCCESS, OP_LOGE(context, "bad sumOut shape"),
                    return ge::GRAPH_FAILED);
        int64_t expect = 2 * numIters * T * n;
        OP_CHECK_IF(sumNumel != expect, OP_LOGE(context, "sumOut numel %ld != %ld", sumNumel, expect),
                    return ge::GRAPH_FAILED);
    }

    // samples are independent: give every core one contiguous block
    int64_t cores = coreNum;
    if (cores > T) {
        cores = T;
    }
    if (cores < 1) {
        cores = 1;
    }
    int64_t bf = (T + cores - 1) / cores;
    if (bf < 1) {
        bf = 1;
    }
    cores = (T + bf - 1) / bf;
    if (cores < 1) {
        cores = 1;
    }

    tiling->totalNum = goNumel;
    tiling->blockFactor = bf;
    tiling->ubFactor = 0;
    tiling->n = n;
    tiling->T = T;
    tiling->numIters = numIters;
    tiling->usedCores = cores;

    context->SetBlockDim(static_cast<uint32_t>(cores));

    // The operator definition only registers fp32, so there is a single schedule mode.
    context->SetTilingKey(GET_TPL_TILING_KEY(MHCSINKHORNGRAD_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMhcSinkhornGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MhcSinkhornGradCompileInfo {};

IMPL_OP_OPTILING(MhcSinkhornGrad).Tiling(MhcSinkhornGradTilingFunc).TilingParse<MhcSinkhornGradCompileInfo>(TilingParseForMhcSinkhornGrad);

} // namespace optiling

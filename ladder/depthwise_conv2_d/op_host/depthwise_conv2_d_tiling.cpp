/*!
 * \file depthwise_conv2_d_tiling.cpp
 * \brief DepthwiseConv2D tiling.
 *
 * Work distribution: output rows (n, c, oh) are split into disjoint contiguous
 * ranges across AIV cores. Each core fully owns the rows of its range, so there
 * is no cross-core reduction, no synchronisation and no workspace usage; the
 * result is deterministic by construction.
 *
 * The vector path needs a per-kw gather offset table of kw*accSize int32 plus a
 * few row sized float buffers. The host checks that footprint against the UB
 * budget and falls back to the element-wise scalar path when it does not fit.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/depthwise_conv2_d_tiling_data.h"
#include "../op_kernel/depthwise_conv2_d_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t STAGE_CHUNK = 1024;   /* scalar path staging chunk, mult of 64 */
constexpr int64_t MIN_VEC_ELEMS = 64;   /* smallest allowed vector op count */
constexpr int64_t MAX_VEC_COUNT = 16320;/* 255 repeats * 64 elements */
constexpr int64_t UB_BUDGET_NUM = 7;    /* used fraction of UB: 7/10 */
constexpr int64_t UB_BUDGET_DEN = 10;

static inline int64_t Align64(int64_t v)
{
    return ((v + 63) / 64) * 64;
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

static ge::graphStatus DepthwiseConv2DTilingFunc(gert::TilingContext* context)
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

    DepthwiseConv2DTilingData* tiling = context->GetTilingData<DepthwiseConv2DTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto xShapePtr = context->GetInputShape(0);
    auto wShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    const gert::Shape& xs = xShapePtr->GetStorageShape();
    const gert::Shape& ws = wShapePtr->GetStorageShape();
    OP_CHECK_IF(xs.GetDimNum() != 4 || ws.GetDimNum() != 4,
                OP_LOGE(context, "DepthwiseConv2D expects 4-D x and w"), return ge::GRAPH_FAILED);

    const int64_t n = xs.GetDim(0);
    const int64_t chIn = xs.GetDim(1);
    const int64_t hin = xs.GetDim(2);
    const int64_t win = xs.GetDim(3);
    const int64_t wCin = ws.GetDim(0);
    const int64_t wGroup = ws.GetDim(1);
    const int64_t kh = ws.GetDim(2);
    const int64_t kw = ws.GetDim(3);

    /* the kernel instantiates a single element type for both x and w, so they
     * have to share one dtype; reject fp16/fp32 mixtures before any access */
    auto xDesc = context->GetInputDesc(0);
    auto wDesc = context->GetInputDesc(1);
    OP_CHECK_IF(xDesc == nullptr || wDesc == nullptr,
                OP_LOGE(context, "DepthwiseConv2D missing input tensor desc"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDesc->GetDataType() != wDesc->GetDataType(),
                OP_LOGE(context, "DepthwiseConv2D requires x and w to have the same dtype"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(wCin != chIn || wGroup != 1,
                OP_LOGE(context, "DepthwiseConv2D w must be (C, 1, kH, kW)"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(n < 1 || chIn < 1 || hin < 1 || win < 1 || kh < 1 || kw < 1,
                OP_LOGE(context, "DepthwiseConv2D dims must be positive"), return ge::GRAPH_FAILED);

    int64_t stride = 1;
    int64_t pad = 0;
    int64_t dil = 1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            stride = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(1);
        if (p != nullptr) {
            pad = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(2);
        if (p != nullptr) {
            dil = *p;
        }
    }
    OP_CHECK_IF(stride < 1 || dil < 1 || pad < 0,
                OP_LOGE(context, "DepthwiseConv2D invalid stride/pad/dilation"), return ge::GRAPH_FAILED);

    const int64_t spanH = hin + 2 * pad - dil * (kh - 1) - 1;
    const int64_t spanW = win + 2 * pad - dil * (kw - 1) - 1;
    OP_CHECK_IF(spanH < 0 || spanW < 0,
                OP_LOGE(context, "DepthwiseConv2D kernel larger than padded input"), return ge::GRAPH_FAILED);
    const int64_t hout = spanH / stride + 1;
    const int64_t wout = spanW / stride + 1;
    OP_CHECK_IF(hout <= 0 || wout <= 0,
                OP_LOGE(context, "DepthwiseConv2D output H/W must be positive"), return ge::GRAPH_FAILED);

    int32_t hasBias = 0;
    auto bShapePtr = context->GetOptionalInputShape(2);
    if (bShapePtr != nullptr && bShapePtr->GetStorageShape().GetDimNum() >= 1) {
        int64_t bSize = bShapePtr->GetStorageShape().GetShapeSize();
        if (bSize > 0) {
            OP_CHECK_IF(bSize != chIn,
                        OP_LOGE(context, "DepthwiseConv2D bias length must equal C"), return ge::GRAPH_FAILED);
            hasBias = 1;
        }
    }

    const int64_t totalOut = n * chIn * hout * wout;
    const int64_t totalRows = n * chIn * hout;

    auto inputDesc = context->GetInputDesc(0);
    const bool isF16 = (inputDesc != nullptr &&
                        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16));

    /* vector path geometry mirrored by the kernel */
    const int64_t accSize = Align64(wout);
    const int64_t zoff = Align64(win) + 64; /* one spare block above the row */
    const int64_t xSize = zoff + 64;
    const int64_t wPad = Align64(kh * kw);

    int64_t vecFloats = xSize + accSize + accSize + wPad + kw * accSize;
    if (isF16) {
        vecFloats += (xSize + wPad + accSize) / 2; /* io dtype staging buffers */
    }
    const int64_t ubBudget = static_cast<int64_t>(ubSize) * UB_BUDGET_NUM / UB_BUDGET_DEN;

    int32_t useScalar = 0;
    if (accSize > MAX_VEC_COUNT || accSize < MIN_VEC_ELEMS || vecFloats * 4 > ubBudget) {
        useScalar = 1;
    }

    int64_t cores = coreNum;
    if (useScalar == 0) {
        if (cores > totalRows) {
            cores = totalRows;
        }
    } else {
        int64_t maxCores = totalOut / 64;
        if (maxCores < 1) {
            maxCores = 1;
        }
        if (cores > maxCores) {
            cores = maxCores;
        }
    }
    if (cores < 1) {
        cores = 1;
    }
    int64_t rowsPerCore = CeilDiv<int64_t>(useScalar == 0 ? totalRows : totalOut, cores);
    if (rowsPerCore < 1) {
        rowsPerCore = 1;
    }

    tiling->totalNum = totalOut;
    tiling->blockFactor = rowsPerCore;
    tiling->ubFactor = STAGE_CHUNK;
    tiling->n = n;
    tiling->chIn = chIn;
    tiling->hin = hin;
    tiling->win = win;
    tiling->hout = hout;
    tiling->wout = wout;
    tiling->kh = kh;
    tiling->kw = kw;
    tiling->stride = stride;
    tiling->pad = pad;
    tiling->dil = dil;
    tiling->totalRows = totalRows;
    tiling->accSize = accSize;
    tiling->xSize = xSize;
    tiling->zoff = zoff;
    tiling->wPad = wPad;
    tiling->hasBias = hasBias;
    tiling->usedCores = static_cast<int32_t>(cores);
    tiling->useScalar = useScalar;

    context->SetBlockDim(static_cast<uint32_t>(cores));

    uint64_t tilingKey;
    if (isF16) {
        tilingKey = GET_TPL_TILING_KEY(DEPTHWISECONV2D_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(DEPTHWISECONV2D_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDepthwiseConv2D([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DepthwiseConv2DCompileInfo {};

IMPL_OP_OPTILING(DepthwiseConv2D).Tiling(DepthwiseConv2DTilingFunc).TilingParse<DepthwiseConv2DCompileInfo>(TilingParseForDepthwiseConv2D);

} // namespace optiling

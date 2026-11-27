/*!
 * \file depthwise_conv2_d_tiling.cpp
 * \brief DepthwiseConv2D tiling.
 *
 * Work distribution: the output is split into "row groups".  A group produces
 * R consecutive output rows of one (n, c) channel plane at once, packed into a
 * single vector of length count = align64(R*wout).  Grouping amortises the
 * per-row fixed cost (barriers, tiny GM copies, short vectors) that dominates
 * when wout is small, and lengthens every vector op.  Groups are partitioned
 * into disjoint contiguous ranges across AIV cores: no cross-core reduction,
 * no synchronisation, no workspace.
 *
 * The vector path also needs a single gather offset table of `count` int32 and
 * one input tile of rowSpan*WPtile floats.  The host checks that footprint
 * against the UB budget and falls back to an element-wise scalar path when it
 * does not fit.
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
constexpr int64_t STAGE_CHUNK = 1024;    /* scalar path staging chunk, mult of 64 */
constexpr int64_t MIN_VEC_ELEMS = 64;    /* smallest allowed vector op count */
constexpr int64_t MAX_VEC_COUNT = 16320; /* 255 repeats * 64 elements */
constexpr int64_t PREF_VEC_COUNT = 8192; /* preferred upper bound on the vector length */
constexpr int64_t MAX_R = 64;            /* hard cap on rows per group */
constexpr int64_t UB_BUDGET_NUM = 9;     /* used fraction of UB: 9/10 */
constexpr int64_t UB_BUDGET_DEN = 10;

static inline int64_t Align64(int64_t v)
{
    return ((v + 63) / 64) * 64;
}
static inline int64_t Align8(int64_t v)
{
    return ((v + 7) / 8) * 8;
}
static inline int64_t Align16(int64_t v)
{
    return ((v + 15) / 16) * 16;
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
    auto inputDesc = context->GetInputDesc(0);
    const bool isF16 = (inputDesc != nullptr &&
                        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16));

    /* ---- geometry mirrored exactly by the kernel ---- */
    /* LP is a multiple of 16 so that half-typed tile rows written directly into
     * UB are 32B aligned; WPtile is a multiple of 64 so a whole-tile Cast has a
     * length that is a multiple of 64. */
    const int64_t LP = Align16(pad);
    int64_t colMax = (wout - 1) * stride + (kw - 1) * dil - pad + LP;
    int64_t needW = colMax + 1;
    int64_t needW2 = LP + win + 64;
    const int64_t WPtile = Align64((needW > needW2) ? needW : needW2);
    const int64_t wPad = Align64(kh * kw);

    const int64_t ubBudget = static_cast<int64_t>(ubSize) * UB_BUDGET_NUM / UB_BUDGET_DEN;

    /* choose R: largest row-group size that fits UB and the vector-length limits
     * and still leaves at least coreNum groups whenever possible */
    int64_t rByCount = PREF_VEC_COUNT / (wout > 0 ? wout : 1);
    if (rByCount < 1) {
        rByCount = 1;
    }
    int64_t rCap = hout;
    if (rCap > rByCount) {
        rCap = rByCount;
    }
    if (rCap > MAX_R) {
        rCap = MAX_R;
    }
    if (rCap < 1) {
        rCap = 1;
    }

    int32_t useScalar = 0;
    int64_t R = 0;
    int64_t rowSpan = 1;
    int64_t count = 64;
    bool found = false;
    for (int64_t r = rCap; r >= 1; --r) {
        int64_t rs = (r - 1) * stride + (kh - 1) * dil + 1;
        int64_t ct = Align64(r * wout);
        if (ct > MAX_VEC_COUNT) {
            continue;
        }
        int64_t tileB = rs * WPtile * 4;
        int64_t accB = ct * 4 * 3; /* acc + gather dst + offset table */
        int64_t wB = wPad * 4;
        int64_t extra = 0;
        if (isF16) {
            /* full io-dtype tile staging (whole-tile Cast) + weight staging + output staging */
            extra = rs * WPtile * 2 + wPad * 2 + ct * 2;
        }
        int64_t total = tileB + accB + wB + extra;
        int64_t groups = n * chIn * CeilDiv<int64_t>(hout, r);
        bool enoughCores = (groups >= coreNum) || (r == 1);
        if (total <= ubBudget && enoughCores) {
            R = r;
            rowSpan = rs;
            count = ct;
            found = true;
            break;
        }
    }
    if (!found) {
        useScalar = 1;
        R = 1;
        rowSpan = (kh - 1) * dil + 1;
        count = Align64(wout < MIN_VEC_ELEMS ? MIN_VEC_ELEMS : wout);
    }
    if (count < MIN_VEC_ELEMS) {
        count = MIN_VEC_ELEMS;
    }

    const int64_t gpc = CeilDiv<int64_t>(hout, R);
    const int64_t totalGroups = n * chIn * gpc;

    int64_t cores = coreNum;
    if (useScalar == 0) {
        if (cores > totalGroups) {
            cores = totalGroups;
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

    tiling->totalNum = totalOut;
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
    tiling->R = R;
    tiling->rowSpan = rowSpan;
    tiling->LP = LP;
    tiling->WPtile = WPtile;
    tiling->count = count;
    tiling->gpc = gpc;
    tiling->totalGroups = totalGroups;
    tiling->wPad = wPad;
    tiling->hasBias = hasBias;
    tiling->useScalar = useScalar;
    tiling->usedCores = static_cast<int32_t>(cores);

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

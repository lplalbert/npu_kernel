/*!
 * \file index_tiling.cpp
 * \brief Index  Tiling
 *
 *  indexed_sizes  tiling
 * //dtype  tiling x indices
 * mask  offset  kernel  indexed_sizes
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/index_tiling_data.h"
#include "../op_kernel/index_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t kMinElemsPerCore = 1024;

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

static int64_t ShapeElems(const gert::Shape& shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t d = shape.GetDim(i);
        if (d < 0) {
            d = 0;
        }
        n *= d;
    }
    return n;
}

static ge::graphStatus IndexTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    IndexTilingData* tiling = context->GetTilingData<IndexTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    if (dimNum > INDEX_MAX_DIM) {
        OP_LOGE(context, "Index: x rank %ld exceeds max supported %d", static_cast<long>(dimNum), INDEX_MAX_DIM);
        return ge::GRAPH_FAILED;
    }
    int64_t shape[INDEX_MAX_DIM] = {0};
    int64_t stride[INDEX_MAX_DIM] = {0};
    int64_t xElems = 1;
    for (int64_t i = 0; i < dimNum; ++i) {
        int64_t d = xShape.GetDim(i);
        if (d < 0) {
            d = 0; //  0 /
        }
        shape[i] = d;
        xElems *= d;
    }
    int64_t acc = 1;
    for (int64_t i = dimNum - 1; i >= 0; --i) {
        stride[i] = acc;
        acc *= shape[i];
    }

    int64_t idxNum = 1;
    auto idxShapePtr = context->GetInputShape(2);
    if (idxShapePtr != nullptr) {
        idxNum = ShapeElems(idxShapePtr->GetStorageShape());
        if (idxNum < 0) {
            idxNum = 1;
        }
    }

    int64_t outElems = 0;
    auto yShapePtr = context->GetOutputShape(0);
    if (yShapePtr != nullptr) {
        outElems = ShapeElems(yShapePtr->GetStorageShape());
    }
    // / indices  outElems  0 0 kernel

    int64_t idxBytes = 8;
    auto idxDesc = context->GetInputDesc(2);
    if (idxDesc != nullptr && idxDesc->GetDataType() == ge::DT_INT32) {
        idxBytes = 4;
    }

    uint64_t tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_1);
    auto xDesc = context->GetInputDesc(0);
    if (xDesc != nullptr) {
        ge::DataType dt = xDesc->GetDataType();
        if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_0);
        } else if (dt == ge::DT_INT64) {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_2);
        } else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8 || dt == ge::DT_BOOL) {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_3);
        } else {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_1);
        }
    }

    int64_t cores = coreNum;
    int64_t need = (outElems + kMinElemsPerCore - 1) / kMinElemsPerCore;
    if (need < cores) {
        cores = need;
    }
    if (cores < 1) {
        cores = 1;
    }

    tiling->dimNum = dimNum;
    for (int64_t i = 0; i < INDEX_MAX_DIM; ++i) {
        tiling->shape[i] = shape[i];
        tiling->stride[i] = stride[i];
    }
    tiling->idxNum = idxNum;
    tiling->outElems = outElems;
    tiling->xElems = xElems;
    tiling->idxBytes = idxBytes;
    tiling->usedCores = cores;

    context->SetBlockDim(static_cast<uint32_t>(cores));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIndex([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct IndexCompileInfo {};

IMPL_OP_OPTILING(Index).Tiling(IndexTilingFunc).TilingParse<IndexCompileInfo>(TilingParseForIndex);

} // namespace optiling

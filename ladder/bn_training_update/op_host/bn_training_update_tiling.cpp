/*!
 * \file bn_training_update_tiling.cpp
 * \brief BnTrainingUpdate tiling implementation
 *
 * The flat element space is split along rows, where a row is one contiguous
 * (n, c) plane of S = H*W elements. Each core owns a contiguous block of rows,
 * so it also owns a contiguous element range and no cross core combine is
 * needed. Sizes handed to the kernel:
 *   blockFactor   rows per core
 *   ubFactor      elements per inner UB chunk (multiple of 64)
 *   channelNum    C, rowLength S, totalRows N*C, elemsPerChannel N*S
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/bn_training_update_tiling_data.h"
#include "../op_kernel/bn_training_update_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t kAlign = 64;      // vector op count alignment (one 256B repeat)
constexpr int64_t kMaxUb = 2048;    // upper bound on the inner chunk
constexpr int64_t kMinUb = 64;      // lower bound on the inner chunk
constexpr int64_t kChanBytes = 20;  // 5 per channel fp32 tables: mean/var/den/scale/offset
constexpr int64_t kMaxChan = 8192;  // guard for the per channel UB tables

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

static ge::graphStatus BnTrainingUpdateTilingFunc(gert::TilingContext* context)
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

    BnTrainingUpdateTilingData* tiling = context->GetTilingData<BnTrainingUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());

    int64_t n = 1;
    int64_t c = 1;
    int64_t s = 1;
    if (rank >= 2) {
        n = xShape.GetDim(0);
        c = xShape.GetDim(1);
        for (int64_t i = 2; i < rank; ++i) {
            s *= xShape.GetDim(i);
        }
    } else if (rank == 1) {
        c = xShape.GetDim(0);
    }
    if (n < 1) {
        n = 1;
    }
    if (c < 1) {
        c = 1;
    }
    if (s < 1) {
        s = 1;
    }
    OP_CHECK_IF(c > kMaxChan, OP_LOGE(context, "channelNum is too large"), return ge::GRAPH_FAILED);

    const int64_t rows = n * c;
    const int64_t nElem = n * s;
    const int64_t total = rows * s;

    uint64_t tilingKey;
    bool isF16 = false;
    auto inputDesc = context->GetInputDesc(0);
    // Only the dtypes declared in the op def are dispatched here: float16 uses
    // the half kernel template, everything else (float32) the float template.
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(BNTRAININGUPDATE_TPL_SCH_MODE_0);
        isF16 = true;
    } else {
        tilingKey = GET_TPL_TILING_KEY(BNTRAININGUPDATE_TPL_SCH_MODE_1);
    }

    float epsilon = 1e-5f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* p = attrs->GetAttrPointer<float>(0);
        if (p != nullptr) {
            epsilon = *p;
        }
    }

    // Rows per core (each core gets at least one row, and no core is spawned
    // for an empty range).
    int64_t cores = (coreNum > 0) ? coreNum : 1;
    int64_t rowsPerCore = CeilDiv(rows, cores);
    if (rowsPerCore < 1) {
        rowsPerCore = 1;
    }
    int64_t usedCores = CeilDiv(rows, rowsPerCore);
    if (usedCores < 1) {
        usedCores = 1;
    }
    if (usedCores > cores) {
        usedCores = cores;
    }

    // Inner UB chunk. Buffers: input tile + output tile (+ fp32 work tile for
    // fp16) plus the five per channel tables.
    const int64_t perElemBytes = 2 * static_cast<int64_t>(isF16 ? 2 : 4) +
                                 (isF16 ? static_cast<int64_t>(sizeof(float)) : 0);
    const int64_t tblBytes = CeilDiv(c, kAlign) * kAlign * kChanBytes;
    int64_t avail = static_cast<int64_t>(ubSize * 9 / 10) - tblBytes;
    int64_t ub = kMinUb;
    if (avail > 0) {
        ub = (avail / perElemBytes) / kAlign * kAlign;
        if (ub > kMaxUb) {
            ub = kMaxUb;
        }
        if (ub < kMinUb) {
            ub = kMinUb;
        }
    }

    tiling->totalNum = total;
    tiling->blockFactor = rowsPerCore;
    tiling->ubFactor = ub;
    tiling->channelNum = c;
    tiling->rowLength = s;
    tiling->totalRows = rows;
    tiling->elemsPerChannel = nElem;
    tiling->epsilon = epsilon;
    tiling->usedCores = static_cast<int32_t>(usedCores);

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBnTrainingUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BnTrainingUpdateCompileInfo {};

IMPL_OP_OPTILING(BnTrainingUpdate).Tiling(BnTrainingUpdateTilingFunc).TilingParse<BnTrainingUpdateCompileInfo>(TilingParseForBnTrainingUpdate);

}  // namespace optiling

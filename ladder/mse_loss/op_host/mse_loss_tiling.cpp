/*!
 * \file mse_loss_tiling.cpp
 * \brief MseLoss tiling implementation
 *
 * Splits the flat element dim across AI vector cores. Inner UB chunk (ubFactor)
 * is always a multiple of 64 elements so that every vector op count is one or
 * more full 256B repeats (fast path). fp32 reduction uses all cores with a 32B
 * atomic add for the cross-core combine; fp16 reduction runs single core.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mse_loss_tiling_data.h"
#include "../op_kernel/mse_loss_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t kElemAlign = 64;      // vector op count alignment
constexpr int64_t kMaxUb = 8192;        // upper bound on inner chunk elements
constexpr int64_t kMinPerCore = 8192;   // do not spawn a core for < this many elem

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

static ge::graphStatus MseLossTilingFunc(gert::TilingContext* context)
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

    MseLossTilingData* tiling = context->GetTilingData<MseLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& shape = inputShapePtr->GetStorageShape();
    int64_t total = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        total *= shape.GetDim(i);
    }
    if (total < 1) {
        total = 1;
    }

    uint64_t tilingKey;
    bool isF16 = false;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_0);
        isF16 = true;
    } else {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_1);
    }

    int64_t reduction = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            reduction = *p;
        }
    }
    if (reduction < 0 || reduction > 2) {
        reduction = 1;
    }

    // Inner chunk: every per-element buffer fits inside 9/10 of UB. Buffers use
    // BUFFER_NUM == 3 double/triple buffering in the kernel.
    //   none   : 3 queues * 3 * sizeof(T) + (fp16: 3 float cast/compute buffers)
    //   reduce : 2 queues * 3 * sizeof(T) + 1 float compute + 1 float accumulator
    //            (+ 2 float cast buffers for fp16)
    int64_t perElemBytes;
    if (reduction == 0) {
        perElemBytes = isF16 ? 30 : 36;
    } else {
        perElemBytes = isF16 ? 28 : 32;
    }
    int64_t raw = static_cast<int64_t>((ubSize * 9 / 10) / static_cast<uint64_t>(perElemBytes));
    if (raw > kMaxUb) {
        raw = kMaxUb;
    }
    int64_t ub;
    if (reduction == 0) {
        // elementwise: no lane fold, so a plain multiple of 64 up to raw is fine
        ub = (raw / kElemAlign) * kElemAlign;
        if (ub < kElemAlign) {
            ub = kElemAlign;
        }
        // do not compute a huge padded chunk for a tiny tensor
        int64_t need = ((total + kElemAlign - 1) / kElemAlign) * kElemAlign;
        if (need < ub) {
            ub = need;
        }
        if (ub < kElemAlign) {
            ub = kElemAlign;
        }
    } else {
        // reduction: use a power of two so the final lane fold keeps power-of-two
        // (hence multiple of 64) vector counts all the way down to 128
        ub = kElemAlign;
        while (ub * 2 <= raw) {
            ub *= 2;
        }
        // shrink toward total so a small tensor is finished in one chunk
        while (ub > kElemAlign && (ub / 2) >= total) {
            ub /= 2;
        }
    }

    int64_t cores = coreNum;
    if (reduction != 0 && isF16) {
        cores = 1; // half atomic add is unreliable -> single core for fp16 reduction
    }
    int64_t need = (total + kMinPerCore - 1) / kMinPerCore;
    if (need < cores) {
        cores = need;
    }
    if (cores < 1) {
        cores = 1;
    }
    int64_t per = (total + cores - 1) / cores;
    if (per < 1) {
        per = 1;
    }

    tiling->totalNum = total;
    tiling->blockFactor = per;
    tiling->ubFactor = ub;
    tiling->reduction = reduction;
    tiling->usedCores = static_cast<int32_t>(cores);

    context->SetBlockDim(static_cast<uint32_t>(cores));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMseLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MseLossCompileInfo {};

IMPL_OP_OPTILING(MseLoss).Tiling(MseLossTilingFunc).TilingParse<MseLossCompileInfo>(TilingParseForMseLoss);

} // namespace optiling

/*!
 * \file multi_add_rms_norm_dynamic_quant_tiling.cpp
 * \brief MultiAddRmsNormDynamicQuant tiling implementation
 *
 * Row parallel: each core owns rowsPerCore rows; inside a row the data is split
 * into tileLen-sized tiles (multiples of 64).
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/multi_add_rms_norm_dynamic_quant_tiling_data.h"
#include "../op_kernel/multi_add_rms_norm_dynamic_quant_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr int64_t kAlign = 64;
constexpr int64_t kBytesPerTilePre = 32;    // preload path: UB bytes per tile element (no gamma/smooth tile, incl. i32)
constexpr int64_t kBytesPerTileNoPre = 48;  // non-preload path (adds gamma/smooth tiles + i32)
constexpr int64_t kPreBytesPerH = 12;       // preloaded gamma/smooth1/smooth2 as fp32: 3*4 bytes/element
constexpr int64_t kReserve = 8192;          // reserve: scBuf/oneBuf/slack

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum,
                                       size_t& sysWs)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWs = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MultiAddRmsNormDynamicQuantTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    size_t sysWs = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum, sysWs) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dt = inputDesc->GetDataType();
    uint64_t tilingKey;
    if (dt == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(MULTIADDRMSNORMDYNAMICQUANT_TPL_SCH_MODE_1);
    } else if (dt == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(MULTIADDRMSNORMDYNAMICQUANT_TPL_SCH_MODE_0);
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& shape = context->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "expect 2-D (B,H)"), return ge::GRAPH_FAILED);
    int64_t B = shape.GetDim(0);
    int64_t H = shape.GetDim(1);
    OP_CHECK_IF(B <= 0 || H <= 0, OP_LOGE(context, "bad B/H"), return ge::GRAPH_FAILED);
    int64_t Hpad = ((H + kAlign - 1) / kAlign) * kAlign;

    // add list presence: bit0..4 = x1a..x1e, bit5 = x2
    uint32_t inMask = (1u << 0);  // x1a is mandatory
    for (int i = 1; i <= 4; ++i) {
        if (context->GetOptionalInputTensor(i) != nullptr) {
            inMask |= (1u << i);
        }
    }
    inMask |= (1u << 5);  // x2 is mandatory

    float epsilon = 1e-6f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* pe = attrs->GetAttrPointer<float>(0);
        if (pe != nullptr) {
            epsilon = *pe;
        }
    }

    int32_t cores = coreNum > 0 ? static_cast<int32_t>(coreNum) : 1;
    if (static_cast<int64_t>(cores) > B) {
        cores = static_cast<int32_t>(B);
    }
    if (cores < 1) {
        cores = 1;
    }
    int64_t rowsPerCore = (B + cores - 1) / cores;

    int64_t budget = static_cast<int64_t>(ubSize) * 3 / 4;
    int64_t rowBytes = 4 * Hpad;
    int64_t preBytes = kPreBytesPerH * Hpad;
    // Preload gamma/smooth1/smooth2 into UB as fp32. Requires that the whole row
    // still fits one tile (splitting it into tiles would only slow things down) and
    // that each core owns at least two rows (so the preload is amortized).
    int64_t capPre = (budget - rowBytes - preBytes - kReserve) / kBytesPerTilePre;
    capPre = (capPre / kAlign) * kAlign;
    bool preload = (rowsPerCore >= 2) && (capPre >= Hpad);
    int64_t tileLen;
    if (preload) {
        tileLen = Hpad;  // one tile per row
    } else {
        int64_t cap = (budget - rowBytes - kReserve) / kBytesPerTileNoPre;
        cap = (cap / kAlign) * kAlign;
        if (cap < kAlign) {
            cap = kAlign;
        }
        tileLen = (Hpad < cap ? Hpad : cap);
        tileLen = (tileLen / kAlign) * kAlign;
        if (tileLen < kAlign) {
            tileLen = kAlign;
        }
    }

    MultiAddRmsNormDynamicQuantTilingData* tiling = context->GetTilingData<MultiAddRmsNormDynamicQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->B = B;
    tiling->H = H;
    tiling->Hpad = Hpad;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tileLen = tileLen;
    tiling->usedCores = cores;
    tiling->inMask = inMask;
    tiling->preload = preload ? 1u : 0u;
    tiling->epsilon = epsilon;

    context->SetBlockDim(static_cast<uint32_t>(cores));
    context->SetTilingKey(tilingKey);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWs;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMultiAddRmsNormDynamicQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MultiAddRmsNormDynamicQuantCompileInfo {};

IMPL_OP_OPTILING(MultiAddRmsNormDynamicQuant).Tiling(MultiAddRmsNormDynamicQuantTilingFunc).TilingParse<MultiAddRmsNormDynamicQuantCompileInfo>(TilingParseForMultiAddRmsNormDynamicQuant);

} // namespace optiling

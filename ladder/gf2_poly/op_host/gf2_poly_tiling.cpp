/*!
 * \file gf2_poly_tiling.cpp
 * \brief Gf2Poly tiling implementation
 *
 * Model: one GF(2) polynomial per row, rows are independent and share one
 * runtime modulus P. Rows are split across AIV cores (no workspace, no
 * cross-core sync, no atomics); each core streams tiles of tileRows rows
 * through UB. tileBytes targets 8KB and is bounded by the UB budget.
 *
 * Input contract (same as the problem reference implementation):
 *   a, b : uint8 [N, W],  t : uint8 [N, 2W],  mod_poly : uint8 [W+1] (monic),
 *   W = m/8, m in {8,32,64,128,256}, mode in {mul, reduce, mul_reduce}
 * Output y : uint8 [N, 2W] for mode "mul", [N, W] otherwise.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/gf2_poly_tiling_data.h"
#include "../op_kernel/gf2_poly_tiling_key.h"
#include <cstring>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TARGET_TILE_BYTES = 8192;
constexpr int64_t MAX_ROWS = 1LL << 20;

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

// Shared by tiling and shape inference so both always agree.
static inline bool ValidM(int64_t mVal)
{
    return mVal == 8 || mVal == 32 || mVal == 64 || mVal == 128 || mVal == 256;
}

static ge::graphStatus Gf2PolyTilingFunc(gert::TilingContext* context)
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

    Gf2PolyTilingData* tiling = context->GetTilingData<Gf2PolyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- attributes: m (int, required), mode (string, optional) ----
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* mPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, mPtr);
    const int64_t mVal = *mPtr;
    OP_CHECK_IF(!ValidM(mVal), OP_LOGE(context, "attr m must be 8/32/64/128/256"), return ge::GRAPH_FAILED);

    int64_t mode = 2;  // 0 mul, 1 reduce, 2 mul_reduce (default)
    const char* modePtr = attrs->GetAttrPointer<char>(1);
    if (modePtr != nullptr) {
        if (std::strcmp(modePtr, "mul") == 0) {
            mode = 0;
        } else if (std::strcmp(modePtr, "reduce") == 0) {
            mode = 1;
        } else if (std::strcmp(modePtr, "mul_reduce") == 0) {
            mode = 2;
        } else {
            OP_LOGE(context, "attr mode must be mul/reduce/mul_reduce");
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t width = mVal / 8;
    const int64_t outWidth = (mode == 0) ? (2 * width) : width;

    // ---- dtype: only uint8 is registered in the op definition ----
    for (size_t i = 0; i < 4; ++i) {
        auto inDesc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, inDesc);
        OP_CHECK_IF(
            inDesc->GetDataType() != ge::DT_UINT8,
            OP_LOGE(context, "only uint8 inputs are supported"),
            return ge::GRAPH_FAILED);
    }
    auto outDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    OP_CHECK_IF(
        outDesc->GetDataType() != ge::DT_UINT8,
        OP_LOGE(context, "only uint8 output is supported"),
        return ge::GRAPH_FAILED);

    // ---- shape contract, checked explicitly instead of truncating ----
    const gert::StorageShape* aStorage = context->GetInputShape(0);
    const gert::StorageShape* bStorage = context->GetInputShape(1);
    const gert::StorageShape* tStorage = context->GetInputShape(2);
    const gert::StorageShape* pStorage = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, aStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, bStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, tStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, pStorage);
    const gert::Shape& aShape = aStorage->GetStorageShape();
    const gert::Shape& bShape = bStorage->GetStorageShape();
    const gert::Shape& tShape = tStorage->GetStorageShape();
    const gert::Shape& pShape = pStorage->GetStorageShape();

    OP_CHECK_IF(aShape.GetDimNum() != 2, OP_LOGE(context, "a must be rank 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(bShape.GetDimNum() != 2, OP_LOGE(context, "b must be rank 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tShape.GetDimNum() != 2, OP_LOGE(context, "t must be rank 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(pShape.GetDimNum() != 1, OP_LOGE(context, "mod_poly must be rank 1"), return ge::GRAPH_FAILED);

    const int64_t totalRows = aShape.GetDim(0);
    OP_CHECK_IF(
        totalRows < 0 || totalRows > MAX_ROWS,
        OP_LOGE(context, "N out of range [0, 2^20]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        aShape.GetDim(1) != width || bShape.GetDim(0) != totalRows || bShape.GetDim(1) != width,
        OP_LOGE(context, "a/b must be [N, m/8] with equal shapes"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tShape.GetDim(0) != totalRows || tShape.GetDim(1) != 2 * width,
        OP_LOGE(context, "t must be [N, 2*m/8]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        pShape.GetDim(0) != width + 1,
        OP_LOGE(context, "mod_poly must be m/8 + 1 bytes"),
        return ge::GRAPH_FAILED);

    // ---- UB tile sized to ~8KB, bounded by the UB budget ----
    int64_t limit = (int64_t)(ubSize / 8);
    limit = (limit / 32) * 32;
    if (limit < 32) {
        limit = 32;
    }
    int64_t tileBytes = TARGET_TILE_BYTES;
    if (tileBytes > limit) {
        tileBytes = limit;
    }
    tileBytes = (tileBytes / width) * width;
    if (tileBytes < width) {
        tileBytes = width;
    }
    tileBytes = ((tileBytes + 31) / 32) * 32;  // keep InitBuffer 32B aligned
    int64_t tileRows = tileBytes / width;
    if (tileRows < 1) {
        tileRows = 1;
    }

    tiling->totalRows = totalRows;
    tiling->width = width;
    tiling->outWidth = outWidth;
    tiling->m = mVal;
    tiling->mode = mode;
    tiling->tileRows = tileRows;
    tiling->reserved0 = 0;
    tiling->reserved1 = 0;

    // ---- empty tensor: launch a single core that does nothing ----
    if (totalRows == 0) {
        tiling->rowsPerCore = 0;
        context->SetBlockDim(1U);
        context->SetTilingKey(GET_TPL_TILING_KEY(GF2POLY_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // ---- multi-core split by rows ----
    int64_t bd = coreNum;
    if (bd > totalRows) {
        bd = totalRows;
    }
    if (bd < 1) {
        bd = 1;
    }
    int64_t rowsPerCore = CeilDiv(totalRows, bd);
    if (rowsPerCore < 1) {
        rowsPerCore = 1;
    }
    bd = CeilDiv(totalRows, rowsPerCore);
    if (bd < 1) {
        bd = 1;
    }
    tiling->rowsPerCore = rowsPerCore;

    context->SetBlockDim(static_cast<uint32_t>(bd));
    context->SetTilingKey(GET_TPL_TILING_KEY(GF2POLY_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGf2Poly([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct Gf2PolyCompileInfo {};

IMPL_OP_OPTILING(Gf2Poly).Tiling(Gf2PolyTilingFunc).TilingParse<Gf2PolyCompileInfo>(TilingParseForGf2Poly);

}  // namespace optiling

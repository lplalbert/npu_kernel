/*!
 * \file gf2k_mul_square_tiling.cpp
 * \brief Gf2kMulSquare tiling implementation
 *
 * Model: one GF(2^k) element per row, rows are independent. Rows are split
 * across AIV cores (no workspace, no cross-core sync); each core streams tiles
 * of tileRows rows through UB. tileBytes targets 16KB and is bounded by the UB
 * budget of three buffers.
 *
 * Input contract (same as the problem reference implementation):
 *   a, b : uint8, rank 2, shape [N, k/8], 0 <= N <= 2^20
 *   k    : 8 or 128; operation : "mul" or "square"
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/gf2k_mul_square_tiling_data.h"
#include "../op_kernel/gf2k_mul_square_tiling_key.h"
#include <cstring>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TARGET_TILE_BYTES = 16384;
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

static ge::graphStatus Gf2kMulSquareTilingFunc(gert::TilingContext* context)
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

    Gf2kMulSquareTilingData* tiling = context->GetTilingData<Gf2kMulSquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- attributes: k (int), operation (string, optional, default "mul") ----
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    const int64_t kVal = *kPtr;
    OP_CHECK_IF(
        kVal != 8 && kVal != 128,
        OP_LOGE(context, "attr k must be 8 or 128"),
        return ge::GRAPH_FAILED);

    // Optional attribute: absent/empty falls back to the registered default "mul".
    const char* opPtr = attrs->GetAttrPointer<char>(1);
    bool isSquare = false;
    if (opPtr != nullptr && opPtr[0] != '\0') {
        if (std::strcmp(opPtr, "square") == 0) {
            isSquare = true;
        } else if (std::strcmp(opPtr, "mul") != 0) {
            OP_LOGE(context, "attr operation must be mul or square");
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t width = kVal / 8;  // 1 or 16

    // ---- only uint8 is registered in the op definition ----
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_UINT8 || outputDesc->GetDataType() != ge::DT_UINT8,
        OP_LOGE(context, "only uint8 is supported"),
        return ge::GRAPH_FAILED);

    // ---- shape contract: a/b are [N, width], no division / truncation ----
    const gert::StorageShape* aStorage = context->GetInputShape(0);
    const gert::StorageShape* bStorage = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, aStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, bStorage);
    const gert::Shape& aShape = aStorage->GetStorageShape();
    const gert::Shape& bShape = bStorage->GetStorageShape();
    OP_CHECK_IF(aShape.GetDimNum() != 2, OP_LOGE(context, "a must be rank 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(bShape.GetDimNum() != 2, OP_LOGE(context, "b must be rank 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        aShape.GetDim(1) != width,
        OP_LOGE(context, "a.shape[1] must be k/8"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        bShape.GetDim(0) != aShape.GetDim(0) || bShape.GetDim(1) != width,
        OP_LOGE(context, "b shape must equal a shape"),
        return ge::GRAPH_FAILED);
    const int64_t totalRows = aShape.GetDim(0);
    OP_CHECK_IF(
        totalRows < 0 || totalRows > MAX_ROWS,
        OP_LOGE(context, "N out of range [0, 2^20]"),
        return ge::GRAPH_FAILED);

    // ---- UB tile: 3 buffers, tileBytes bounded by UB / 6 and rounded to 32B ----
    int64_t limit = (int64_t)(ubSize / 6);
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
    tiling->k = kVal;
    tiling->op = isSquare ? 1 : 0;
    tiling->tileRows = tileRows;

    // ---- empty tensor: launch a single core that does nothing ----
    if (totalRows == 0) {
        tiling->rowsPerCore = 0;
        context->SetBlockDim(1U);
        context->SetTilingKey(GET_TPL_TILING_KEY(GF2KMULSQUARE_TPL_SCH_MODE_0));
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
    context->SetTilingKey(GET_TPL_TILING_KEY(GF2KMULSQUARE_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGf2kMulSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct Gf2kMulSquareCompileInfo {};

IMPL_OP_OPTILING(Gf2kMulSquare).Tiling(Gf2kMulSquareTilingFunc).TilingParse<Gf2kMulSquareCompileInfo>(TilingParseForGf2kMulSquare);

} // namespace optiling

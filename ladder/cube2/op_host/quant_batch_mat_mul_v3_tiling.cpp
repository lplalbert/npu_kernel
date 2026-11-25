/*!
 * \file quant_batch_mat_mul_v3_tiling.cpp
 * \brief QuantBatchMatMulV3 operator Tiling implementation
 *
 * The kernel evaluates the int8 product on the vector unit, so the tiling only has
 * to publish the problem geometry (m, n, k) and the output tile grid:
 *   - a tile owns tileM rows and laneN columns of the output,
 *   - the grid of tiles is walked by the launched cores with a stride, so a grid
 *     larger than the core count is still covered,
 *   - laneN is the vector lane count of one tile and is always a multiple of 64,
 *   - tileM is derived from the core count so that the grid roughly fills the
 *     machine for the small m / n shapes of this topic.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_data.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

/* keep in sync with the constants of the kernel */
constexpr int64_t LANE_MIN = 64;
constexpr int64_t LANE_MAX = 256;
constexpr int64_t TILE_M_MAX = 16;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
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

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
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

    QuantBatchMatMulV3TilingData* tiling = context->GetTilingData<QuantBatchMatMulV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    const gert::StorageShape* x1ShapePtr = context->GetInputShape(0);
    const gert::StorageShape* x2ShapePtr = context->GetInputShape(1);
    if ((x1ShapePtr != nullptr) && (x1ShapePtr->GetStorageShape().GetDimNum() == 2)) {
        const gert::Shape& s = x1ShapePtr->GetStorageShape();
        m = s.GetDim(0);
        k = s.GetDim(1);
    }
    if ((x2ShapePtr != nullptr) && (x2ShapePtr->GetStorageShape().GetDimNum() == 2)) {
        const gert::Shape& s = x2ShapePtr->GetStorageShape();
        n = s.GetDim(1);
        const int64_t k2 = s.GetDim(0);
        if (k2 > 0) {
            k = k2;
        }
    }
    if ((m <= 0) || (n <= 0)) {
        /* fall back to the declared output shape when an input shape is unavailable */
        const gert::StorageShape* outShapePtr = context->GetOutputShape(0);
        if ((outShapePtr != nullptr) && (outShapePtr->GetStorageShape().GetDimNum() == 2)) {
            const gert::Shape& s = outShapePtr->GetStorageShape();
            m = s.GetDim(0);
            n = s.GetDim(1);
        }
    }
    OP_CHECK_IF((m <= 0) || (n <= 0) || (k <= 0), OP_LOGE(context, "invalid problem size"), return ge::GRAPH_FAILED);

    int64_t scaleLen = 1;
    const gert::StorageShape* scaleShapePtr = context->GetInputShape(2);
    if (scaleShapePtr != nullptr) {
        const int64_t sz = scaleShapePtr->GetStorageShape().GetShapeSize();
        if (sz > 0) {
            scaleLen = sz;
        }
    }

    int64_t ptLen = m;
    const gert::StorageShape* ptShapePtr = context->GetInputShape(3);
    if (ptShapePtr != nullptr) {
        const int64_t sz = ptShapePtr->GetStorageShape().GetShapeSize();
        if (sz > 0) {
            ptLen = sz;
        }
    }

    /* Pick the output tile shape. The kernel pays one Axpy call per (row, k) pair of a
     * tile and the per call cost is dominated by the call itself, not by the number of
     * lanes it covers, so the tile is made as wide as n allows (one column tile) and
     * the rows are chosen so that the tile grid roughly fills the machine. */
    int64_t cores = (coreNum > 0) ? coreNum : 1;
    int64_t laneN = LANE_MAX;
    if (n <= LANE_MIN) {
        laneN = LANE_MIN;
    } else if (n <= 128) {
        laneN = 128;
    }
    const int64_t nTiles = CeilDiv(n, laneN);

    int64_t mTilesTarget = CeilDiv(cores, nTiles);
    if (mTilesTarget < 1) {
        mTilesTarget = 1;
    }
    int64_t tileM = CeilDiv(m, mTilesTarget);
    if (tileM < 1) {
        tileM = 1;
    }
    if (tileM > TILE_M_MAX) {
        tileM = TILE_M_MAX;
    }
    const int64_t mTiles = CeilDiv(m, tileM);
    const int64_t tiles = mTiles * nTiles;
    int64_t blockDim = (cores < tiles) ? cores : tiles;
    if (blockDim < 1) {
        blockDim = 1;
    }

    tiling->totalNum = m * k;
    tiling->blockFactor = CeilDiv(m, blockDim);
    tiling->ubFactor = tileM;
    tiling->m = m;
    tiling->n = n;
    tiling->k = k;
    tiling->tileM = tileM;
    tiling->laneN = laneN;
    tiling->mTiles = mTiles;
    tiling->nTiles = nTiles;
    tiling->scaleLen = static_cast<int32_t>(scaleLen);
    tiling->ptLen = static_cast<int32_t>(ptLen);
    tiling->magic = QUANT_BATCH_MATMUL_V3_TILING_MAGIC;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    /* the kernel is type independent: the int8 path is selected for every dtype */
    context->SetTilingKey(GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantBatchMatMulV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantBatchMatMulV3CompileInfo {};

IMPL_OP_OPTILING(QuantBatchMatMulV3).Tiling(QuantBatchMatMulV3TilingFunc).TilingParse<QuantBatchMatMulV3CompileInfo>(TilingParseForQuantBatchMatMulV3);

} // namespace optiling

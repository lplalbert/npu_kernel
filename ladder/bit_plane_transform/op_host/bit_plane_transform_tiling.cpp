/*!
 * \file bit_plane_transform_tiling.cpp
 * \brief BitPlaneTransform operator tiling implementation.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <cstring>
#include "../op_kernel/bit_plane_transform_tiling_data.h"
#include "../op_kernel/bit_plane_transform_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static ge::graphStatus BitPlaneTransformTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    BitPlaneTransformTilingData* tiling = context->GetTilingData<BitPlaneTransformTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // attributes
    const char* dir = "transpose";
    int64_t n = -1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char* d = attrs->GetAttrPointer<char>(0);
        if (d != nullptr) {
            dir = d;
        }
        const int64_t* np = attrs->GetAttrPointer<int64_t>(1);
        if (np != nullptr) {
            n = *np;
        }
    }
    // strict, full-string direction match
    bool isInverse = false;
    if (std::strcmp(dir, "transpose") == 0) {
        isInverse = false;
    } else if (std::strcmp(dir, "inverse") == 0) {
        isInverse = true;
    } else {
        OP_LOGE(context, "invalid direction '%s'", dir);
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* stShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, stShape);
    const gert::Shape& inShape = stShape->GetStorageShape();
    const size_t rank = inShape.GetDimNum();

    int64_t C = 0;
    int64_t rows = 0;
    if (!isInverse) {
        if (rank != 2) {
            OP_LOGE(context, "transpose requires rank 2 input");
            return ge::GRAPH_FAILED;
        }
        int64_t N = inShape.GetDim(0);
        C = inShape.GetDim(1);
        if (N <= 0 || C <= 0) {
            OP_LOGE(context, "transpose invalid N=%ld C=%ld", (long)N, (long)C);
            return ge::GRAPH_FAILED;
        }
        rows = N;
    } else {
        if (rank != 3) {
            OP_LOGE(context, "inverse requires rank 3 input");
            return ge::GRAPH_FAILED;
        }
        int64_t Gin = inShape.GetDim(0);
        int64_t Q = inShape.GetDim(1);
        int64_t lastD = inShape.GetDim(2);
        if (Gin <= 0 || Q <= 0) {
            OP_LOGE(context, "inverse invalid G=%ld Q=%ld", (long)Gin, (long)Q);
            return ge::GRAPH_FAILED;
        }
        if (lastD != 4) {
            OP_LOGE(context, "inverse requires last dim == 4, got %ld", (long)lastD);
            return ge::GRAPH_FAILED;
        }
        if (Q % 8 != 0) {
            OP_LOGE(context, "inverse requires middle dim multiple of 8, got %ld", (long)Q);
            return ge::GRAPH_FAILED;
        }
        C = Q / 8;
        if (n <= 0) {
            OP_LOGE(context, "inverse requires n > 0, got %ld", (long)n);
            return ge::GRAPH_FAILED;
        }
        int64_t Gproc = (n + 31) / 32;
        // prevent OOB: groups actually read (ceil(n/32)) must not exceed input G
        if (Gproc > Gin) {
            OP_LOGE(context, "inverse n=%ld needs %ld groups but input G=%ld (OOB)",
                    (long)n, (long)Gproc, (long)Gin);
            return ge::GRAPH_FAILED;
        }
        rows = n;
    }

    const int64_t totalGroups = (rows + 31) / 32;
    const int64_t fullGroups = rows / 32;
    const int64_t gb = 32 * C;

    int64_t B = static_cast<int64_t>((ubSize * 2ULL / 5ULL) / static_cast<uint64_t>(gb));
    if (B < 1) {
        B = 1;
    }
    if (B > 2048) {
        B = 2048;
    }

    int64_t blockDim = coreNum;
    if (blockDim > totalGroups) {
        blockDim = totalGroups;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    tiling->totalGroups = totalGroups;
    tiling->fullGroups = fullGroups;
    tiling->cols = C;
    tiling->rows = rows;
    tiling->batchGroups = B;
    tiling->dirFlag = isInverse ? 1 : 0;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(GET_TPL_TILING_KEY(BITPLANETRANSFORM_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBitPlaneTransform([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BitPlaneTransformCompileInfo {};

IMPL_OP_OPTILING(BitPlaneTransform)
    .Tiling(BitPlaneTransformTilingFunc)
    .TilingParse<BitPlaneTransformCompileInfo>(TilingParseForBitPlaneTransform);

} // namespace optiling

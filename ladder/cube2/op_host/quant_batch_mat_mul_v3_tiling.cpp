/*!
 * \file quant_batch_mat_mul_v3_tiling.cpp
 * \brief host tiling: Matmul(cube) tiling for an explicit per-core output grid.
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_api.h"
#include <cstdio>
#include <cstdlib>
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_data.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_key.h"

namespace optiling {

constexpr int32_t QBMM_TILING_MAGIC = 0x51424D33;
constexpr int64_t QBMM_KC = 16384;  /* keep in sync with the kernel */

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    /* cube2 op has no transpose attributes: both operands are always ND (m,k)/(k,n). */
    const bool transA = false;
    const bool transB = false;

    int64_t m = 0, n = 0, k = 0;
    const gert::StorageShape* x1 = context->GetInputShape(0);
    const gert::StorageShape* x2 = context->GetInputShape(1);
    if ((x1 != nullptr) && (x1->GetStorageShape().GetDimNum() == 2)) {
        m = transA ? x1->GetStorageShape().GetDim(1) : x1->GetStorageShape().GetDim(0);
        k = transA ? x1->GetStorageShape().GetDim(0) : x1->GetStorageShape().GetDim(1);
    }
    if ((x2 != nullptr) && (x2->GetStorageShape().GetDimNum() == 2)) {
        n = transB ? x2->GetStorageShape().GetDim(0) : x2->GetStorageShape().GetDim(1);
        const int64_t k2 = transB ? x2->GetStorageShape().GetDim(1) : x2->GetStorageShape().GetDim(0);
        if (k2 > 0) { k = k2; }
    }
    OP_CHECK_IF((m <= 0) || (n <= 0) || (k <= 0), OP_LOGE(context, "invalid shape"), return ge::GRAPH_FAILED);

    int64_t scaleLen = 1;
    const gert::StorageShape* sc = context->GetInputShape(2);
    if (sc != nullptr) {
        const int64_t s = sc->GetStorageShape().GetShapeSize();
        if (s > 0) { scaleLen = s; }
    }
    int64_t ptLen = m;
    const gert::StorageShape* pt = context->GetInputShape(3);
    if (pt != nullptr) {
        const int64_t s = pt->GetStorageShape().GetShapeSize();
        if (s > 0) { ptLen = s; }
    }

    QuantBatchMatMulV3TilingData* t = context->GetTilingData<QuantBatchMatMulV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, t);

    int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum <= 0) { aicNum = 1; }

    /* Small M/N, large K: split K across the cores. Every core owns a disjoint K slice,
     * so A and B are each read exactly once over the whole grid (no per-row B re-read),
     * and the exact int32 partials are atomically accumulated into the shared C. */
    int64_t gridK = aicNum;
    if (gridK > k) { gridK = k; }
    if (gridK < 1) { gridK = 1; }
    int64_t kPer = (k + gridK - 1) / gridK;
    kPer = ((kPer + 63) / 64) * 64;
    if (kPer < 64) { kPer = 64; }
    if (kPer > 65534) { kPer = 65534; }
    gridK = (k + kPer - 1) / kPer;
    int64_t mPer = m;
    int64_t nPer = n;
    t->gridK = static_cast<int32_t>(gridK);
    t->gridM = 1;
    t->gridN = 1;
    t->kc = static_cast<int32_t>(kPer);

    /* Tile the matmul for one core's output tile; the runtime SetOrgShape() supplies
     * the real leading dimensions of A / B / C. */
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, transA);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, transB);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    const int64_t kTile = kPer;
    t->slowPath = 0;
    mm.SetShape(static_cast<int32_t>(mPer), static_cast<int32_t>(nPer), static_cast<int32_t>(kTile));
    mm.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mm.SetBias(false);
    mm.SetBufferSpace(-1, -1, -1);
    int64_t rc = mm.GetTiling(t->mm);
    OP_CHECK_IF(rc != 0, OP_LOGE(context, "MatmulApiTiling GetTiling failed"), return ge::GRAPH_FAILED);
    t->mm.usedCoreNum = 1;

    t->m = m;
    t->n = n;
    t->k = k;
    t->scaleLen = static_cast<int32_t>(scaleLen);
    t->ptLen = static_cast<int32_t>(ptLen);
    t->magic = QBMM_TILING_MAGIC;
    t->transposeX1 = transA ? 1 : 0;
    t->transposeX2 = transB ? 1 : 0;
    size_t sysWs = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    t->sysWs = static_cast<int64_t>(sysWs);

    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = sysWs + static_cast<size_t>(m * n) * sizeof(int32_t) + 4096;

    uint32_t blockDim = static_cast<uint32_t>(gridK);


    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantBatchMatMulV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantBatchMatMulV3CompileInfo {};

IMPL_OP_OPTILING(QuantBatchMatMulV3).Tiling(QuantBatchMatMulV3TilingFunc).TilingParse<QuantBatchMatMulV3CompileInfo>(TilingParseForQuantBatchMatMulV3);

}  // namespace optiling




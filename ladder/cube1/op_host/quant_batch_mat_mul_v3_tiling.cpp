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

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    bool transA = false;
    bool transB = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* pa = attrs->GetAttrPointer<bool>(0);
        if (pa != nullptr) { transA = *pa; }
        const bool* pb = attrs->GetAttrPointer<bool>(1);
        if (pb != nullptr) { transB = *pb; }
    }

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

    /* Pick the output grid. Pure M-split across all cores is empirically fastest
     * (a narrow column tile makes the L1 copy of B strided and ~3x slower), so N is
     * only split when there are not enough rows to fill the cores; the column tile is
     * kept >= min(n,512) wide. */
    int64_t nPerFloor = (n < 512) ? n : 512;
    int64_t bestTm = 1, bestTn = 1, bestCost = -1;
    for (int64_t tn = 1; tn <= aicNum && tn <= n; ++tn) {
        int64_t tm = aicNum / tn;
        if (tm > m) { tm = m; }
        if (tm < 1) { continue; }
        int64_t mPer = (m + tm - 1) / tm;
        int64_t nPer = (n + tn - 1) / tn;
        if (nPer < nPerFloor) { continue; }
        int64_t cost = mPer + nPer;
        if (bestCost < 0 || cost < bestCost) { bestCost = cost; bestTm = tm; bestTn = tn; }
    }
    int64_t gridM = bestTm;
    int64_t gridN = bestTn;
    int64_t mPer = (m + gridM - 1) / gridM;
    int64_t nPer = (n + gridN - 1) / gridN;
    t->gridM = static_cast<int32_t>(gridM);
    t->gridN = static_cast<int32_t>(gridN);

    /* Tile the matmul for one core's output tile; the runtime SetOrgShape() supplies
     * the real leading dimensions of A / B / C. */
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, transA);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, transB);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetShape(static_cast<int32_t>(mPer), static_cast<int32_t>(nPer), static_cast<int32_t>(k));
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

    uint32_t blockDim = static_cast<uint32_t>(gridM * gridN);


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








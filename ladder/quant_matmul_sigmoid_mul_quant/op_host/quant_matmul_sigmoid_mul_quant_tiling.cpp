/*!
 * \file quant_matmul_sigmoid_mul_quant_tiling.cpp
 * \brief host tiling: Matmul(cube) tiling + workspace for the int32 result buffer.
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_api.h"
#include <cstring>
#include "../op_kernel/quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "../op_kernel/quant_matmul_sigmoid_mul_quant_tiling_key.h"

namespace optiling {

constexpr int64_t ALIGN_BYTES = 512;
constexpr int64_t TILE_LEN = 2048;

static inline int64_t AlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

static ge::graphStatus QuantMatmulSigmoidMulQuantTilingFunc(gert::TilingContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    const gert::StorageShape* x1ss = context->GetInputShape(0);
    const gert::StorageShape* x2ss = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ss);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ss);
    const gert::Shape& x1s = x1ss->GetStorageShape();
    const gert::Shape& x2s = x2ss->GetStorageShape();

    bool tA = false;
    bool tB = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* pa = attrs->GetAttrPointer<bool>(0);
        const bool* pb = attrs->GetAttrPointer<bool>(1);
        if (pa != nullptr) {
            tA = *pa;
        }
        if (pb != nullptr) {
            tB = *pb;
        }
    }

    /* physical x1: (M,K) or (K,M); physical x2: (K,N) or (N,K) */
    const int64_t x1d0 = x1s.GetDim(0);
    const int64_t x1d1 = x1s.GetDim(1);
    const int64_t x2d0 = x2s.GetDim(0);
    const int64_t x2d1 = x2s.GetDim(1);
    const int64_t M = tA ? x1d1 : x1d0;
    const int64_t Ka = tA ? x1d0 : x1d1;
    const int64_t N = tB ? x2d0 : x2d1;
    const int64_t Kb = tB ? x2d1 : x2d0;
    OP_CHECK_IF((M <= 0 || N <= 0 || Ka <= 0 || Kb <= 0 || Ka != Kb),
                OP_LOGE(context, "invalid shapes M=%ld N=%ld Ka=%ld Kb=%ld", M, N, Ka, Kb),
                return ge::GRAPH_FAILED);
    const int64_t K = Ka;

    QuantMatmulSigmoidMulQuantTilingData* t = context->GetTilingData<QuantMatmulSigmoidMulQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, t);
    memset(&t->mm, 0, sizeof(TCubeTiling));

    /* Output grid (one tile per core). Empirically a pure M split wins whenever there are
     * enough rows (splitting N makes every core re-read all of B and was ~2x slower on
     * 2048^3); N is only split when the row count cannot fill the cores. Below a few M MAC
     * the cross-core barrier costs more than it saves, so the work stays on one core. */
    int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum <= 0) {
        aicNum = 1;
    }
    int64_t gridM = 1;
    int64_t gridN = 1;
    if (M * N * K > static_cast<int64_t>(4) * 1024 * 1024) {
        gridM = M / 32;
        if (gridM > aicNum) {
            gridM = aicNum;
        }
        if (gridM < 2) {
            /* too few rows to split: split the columns instead */
            gridM = 1;
            gridN = N / 512;
            if (gridN > aicNum) {
                gridN = aicNum;
            }
            if (gridN < 1) {
                gridN = 1;
            }
        }
    }
    const int64_t mPer = (M + gridM - 1) / gridM;
    const int64_t nPer = (N + gridN - 1) / gridN;

    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, tA);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, tB);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetShape(static_cast<int32_t>(mPer), static_cast<int32_t>(nPer), static_cast<int32_t>(K));
    mm.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    mm.SetBias(false);
    mm.SetBufferSpace(-1, -1, -1);
    int64_t rc = mm.GetTiling(t->mm);
    OP_CHECK_IF(rc != 0, OP_LOGE(context, "MatmulApiTiling GetTiling failed"), return ge::GRAPH_FAILED);
    t->mm.usedCoreNum = 1;

    t->M = M;
    t->N = N;
    t->K = K;
    t->totalNum = M * N;
    t->tileLen = static_cast<int32_t>(TILE_LEN);
    t->transA = tA ? 1 : 0;
    t->transB = tB ? 1 : 0;
    t->gridM = static_cast<int32_t>(gridM);
    t->gridN = static_cast<int32_t>(gridN);

    size_t sysWs = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    t->sysWs = static_cast<int64_t>(sysWs);

    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    const int64_t cBytes = AlignUp(M * N * static_cast<int64_t>(sizeof(int32_t)), ALIGN_BYTES);
    const int64_t uBytes = AlignUp(M * N * static_cast<int64_t>(sizeof(uint16_t)), ALIGN_BYTES);
    ws[0] = sysWs + static_cast<size_t>(cBytes + uBytes + 4096);

    context->SetBlockDim(static_cast<uint32_t>(gridM * gridN));
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantMatmulSigmoidMulQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantMatmulSigmoidMulQuantCompileInfo {};

IMPL_OP_OPTILING(QuantMatmulSigmoidMulQuant)
    .Tiling(QuantMatmulSigmoidMulQuantTilingFunc)
    .TilingParse<QuantMatmulSigmoidMulQuantCompileInfo>(TilingParseForQuantMatmulSigmoidMulQuant);

} // namespace optiling

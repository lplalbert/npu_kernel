/*!
 * \file quant_matmul_sigmoid_mul_quant_tiling.cpp
 * \brief QuantMatmulSigmoidMulQuant tiling implementation
 */

#include "register/op_def_registry.h"
#include <cstring>
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_api.h"
#include "../op_kernel/quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "../op_kernel/quant_matmul_sigmoid_mul_quant_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TILE_LEN = 2048;
constexpr int64_t ALIGN_BYTES = 512;

static inline int64_t AlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantMatmulSigmoidMulQuantTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

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

    // physical x1: (M,K) or (K,M); physical x2: (K,N) or (N,K)
    const int64_t x1d0 = x1s.GetDim(0);
    const int64_t x1d1 = x1s.GetDim(1);
    const int64_t x2d0 = x2s.GetDim(0);
    const int64_t x2d1 = x2s.GetDim(1);
    const int64_t M = tA ? x1d1 : x1d0;
    const int64_t Ka = tA ? x1d0 : x1d1;
    const int64_t N = tB ? x2d0 : x2d1;
    const int64_t Kb = tB ? x2d1 : x2d0;
    OP_CHECK_IF(
        (M <= 0 || N <= 0 || Ka <= 0 || Kb <= 0 || Ka != Kb),
        OP_LOGE(context, "invalid shapes M=%ld N=%ld Ka=%ld Kb=%ld", M, N, Ka, Kb),
        return ge::GRAPH_FAILED);
    const int64_t K = Ka;

    QuantMatmulSigmoidMulQuantTilingData* tiling =
        context->GetTilingData<QuantMatmulSigmoidMulQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- tiling (no cube tiling in DEBUG test) ----
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    memset(&tiling->matmulTiling, 0, sizeof(TCubeTiling));

    tiling->M = M;
    tiling->N = N;
    tiling->K = K;
    tiling->totalNum = M * N;
    tiling->blockFactor = M * N;
    tiling->ubFactor = TILE_LEN;
    tiling->tileLen = static_cast<int32_t>(TILE_LEN);
    tiling->transA = tA ? 1 : 0;
    tiling->transB = tB ? 1 : 0;

    // ---- workspace: int32 C buffer + half U buffer ----
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    const int64_t cBytes = AlignUp(M * N * static_cast<int64_t>(sizeof(int32_t)), ALIGN_BYTES);
    const int64_t uBytes = AlignUp(M * N * static_cast<int64_t>(sizeof(uint16_t)), ALIGN_BYTES);
    const size_t sysWs = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    currentWorkspace[0] = sysWs + static_cast<size_t>(cBytes + uBytes + ALIGN_BYTES) + WS_SYS_SIZE;

    context->SetBlockDim(1);
    context->SetTilingKey(GET_TPL_TILING_KEY(QUANTMATMULSIGMOIDMULQUANT_TPL_SCH_MODE_1));
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

/*!
 * \file confusion_mul_grad_tiling.cpp
 * \brief ConfusionMulGrad tiling implementation
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/confusion_mul_grad_tiling_data.h"
#include "../op_kernel/confusion_mul_grad_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BYTES_PER_ELEM = 40;  // matches kernel InitBuffer budget
// One-line fallback switch: set false to revert axis=[0] fp32 to the (deterministic) column-split path.
static constexpr bool CMG_ENABLE_ROWBLOCK = true;

static int64_t CeilDivI(int64_t a, int64_t b) { return b <= 0 ? a : (a + b - 1) / b; }
static int64_t AlignI(int64_t a, int64_t b) { return ((a + b - 1) / b) * b; }
static int64_t NextPow2Mult64(int64_t x) {
    if (x < 64) x = 64;
    int64_t v = 64;
    while (v < x && v < 2048) v <<= 1;
    return v;
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

static ge::graphStatus ConfusionMulGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ConfusionMulGradTilingData* tiling = context->GetTilingData<ConfusionMulGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::Tensor* tensorIn = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorIn);
    ge::DataType dtype = tensorIn->GetDataType();
    const bool isFp16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);

    auto inShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    const gert::Shape& shp = inShapePtr->GetStorageShape();
    int64_t M = shp.GetDimNum() >= 1 ? shp.GetDim(0) : 1;
    int64_t N = shp.GetDimNum() >= 2 ? shp.GetDim(1) : 1;
    if (M <= 0) M = 1;
    if (N <= 0) N = 1;
    const int64_t total = M * N;

    // ---- axis -> mode ----
    int64_t mode = 2;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const gert::TypedContinuousVector<int64_t>* axisV = attrs->GetListInt(0);
        if (axisV != nullptr) {
            bool red[2] = {false, false};
            const int64_t* ad = axisV->GetData();
            const int64_t an = static_cast<int64_t>(axisV->GetSize());
            for (int64_t i = 0; i < an; ++i) {
                int64_t v = ad[i] % 2;
                if (v < 0) v += 2;
                red[v] = true;
            }
            if (red[0] && red[1]) {
                mode = 2;
            } else if (red[1]) {
                mode = 0;
            } else if (red[0]) {
                mode = 1;
            } else {
                mode = 2;
            }
        }
    }

    const int64_t bpe = (mode == 0) ? 40 : 52;  // mode 1/2 need 3 extra Kahan buffers
    int64_t tile = static_cast<int64_t>((ubSize * 3 / 4) / static_cast<uint64_t>(bpe));
    tile = (tile / 64) * 64;
    if (tile < 64) tile = 64;
    if (tile > 8192) tile = 8192;

    tiling->M = M;
    tiling->N = N;
    tiling->totalNum = total;
    tiling->axisMode = mode;

    int64_t used = 1;
    int64_t rowPerCore = 0;
    int64_t colPerCore = 0;
    int64_t accLen = 64;

    if (mode == 0) {
        int64_t cores = coreNum;
        if (cores > M) cores = M;
        if (cores < 1) cores = 1;
        int64_t rpc = CeilDivI(M, cores);
        rpc = AlignI(rpc, 8);
        if (rpc < 8) rpc = 8;
        used = CeilDivI(M, rpc);
        if (used > coreNum) used = coreNum;
        if (used < 1) used = 1;
        rowPerCore = rpc;
        accLen = NextPow2Mult64(N < 2048 ? N : 2048);
    } else if (mode == 2) {
        if (!isFp16) {
            int64_t cores = coreNum;
            int64_t per = CeilDivI(total, cores);
            per = AlignI(per, 64);
            used = CeilDivI(total, per);
            if (used > coreNum) used = coreNum;
            if (used < 1) used = 1;
            rowPerCore = per;
        } else {
            used = 1;
            rowPerCore = total;
        }
        accLen = NextPow2Mult64(total < 2048 ? total : 2048);
    } else {
        // axis=[0]. Preferred path (fp32, N>=64, fits UB): row-block split with
        // *contiguous full-row* reads (strided column slices only reach ~1.4 GB/s/core,
        // contiguous rows reach hundreds of GB/s). Partial column sums merged across
        // cores with a 32B atomic add using the (P0 - g) offset trick (no SyncAll).
        int64_t nPad = AlignI(N, 64);                   // = accLen = tile in this path
        // Honest UB footprint of ConfusionMulGrad<float>: q0/q1/q2 (2 buffers each),
        // oq (2), accBuf, tP, tComp, tK, tKT, gBuf => 14 fp32 tiles = 56 bytes/elem,
        // plus the fixed rsvBuf/rsBuf/tOutH buffers (~4.3KB, 8KB reserved here).
        constexpr int64_t kRowBlockBytesPerElem = 56;
        constexpr int64_t kRowBlockFixedBytes = 8192;
        int64_t needBytes = kRowBlockBytesPerElem * nPad + kRowBlockFixedBytes;
        int64_t ubCap = static_cast<int64_t>(ubSize) * 15 / 16;
        bool rowBlockOk = CMG_ENABLE_ROWBLOCK && (!isFp16) && (N >= 64) && (N <= 65536) &&
                          (needBytes <= ubCap);
        if (rowBlockOk) {
            used = coreNum;
            if (used > M) used = M;
            if (used < 1) used = 1;
            int64_t rpc = CeilDivI(M, used);
            rowPerCore = rpc;
            colPerCore = 0;
            accLen = AlignI(N, 64);
            tile = AlignI(N, 64);
            tiling->useRowBlock = 1;
        } else {
            // Column split: every core owns colPerCore contiguous columns and reduces all
            // M rows for them. Column parallelism is bounded by N, never by M (capping by M
            // made the single-core slice larger than the UB tile for M=1), and alignE is 64
            // so that every vector op count stays a multiple of 64.
            const int64_t alignE = 64;
            int64_t maxByN = N / 64;
            if (maxByN < 1) maxByN = 1;
            int64_t cores = coreNum;
            if (cores > maxByN) cores = maxByN;
            if (cores < 1) cores = 1;
            int64_t cw = CeilDivI(N, cores);
            cw = AlignI(cw, alignE);
            if (cw < 64) cw = 64;
            colPerCore = cw;
            accLen = 64;
            tiling->useRowBlock = 0;
            // ProcessCol splits its own range into tile-sized chunks, so it is always
            // safe to cover all N columns with at most coreNum cores.
            used = CeilDivI(N, cw);
            if (used < 1) used = 1;
            if (used > coreNum) used = coreNum;
        }
    }

    tiling->usedCores = used;
    tiling->rowPerCore = rowPerCore;
    tiling->colPerCore = colPerCore;
    tiling->accLen = accLen;
    // Must be written AFTER the branch: the row-block path enlarges tile to
    // Align(N,64) so that a full row fits one UB tile, and the kernel sizes every
    // queue from tileLength.
    tiling->tileLength = tile;

    context->SetBlockDim(static_cast<uint32_t>(used));

    uint64_t tilingKey;
    if (isFp16) {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONMULGRAD_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONMULGRAD_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConfusionMulGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ConfusionMulGradCompileInfo {};

IMPL_OP_OPTILING(ConfusionMulGrad)
    .Tiling(ConfusionMulGradTilingFunc)
    .TilingParse<ConfusionMulGradCompileInfo>(TilingParseForConfusionMulGrad);

}  // namespace optiling

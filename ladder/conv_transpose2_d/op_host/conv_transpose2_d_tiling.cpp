/*!
 * \file conv_transpose2_d_tiling.cpp
 * \brief ConvTranspose2D Tiling
 *
 * Simple, correct scalar-GM gather implementation. Output elements
 * (n, co, oh, ow) are split into disjoint ranges across AIV cores; each core
 * computes its outputs independently (no cross-core sync, deterministic).
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/conv_transpose2_d_tiling_data.h"
#include "../op_kernel/conv_transpose2_d_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t STAGE_CHUNK = 1024;  // staging chunk (multiple of 64)
constexpr int64_t MIN_OUT_PER_CORE = 64;

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

static ge::graphStatus ConvTranspose2DTilingFunc(gert::TilingContext* context)
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

    ConvTranspose2DTilingData* tiling = context->GetTilingData<ConvTranspose2DTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto xShapePtr = context->GetInputShape(0);
    auto wShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    const gert::Shape& xs = xShapePtr->GetStorageShape();
    const gert::Shape& ws = wShapePtr->GetStorageShape();
    OP_CHECK_IF(xs.GetDimNum() != 4 || ws.GetDimNum() != 4,
                OP_LOGE(context, "ConvTranspose2D expects 4-D x and w"), return ge::GRAPH_FAILED);

    int64_t n = xs.GetDim(0);
    int64_t chIn = xs.GetDim(1);
    int64_t hin = xs.GetDim(2);
    int64_t win = xs.GetDim(3);
    int64_t wCin = ws.GetDim(0);
    int64_t chOut = ws.GetDim(1);
    int64_t kh = ws.GetDim(2);
    int64_t kw = ws.GetDim(3);

    // dtype contract: kernel instantiates a single T for both x and w, so they
    // must share one dtype. Reject fp16/fp32 mixtures (otherwise w is read with
    // the wrong element size and the access overruns).
    auto xDesc = context->GetInputDesc(0);
    auto wDesc = context->GetInputDesc(1);
    OP_CHECK_IF(xDesc == nullptr || wDesc == nullptr,
                OP_LOGE(context, "ConvTranspose2D missing input tensor desc"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDesc->GetDataType() != wDesc->GetDataType(),
                OP_LOGE(context, "ConvTranspose2D requires x and w to have the same dtype"),
                return ge::GRAPH_FAILED);

    // shape contract
    OP_CHECK_IF(wCin != chIn,
                OP_LOGE(context, "ConvTranspose2D w.shape[0] must equal C_in"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(n < 1 || chIn < 1 || hin < 1 || win < 1 || chOut < 1 || kh < 1 || kw < 1,
                OP_LOGE(context, "ConvTranspose2D dims must be positive"), return ge::GRAPH_FAILED);

    int64_t stride = 1;
    int64_t pad = 0;
    int64_t outPad = 0;
    int64_t dil = 1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            stride = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(1);
        if (p != nullptr) {
            pad = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(2);
        if (p != nullptr) {
            outPad = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(3);
        if (p != nullptr) {
            dil = *p;
        }
    }
    OP_CHECK_IF(stride < 1 || dil < 1 || pad < 0 || outPad < 0,
                OP_LOGE(context, "ConvTranspose2D invalid stride/dilation/padding"), return ge::GRAPH_FAILED);

    int64_t hout = (hin - 1) * stride - 2 * pad + dil * (kh - 1) + outPad + 1;
    int64_t wout = (win - 1) * stride - 2 * pad + dil * (kw - 1) + outPad + 1;
    OP_CHECK_IF(hout <= 0 || wout <= 0,
                OP_LOGE(context, "ConvTranspose2D output H/W must be positive"), return ge::GRAPH_FAILED);

    int32_t hasBias = 0;
    auto bShapePtr = context->GetOptionalInputShape(2);
    if (bShapePtr != nullptr && bShapePtr->GetStorageShape().GetDimNum() >= 1) {
        int64_t bSize = bShapePtr->GetStorageShape().GetShapeSize();
        if (bSize > 0) {
            OP_CHECK_IF(bSize != chOut,
                        OP_LOGE(context, "ConvTranspose2D bias length must equal C_out"), return ge::GRAPH_FAILED);
            hasBias = 1;
        }
    }

    int64_t totalOut = n * chOut * hout * wout;
    if (totalOut < 0) {
        totalOut = 0;
    }
    int64_t rowsTotal = n * chOut * hout;
    if (rowsTotal < 0) {
        rowsTotal = 0;
    }

    auto inputDesc = context->GetInputDesc(0);
    bool isF16 = (inputDesc != nullptr &&
                  (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16));

    // Decide vector vs scalar path by UB budget.
    int64_t woutPad = ((wout + 63) / 64) * 64;
    if (woutPad < 64) {
        woutPad = 64;
    }
    int64_t ubBudget = static_cast<int64_t>(ubSize) * 7 / 10;  // bytes
    int64_t p2Row = ((win + 64 + 63) / 64) * 64;                 // whole x-row buffer (floats)

    auto pickWb = [](int64_t a) -> int64_t {
        if (a >= 256 && (a % 256) == 0) {
            return 256;
        }
        if (a >= 128 && (a % 128) == 0) {
            return 128;
        }
        return 64;
    };
    // UB footprint of the vector path, in fp32 slots, mirroring what the kernel
    // allocates: row accumulator, per-kw gather table (int32), gathered block,
    // x buffer, plus the fp16 staging buffers when the io dtype is fp16.
    auto commonFloatsOf = [&](int64_t a, int64_t w, int64_t xbuf) -> int64_t {
        int64_t f16 = isF16 ? ((a + xbuf) / 2) : 0;
        return a + kw * a + w + xbuf + f16;
    };

    int32_t useScalar = 1;
    int32_t accMode = 0;
    int32_t winSeg = 0;
    int64_t owBlockSize = 0;
    int64_t wbSize = 0;
    int64_t winSegSize = 0;

    if (wout >= 1 && rowsTotal >= 1) {
        // Fast path: whole padded output row AND whole x row fit in UB -> keep
        // the historical single-block tiling (bit-identical for shapes that
        // already worked).
        int64_t wbRow = woutPad < 256 ? woutPad : 256;
        if (wbRow < 64) {
            wbRow = 64;
        }
        int64_t accRow = ((woutPad + wbRow - 1) / wbRow) * wbRow;
        if (commonFloatsOf(accRow, wbRow, p2Row) * 4 <= ubBudget) {
            useScalar = 0;
            owBlockSize = accRow;
            wbSize = wbRow;
        } else {
            // Segmented path: split the output width into <=4096 blocks AND load
            // only the needed slice of the x row per block, so the row
            // accumulator and the x buffer both stay small regardless of
            // wout/win.  This keeps the per-tap reference accumulation order.
            int64_t cap = woutPad < 4096 ? woutPad : 4096;
            int64_t bestExact = 0;  // largest block that fits per-tap (accMode 2)
            int64_t bestAny = 0;    // largest block that fits at least accMode 3
            int64_t bestSeg = 0;
            for (int64_t a = cap; a >= 64; a -= 64) {
                int64_t segLenMax = ((a - 1) + (kw - 1) * dil) / stride + 1;
                if (segLenMax < 1) {
                    segLenMax = 1;
                }
                int64_t segP2 = ((segLenMax + 63) / 64) * 64;
                int64_t w = pickWb(a);
                int64_t common = commonFloatsOf(a, w, segP2 + 64);
                if (bestAny == 0 && common * 4 <= ubBudget) {
                    bestAny = a;
                    bestSeg = segP2;
                }
                if (bestExact == 0 && (common + kh * kw * a) * 4 <= ubBudget) {
                    bestExact = a;
                }
                if (bestExact != 0 && bestAny != 0) {
                    break;
                }
            }
            if (bestExact != 0) {
                bestAny = bestExact;
                int64_t segLenMax = ((bestExact - 1) + (kw - 1) * dil) / stride + 1;
                if (segLenMax < 1) {
                    segLenMax = 1;
                }
                bestSeg = ((segLenMax + 63) / 64) * 64;
            }
            if (bestAny != 0) {
                useScalar = 0;
                winSeg = 1;
                owBlockSize = bestAny;
                wbSize = pickWb(bestAny);
                winSegSize = bestSeg;
            }
        }
    }

    if (useScalar == 0) {
        int64_t xbuf = (winSeg != 0) ? (winSegSize + 64) : p2Row;
        int64_t commonFloats = commonFloatsOf(owBlockSize, wbSize, xbuf);
        int64_t tapFloats = commonFloats + kh * kw * owBlockSize;
        int64_t kahanFloats = commonFloats + 2 * owBlockSize + 2 * wbSize;
        int64_t plainFloats = commonFloats + owBlockSize;
        if (tapFloats * 4 <= ubBudget) {
            accMode = 2;
        } else if (kahanFloats * 4 <= ubBudget) {
            accMode = 1;
        } else if (plainFloats * 4 <= ubBudget) {
            accMode = 0;
        } else {
            accMode = 3;  // single accumulator, no extra chunk buffer
        }
    }

    int64_t cores = coreNum;
    if (cores < 1) {
        cores = 1;
    }
    int64_t per = 1;
    if (useScalar == 0) {
        if (cores > rowsTotal) {
            cores = rowsTotal;
        }
        if (cores < 1) {
            cores = 1;
        }
        per = CeilDiv<int64_t>(rowsTotal, cores);
        if (per < 1) {
            per = 1;
        }
    } else {
        int64_t maxCores = totalOut / MIN_OUT_PER_CORE;
        if (maxCores < 1) {
            maxCores = 1;
        }
        if (cores > maxCores) {
            cores = maxCores;
        }
        per = CeilDiv<int64_t>(totalOut, cores);
        if (per < 1) {
            per = 1;
        }
    }
    int64_t usedCores = cores;
    if (usedCores < 1) {
        usedCores = 1;
    }
    if (usedCores > coreNum) {
        usedCores = coreNum;
    }

    tiling->totalNum = totalOut;
    tiling->blockFactor = per;
    tiling->ubFactor = STAGE_CHUNK;
    tiling->n = n;
    tiling->chIn = chIn;
    tiling->chOut = chOut;
    tiling->hin = hin;
    tiling->win = win;
    tiling->hout = hout;
    tiling->wout = wout;
    tiling->kh = kh;
    tiling->kw = kw;
    tiling->stride = stride;
    tiling->pad = pad;
    tiling->outPad = outPad;
    tiling->dil = dil;
    tiling->hasBias = hasBias;
    tiling->usedCores = static_cast<int32_t>(usedCores);
    tiling->useScalar = useScalar;
    tiling->accMode = accMode;
    tiling->owBlockSize = owBlockSize;
    tiling->wbSize = wbSize;
    tiling->winSeg = winSeg;
    tiling->winSegSize = winSegSize;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    uint64_t tilingKey;
    if (isF16) {
        tilingKey = GET_TPL_TILING_KEY(CONVTRANSPOSE2D_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(CONVTRANSPOSE2D_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConvTranspose2D([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ConvTranspose2DCompileInfo {};

IMPL_OP_OPTILING(ConvTranspose2D).Tiling(ConvTranspose2DTilingFunc).TilingParse<ConvTranspose2DCompileInfo>(TilingParseForConvTranspose2D);

} // namespace optiling

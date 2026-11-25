/*!
 * \file poisson_nll_loss_tiling.cpp
 * \brief PoissonNllLoss 绠楀瓙 Tiling 瀹炵幇
 *
 * 鍗曟牳淇濊瘉姝ｇ‘鎬э紙blockDim=1锛夛紝灞炴€у叏閮ㄧ粡 tiling 鏁版嵁涓嬪彂 kernel銆? */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/poisson_nll_loss_tiling_data.h"
#include "../op_kernel/poisson_nll_loss_tiling_key.h"
#include <string>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE_F32 = 4;
constexpr int64_t MAX_UB_LEN = 6144;

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, int64_t coreNum)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 姣忔牳涓€涓?float 閮ㄥ垎鍜岋紙鎸?coreNum 鐢宠锛岄槻澶氭牳鍚姩鏃惰秺鐣岋級
    currentWorkspace[0] = static_cast<size_t>(coreNum) * 8U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PoissonNllLossTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    PoissonNllLossTilingData* tiling = context->GetTilingData<PoissonNllLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 鍏冪礌鎬绘暟 ----
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& shape = inputShapePtr->GetStorageShape();
    int64_t totalNum = 1;
    for (int64_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    if (shape.GetDimNum() == 0) {
        totalNum = 1;
    }
    if (totalNum < 1) {
        totalNum = 1;
    }
    tiling->totalNum = totalNum;

    // ---- dtype -> tilingKey / typeSize ----
    uint64_t tilingKey;
    int64_t typeSize = TYPE_SIZE_F32;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(POISSONNLLLOSS_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(POISSONNLLLOSS_TPL_SCH_MODE_2);
        typeSize = 2;
    } else {
        tilingKey = GET_TPL_TILING_KEY(POISSONNLLLOSS_TPL_SCH_MODE_1);
        typeSize = 4;
    }

    // ---- 灞炴€?----
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* logInputPtr = attrs->GetBool(0);
    const bool* fullPtr = attrs->GetBool(1);
    const float* epsPtr = attrs->GetFloat(2);
    const char* reductionPtr = attrs->GetStr(3);
    int32_t logInput = (logInputPtr != nullptr && !(*logInputPtr)) ? 0 : 1;
    int32_t full = (fullPtr != nullptr && *fullPtr) ? 1 : 0;
    float eps = (epsPtr != nullptr) ? *epsPtr : 1e-8f;
    int32_t reduction = 2; // mean
    if (reductionPtr != nullptr) {
        std::string red(reductionPtr);
        if (red == "none") {
            reduction = 0;
        } else if (red == "sum") {
            reduction = 1;
        } else {
            reduction = 2;
        }
    }
    tiling->logInput = logInput;
    tiling->full = full;
    tiling->eps = eps;
    tiling->reduction = reduction;
    tiling->invN = 1.0f / static_cast<float>(totalNum);

    // ---- 鏍告暟锛歳eduction=none 璧伴€愬厓绱犲鏍革紙鏃犺法鏍稿悓姝ワ級锛泂um/mean 闇€璺ㄦ牳鍚堝苟锛屽浐瀹氬崟鏍?----
    constexpr int64_t ALIGN_ELEM = 128;
    constexpr int64_t MIN_PER_CORE_ELEM = 2048;
    constexpr int64_t MIN_PER_CORE_REDUCE = 65536;
    (void)MIN_PER_CORE_REDUCE;
    int64_t blockNum = 1;
    if (reduction == 0) {
        // 閫愬厓绱犲鏍革細鍚勬牳澶勭悊涓嶉噸鍙犲尯闂达紝鏃犺法鏍稿悓姝?        blockNum = (totalNum + MIN_PER_CORE_ELEM - 1) / MIN_PER_CORE_ELEM;
        if (blockNum > coreNum) {
            blockNum = coreNum;
        }
        if (blockNum < 1) {
            blockNum = 1;
        }
    }
    // 娉ㄦ剰锛歴um/mean 鐨勮法鏍稿悎骞堕渶瑕?SyncAll锛屾湰鏈哄繀鐜?507035锛屾晠鍥哄畾鍗曟牳
    context->SetBlockDim(blockNum);
    int64_t blockFactor = (totalNum + blockNum - 1) / blockNum;
    blockFactor = ((blockFactor + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
    if (blockFactor < ALIGN_ELEM) {
        blockFactor = ALIGN_ELEM;
    }
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = totalNum;

    // ---- UB 鍒嗗潡闀垮害锛? 涓?float 缂撳啿锛堝惈 sum/mean 鐨?lane 绱姞鍣級+ 16bit 鏀跺彂缂撳瓨 ----
    int64_t bytesPerElem = 5 * TYPE_SIZE_F32 + ((typeSize < 4) ? 2 * typeSize : 0);
    int64_t ubLen = static_cast<int64_t>(ubSize) / bytesPerElem;
    if (ubLen > MAX_UB_LEN) {
        ubLen = MAX_UB_LEN;
    }
    ubLen = (ubLen / 64) * 64;
    if (ubLen < 64) {
        ubLen = 64;
    }
    tiling->ubLen = ubLen;

    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPoissonNllLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PoissonNllLossCompileInfo {};

IMPL_OP_OPTILING(PoissonNllLoss).Tiling(PoissonNllLossTilingFunc).TilingParse<PoissonNllLossCompileInfo>(TilingParseForPoissonNllLoss);

} // namespace optiling

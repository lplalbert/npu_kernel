/*!
 * \file dot_tiling.cpp
 * \brief Dot 算子 Tiling 实现
 *
 * x1/x2 为 1-D [N]，y 为 [1]。
 * 多核按 N 切分（blockFactor），各核 float/int64 部分和写用户 workspace，
 * kernel 内 SyncAll 后由 core0 合并。低精度浮点升 float32 累加；int32 用 int64 乘加。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/dot_tiling_data.h"
#include "../op_kernel/dot_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus DotTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    DotTilingData* tiling = context->GetTilingData<DotTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- x1 长度 N（1-D）----
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
    tiling->totalNum = totalNum;

    // ---- dtype -> tilingKey / typeSize ----
    uint64_t tilingKey;
    int64_t typeSize = TYPE_SIZE;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(DOT_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(DOT_TPL_SCH_MODE_2);
        typeSize = 2;
    } else if (dtype == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(DOT_TPL_SCH_MODE_3);
        typeSize = 4;
    } else {
        tilingKey = GET_TPL_TILING_KEY(DOT_TPL_SCH_MODE_1);
        typeSize = 4;
    }

    // ---- 核数：多核归约（部分和 -> SyncAll -> core0 汇总）----
    // 关键修复：kernel 必须用 GetUserWorkspace(workspace) 拿用户区；直接写 workspace（系统 KFC 区）
    // 会破坏 SyncAll 的保留区 -> ACL 507035。修好后多核可用。
    constexpr int64_t ALIGN_ELEM = 128;
    constexpr int64_t MIN_PER_CORE = 1; // 探测：blockNum 直接吃满 coreNum，验证 SyncAll 在满核下是否退化
    int64_t blockNum = (totalNum + MIN_PER_CORE - 1) / MIN_PER_CORE;
    if (blockNum > coreNum) {
        blockNum = coreNum;
    }

    if (blockNum < 1) {
        blockNum = 1;
    }
    // SyncAll/多核归约必须独占核资源：默认调度模式下各核可能不被同批调度，
    // 跨核栅障永远等不齐（507035 死锁）或退化为串行执行。
    // BATCH_MODE_SCHEDULE = 1（CANN 9.2.0 头文件未导出该常量，按官方取值自定义）。
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    int64_t blockFactor = (totalNum + blockNum - 1) / blockNum;
    blockFactor = ((blockFactor + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
    if (blockFactor < 1) {
        blockFactor = 1;
    }
    tiling->blockFactor = blockFactor;

    // ---- UB 分块长度：f1/f2/lane 累加器各 ubLen*float（16bit 再加收发缓存）----
    // kernel 里用折半树归约，ubLen 必须是 2 的幂
    int64_t bytesPerElem = 3 * TYPE_SIZE;
    if (typeSize < TYPE_SIZE) {
        bytesPerElem += typeSize;
    }
    int64_t ubLimit = static_cast<int64_t>(ubSize) / bytesPerElem;
    if (ubLimit > 8192) {
        ubLimit = 8192;
    }
    int64_t ubLen = 64;
    while (ubLen * 2 <= ubLimit) {
        ubLen *= 2;
    }
    tiling->ubLen = ubLen;

    // ---- workspace：用户区 + **系统区** ----
    // 关键：kernel 的 GetUserWorkspace(workspace) = kfc_workspace + RESERVED_WORKSPACE(16MB)，
    // 所以必须把 GetLibApiWorkSpaceSize()（>=16MB）算进申请量，否则用户区指针落在分配区外
    // -> 一写就 ACL 507035。用户区 = 每核一个 1KB 槽。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    size_t userSize = static_cast<size_t>(coreNum + 1) * 128U * 8U;
    currentWorkspace[0] = 512U; // no GM workspace needed (atomic into y)

    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDot([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DotCompileInfo {};

IMPL_OP_OPTILING(Dot).Tiling(DotTilingFunc).TilingParse<DotCompileInfo>(TilingParseForDot);

} // namespace optiling

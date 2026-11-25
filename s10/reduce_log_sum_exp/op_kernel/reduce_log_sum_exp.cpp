/*!
 * \file reduce_log_sum_exp.cpp
 * \brief ReduceLogSumExp 算子 kernel 入口
 */

#include "reduce_log_sum_exp.h"

enum class ReduceLogSumExpTilingKey : uint32_t
{
    TILING_KEY_REDUCELOGSUMEXP_MODE_0 = 0, // float16
    TILING_KEY_REDUCELOGSUMEXP_MODE_1 = 1, // float32
    TILING_KEY_REDUCELOGSUMEXP_MODE_2 = 2, // bfloat16
};

template <uint32_t schMode>
__global__ __aicore__ void reduce_log_sum_exp(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReduceLogSumExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceLogSumExpTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ReduceLogSumExpTilingKey::TILING_KEY_REDUCELOGSUMEXP_MODE_0)) {
        NsReduceLogSumExp::ReduceLogSumExp<half> op;
        op.Init(x, axes, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReduceLogSumExpTilingKey::TILING_KEY_REDUCELOGSUMEXP_MODE_1)) {
        NsReduceLogSumExp::ReduceLogSumExp<float> op;
        op.Init(x, axes, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReduceLogSumExpTilingKey::TILING_KEY_REDUCELOGSUMEXP_MODE_2)) {
        NsReduceLogSumExp::ReduceLogSumExp<bfloat16_t> op;
        op.Init(x, axes, y, &tilingData);
        op.Process();
    }
}

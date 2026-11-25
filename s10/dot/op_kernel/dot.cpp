/*!
 * \file dot.cpp
 * \brief Dot 算子 kernel 入口
 */

#include "dot.h"

enum class DotTilingKey : uint32_t
{
    TILING_KEY_DOT_MODE_0 = 0, // float16
    TILING_KEY_DOT_MODE_1 = 1, // float32
    TILING_KEY_DOT_MODE_2 = 2, // bfloat16
    TILING_KEY_DOT_MODE_3 = 3, // int32
};

template <uint32_t schMode>
__global__ __aicore__ void dot(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DotTilingData);
    GET_TILING_DATA_WITH_STRUCT(DotTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(DotTilingKey::TILING_KEY_DOT_MODE_0)) {
        NsDot::Dot<half> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(DotTilingKey::TILING_KEY_DOT_MODE_1)) {
        NsDot::Dot<float> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(DotTilingKey::TILING_KEY_DOT_MODE_2)) {
        NsDot::Dot<bfloat16_t> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(DotTilingKey::TILING_KEY_DOT_MODE_3)) {
        NsDot::Dot<int32_t> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
}

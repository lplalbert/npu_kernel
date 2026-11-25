/*!
 * \file poisson_nll_loss.cpp
 * \brief PoissonNllLoss 算子 kernel 入口
 */

#include "poisson_nll_loss.h"

enum class PoissonNllLossTilingKey : uint32_t
{
    TILING_KEY_POISSONNLLLOSS_MODE_0 = 0, // float16
    TILING_KEY_POISSONNLLLOSS_MODE_1 = 1, // float32
    TILING_KEY_POISSONNLLLOSS_MODE_2 = 2, // bfloat16
};

template <uint32_t schMode>
__global__ __aicore__ void poisson_nll_loss(GM_ADDR input_x, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PoissonNllLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(PoissonNllLossTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(PoissonNllLossTilingKey::TILING_KEY_POISSONNLLLOSS_MODE_0)) {
        NsPoissonNllLoss::PoissonNllLoss<half> op;
        op.Init(input_x, target, loss, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(PoissonNllLossTilingKey::TILING_KEY_POISSONNLLLOSS_MODE_1)) {
        NsPoissonNllLoss::PoissonNllLoss<float> op;
        op.Init(input_x, target, loss, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(PoissonNllLossTilingKey::TILING_KEY_POISSONNLLLOSS_MODE_2)) {
        NsPoissonNllLoss::PoissonNllLoss<bfloat16_t> op;
        op.Init(input_x, target, loss, workspace, &tilingData);
        op.Process();
    }
}

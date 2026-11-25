/*!
 * \file mhc_sinkhorn_grad.cpp
 * \brief MhcSinkhornGrad kernel entry
 */

#include "mhc_sinkhorn_grad.h"

enum class MhcSinkhornGradTilingKey : uint32_t
{
    TILING_KEY_MHCSINKHORNGRAD_MODE_0 = 0,
    TILING_KEY_MHCSINKHORNGRAD_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void mhc_sinkhorn_grad(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut, GM_ADDR gradInput,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcSinkhornGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornGradTilingData, tilingData, tiling);
    // The operator definition registers fp32 only; the schedule mode is a build
    // time constant and does not change the computation.
    (void)schMode;
    NsMhcSinkhornGrad::MhcSinkhornGrad op;
    op.Init(gradOutput, normOut, sumOut, gradInput, &tilingData);
    op.Process();
}

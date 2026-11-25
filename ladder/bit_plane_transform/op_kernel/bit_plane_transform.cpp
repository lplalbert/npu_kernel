/*!
 * \file bit_plane_transform.cpp
 * \brief BitPlaneTransform 算子 kernel 入口
 */

#include "bit_plane_transform.h"

template <uint32_t schMode>
__global__ __aicore__ void bit_plane_transform(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BitPlaneTransformTilingData);
    GET_TILING_DATA_WITH_STRUCT(BitPlaneTransformTilingData, tilingData, tiling);
    if constexpr (schMode == 0U) {
        NsBitPlaneTransform::BitPlaneTransform<uint8_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

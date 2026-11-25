/*!
 * \file bessel_i0e.cpp
 * \brief BesselI0e 算子 kernel 入口
 */

#include "bessel_i0e.h"

enum class BesselI0eTilingKey : uint32_t
{
    TILING_KEY_BESSELI0E_MODE_0 = 0, // float16
    TILING_KEY_BESSELI0E_MODE_1 = 1, // float32
    TILING_KEY_BESSELI0E_MODE_2 = 2, // bfloat16
};

template <uint32_t schMode>
__global__ __aicore__ void bessel_i0e(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BesselI0eTilingData);
    GET_TILING_DATA_WITH_STRUCT(BesselI0eTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(BesselI0eTilingKey::TILING_KEY_BESSELI0E_MODE_0)) {
        NsBesselI0e::BesselI0e<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(BesselI0eTilingKey::TILING_KEY_BESSELI0E_MODE_1)) {
        NsBesselI0e::BesselI0e<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(BesselI0eTilingKey::TILING_KEY_BESSELI0E_MODE_2)) {
        NsBesselI0e::BesselI0e<bfloat16_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

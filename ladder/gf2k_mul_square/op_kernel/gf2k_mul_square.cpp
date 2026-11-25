/*!
 * \file gf2k_mul_square.cpp
 * \brief Gf2kMulSquare kernel entry
 *
 * Only uint8 (the single registered dtype) is instantiated.
 */

#include "gf2k_mul_square.h"

enum class Gf2kMulSquareTilingKey : uint32_t
{
    TILING_KEY_GF2KMULSQUARE_MODE_0 = 0,
};

template <uint32_t schMode>
__global__ __aicore__ void gf2k_mul_square(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Gf2kMulSquareTilingData);
    GET_TILING_DATA_WITH_STRUCT(Gf2kMulSquareTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(Gf2kMulSquareTilingKey::TILING_KEY_GF2KMULSQUARE_MODE_0)) {
        NsGf2kMulSquare::Gf2kMulSquare<uint8_t> op;
        op.Init(a, b, c, &tilingData);
        op.Process();
    }
}

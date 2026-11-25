/*!
 * \file gf2_poly.cpp
 * \brief Gf2Poly kernel entry
 *
 * Only uint8 (the single registered dtype) is instantiated.
 */

#include "gf2_poly.h"

enum class Gf2PolyTilingKey : uint32_t
{
    TILING_KEY_GF2POLY_MODE_0 = 0,
};

template <uint32_t schMode>
__global__ __aicore__ void gf2_poly(GM_ADDR a, GM_ADDR b, GM_ADDR t, GM_ADDR mod_poly, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Gf2PolyTilingData);
    GET_TILING_DATA_WITH_STRUCT(Gf2PolyTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(Gf2PolyTilingKey::TILING_KEY_GF2POLY_MODE_0)) {
        NsGf2Poly::Gf2Poly<uint8_t> op;
        op.Init(a, b, t, mod_poly, y, &tilingData);
        op.Process();
    }
}

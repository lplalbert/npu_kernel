/*!
 * \file quant_matmul_sigmoid_mul_quant.cpp
 * \brief QuantMatmulSigmoidMulQuant operator kernel entry (cube / Matmul high level API).
 *        transposeX1 / transposeX2 are compile-time template parameters of the Matmul
 *        types, so the four layout combinations are dispatched on the tiling flags.
 */

#include "quant_matmul_sigmoid_mul_quant.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void quant_matmul_sigmoid_mul_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale,
                                                                     GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y,
                                                                     GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(QuantMatmulSigmoidMulQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulSigmoidMulQuantTilingData, tilingData, tiling);
    (void)bias;
    /* The lib-api (system) workspace precedes the user area, and only the runtime can
     * tell us where the prefix really ends: AscendC::GetUserWorkspace() must be called
     * exactly ONCE per kernel (a second call moves the pointer by another prefix). */
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if ((static_cast<uint64_t>(reinterpret_cast<uintptr_t>(userWS)) >> 32) == 0) {
        return;
    }
    const bool tA = (tilingData.transA != 0);
    const bool tB = (tilingData.transB != 0);
    if (!tA && !tB) {
        NsQuantMatmulSigmoidMulQuant::RunQmsq<false, false>(x1, x2, x1Scale, x2Scale, y, yScale, userWS, &tilingData);
    } else if (tA && !tB) {
        NsQuantMatmulSigmoidMulQuant::RunQmsq<true, false>(x1, x2, x1Scale, x2Scale, y, yScale, userWS, &tilingData);
    } else if (!tA && tB) {
        NsQuantMatmulSigmoidMulQuant::RunQmsq<false, true>(x1, x2, x1Scale, x2Scale, y, yScale, userWS, &tilingData);
    } else {
        NsQuantMatmulSigmoidMulQuant::RunQmsq<true, true>(x1, x2, x1Scale, x2Scale, y, yScale, userWS, &tilingData);
    }
}

/*!
 * \file quant_matmul_sigmoid_mul_quant.cpp
 * \brief QuantMatmulSigmoidMulQuant kernel entry (vector GEMM)
 */

#include "quant_matmul_sigmoid_mul_quant.h"

template <uint32_t schMode>
__global__ __aicore__ void quant_matmul_sigmoid_mul_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                                          GM_ADDR bias, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantMatmulSigmoidMulQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulSigmoidMulQuantTilingData, tilingData, tiling);
    (void)schMode;

    NsQuantMatmulSigmoidMulQuant::QuantMatmulSigmoidMulQuant<int8_t> op;
    op.Init(x1, x2, x1Scale, x2Scale, bias, y, yScale, AscendC::GetUserWorkspace(workspace), &tilingData);
    op.Process();
}

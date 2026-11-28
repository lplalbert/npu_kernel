/*!
 * \file quant_matmul_sigmoid_mul_quant_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _QUANTMATMULSIGMOIDMULQUANT_TILING_DATA_H_
#define _QUANTMATMULSIGMOIDMULQUANT_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct QuantMatmulSigmoidMulQuantTilingData {
    TCubeTiling matmulTiling;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t ubFactor = 0;
    int32_t tileLen = 2048;
    int32_t transA = 0;
    int32_t transB = 0;
};
#endif

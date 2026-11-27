/*!
 * \file quant_matmul_sigmoid_mul_quant_tiling_data.h
 * \brief tiling data for the Matmul(cube) implementation
 */
#ifndef _QUANTMATMULSIGMOIDMULQUANT_TILING_DATA_H_
#define _QUANTMATMULSIGMOIDMULQUANT_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) QuantMatmulSigmoidMulQuantTilingData {
    TCubeTiling mm;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t totalNum = 0;
    int64_t sysWs = 0;
    int32_t tileLen = 2048;
    int32_t transA = 0;
    int32_t transB = 0;
    int32_t gridM = 1;
    int32_t gridN = 1;
};
#pragma pack(pop)

#endif

/*!
 * \file quant_batch_mat_mul_v3_tiling_data.h
 * \brief tiling data for the Matmul(cube) implementation
 */
#ifndef QBMM_TILING_DATA_H
#define QBMM_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) QuantBatchMatMulV3TilingData {
    TCubeTiling mm;      // cube tiling computed for one core's output tile
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int32_t scaleLen = 1;
    int32_t ptLen = 1;
    int32_t magic = 0;
    int32_t transposeX1 = 0;
    int32_t transposeX2 = 0;
    int64_t sysWs = 0;   // lib-api workspace bytes that precede the user area
    int32_t gridM = 1;   // number of row tiles of the output grid
    int32_t gridN = 1;   // number of column tiles of the output grid
};
#pragma pack(pop)

#endif

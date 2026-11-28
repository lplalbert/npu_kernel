/*!
 * \file quant_batch_mat_mul_v3_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _QUANTBATCHMATMULV3_TILING_DATA_H_
#define _QUANTBATCHMATMULV3_TILING_DATA_H_

/* Marks a tiling buffer that was filled by the host tiling function. */
constexpr int32_t QUANT_BATCH_MATMUL_V3_TILING_MAGIC = 0x51424D33;

struct QuantBatchMatMulV3TilingData {
    int64_t totalNum = 0;      // element count of x1 (m * k)
    int64_t blockFactor = 1;   // output rows owned by one core
    int64_t ubFactor = 0;      // output rows per inner tile (informative)
    int64_t m = 0;             // output rows
    int64_t n = 0;             // output columns
    int64_t k = 0;             // reduction length
    int32_t transposeX1 = 0;   // x1 is stored as (k, m) when set
    int32_t transposeX2 = 0;   // x2 is stored as (n, k) when set
    int32_t scaleLen = 1;      // element count of scale (1 means broadcast)
    int32_t magic = 0;         // QUANT_BATCH_MATMUL_V3_TILING_MAGIC when valid
};
#endif

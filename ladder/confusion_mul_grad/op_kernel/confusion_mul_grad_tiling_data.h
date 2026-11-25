/*!
 * \file confusion_mul_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CONFUSIONMULGRAD_TILING_DATA_H_
#define _CONFUSIONMULGRAD_TILING_DATA_H_

struct ConfusionMulGradTilingData {
    int64_t M = 1;             // rows
    int64_t N = 1;             // cols
    int64_t totalNum = 0;      // M*N
    int64_t tileLength = 1024; // elements per UB tile (multiple of 64)
    int64_t accLen = 64;       // reducer fold length (multiple of 64, power of two)
    int64_t usedCores = 1;
    int64_t axisMode = 2;      // 0: axis=[1] row reduce, 1: axis=[0] col reduce, 2: axis=[0,1]
    int64_t rowPerCore = 0;    // axisMode 0: rows per core; axisMode 2: flat elems per core
    int64_t colPerCore = 0;    // axisMode 1 (old path): columns per core
    int64_t useRowBlock = 0;   // 1: axis=[0] fp32 row-block contiguous + atomic merge
};
#endif

/*!
 * \file mhc_sinkhorn_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _MHCSINKHORNGRAD_TILING_DATA_H_
#define _MHCSINKHORNGRAD_TILING_DATA_H_

struct MhcSinkhornGradTilingData {
    int64_t totalNum = 0;     // total gradient elements = T*n*n
    int64_t blockFactor = 1;  // samples per core
    int64_t ubFactor = 0;     // reserved
    int64_t n = 4;            // matrix order n, 1..8
    int64_t T = 1;            // sample count B*S
    int64_t numIters = 1;     // Sinkhorn iterations saved by the forward pass
    int64_t usedCores = 1;    // number of cores used
};
#endif

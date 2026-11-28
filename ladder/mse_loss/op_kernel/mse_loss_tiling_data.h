/*!
 * \file mse_loss_tiling_data.h
 * \brief tiling data struct for MseLoss
 */

#ifndef _MSELOSS_TILING_DATA_H_
#define _MSELOSS_TILING_DATA_H_

struct MseLossTilingData {
    int32_t totalNum = 0;     // total element count
    int32_t blockFactor = 1;  // elements handled by one core
    int32_t ubFactor = 0;     // elements per inner UB chunk (multiple of 64)
    int32_t reduction = 1;    // 0 = none, 1 = mean, 2 = sum
    int32_t usedCores = 1;    // active cores
};
#endif

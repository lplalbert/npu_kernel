/*!
 * \file bn_training_update_tiling_data.h
 * \brief tiling data struct (v3: int32 fields, only what the kernel needs)
 */

#ifndef _BNTRAININGUPDATE_TILING_DATA_H_
#define _BNTRAININGUPDATE_TILING_DATA_H_

struct BnTrainingUpdateTilingData {
    int32_t blockFactor = 1;      // rows ((n,c) planes) handled by one core
    int32_t ubFactor = 0;         // elements per inner UB chunk (multiple of 64)
    int32_t channelNum = 1;       // C
    int32_t rowLength = 1;        // S = H*W
    int32_t totalRows = 1;        // N*C
    int32_t elemsPerChannel = 1;  // N_c = N*S
    float epsilon = 1e-5f;
    int32_t usedCores = 1;
};
#endif

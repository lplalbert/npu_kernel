/*!
 * \file bn_training_update_tiling_data.h
 * \brief tiling data struct for BnTrainingUpdate
 */

#ifndef _BNTRAININGUPDATE_TILING_DATA_H_
#define _BNTRAININGUPDATE_TILING_DATA_H_

struct BnTrainingUpdateTilingData {
    int64_t totalNum = 0;         // total element count N*C*S
    int64_t blockFactor = 1;      // rows ((n,c) planes) handled by one core
    int64_t ubFactor = 0;         // elements per inner UB chunk (multiple of 64)
    int64_t channelNum = 1;       // C
    int64_t rowLength = 1;        // S = H*W, contiguous elements of one (n,c) plane
    int64_t totalRows = 1;        // N*C
    int64_t elemsPerChannel = 1;  // N_c = N*S
    float epsilon = 1e-5f;        // numerical stability constant
    int32_t usedCores = 1;        // number of active cores
};
#endif

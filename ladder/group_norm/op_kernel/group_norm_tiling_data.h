/*!
 * \file group_norm_tiling_data.h
 * \brief tiling data struct (v3: int32 fields, only what the kernel needs)
 */

#ifndef _GROUPNORM_TILING_DATA_H_
#define _GROUPNORM_TILING_DATA_H_

struct GroupNormTilingData {
    int32_t totalGroups = 1;  // N * G
    int32_t G = 1;            // groups per sample
    int32_t CpG = 1;          // channels per group = C / G
    int32_t HW = 1;           // product of the spatial dims
    int32_t chunk = 64;       // elements per UB tile (multiple of 64)
    float epsilon = 1e-5f;
};
#endif

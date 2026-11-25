/*!
 * \file group_norm_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _GROUPNORM_TILING_DATA_H_
#define _GROUPNORM_TILING_DATA_H_

struct GroupNormTilingData {
    int64_t N = 1;
    int64_t C = 1;
    int64_t G = 1;
    int64_t CpG = 1;        // channels per group = C / G
    int64_t HW = 1;         // product of spatial dims
    int64_t gsz = 1;        // elements per group = CpG * HW
    int64_t totalGroups = 1; // N * G
    int64_t chunk = 1;      // elements per UB tile (multiple of 64)
    float epsilon = 1e-5f;
};
#endif

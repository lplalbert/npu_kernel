/*!
 * \file bit_plane_transform_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _BITPLANETRANSFORM_TILING_DATA_H_
#define _BITPLANETRANSFORM_TILING_DATA_H_

struct BitPlaneTransformTilingData {
    int64_t totalGroups = 0;   // G = ceil(rows/32)
    int64_t fullGroups = 0;    // rows/32 (fully valid 32-lane groups)
    int64_t cols = 1;          // C
    int64_t rows = 0;          // N (transpose) or n (inverse)
    int64_t batchGroups = 1;   // groups processed per UB tile
    int64_t dirFlag = 0;       // 0 = transpose, 1 = inverse
};
#endif

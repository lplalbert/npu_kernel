/*!
 * \file index_tiling_data.h
 * \brief tiling data struct for Index
 */

#ifndef _INDEX_TILING_DATA_H_
#define _INDEX_TILING_DATA_H_

constexpr int INDEX_MAX_DIM = 10;

struct IndexTilingData {
    int64_t dimNum = 0;                  // x  N
    int64_t shape[INDEX_MAX_DIM] = {0};  // x
    int64_t stride[INDEX_MAX_DIM] = {0}; // x
    int64_t idxNum = 0;                  // indices  M
    int64_t outElems = 0;                //
    int64_t xElems = 0;                  //
    int64_t idxBytes = 8;                // indices
    int64_t usedCores = 1;               //
};
#endif

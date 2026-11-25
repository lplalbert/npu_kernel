/*!
 * \file gf2k_mul_square_tiling_data.h
 * \brief Gf2kMulSquare tiling data struct
 */

#ifndef _GF2KMULSQUARE_TILING_DATA_H_
#define _GF2KMULSQUARE_TILING_DATA_H_

struct Gf2kMulSquareTilingData {
    int64_t totalRows = 0;    // N: number of rows (one field element per row)
    int64_t width = 1;        // W = k/8: bytes per row (1 or 16)
    int64_t k = 8;            // field size, 8 or 128
    int64_t op = 0;           // 0 = mul, 1 = square
    int64_t rowsPerCore = 1;  // rows handled by each core
    int64_t tileRows = 1;     // rows handled per UB tile
};
#endif

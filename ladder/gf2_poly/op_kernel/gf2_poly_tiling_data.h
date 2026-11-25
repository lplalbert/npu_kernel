/*!
 * \file gf2_poly_tiling_data.h
 * \brief tiling data struct for Gf2Poly
 */

#ifndef _GF2POLY_TILING_DATA_H_
#define _GF2POLY_TILING_DATA_H_

struct Gf2PolyTilingData {
    int64_t totalRows = 0;    // N: number of rows, each row one polynomial
    int64_t width = 1;        // W = m/8 bytes per row for a/b
    int64_t outWidth = 1;     // 2*W for mul, W otherwise
    int64_t m = 8;            // degree of modulus, 8/32/64/128/256
    int64_t mode = 2;         // 0 = mul, 1 = reduce, 2 = mul_reduce
    int64_t rowsPerCore = 1;  // rows handled by each core
    int64_t tileRows = 1;     // rows handled per UB tile
    int64_t reserved0 = 0;
    int64_t reserved1 = 0;
};
#endif

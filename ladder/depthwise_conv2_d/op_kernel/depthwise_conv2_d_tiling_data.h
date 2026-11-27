/*!
 * \file depthwise_conv2_d_tiling_data.h
 * \brief Tiling data exchanged between the host tiling function and the kernel.
 */

#ifndef _DEPTHWISECONV2D_TILING_DATA_H_
#define _DEPTHWISECONV2D_TILING_DATA_H_

struct DepthwiseConv2DTilingData {
    int64_t totalNum = 0;     /* total output element count N*C*Hout*Wout */
    int64_t ubFactor = 1024;  /* staging chunk (scalar path only) */
    /* tensor shape */
    int64_t n = 1;
    int64_t chIn = 1;
    int64_t hin = 1;
    int64_t win = 1;
    int64_t hout = 1;
    int64_t wout = 1;
    int64_t kh = 1;
    int64_t kw = 1;
    /* attributes */
    int64_t stride = 1;
    int64_t pad = 0;
    int64_t dil = 1;
    /* row-group vector path geometry */
    int64_t R = 1;          /* output rows produced per group */
    int64_t rowSpan = 1;    /* input rows loaded into one tile */
    int64_t LP = 0;         /* left pad slots (multiple of 8) inside a tile row */
    int64_t WPtile = 64;    /* tile row pitch in elements (multiple of 8) */
    int64_t count = 64;     /* vector length, multiple of 64 */
    int64_t gpc = 1;        /* groups per channel = ceil(hout / R) */
    int64_t totalGroups = 1;/* n * chIn * gpc */
    int64_t wPad = 64;      /* padded kh*kw weight slots */
    int32_t hasBias = 0;
    int32_t useScalar = 0;
    int32_t usedCores = 1;
};
#endif

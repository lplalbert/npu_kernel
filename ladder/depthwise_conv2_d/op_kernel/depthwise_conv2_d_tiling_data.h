/*!
 * \file depthwise_conv2_d_tiling_data.h
 * \brief Tiling data exchanged between the host tiling function and the kernel.
 */

#ifndef _DEPTHWISECONV2D_TILING_DATA_H_
#define _DEPTHWISECONV2D_TILING_DATA_H_

struct DepthwiseConv2DTilingData {
    int64_t totalNum = 0;     /* total output element count N*C*Hout*Wout */
    int64_t blockFactor = 1;  /* number of output rows handled by one core */
    int64_t ubFactor = 0;     /* staging chunk size (scalar path) */
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
    /* derived vector path geometry */
    int64_t totalRows = 1; /* n * chIn * hout */
    int64_t accSize = 64;  /* padded output row length, multiple of 64 */
    int64_t xSize = 128;   /* padded input row buffer length, float slots */
    int64_t zoff = 64;     /* index of the all-zero gather lane */
    int64_t wPad = 64;     /* padded kh*kw weight slots */
    int32_t hasBias = 0;
    int32_t usedCores = 1;
    int32_t useScalar = 0;
};
#endif

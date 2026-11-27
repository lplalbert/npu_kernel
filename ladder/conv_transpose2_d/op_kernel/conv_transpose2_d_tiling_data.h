/*!
 * \file conv_transpose2_d_tiling_data.h
 * \brief tiling data struct for ConvTranspose2D
 */

#ifndef _CONVTRANSPOSE2D_TILING_DATA_H_
#define _CONVTRANSPOSE2D_TILING_DATA_H_

struct ConvTranspose2DTilingData {
    int64_t totalNum = 0;     // total output element count N*Cout*Hout*Wout
    int64_t blockFactor = 1;  // output elements handled per core
    int64_t ubFactor = 0;     // staging chunk size (multiple of 64)
    // shape
    int64_t n = 1;
    int64_t chIn = 1;
    int64_t chOut = 1;
    int64_t hin = 1;
    int64_t win = 1;
    int64_t hout = 1;
    int64_t wout = 1;
    int64_t kh = 1;
    int64_t kw = 1;
    // attributes
    int64_t stride = 1;
    int64_t pad = 0;
    int64_t outPad = 0;
    int64_t dil = 1;
    int32_t hasBias = 0;
    int32_t usedCores = 1;
    int32_t useScalar = 0;
    int32_t accMode = 0;  // 2: per tap accumulators (reference order), 1: kahan folding, 0: plain
    int64_t owBlockSize = 0;  // output-width block size used by the vector path
    int64_t wbSize = 0;       // inner vector chunk width (multiple of 64)
    int32_t winSeg = 0;       // 1 = load only the needed x slice per block
    int32_t winPad = 0;       // (reserved)
    int64_t winSegSize = 0;   // padded max x-slice length when winSeg == 1
};
#endif

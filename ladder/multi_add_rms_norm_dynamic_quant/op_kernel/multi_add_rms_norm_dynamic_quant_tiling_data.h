/*!
 * \file multi_add_rms_norm_dynamic_quant_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _MULTIADDRMSNORMDYNAMICQUANT_TILING_DATA_H_
#define _MULTIADDRMSNORMDYNAMICQUANT_TILING_DATA_H_

struct MultiAddRmsNormDynamicQuantTilingData {
    int64_t B = 0;            // rows
    int64_t H = 0;            // reduction axis length
    int64_t Hpad = 0;         // ceil(H/64)*64
    int64_t rowsPerCore = 1;  // rows handled by one core
    int64_t tileLen = 64;     // in-row tile length (multiple of 64)
    int32_t usedCores = 1;    // number of AI cores used
    uint32_t inMask = 0;      // bit0..4 = x1a..x1e, bit5 = x2
    uint32_t preload = 0;     // 1 = gamma/smooth1/smooth2 preloaded to UB as fp32
    float epsilon = 1e-6f;    // eps
};
#endif

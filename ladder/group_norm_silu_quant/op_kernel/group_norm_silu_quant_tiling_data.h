/*!
 * \file group_norm_silu_quant_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _GROUPNORMSILUQUANT_TILING_DATA_H_
#define _GROUPNORMSILUQUANT_TILING_DATA_H_

struct GroupNormSiluQuantTilingData {
    int64_t N = 0;              // batch
    int64_t C = 0;              // channels
    int64_t G = 1;              // groups
    int64_t CpG = 1;            // channels per group
    int64_t HW = 1;             // H*W
    int64_t gsz = 1;            // elements per group = CpG*HW
    int64_t totalGroups = 1;    // N*G
    int64_t chunk = 64;         // elements handled per UB loop (multiple of 64)
    int64_t qsPerChannel = 0;   // 1 if quantScale is per-channel (C), else scalar
    int64_t activateSilu = 1;   // 1 to enable SiLU
    float epsilon = 1e-5f;      // numerical stability constant
};
#endif

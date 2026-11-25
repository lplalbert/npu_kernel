/*!
 * \file flash_attention_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _FLASHATTENTION_TILING_DATA_H_
#define _FLASHATTENTION_TILING_DATA_H_

struct FlashAttentionTilingData {
    int64_t B = 1;
    int64_t N = 1;
    int64_t S = 1;
    int64_t D = 1;
    int64_t isCausal = 0;
    int64_t hasMask = 0;
    int64_t totalRows = 0;   // B*N*S
    int64_t keyBlock = 1;    // keys per UB tile (BK)
    float softmaxScale = 0.0f;
};
#endif


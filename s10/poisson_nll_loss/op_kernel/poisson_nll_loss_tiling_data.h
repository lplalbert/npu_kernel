/*!
 * \file poisson_nll_loss_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _POISSONNLLLOSS_TILING_DATA_H_
#define _POISSONNLLLOSS_TILING_DATA_H_

struct PoissonNllLossTilingData {
    int64_t totalNum = 0;    // 总元素数量
    int64_t blockFactor = 1; // 每个核处理的元素数量
    int64_t ubFactor = 0;    // 每次 UB 循环处理的元素数量
    int64_t ubLen = 8192;    // 单次 UB 分块长度（64 元素对齐）
    int32_t logInput = 1;    // attr log_input
    int32_t full = 0;        // attr full
    int32_t reduction = 2;   // 0=none 1=sum 2=mean
    int32_t reserved = 0;
    float eps = 1e-8f;       // attr eps
    float invN = 1.0f;       // 1/totalNum（mean 用）
};
#endif

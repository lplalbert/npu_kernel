/*!
 * \file bessel_i0e_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _BESSELI0E_TILING_DATA_H_
#define _BESSELI0E_TILING_DATA_H_

struct BesselI0eTilingData {
    int64_t totalNum = 0;    // 总元素数量
    int64_t blockFactor = 1; // 每个核处理的元素数量
    int64_t ubFactor = 0;    // 每次 UB 循环处理的元素数量
    int64_t ubLen = 8192;    // 单次 UB 分块长度（64 元素对齐）
};
#endif

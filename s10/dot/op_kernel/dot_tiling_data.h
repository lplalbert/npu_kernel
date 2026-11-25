/*!
 * \file dot_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _DOT_TILING_DATA_H_
#define _DOT_TILING_DATA_H_

struct DotTilingData {
    int64_t totalNum = 0;   // 向量长度 N
    int64_t blockFactor = 1; // 每核处理元素数
    int64_t ubLen = 1024;   // 单次 UB 分块长度
};
#endif

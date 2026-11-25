/*!
 * \file trace_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TRACE_TILING_DATA_H_
#define _TRACE_TILING_DATA_H_

struct TraceTilingData {
    int64_t totalNum = 0;     // 输入元素总数 M*N
    int64_t blockFactor = 1;  // 每个核处理的对角行数
    int64_t ubFactor = 0;     // 每次 UB 循环处理的对角行数（chunk，64 的整数倍）
    int64_t m = 0;            // 行数
    int64_t n = 0;            // 列数
    int64_t k = 0;            // 对角线长度 min(M,N)
    int64_t chunk = 1024;     // 单次搬运/计算的对角行数
    int32_t usedCores = 1;    // 实际启用核数
};
#endif

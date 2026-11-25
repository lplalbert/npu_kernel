// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct SpspmmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t U;
    uint32_t e1;
    uint32_t e2;
    uint32_t coreNum;
    uint32_t accN;   // UB 行累加器容量 (>=64, 64 对齐, <= n 向上取)
    uint32_t bsIter; // B 键二分查找迭代次数
    uint32_t chN;    // = ceil(n / accN)
};

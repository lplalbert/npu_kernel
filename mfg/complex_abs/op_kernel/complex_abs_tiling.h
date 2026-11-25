// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ComplexAbsTilingData {
    uint64_t length;      // 复数元素总个数 (N*M)
    uint64_t blockFactor; // 每个核处理的复数元素个数 (32 的整数倍)
    uint32_t tileLength;  // 段内每次搬运/计算的复数元素个数 (32 的整数倍)
    uint32_t usedCores;   // 实际启用核数
};

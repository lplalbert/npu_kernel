// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GridClusterTilingData {
    uint64_t n;         // 点数 N
    uint64_t blockFactor;  // 每核处理点数 (64 的整数倍)
    uint32_t chunk;     // 每次处理的点数 (64 的整数倍)
    uint32_t dim;       // D = 2 或 3
    uint32_t dtypeMode; // 0=float16, 1=float32
    uint32_t rsv;
};

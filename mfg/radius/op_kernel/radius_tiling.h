// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct RadiusTilingData {
    uint64_t n;        // 参考点数 N
    uint64_t m;        // 查询数 M
    uint64_t dim;      // D = 2/3
    uint64_t k;        // max_num_neighbors
    uint64_t qPerCore; // 每核查询数
    uint32_t chunk;    // 每次扫描的参考点数
    uint32_t dtypeMode;   // 0=float16 1=float32
    uint32_t ignoreSame;  // 1 = 跳过 j==i
    uint32_t rsv;
    float r2;          // r * r
};

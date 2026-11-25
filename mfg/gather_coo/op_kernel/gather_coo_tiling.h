// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GatherCooTilingData {
    uint64_t m;         // src 行数 M
    uint64_t k;         // 行宽 K
    uint64_t e;         // 输出行数 E
    uint64_t rowsPerCore;  // 每核处理的行数 (8 的整数倍)
    uint32_t groupRows;    // 每批处理多少行 (UB 分组)
};

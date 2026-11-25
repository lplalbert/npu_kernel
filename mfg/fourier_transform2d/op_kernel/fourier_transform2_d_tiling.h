// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FourierTransform2DTilingData {
    uint32_t length;
    uint32_t b;
    uint32_t n;
    uint32_t m;
    uint32_t logN;
    uint32_t logM;
    uint32_t coreNum;
    uint32_t inverse;
    uint32_t rowsPerCore;
    uint32_t probe;
};

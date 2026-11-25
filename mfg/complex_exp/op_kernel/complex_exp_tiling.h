// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ComplexExpTilingData {
    uint64_t length;      // 元素个数 (实数) / 复数个数 (complex64)
    uint64_t blockFactor; // 每核处理的元素(复数)个数
    uint32_t tileLength;  // 段内每块元素(复数)个数
    uint32_t dtypeMode;   // 0=float16, 1=float32, 2=complex64
};

// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ComplexDivTilingData {
    uint64_t aN;   // dividend 行数
    uint64_t aM;   // dividend 列数
    uint64_t bN;   // divisor 行数
    uint64_t bM;   // divisor 列数
    uint64_t outN; // 输出行数 = max(aN,bN)
    uint64_t outM; // 输出列数 = max(aM,bM)
    uint64_t rowsPerCore;
    uint32_t chunk;     // 每次处理的列数
    uint32_t dtypeMode; // 0=float16 1=float32 2=complex64
};

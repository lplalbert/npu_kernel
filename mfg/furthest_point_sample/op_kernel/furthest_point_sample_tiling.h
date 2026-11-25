// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FpsTilingData {
    uint64_t b;       // batch 数
    uint64_t n;       // 每 batch 点数
    uint64_t npoint;  // 每 batch 采样点数
    uint64_t nPad;    // n 向上取整到 64 (向量 op 的 count 需 >=64)
    uint32_t dtypeMode;  // 0=float16 1=float32
    uint32_t rsv0;
    uint64_t rsv1;
};

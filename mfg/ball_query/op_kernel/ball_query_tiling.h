// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct BallQueryTilingData {
    uint64_t n;          // 参考点数 N
    uint64_t m;          // 查询数 M
    uint64_t batch;      // B
    uint64_t nsample;    // 每个查询最多返回的索引数
    float radius2;       // radius * radius (float32)
    uint32_t dtypeMode;  // 0=float16, 1=float32
    uint32_t chunk;      // 参考点扫描分块大小 (点数)
    uint32_t qPerCore;   // 每个核每个 batch 处理的查询数
};

// 每个核内最多处理的查询数 (受 blockDim 与 M 限制, 这里给足够的静态数组空间)
constexpr int32_t BQ_MAX_Q_PER_CORE = 8;

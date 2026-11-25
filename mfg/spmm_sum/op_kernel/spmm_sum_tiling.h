// Tiling structure for SpmmSum
#pragma once

#include <cstdint>

struct SpmmSumTilingData {
    uint64_t m;          // rows of out
    uint64_t n;          // rows of mat (unused in kernel except bound)
    uint64_t k;          // cols of mat
    uint64_t e;          // nnz
    uint32_t rowsPerCore;
    uint32_t coreStride;  // interleaved assignment: core c handles rows c, c+coreStride, ...
    uint32_t ck;         // column chunk (multiple of 64)
    uint32_t dtypeMode;  // 0=float16 1=float32
    uint32_t rsv;
};

constexpr int32_t SPMM_MIN_CHUNK = 64;

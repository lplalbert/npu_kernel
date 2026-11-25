// Tiling structure for SegmentMinCsr
#pragma once

#include <cstdint>

struct SegmentMinCsrTilingData {
    uint64_t n;          // number of segments (rows of out)
    uint64_t e;          // number of source rows (nnz)
    uint64_t k;          // columns
    uint64_t rowsPerCore;
    uint32_t ck;         // column chunk (multiple of 64)
    uint32_t dtypeMode;  // 0=float16 1=float32
    uint32_t rsv0;
    uint32_t rsv1;
};

constexpr int32_t SEG_MIN_CHUNK = 64;

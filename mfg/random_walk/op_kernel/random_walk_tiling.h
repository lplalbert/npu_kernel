// Tiling structure for RandomWalk
#pragma once

#include <cstdint>

struct RandomWalkTilingData {
    uint64_t n;         // nodes
    uint64_t e;         // edges
    uint64_t b;         // walkers
    uint64_t wl;        // walk_length
    float p;
    float q;
    uint32_t walkersPerCore;
    uint32_t chunkW;    // walkers per UB block
};

constexpr int32_t RW_CHUNK_W = 32;

// Tiling data shared by host and kernel for FusedPropStep
#pragma once

#include <cstdint>

struct FusedPropStepTilingData {
    uint32_t B;          // batch of field / out
    uint32_t N;          // rows (transform length of the column pass)
    uint32_t M;          // cols (transform length of the row pass)
    uint32_t Bh;         // batch of h (1 or B)
    uint32_t dtypeCode;  // 0 = float16, 1 = float32, 2 = complex64
    uint32_t norm;       // 1: inverse transform carries 1/(N*M)
    uint32_t logN;
    uint32_t logM;
    uint32_t maxNM;      // max(N, M), sizes the FFT scratch buffer
    uint32_t length;     // B*N*M element count
    uint32_t useUb;      // 1: spectrum staged in UB (no workspace needed)
};

namespace fps_tiling {
// UB bytes available for the spectrum on top of the FFT scratch and twiddle tables
constexpr uint32_t kUbBudgetBytes = 160u * 1024u;
// offset of the user workspace slice inside the workspace block (the first bytes are
// the reserved system area the runtime keeps for its own bookkeeping)
constexpr uint32_t kWorkspaceOffset = 16u * 1024u * 1024u;
}  // namespace fps_tiling

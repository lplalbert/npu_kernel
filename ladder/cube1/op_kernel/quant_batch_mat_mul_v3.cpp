/*!
 * \file quant_batch_mat_mul_v3.cpp
 * \brief QuantBatchMatMulV3 operator kernel entry (cube / Matmul high level API).
 *        The transpose combination and the Matmul config (fast / loop-based fallback,
 *        chosen by the host from the physical leading dims) are compile-time template
 *        parameters, dispatched on the tiling flags.
 */
#include "quant_batch_mat_mul_v3.h"

using namespace AscendC;

template <bool AT, bool BT, bool SLOW>
__aicore__ inline void RunQBMM(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out,
                               GM_ADDR userWS, QuantBatchMatMulV3TilingData* td)
{
    NsQBMM::QBMM<AT, BT, SLOW> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &(td->mm));
    op.Init(x1, x2, scale, pertokenScale, out, userWS, td);
    op.Process();
}

extern "C" __global__ __aicore__ void quant_batch_mat_mul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale,
                                                             GM_ADDR pertokenScale, GM_ADDR out, GM_ADDR workspace,
                                                             GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(QuantBatchMatMulV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantBatchMatMulV3TilingData, tilingData, tiling);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if ((static_cast<uint64_t>(reinterpret_cast<uintptr_t>(userWS)) >> 32) == 0) {
        return;
    }
    const bool tA = (tilingData.transposeX1 != 0);
    const bool tB = (tilingData.transposeX2 != 0);
    const bool slow = (tilingData.slowPath != 0);
    if (!slow) {
        if (!tA && !tB) {
            RunQBMM<false, false, false>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        } else if (tA && !tB) {
            RunQBMM<true, false, false>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        } else if (!tA && tB) {
            RunQBMM<false, true, false>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        } else {
            RunQBMM<true, true, false>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        }
    } else {
        if (!tA && !tB) {
            RunQBMM<false, false, true>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        } else if (tA && !tB) {
            RunQBMM<true, false, true>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        } else if (!tA && tB) {
            RunQBMM<false, true, true>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        } else {
            RunQBMM<true, true, true>(x1, x2, scale, pertokenScale, out, userWS, &tilingData);
        }
    }
}

/*!
 * \file mse_loss_tiling_key.h
 * \brief Tiling template argument definition
 */

#ifndef __MSELOSS_TILING_KEY_H__
#define __MSELOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MSELOSS_TPL_SCH_MODE_0 0 /* float16 */
#define MSELOSS_TPL_SCH_MODE_1 1 /* float32 */

/*
 * Reduction determinism switch (shared by host tiling and the kernel).
 *
 *   0 : multi-core partial sums combined with a 32B atomic add (default).
 *       Fastest, but the float add order across cores is chosen by the
 *       scheduler, so two runs may differ in the last bits (~1e-6 relative,
 *       far inside the spec tolerance of 1e-4 but not bit-identical).
 *   1 : the reduction runs on one core, so the accumulation order is fixed and
 *       repeated runs are bit-identical. Costs most of the read bandwidth of
 *       the larger reduce cases.
 */
#define MSELOSS_DET_REDUCE 0

ASCENDC_TPL_ARGS_DECL(
    MseLoss,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, MSELOSS_TPL_SCH_MODE_0, MSELOSS_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, MSELOSS_TPL_SCH_MODE_0, MSELOSS_TPL_SCH_MODE_1)));

#endif

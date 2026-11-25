/*!
 * \file poisson_nll_loss_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __POISSONNLLLOSS_TILING_KEY_H__
#define __POISSONNLLLOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define POISSONNLLLOSS_TPL_SCH_MODE_0 0 // float16
#define POISSONNLLLOSS_TPL_SCH_MODE_1 1 // float32
#define POISSONNLLLOSS_TPL_SCH_MODE_2 2 // bfloat16

ASCENDC_TPL_ARGS_DECL(
    PoissonNllLoss,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          POISSONNLLLOSS_TPL_SCH_MODE_0, POISSONNLLLOSS_TPL_SCH_MODE_1,
                          POISSONNLLLOSS_TPL_SCH_MODE_2));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         POISSONNLLLOSS_TPL_SCH_MODE_0, POISSONNLLLOSS_TPL_SCH_MODE_1,
                         POISSONNLLLOSS_TPL_SCH_MODE_2)));

#endif

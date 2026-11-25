/*!
 * \file group_norm_tiling_key.h
 * \brief Tiling template argument definition
 */

#ifndef __GROUPNORM_TILING_KEY_H__
#define __GROUPNORM_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define GROUPNORM_TPL_SCH_MODE_0 0 /* float16 */
#define GROUPNORM_TPL_SCH_MODE_1 1 /* float32 */
#define GROUPNORM_TPL_SCH_MODE_2 2 /* bfloat16 */

ASCENDC_TPL_ARGS_DECL(
    GroupNorm,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          GROUPNORM_TPL_SCH_MODE_0, GROUPNORM_TPL_SCH_MODE_1, GROUPNORM_TPL_SCH_MODE_2));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         GROUPNORM_TPL_SCH_MODE_0, GROUPNORM_TPL_SCH_MODE_1, GROUPNORM_TPL_SCH_MODE_2)));

#endif

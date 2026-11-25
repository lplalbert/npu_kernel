/*!
 * \file dot_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __DOT_TILING_KEY_H__
#define __DOT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define DOT_TPL_SCH_MODE_0 0 // float16
#define DOT_TPL_SCH_MODE_1 1 // float32
#define DOT_TPL_SCH_MODE_2 2 // bfloat16
#define DOT_TPL_SCH_MODE_3 3 // int32

ASCENDC_TPL_ARGS_DECL(
    Dot,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          DOT_TPL_SCH_MODE_0, DOT_TPL_SCH_MODE_1, DOT_TPL_SCH_MODE_2, DOT_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         DOT_TPL_SCH_MODE_0, DOT_TPL_SCH_MODE_1, DOT_TPL_SCH_MODE_2, DOT_TPL_SCH_MODE_3)));

#endif

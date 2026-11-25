/*!
 * \file index_tiling_key.h
 * \brief Tiling
 */

#ifndef __INDEX_TILING_KEY_H__
#define __INDEX_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/*  x  kernel
 *   MODE_0 = 2  (float16/bfloat16)
 *   MODE_1 = 4  (float32/int32)
 *   MODE_2 = 8  (int64)
 *   MODE_3 = 1  (int8/uint8/bool)
 *  */
#define INDEX_TPL_SCH_MODE_0 0
#define INDEX_TPL_SCH_MODE_1 1
#define INDEX_TPL_SCH_MODE_2 2
#define INDEX_TPL_SCH_MODE_3 3

ASCENDC_TPL_ARGS_DECL(
    Index,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          INDEX_TPL_SCH_MODE_0, INDEX_TPL_SCH_MODE_1,
                          INDEX_TPL_SCH_MODE_2, INDEX_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         INDEX_TPL_SCH_MODE_0, INDEX_TPL_SCH_MODE_1,
                         INDEX_TPL_SCH_MODE_2, INDEX_TPL_SCH_MODE_3)));

#endif

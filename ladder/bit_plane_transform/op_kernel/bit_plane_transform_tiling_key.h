/*!
 * \file bit_plane_transform_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __BITPLANETRANSFORM_TILING_KEY_H__
#define __BITPLANETRANSFORM_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define BITPLANETRANSFORM_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(
    BitPlaneTransform,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, BITPLANETRANSFORM_TPL_SCH_MODE_0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, BITPLANETRANSFORM_TPL_SCH_MODE_0)));

#endif

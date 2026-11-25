/*!
 * \file gf2_poly_tiling_key.h
 * \brief Tiling template argument definition
 *
 * The op definition only registers uint8, so a single schedule (schMode 0) is
 * exposed; the previous float16 key was unreachable dead code.
 */

#ifndef __GF2POLY_TILING_KEY_H__
#define __GF2POLY_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define GF2POLY_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(
    Gf2Poly,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, GF2POLY_TPL_SCH_MODE_0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, GF2POLY_TPL_SCH_MODE_0)));

#endif

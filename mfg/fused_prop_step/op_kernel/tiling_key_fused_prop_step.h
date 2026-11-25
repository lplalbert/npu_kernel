// TilingKey template definition for FusedPropStep
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(FusedPropStep,
    ASCENDC_TPL_DATATYPE_DECL(DT_FIELD, C_DT_FLOAT16, C_DT_FLOAT, C_DT_COMPLEX64),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_FIELD, C_DT_FLOAT16, C_DT_FLOAT, C_DT_COMPLEX64),
    ),
);

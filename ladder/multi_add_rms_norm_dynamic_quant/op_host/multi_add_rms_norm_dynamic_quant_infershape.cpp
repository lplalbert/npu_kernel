/*!
 * \file multi_add_rms_norm_dynamic_quant_infershape.cpp
 * \brief MultiAddRmsNormDynamicQuant shape inference
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMultiAddRmsNormDynamicQuant(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t b = inShape->GetDim(0);
    const int64_t h = inShape->GetDim(1);
    // y1, y2, x, y  -> (B, H)
    for (size_t i = 0; i < 4; i++) {
        gert::Shape* o = context->GetOutputShape(i);
        if (o == nullptr) {
            return ge::GRAPH_FAILED;
        }
        o->SetDimNum(2);
        o->SetDim(0, b);
        o->SetDim(1, h);
    }
    // scale1, scale2 -> (B,)
    for (size_t i = 4; i < 6; i++) {
        gert::Shape* o = context->GetOutputShape(i);
        if (o == nullptr) {
            return ge::GRAPH_FAILED;
        }
        o->SetDimNum(1);
        o->SetDim(0, b);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiAddRmsNormDynamicQuant).InferShape(InferShapeMultiAddRmsNormDynamicQuant);

} // namespace ops

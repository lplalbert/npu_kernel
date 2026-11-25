/*!
 * \file bessel_i0e_infershape.cpp
 * \brief BesselI0e 算子形状推导实现（逐元素，输出同输入）
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeBesselI0e(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BesselI0e).InferShape(InferShapeBesselI0e);

} // namespace ops

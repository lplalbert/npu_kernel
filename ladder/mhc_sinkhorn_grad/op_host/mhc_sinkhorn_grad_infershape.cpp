/*!
 * \file mhc_sinkhorn_grad_infershape.cpp
 * \brief MhcSinkhornGrad shape inference
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMhcSinkhornGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // gradInput has exactly the same shape as gradOutput
    const gert::Shape* inShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (inShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcSinkhornGrad).InferShape(InferShapeMhcSinkhornGrad);

} // namespace ops

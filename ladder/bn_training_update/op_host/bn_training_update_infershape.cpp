/*!
 * \file bn_training_update_infershape.cpp
 * \brief BnTrainingUpdate shape inference
 *
 *   y             : same shape as x
 *   save_mean     : (C,)
 *   save_variance : (C,)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeBnTrainingUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;

    // The per channel statistics follow the (C,) shape of the reduce inputs.
    const gert::Shape* statShape = context->GetInputShape(1);
    for (size_t i = 1; i < 3; ++i) {
        gert::Shape* outShape = context->GetOutputShape(i);
        if (outShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (statShape != nullptr) {
            *outShape = *statShape;
        } else {
            outShape->SetDimNum(1);
            outShape->SetDim(0, xShape->GetDim(1));
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BnTrainingUpdate).InferShape(InferShapeBnTrainingUpdate);

}  // namespace ops

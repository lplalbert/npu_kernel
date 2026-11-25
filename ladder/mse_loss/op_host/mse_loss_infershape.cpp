/*!
 * \file mse_loss_infershape.cpp
 * \brief MseLoss shape inference
 *
 *   reduction = 0 (none)  -> out shape equals input shape
 *   reduction = 1 / 2     -> out shape is [1]
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMseLoss(gert::InferShapeContext* context)
{
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t reduction = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            reduction = *p;
        }
    }

    if (reduction == 0) {
        const gert::Shape* inShape = context->GetInputShape(0);
        if (inShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outShape = *inShape;
    } else {
        outShape->SetDimNum(1);
        outShape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MseLoss).InferShape(InferShapeMseLoss);

} // namespace ops

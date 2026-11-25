/*!
 * \file poisson_nll_loss_infershape.cpp
 * \brief PoissonNllLoss 算子形状推导实现
 *
 * reduction=none -> 输出 shape 同输入；sum/mean -> 输出 [1]。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <cstring>

using namespace ge;

namespace ops {

static ge::graphStatus InferShapePoissonNllLoss(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char* reduction = "mean";
    if (context->GetAttrs() != nullptr) {
        const char* attr = context->GetAttrs()->GetStr(3);
        if (attr != nullptr) {
            reduction = attr;
        }
    }

    if (std::strcmp(reduction, "none") == 0) {
        *output_shape = *input_shape;
    } else {
        output_shape->SetDimNum(1);
        output_shape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PoissonNllLoss).InferShape(InferShapePoissonNllLoss);

} // namespace ops

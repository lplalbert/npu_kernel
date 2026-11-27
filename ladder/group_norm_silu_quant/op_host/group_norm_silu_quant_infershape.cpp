/*!
 * \file group_norm_silu_quant_infershape.cpp
 * \brief GroupNormSiluQuant
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeGroupNormSiluQuant(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    if (selfShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const size_t rank = selfShape->GetDimNum();
    if (rank < 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t N = selfShape->GetDim(0);
    const int64_t C = selfShape->GetDim(1);

    int64_t G = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* gp = attrs->GetAttrPointer<int64_t>(0);
        if (gp != nullptr && *gp > 0) {
            G = *gp;
        }
    }
    if (G < 1) G = 1;
    if (G > C) G = C;
    while (C % G != 0 && G > 1) {
        --G;
    }

    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* meanShape = context->GetOutputShape(1);
    gert::Shape* rstdShape = context->GetOutputShape(2);
    if (outShape == nullptr || meanShape == nullptr || rstdShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *selfShape;
    meanShape->SetDimNum(2);
    meanShape->SetDim(0, N);
    meanShape->SetDim(1, G);
    *rstdShape = *meanShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSiluQuant).InferShape(InferShapeGroupNormSiluQuant);

} // namespace ops

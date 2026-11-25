/*!
 * \file flash_attention_infershape.cpp
 * \brief FlashAttention 算子形状推导实现
 *
 *  out shape == q shape (B, N, S, D), dtype same as q.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeFlashAttention(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0);
    if (qShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *qShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttention).InferShape(InferShapeFlashAttention);

} // namespace ops


/*!
 * \file trace_infershape.cpp
 * \brief Trace 算子形状推导实现：输出固定为 shape (1,)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTrace(gert::InferShapeContext* context)
{
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(1);
    outShape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Trace).InferShape(InferShapeTrace);

} // namespace ops

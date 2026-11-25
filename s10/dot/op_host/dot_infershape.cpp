/*!
 * \file dot_infershape.cpp
 * \brief Dot 算子形状推导实现
 *
 * 输出 y 固定为 [1]（标量以 1 元素 1-D 表示，与 torch.dot 的 0-D 结果对齐）。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeDot(gert::InferShapeContext* context)
{
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(1);
    outShape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Dot).InferShape(InferShapeDot);

} // namespace ops

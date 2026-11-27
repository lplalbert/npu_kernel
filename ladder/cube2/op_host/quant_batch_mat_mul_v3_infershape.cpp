/*!
 * \file quant_batch_mat_mul_v3_infershape.cpp
 * \brief QuantBatchMatMulV3 shape inference: out = (m, n).
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeQuantBatchMatMulV3(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    if ((x1Shape == nullptr) || (x2Shape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if ((x1Shape->GetDimNum() != 2) || (x2Shape->GetDimNum() != 2)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t m = x1Shape->GetDim(0);
    const int64_t n = x2Shape->GetDim(1);
    if ((m <= 0) || (n <= 0)) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(2);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatMulV3).InferShape(InferShapeQuantBatchMatMulV3);

} // namespace ops

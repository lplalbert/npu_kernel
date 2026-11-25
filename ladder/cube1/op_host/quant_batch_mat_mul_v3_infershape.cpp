/*!
 * \file quant_batch_mat_mul_v3_infershape.cpp
 * \brief QuantBatchMatMulV3 shape inference: out = (m, n)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeQuantBatchMatMulV3(gert::InferShapeContext* context)
{
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    bool transA = false;
    bool transB = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* pa = attrs->GetAttrPointer<bool>(0);
        if (pa != nullptr) {
            transA = *pa;
        }
        const bool* pb = attrs->GetAttrPointer<bool>(1);
        if (pb != nullptr) {
            transB = *pb;
        }
    }

    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    if ((x1Shape == nullptr) || (x2Shape == nullptr)) {
        return ge::GRAPH_FAILED;
    }

    int64_t m = 0;
    int64_t n = 0;
    if (x1Shape->GetDimNum() == 2) {
        m = transA ? x1Shape->GetDim(1) : x1Shape->GetDim(0);
    }
    if (x2Shape->GetDimNum() == 2) {
        n = transB ? x2Shape->GetDim(0) : x2Shape->GetDim(1);
    }
    if ((m <= 0) || (n <= 0)) {
        return ge::GRAPH_FAILED;
    }

    outShape->SetDimNum(2);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatMulV3).InferShape(InferShapeQuantBatchMatMulV3);

} // namespace ops

/*!
 * \file quant_matmul_sigmoid_mul_quant_infershape.cpp
 * \brief QuantMatmulSigmoidMulQuant shape inference
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeQuantMatmulSigmoidMulQuant(gert::InferShapeContext* context)
{
    const gert::Shape* x1s = context->GetInputShape(0);
    const gert::Shape* x2s = context->GetInputShape(1);
    if (x1s == nullptr || x2s == nullptr) {
        return ge::GRAPH_FAILED;
    }

    bool tA = false;
    bool tB = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* pa = attrs->GetAttrPointer<bool>(0);
        const bool* pb = attrs->GetAttrPointer<bool>(1);
        if (pa != nullptr) {
            tA = *pa;
        }
        if (pb != nullptr) {
            tB = *pb;
        }
    }

    // logical A is (M, K), logical B is (K, N)
    const int64_t m = tA ? x1s->GetDim(1) : x1s->GetDim(0);
    const int64_t n = tB ? x2s->GetDim(0) : x2s->GetDim(1);

    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* sShape = context->GetOutputShape(1);
    if (yShape == nullptr || sShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(2);
    yShape->SetDim(0, m);
    yShape->SetDim(1, n);
    sShape->SetDimNum(1);
    sShape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantMatmulSigmoidMulQuant).InferShape(InferShapeQuantMatmulSigmoidMulQuant);

} // namespace ops

/*!
 * \file depthwise_conv2_d_infershape.cpp
 * \brief DepthwiseConv2D shape inference with strict validation.
 *
 * Validates: x rank 4, w rank 4, w.shape[0] == C, w.shape[1] == 1, all dims
 * positive, attrs sane, and the computed Hout/Wout positive. Returns
 * GRAPH_FAILED on any violation so mismatched input shapes are rejected instead
 * of being over-read by the kernel.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeDepthwiseConv2D(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || wShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 4 || wShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t n = xShape->GetDim(0);
    const int64_t chIn = xShape->GetDim(1);
    const int64_t hin = xShape->GetDim(2);
    const int64_t win = xShape->GetDim(3);
    const int64_t wCin = wShape->GetDim(0);
    const int64_t wGroup = wShape->GetDim(1);
    const int64_t kh = wShape->GetDim(2);
    const int64_t kw = wShape->GetDim(3);

    /* depthwise contract: one kernel per input channel */
    if (wCin != chIn || wGroup != 1) {
        return ge::GRAPH_FAILED;
    }
    if (n < 1 || chIn < 1 || hin < 1 || win < 1 || kh < 1 || kw < 1) {
        return ge::GRAPH_FAILED;
    }

    int64_t stride = 1;
    int64_t pad = 0;
    int64_t dil = 1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            stride = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(1);
        if (p != nullptr) {
            pad = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(2);
        if (p != nullptr) {
            dil = *p;
        }
    }
    if (stride < 1 || dil < 1 || pad < 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t spanH = hin + 2 * pad - dil * (kh - 1) - 1;
    const int64_t spanW = win + 2 * pad - dil * (kw - 1) - 1;
    if (spanH < 0 || spanW < 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t hout = spanH / stride + 1;
    const int64_t wout = spanW / stride + 1;
    if (hout <= 0 || wout <= 0) {
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(4);
    yShape->SetDim(0, n);
    yShape->SetDim(1, chIn);
    yShape->SetDim(2, hout);
    yShape->SetDim(3, wout);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DepthwiseConv2D).InferShape(InferShapeDepthwiseConv2D);

} // namespace ops

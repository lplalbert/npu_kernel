/*!
 * \file conv_transpose2_d_infershape.cpp
 * \brief ConvTranspose2D shape inference (with strict validation)
 *
 * Validates: x rank4, w rank4, w.shape[0]==C_in, all dims positive,
 * attrs positive, and computed H_out/W_out > 0. Returns GRAPH_FAILED on any
 * violation so mismatched-shape inputs are rejected instead of over-reading w.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeConvTranspose2D(gert::InferShapeContext* context)
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
    int64_t n = xShape->GetDim(0);
    int64_t chIn = xShape->GetDim(1);
    int64_t hin = xShape->GetDim(2);
    int64_t win = xShape->GetDim(3);
    int64_t wCin = wShape->GetDim(0);
    int64_t chOut = wShape->GetDim(1);
    int64_t kh = wShape->GetDim(2);
    int64_t kw = wShape->GetDim(3);

    // w.shape[0] must equal C_in (groups = 1)
    if (wCin != chIn) {
        return ge::GRAPH_FAILED;
    }
    if (n < 1 || chIn < 1 || hin < 1 || win < 1 || chOut < 1 || kh < 1 || kw < 1) {
        return ge::GRAPH_FAILED;
    }

    int64_t stride = 1;
    int64_t pad = 0;
    int64_t outPad = 0;
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
            outPad = *p;
        }
        p = attrs->GetAttrPointer<int64_t>(3);
        if (p != nullptr) {
            dil = *p;
        }
    }
    if (stride < 1 || dil < 1 || pad < 0 || outPad < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t hout = (hin - 1) * stride - 2 * pad + dil * (kh - 1) + outPad + 1;
    int64_t wout = (win - 1) * stride - 2 * pad + dil * (kw - 1) + outPad + 1;
    if (hout <= 0 || wout <= 0) {
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(4);
    yShape->SetDim(0, n);
    yShape->SetDim(1, chOut);
    yShape->SetDim(2, hout);
    yShape->SetDim(3, wout);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConvTranspose2D).InferShape(InferShapeConvTranspose2D);

} // namespace ops

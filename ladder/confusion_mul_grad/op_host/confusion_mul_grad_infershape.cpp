/*!
 * \file confusion_mul_grad_infershape.cpp
 * \brief ConfusionMulGrad shape / dtype inference
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeConfusionMulGrad(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* out0 = context->GetOutputShape(0);
    gert::Shape* out1 = context->GetOutputShape(1);
    if (out0 == nullptr || out1 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *out0 = *inShape;

    int64_t M = inShape->GetDimNum() >= 1 ? inShape->GetDim(0) : 1;
    int64_t N = inShape->GetDimNum() >= 2 ? inShape->GetDim(1) : 1;

    bool keep = false;
    bool red[2] = {false, false};
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* kp = attrs->GetAttrPointer<bool>(1);
        if (kp != nullptr) {
            keep = *kp;
        }
        const gert::TypedContinuousVector<int64_t>* axisV = attrs->GetListInt(0);
        if (axisV != nullptr) {
            const int64_t* ad = axisV->GetData();
            const int64_t an = static_cast<int64_t>(axisV->GetSize());
            for (int64_t i = 0; i < an; ++i) {
                int64_t v = ad[i] % 2;
                if (v < 0) v += 2;
                red[v] = true;
            }
        }
    }

    if (red[0] && red[1]) {
        if (keep) {
            out1->SetDimNum(2);
            out1->SetDim(0, 1);
            out1->SetDim(1, 1);
        } else {
            out1->SetDimNum(0);
        }
    } else if (red[1]) {
        if (keep) {
            out1->SetDimNum(2);
            out1->SetDim(0, M);
            out1->SetDim(1, 1);
        } else {
            out1->SetDimNum(1);
            out1->SetDim(0, M);
        }
    } else if (red[0]) {
        if (keep) {
            out1->SetDimNum(2);
            out1->SetDim(0, 1);
            out1->SetDim(1, N);
        } else {
            out1->SetDimNum(1);
            out1->SetDim(0, N);
        }
    } else {
        out1->SetDimNum(0);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeConfusionMulGrad(gert::InferDataTypeContext* context)
{
    ge::DataType dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    context->SetOutputDataType(1, dt);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionMulGrad)
    .InferShape(InferShapeConfusionMulGrad)
    .InferDataType(InferDataTypeConfusionMulGrad);

}  // namespace ops

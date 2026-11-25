/*!
 * \file gf2k_mul_square_infershape.cpp
 * \brief Gf2kMulSquare shape inference
 *
 * Output c has exactly the shape of a ([N, k/8]). Invalid attribute values or
 * shapes are rejected here so that inference never disagrees with tiling.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeGf2kMulSquare(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (aShape == nullptr || bShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t kVal = 8;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
        if (kPtr != nullptr) {
            kVal = *kPtr;
        }
    }
    if (kVal != 8 && kVal != 128) {
        return ge::GRAPH_FAILED;
    }
    const int64_t width = kVal / 8;
    if (aShape->GetDimNum() != 2 || bShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (aShape->GetDim(1) != width || bShape->GetDim(1) != width) {
        return ge::GRAPH_FAILED;
    }
    if (aShape->GetDim(0) != bShape->GetDim(0) || aShape->GetDim(0) < 0) {
        return ge::GRAPH_FAILED;
    }

    *outShape = *aShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Gf2kMulSquare).InferShape(InferShapeGf2kMulSquare);

} // namespace ops

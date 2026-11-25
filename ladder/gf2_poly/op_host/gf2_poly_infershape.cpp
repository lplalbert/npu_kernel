/*!
 * \file gf2_poly_infershape.cpp
 * \brief Gf2Poly shape inference
 *
 * Output y shape depends on the attributes, using exactly the same rules as
 * tiling:
 *   mode == "mul" -> [N, 2W]
 *   otherwise     -> [N, W]      (W = m/8)
 * N is derived from the input element count divided by W, so inference and
 * tiling can never disagree on the number of rows.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <cstring>

using namespace ge;

namespace ops {

static inline bool ValidM(int64_t mVal)
{
    return mVal == 8 || mVal == 32 || mVal == 64 || mVal == 128 || mVal == 256;
}

static ge::graphStatus InferShapeGf2Poly(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* aShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (aShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t mVal = 8;
    int64_t mode = 2;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* mPtr = attrs->GetAttrPointer<int64_t>(0);
        if (mPtr != nullptr) {
            mVal = *mPtr;
        }
        const char* modePtr = attrs->GetAttrPointer<char>(1);
        if (modePtr != nullptr) {
            if (std::strcmp(modePtr, "mul") == 0) {
                mode = 0;
            } else if (std::strcmp(modePtr, "reduce") == 0) {
                mode = 1;
            } else if (std::strcmp(modePtr, "mul_reduce") == 0) {
                mode = 2;
            } else {
                return ge::GRAPH_FAILED;
            }
        }
    }
    if (!ValidM(mVal)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t w = mVal / 8;
    const int64_t outW = (mode == 0) ? (2 * w) : w;

    const size_t rank = aShape->GetDimNum();
    int64_t totalElems = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t d = aShape->GetDim(i);
        if (d < 0) {
            return ge::GRAPH_FAILED;
        }
        totalElems *= d;
    }
    if (rank == 0 || totalElems % w != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t n = totalElems / w;

    yShape->SetDimNum(2);
    yShape->SetDim(0, n);
    yShape->SetDim(1, outW);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Gf2Poly).InferShape(InferShapeGf2Poly);

}  // namespace ops

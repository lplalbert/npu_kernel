/*!
 * \file bit_plane_transform_infershape.cpp
 * \brief BitPlaneTransform operator shape inference (validated, same rules as tiling).
 *
 *  transpose: x[N,C]  -> y[ceil(N/32), 8C, 4]
 *  inverse:   x[G,8C,4], attr n (>0, ceil(n/32) <= G) -> y[n, C]
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <cstring>

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeBitPlaneTransform(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char* dir = "transpose";
    int64_t n = -1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char* d = attrs->GetAttrPointer<char>(0);
        if (d != nullptr) {
            dir = d;
        }
        const int64_t* np = attrs->GetAttrPointer<int64_t>(1);
        if (np != nullptr) {
            n = *np;
        }
    }

    bool isInverse = false;
    if (std::strcmp(dir, "transpose") == 0) {
        isInverse = false;
    } else if (std::strcmp(dir, "inverse") == 0) {
        isInverse = true;
    } else {
        return ge::GRAPH_FAILED;
    }

    const size_t rank = inShape->GetDimNum();

    if (!isInverse) {
        if (rank != 2) {
            return ge::GRAPH_FAILED;
        }
        const int64_t N = inShape->GetDim(0);
        const int64_t C = inShape->GetDim(1);
        if (N <= 0 || C <= 0) {
            return ge::GRAPH_FAILED;
        }
        const int64_t G = (N + 31) / 32;
        outShape->SetDimNum(3);
        outShape->SetDim(0, G);
        outShape->SetDim(1, 8 * C);
        outShape->SetDim(2, 4);
    } else {
        if (rank != 3) {
            return ge::GRAPH_FAILED;
        }
        const int64_t Gin = inShape->GetDim(0);
        const int64_t Q = inShape->GetDim(1);
        const int64_t lastD = inShape->GetDim(2);
        if (Gin <= 0 || Q <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (lastD != 4) {
            return ge::GRAPH_FAILED;
        }
        if (Q % 8 != 0) {
            return ge::GRAPH_FAILED;
        }
        const int64_t C = Q / 8;
        if (n <= 0) {
            return ge::GRAPH_FAILED;
        }
        const int64_t Gproc = (n + 31) / 32;
        if (Gproc > Gin) {
            return ge::GRAPH_FAILED;  // would read beyond input groups (OOB)
        }
        outShape->SetDimNum(2);
        outShape->SetDim(0, n);
        outShape->SetDim(1, C);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BitPlaneTransform).InferShape(InferShapeBitPlaneTransform);

} // namespace ops

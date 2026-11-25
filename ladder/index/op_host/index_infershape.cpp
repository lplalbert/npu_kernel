/*!
 * \file index_infershape.cpp
 * \brief Index
 *
 *  indexed_sizes
 *   - out = [] + [indices.shape] + []
 *   - out = [indices.shape] + []
 *  indexed_sizes  x
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeIndex(gert::InferShapeContext* context)
{
    auto xShapePtr = context->GetInputShape(0);
    auto yShapePtr = context->GetOutputShape(0);
    if (xShapePtr == nullptr || yShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& xs = *xShapePtr;
    const int64_t ndim = static_cast<int64_t>(xs.GetDimNum());
    if (ndim > 10) {
        return ge::GRAPH_FAILED; //
    }

    const int64_t* mask = nullptr;
    const gert::Tensor* maskTensor = context->GetInputTensor(1);
    if (maskTensor != nullptr) {
        mask = maskTensor->GetData<int64_t>();
    }
    if (mask == nullptr) {
        *yShapePtr = xs;
        return ge::GRAPH_SUCCESS;
    }

    int64_t firstIdx = -1;
    int64_t lastIdx = -1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (mask[i] != 0) {
            if (firstIdx < 0) {
                firstIdx = i;
            }
            lastIdx = i;
        }
    }

    //  indices
    int64_t idxDims[16];
    int64_t idxNdim = 0;
    auto idxShapePtr = context->GetInputShape(2);
    if (idxShapePtr != nullptr) {
        const gert::Shape& is = *idxShapePtr;
        idxNdim = static_cast<int64_t>(is.GetDimNum());
        if (idxNdim > 16) {
            return ge::GRAPH_FAILED; // indices
        }
        for (int64_t i = 0; i < idxNdim; ++i) {
            idxDims[i] = is.GetDim(i);
        }
    }

    yShapePtr->SetDimNum(0);
    if (firstIdx < 0) {
        //  x
        for (int64_t i = 0; i < ndim; ++i) {
            yShapePtr->AppendDim(xs.GetDim(i));
        }
        return ge::GRAPH_SUCCESS;
    }

    bool contig = true;
    for (int64_t i = 0; i < ndim; ++i) {
        bool inBlock = (i >= firstIdx && i <= lastIdx);
        if (inBlock && mask[i] == 0) {
            contig = false;
        }
        if (!inBlock && mask[i] != 0) {
            contig = false;
        }
    }

    if (contig) {
        for (int64_t i = 0; i < firstIdx; ++i) {
            yShapePtr->AppendDim(xs.GetDim(i));
        }
        for (int64_t i = 0; i < idxNdim; ++i) {
            yShapePtr->AppendDim(idxDims[i]);
        }
        for (int64_t i = lastIdx + 1; i < ndim; ++i) {
            yShapePtr->AppendDim(xs.GetDim(i));
        }
    } else {
        for (int64_t i = 0; i < idxNdim; ++i) {
            yShapePtr->AppendDim(idxDims[i]);
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (mask[i] == 0) {
                yShapePtr->AppendDim(xs.GetDim(i));
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Index).InferShape(InferShapeIndex);

} // namespace ops

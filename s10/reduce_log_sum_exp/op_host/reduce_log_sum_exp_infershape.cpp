/*!
 * \file reduce_log_sum_exp_infershape.cpp
 * \brief ReduceLogSumExp 算子形状推导实现
 *
 * y 的 shape 依赖 axes 取值与 keep_dims：
 *   keep_dims=true  归约维保留为 1，其余与 x 一致；
 *   keep_dims=false 删除归约维；全归约时输出 0 维标量；
 *   rank=0 输出 0 维标量。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeReduceLogSumExp(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t rank = xShape->GetDimNum();
    if (rank == 0) {
        outShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    // keep_dims 属性（可选，默认 false）
    bool keepDims = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* keepPtr = attrs->GetAttrPointer<bool>(0);
        if (keepPtr != nullptr) {
            keepDims = *keepPtr;
        }
    }

    // axes 归约标记（负数归一化）
    bool reduceFlags[8] = {false, false, false, false, false, false, false, false};
    const gert::Tensor* axesTensor = context->GetInputTensor(1);
    if (axesTensor != nullptr) {
        const int32_t* axesData = axesTensor->GetData<int32_t>();
        int64_t axesNum = axesTensor->GetShapeSize();
        if (axesData != nullptr) {
            for (int64_t i = 0; i < axesNum && i < 8; ++i) {
                int64_t a = static_cast<int64_t>(axesData[i]);
                if (a < 0) {
                    a += rank;
                }
                if (a >= 0 && a < rank) {
                    reduceFlags[a] = true;
                }
            }
        }
    }

    int64_t outRank = 0;
    int64_t dims[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < rank; ++i) {
        if (reduceFlags[i]) {
            if (keepDims) {
                dims[outRank++] = 1;
            }
        } else {
            dims[outRank++] = xShape->GetDim(i);
        }
    }

    outShape->SetDimNum(outRank);
    for (int64_t i = 0; i < outRank; ++i) {
        outShape->SetDim(i, dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceLogSumExp).InferShape(InferShapeReduceLogSumExp);

} // namespace ops

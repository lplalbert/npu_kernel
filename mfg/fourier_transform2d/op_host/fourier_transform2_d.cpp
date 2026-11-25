// Host侧Tiling实现  FourierTransform2D: y = fft2(x) / ifft2(x)
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fourier_transform2_d_tiling.h"
#include "../op_kernel/tiling_key_fourier_transform2_d.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores = platform.GetCoreNumAiv();
    if (num_cores < 1) num_cores = 1;
    if (num_cores > 64) num_cores = 64;

    const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    int64_t b = 1, n = 64, m = 64;
    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    if (xShape.GetDimNum() >= 3) {
        b = xShape.GetDim(0);
        n = xShape.GetDim(1);
        m = xShape.GetDim(2);
    } else if (xShape.GetDimNum() == 2) {
        n = xShape.GetDim(0);
        m = xShape.GetDim(1);
    }
    if (b < 1) b = 1;
    if (n < 2) n = 2;
    if (m < 2) m = 2;

    bool inverse = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *p = attrs->GetBool(0);
        if (p != nullptr) inverse = *p;
    }
    uint32_t logN = 0, logM = 0;
    while ((int64_t(1) << logN) < n) ++logN;
    while ((int64_t(1) << logM) < m) ++logM;

    FourierTransform2DTilingData* tiling = context->GetTilingData<FourierTransform2DTilingData>();
    tiling->length = static_cast<uint32_t>(tensor_x->GetShapeSize());
    tiling->b = static_cast<uint32_t>(b);
    tiling->n = static_cast<uint32_t>(n);
    tiling->m = static_cast<uint32_t>(m);
    tiling->logN = logN;
    tiling->logM = logM;
    tiling->coreNum = static_cast<uint32_t>(num_cores);
    tiling->inverse = inverse ? 1u : 0u;
    uint32_t tasks = static_cast<uint32_t>(b * n);
    uint32_t rpc = (tasks + static_cast<uint32_t>(num_cores) - 1) / static_cast<uint32_t>(num_cores);
    if (rpc < 1) rpc = 1;
    tiling->rowsPerCore = rpc;
    tiling->probe = 0;
    context->SetBlockDim(static_cast<uint32_t>(num_cores));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FourierTransform2D : public OpDef {
    public:
        explicit FourierTransform2D(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_COMPLEX64})
                .Format({ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_COMPLEX64})
                .Format({ge::FORMAT_ND});
            this->Attr("inverse").AttrType(OPTIONAL).Bool(false);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FourierTransform2D);
}  // namespace ops

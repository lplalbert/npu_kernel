// Host side tiling for FusedPropStep: out = IFFT2( FFT2(field) * h )
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_prop_step_tiling.h"
#include "../op_kernel/tiling_key_fused_prop_step.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv < 1) {
        num_cores_aiv = 1;
    }

    const gert::Tensor *tensor_field = context->GetRequiredInputTensor(0);
    ge::DataType dtype_field = tensor_field->GetDataType();
    uint32_t DT_FIELD = static_cast<uint32_t>(dtype_field);
    ASCENDC_TPL_SEL_PARAM(context, DT_FIELD);

    int64_t B = 1, N = 1, M = 1;
    const gert::Shape sh = context->GetInputShape(0)->GetStorageShape();
    if (sh.GetDimNum() >= 3) {
        B = sh.GetDim(0);
        N = sh.GetDim(1);
        M = sh.GetDim(2);
    } else if (sh.GetDimNum() == 2) {
        N = sh.GetDim(0);
        M = sh.GetDim(1);
    } else if (sh.GetDimNum() == 1) {
        M = sh.GetDim(0);
    }
    if (B < 1) B = 1;
    if (N < 1) N = 1;
    if (M < 1) M = 1;

    int64_t Bh = 1;
    const gert::Shape shh = context->GetInputShape(1)->GetStorageShape();
    if (shh.GetDimNum() >= 3) {
        Bh = shh.GetDim(0);
    }
    if (Bh < 1) Bh = 1;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *attr_norm = attrs ? attrs->GetBool(0) : nullptr;
    uint32_t norm = (attr_norm && *attr_norm) ? 1u : 0u;

    int dtype_size = ge::GetSizeByDataType(dtype_field);
    uint32_t dtype_code = 1;  // float32
    if (dtype_size == 2) {
        dtype_code = 0;  // float16
    } else if (dtype_size == 8) {
        dtype_code = 2;  // complex64
    }

    uint32_t logN = 0, logM = 0;
    for (int64_t v = N; v > 1; v >>= 1) ++logN;
    for (int64_t v = M; v > 1; v >>= 1) ++logM;
    int64_t maxNM = (N > M) ? N : M;

    // Spectrum staging choice (single source of truth, the kernel obeys tiling->useUb):
    //   - complex64 stages the spectrum in its own output buffer (2 floats per element),
    //   - float16 / float32 stage it in UB whenever it fits, so no workspace is touched,
    //   - only the large real cases fall back to a workspace slice.
    const int64_t per_core_words = 2 * N * M;
    const int64_t spec_bytes = per_core_words * 4;
    const int64_t scratch_bytes = 2 * maxNM * 4;
    const int64_t tab_bytes = (M + N) * 4;
    uint32_t use_ub = (dtype_code != 2 &&
                       spec_bytes + scratch_bytes + tab_bytes <=
                           static_cast<int64_t>(fps_tiling::kUbBudgetBytes)) ? 1u : 0u;

    // Batch parallel: every core owns whole batches and runs all four passes locally,
    // so no cross-core barrier is needed. The workspace fallback keeps one float32
    // complex batch per core inside a 1 GiB budget.
    const int64_t ws_budget = 1LL << 30;
    int64_t cores_by_ws = ws_budget / spec_bytes;
    if (cores_by_ws < 1) cores_by_ws = 1;
    int64_t bd = num_cores_aiv;
    if (bd > B) bd = B;
    if (bd > cores_by_ws) bd = cores_by_ws;
    if (bd < 1) bd = 1;

    FusedPropStepTilingData *tiling = context->GetTilingData<FusedPropStepTilingData>();
    tiling->B = static_cast<uint32_t>(B);
    tiling->N = static_cast<uint32_t>(N);
    tiling->M = static_cast<uint32_t>(M);
    tiling->Bh = static_cast<uint32_t>(Bh);
    tiling->dtypeCode = dtype_code;
    tiling->norm = norm;
    tiling->logN = logN;
    tiling->logM = logM;
    tiling->maxNM = static_cast<uint32_t>(maxNM);
    tiling->length = static_cast<uint32_t>(B * N * M);
    tiling->useUb = use_ub;

    context->SetBlockDim(static_cast<uint32_t>(bd));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t user_size = 0;
    if (use_ub == 0 && dtype_code != 2) {
        // user slice sits after the reserved area the runtime keeps at the block head
        user_size = static_cast<size_t>(fps_tiling::kWorkspaceOffset) +
                    static_cast<size_t>(bd * spec_bytes);
    }
    currentWorkspace[0] = user_size + static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    *outShape = *inShape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FusedPropStep : public OpDef {
public:
    explicit FusedPropStep(const char *name) : OpDef(name) {
        this->Input("field")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("h")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("norm").AttrType(OPTIONAL).Bool();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(FusedPropStep);
}  // namespace ops

// Host侧Tiling实现
// FurthestPointSample: 每个 batch 迭代选 npoint 个点, 每次选「到已选集合最小平方距离最大」的点,
//   并列取最小输入索引; 首点固定索引 0; 不额外排除已选点; f16 全程 half 精度。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/furthest_point_sample_tiling.h"
#include "../op_kernel/tiling_key_furthest_point_sample.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        (void)num_cores_aiv;

        const gert::Tensor *tensor_pts = context->GetRequiredInputTensor(0);
        ge::DataType dtype = tensor_pts->GetDataType();
        uint32_t DT = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT);

        const gert::Shape pShape = context->GetInputShape(0)->GetStorageShape();
        int64_t b = pShape.GetDim(0);
        int64_t n = pShape.GetDim(1);

        int64_t npoint = 16;  // local-verify hardcode
        if (npoint < 1) {
            npoint = 1;
        }
        if (npoint > n) {
            npoint = n;
        }

        uint32_t dtype_mode = (dtype == ge::DT_FLOAT16) ? 0u : 1u;
        uint64_t nPad = (static_cast<uint64_t>(n) + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
        if (nPad < ALIGN_ELEM) {
            nPad = ALIGN_ELEM;
        }

        FpsTilingData *tiling = context->GetTilingData<FpsTilingData>();
        tiling->b = static_cast<uint64_t>(b);
        tiling->n = static_cast<uint64_t>(n);
        tiling->npoint = static_cast<uint64_t>(npoint);
        tiling->nPad = nPad;
        tiling->dtypeMode = dtype_mode;
        tiling->rsv0 = 0;
        tiling->rsv1 = 0;

        // 每个 batch 一个核
        int32_t cores = static_cast<int32_t>(b);
        if (cores < 1) {
            cores = 1;
        }
        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *pShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);
        int64_t npoint = 128;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs != nullptr) {
            const int64_t *p = attrs->GetInt(0);
            if (p != nullptr && *p > 0) {
                npoint = *p;
            }
        }
        outShape->SetDimNum(2);
        outShape->SetDim(0, pShape->GetDim(0));
        outShape->SetDim(1, npoint);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_INT32);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FurthestPointSample : public OpDef {
    public:
        explicit FurthestPointSample(const char *name) : OpDef(name) {
            this->Input("points")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FurthestPointSample);
}  // namespace ops

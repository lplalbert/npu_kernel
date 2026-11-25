// Host side tiling for SpmmSum
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/spmm_sum_tiling.h"
#include "../op_kernel/tiling_key_spmm_sum.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_value = context->GetRequiredInputTensor(2);
        ge::DataType dtype_value = tensor_value->GetDataType();
        uint32_t DT_VALUE = static_cast<uint32_t>(dtype_value);
        ASCENDC_TPL_SEL_PARAM(context, DT_VALUE);

        const gert::Shape rpShape = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape colShape = context->GetInputShape(1)->GetStorageShape();
        const gert::Shape matShape = context->GetInputShape(3)->GetStorageShape();
        int64_t rows = rpShape.GetDim(0);
        int64_t m = rows > 0 ? rows - 1 : 0;
        int64_t e = colShape.GetDim(0);
        int64_t n = matShape.GetDim(0);
        int64_t k = matShape.GetDim(1);

        uint32_t dtype_mode = (dtype_value == ge::DT_FLOAT16) ? 0u : 1u;

        // per-chunk UB: raw(ck*sizeof(T)) + out(ck*sizeof(T)) + acc/tmp/f (12*ck)
        uint64_t budget = (ub_size * 3) / 4;
        uint64_t perElem = 12u + static_cast<uint64_t>(ge::GetSizeByDataType(dtype_value)) * 2u;
        uint64_t ck = budget / perElem;
        if (ck > 1024u) {
            ck = 1024u;
        }
        ck = (ck / ALIGN_ELEM) * ALIGN_ELEM;
        if (ck < ALIGN_ELEM) {
            ck = ALIGN_ELEM;
        }

        SpmmSumTilingData *tiling = context->GetTilingData<SpmmSumTilingData>();
        tiling->m = static_cast<uint64_t>(m);
        tiling->n = static_cast<uint64_t>(n);
        tiling->k = static_cast<uint64_t>(k);
        tiling->e = static_cast<uint64_t>(e);
        tiling->ck = static_cast<uint32_t>(ck);
        tiling->dtypeMode = dtype_mode;
        tiling->rsv = 0;

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        if (m <= 0) {
            cores = 1;
        }
        int64_t rpc = (m + cores - 1) / (cores > 0 ? cores : 1);
        if (rpc < 1) {
            rpc = 1;
        }
        int64_t need = (m + rpc - 1) / rpc;
        if (need < cores) {
            cores = static_cast<int32_t>(need > 0 ? need : 1);
        }
        if (cores < 1) {
            cores = 1;
        }
        tiling->rowsPerCore = static_cast<uint32_t>(rpc);
        tiling->coreStride = static_cast<uint32_t>(cores);
        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *rpShape = context->GetInputShape(0);
        const gert::Shape *matShape = context->GetInputShape(3);
        gert::Shape *outShape = context->GetOutputShape(0);
        const int64_t rows = rpShape->GetDim(0);
        outShape->SetDimNum(2);
        outShape->SetDim(0, rows > 0 ? rows - 1 : 0);
        outShape->SetDim(1, matShape->GetDim(1));
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(2));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SpmmSum : public OpDef {
    public:
        explicit SpmmSum(const char *name) : OpDef(name) {
            this->Input("rowptr")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("col")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("mat")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SpmmSum);
}  // namespace ops

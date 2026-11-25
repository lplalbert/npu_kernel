// Host side tiling for SegmentMinCsr
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/segment_min_csr_tiling.h"
#include "../op_kernel/tiling_key_segment_min_csr.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_src = context->GetRequiredInputTensor(0);
        ge::DataType dtype_src = tensor_src->GetDataType();
        uint32_t DT_SRC = static_cast<uint32_t>(dtype_src);
        ASCENDC_TPL_SEL_PARAM(context, DT_SRC);

        const gert::Shape srcShape = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape ptrShape = context->GetInputShape(1)->GetStorageShape();
        int64_t rows = ptrShape.GetDim(0);
        int64_t n = rows > 0 ? rows - 1 : 0;
        int64_t e = srcShape.GetDimNum() >= 1 ? srcShape.GetDim(0) : 0;
        int64_t k = srcShape.GetDimNum() >= 2 ? srcShape.GetDim(1) : 0;

        uint32_t dtype_mode = (dtype_src == ge::DT_FLOAT16) ? 0u : 1u;

        // per chunk: raw(T) + out(T) + 4 f32 + mask(uint8) + ai(int32) + wIn/wOut(2*cn*4 each) + offs
        uint64_t budget = (ub_size * 3) / 4;
        uint64_t perElem = 4u * 4u + 1u + 4u + 16u + static_cast<uint64_t>(ge::GetSizeByDataType(dtype_src)) * 2u;
        uint64_t ck = budget / perElem;
        if (ck > 1024u) {
            ck = 1024u;
        }
        ck = (ck / ALIGN_ELEM) * ALIGN_ELEM;
        if (ck < ALIGN_ELEM) {
            ck = ALIGN_ELEM;
        }

        SegmentMinCsrTilingData *tiling = context->GetTilingData<SegmentMinCsrTilingData>();
        tiling->n = static_cast<uint64_t>(n);
        tiling->e = static_cast<uint64_t>(e);
        tiling->k = static_cast<uint64_t>(k);
        tiling->ck = static_cast<uint32_t>(ck);
        tiling->dtypeMode = dtype_mode;
        tiling->rsv0 = 0;
        tiling->rsv1 = 0;

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        if (n <= 0) {
            cores = 1;
        }
        int64_t rpc = (n + cores - 1) / (cores > 0 ? cores : 1);
        if (rpc < 1) {
            rpc = 1;
        }
        int64_t need = (n + rpc - 1) / rpc;
        if (need < cores) {
            cores = static_cast<int32_t>(need > 0 ? need : 1);
        }
        if (cores < 1) {
            cores = 1;
        }
        tiling->rowsPerCore = static_cast<uint64_t>(rpc);
        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *ptrShape = context->GetInputShape(1);
        const gert::Shape *srcShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);
        gert::Shape *argShape = context->GetOutputShape(1);
        const int64_t rows = ptrShape->GetDim(0);
        const int64_t k = srcShape->GetDimNum() >= 2 ? srcShape->GetDim(1) : 0;
        outShape->SetDimNum(2);
        outShape->SetDim(0, rows > 0 ? rows - 1 : 0);
        outShape->SetDim(1, k);
        argShape->SetDimNum(2);
        argShape->SetDim(0, rows > 0 ? rows - 1 : 0);
        argShape->SetDim(1, k);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        context->SetOutputDataType(1, ge::DT_INT64);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SegmentMinCsr : public OpDef {
    public:
        explicit SegmentMinCsr(const char *name) : OpDef(name) {
            this->Input("src")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("indptr")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("arg_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SegmentMinCsr);
}  // namespace ops

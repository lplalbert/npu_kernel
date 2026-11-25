// Host侧Tiling实现
// GatherCoo: out[i, :] = src[index[i], :]   (i in [0, E))
// 纯带宽算子: 输出连续写, 输入按 index 随机读整行。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gather_coo_tiling.h"
#include "../op_kernel/tiling_key_gather_coo.h"

namespace optiling {
    static constexpr uint64_t GROUP_ROWS_MAX = 4096;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_src = context->GetRequiredInputTensor(0);
        ge::DataType dtype_src = tensor_src->GetDataType();
        int dtype_size_src = ge::GetSizeByDataType(dtype_src);
        uint32_t DT_SRC = static_cast<uint32_t>(dtype_src);
        ASCENDC_TPL_SEL_PARAM(context, DT_SRC);

        const gert::Shape srcShape = context->GetInputShape(0)->GetStorageShape();
        int64_t m = srcShape.GetDim(0);
        int64_t k = srcShape.GetDim(1);
        const gert::Shape idxShape = context->GetInputShape(1)->GetStorageShape();
        int64_t e = idxShape.GetDim(0);
        if (m <= 0 || k <= 0 || e <= 0) {
            return ge::GRAPH_SUCCESS;
        }

        const uint64_t rowBytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(dtype_size_src);
        const uint64_t rowStride = ((rowBytes + 31) / 32) * 32;  // UB 内行首 32B 对齐
        // 分组: 一半 UB 给行缓冲(双缓冲), 另需 groupRows*4 字节放 index
        uint64_t budget = (ub_size / 2);
        uint64_t group = budget / (2 * rowStride + 4);
        if (group < 1) {
            group = 1;
        }
        if (group > GROUP_ROWS_MAX) {
            group = GROUP_ROWS_MAX;
        }

        GatherCooTilingData *tiling = context->GetTilingData<GatherCooTilingData>();

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        int64_t per = (e + cores - 1) / cores;
        per = ((per + 7) / 8) * 8;
        if (per < 8) {
            per = 8;
        }
        int64_t needCores = (e + per - 1) / per;
        if (needCores < static_cast<int64_t>(cores)) {
            cores = needCores > 0 ? static_cast<int32_t>(needCores) : 1;
        }
        if (cores < 1) {
            cores = 1;
        }

        tiling->m = static_cast<uint64_t>(m);
        tiling->k = static_cast<uint64_t>(k);
        tiling->e = static_cast<uint64_t>(e);
        tiling->rowsPerCore = static_cast<uint64_t>(per);
        tiling->groupRows = static_cast<uint32_t>(group);

        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *srcShape = context->GetInputShape(0);
        const gert::Shape *idxShape = context->GetInputShape(1);
        gert::Shape *outShape = context->GetOutputShape(0);
        outShape->SetDimNum(2);
        outShape->SetDim(0, idxShape->GetDim(0));
        outShape->SetDim(1, srcShape->GetDim(1));
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class GatherCoo : public OpDef {
    public:
        explicit GatherCoo(const char *name) : OpDef(name) {
            this->Input("src")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(GatherCoo);
}  // namespace ops

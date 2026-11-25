// Host侧Tiling实现
// ComplexExp: 复数 out = e^a*(cos b + i sin b); 实数 out = e^a
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/complex_exp_tiling.h"
#include "../op_kernel/tiling_key_complex_exp.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;
    // UB 字节/元素(双缓冲+暂存+偏移):
    //   complex64 -> in 8B*2 + out 8B*2 + a/b/c 4B*3 + W 8B + 偏移 24B = 76B
    static constexpr uint64_t BYTES_C64 = 76;
    //   float32   -> in 4B*2 + out 4B*2 = 16B
    static constexpr uint64_t BYTES_F32 = 16;
    //   float16   -> in 2B*2 + out 2B*2 + float 暂存 4B = 12B
    static constexpr uint64_t BYTES_F16 = 12;
    static constexpr uint64_t TILE_MAX = 8192;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t dtype_mode = 1;
        uint64_t bytes_per_elem = BYTES_F32;
        if (dtype_x == ge::DT_COMPLEX64) {
            dtype_mode = 2;
            bytes_per_elem = BYTES_C64;
        } else if (dtype_x == ge::DT_FLOAT16) {
            dtype_mode = 0;
            bytes_per_elem = BYTES_F16;
        }

        int64_t length = static_cast<int64_t>(tensor_x->GetShapeSize());

        uint64_t budget = (ub_size * 3) / 4;
        uint64_t tile = budget / bytes_per_elem;
        if (tile > ALIGN_ELEM) {
            tile -= ALIGN_ELEM;  // 留出尾部预留空间
        }
        tile = (tile / ALIGN_ELEM) * ALIGN_ELEM;
        if (tile < ALIGN_ELEM) {
            tile = ALIGN_ELEM;
        }
        if (tile > TILE_MAX) {
            tile = TILE_MAX;
        }

        ComplexExpTilingData *tiling = context->GetTilingData<ComplexExpTilingData>();

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        int64_t per = (length + cores - 1) / cores;
        per = ((per + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
        if (per < static_cast<int64_t>(ALIGN_ELEM)) {
            per = static_cast<int64_t>(ALIGN_ELEM);
        }
        int64_t need_cores = (length + per - 1) / per;
        if (need_cores < static_cast<int64_t>(cores)) {
            cores = need_cores > 0 ? static_cast<int32_t>(need_cores) : 1;
        }
        if (cores < 1) {
            cores = 1;
        }

        tiling->length = static_cast<uint64_t>(length);
        tiling->blockFactor = static_cast<uint64_t>(per);
        tiling->tileLength = static_cast<uint32_t>(tile);
        tiling->dtypeMode = dtype_mode;

        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);
        *outShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class ComplexExp : public OpDef {
    public:
        explicit ComplexExp(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_COMPLEX64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_COMPLEX64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ComplexExp);
}  // namespace ops

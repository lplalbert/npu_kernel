// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/complex_abs_tiling.h"
#include "../op_kernel/tiling_key_complex_abs.h"

namespace optiling {
    // ComplexAbs 是纯逐点访存受限算子:
    //   out[i] = sqrt(re[i]^2 + im[i]^2)
    // 输入在 GM 中按 [re, im] 交错排布, 每个复数 8 字节。
    // kernel 侧把输入当 float32 数组处理(长度 = 2 * 复数个数)。
    // 每核处理一段连续的复数元素; 段内按 tile 切块做 double buffer。
    static constexpr uint64_t ALIGN_COMPLEX = 64;      // 对齐到 64 个复数(128 float), 满足向量指令 repeat 对齐
    static constexpr uint64_t BYTES_PER_COMPLEX = 40;  // in 8B*2 + out 4B*2 + a/b 4B*2 + 偏移 8B
    static constexpr uint64_t TILE_MAX = 8192;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_src = context->GetRequiredInputTensor(0);
        ge::DataType dtype_src = tensor_src->GetDataType();  // DT_COMPLEX64
        uint32_t DT_SRC = static_cast<uint32_t>(dtype_src);
        ASCENDC_TPL_SEL_PARAM(context, DT_SRC);

        int64_t length = static_cast<int64_t>(tensor_src->GetShapeSize());  // 复数元素个数

        // 按 UB 容量切块: 预留 1/4 余量给框架和同步开销
        uint64_t budget = (ub_size * 3) / 4;
        uint64_t tile = budget / BYTES_PER_COMPLEX;
        if (tile > ALIGN_COMPLEX) {
            tile -= ALIGN_COMPLEX;  // 留出尾部预留空间
        }
        tile = (tile / ALIGN_COMPLEX) * ALIGN_COMPLEX;
        if (tile < ALIGN_COMPLEX) {
            tile = ALIGN_COMPLEX;
        }
        if (tile > TILE_MAX) {
            tile = TILE_MAX;
        }

        ComplexAbsTilingData *tiling = context->GetTilingData<ComplexAbsTilingData>();

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        int64_t per = (length + cores - 1) / cores;
        per = ((per + ALIGN_COMPLEX - 1) / ALIGN_COMPLEX) * ALIGN_COMPLEX;
        if (per < static_cast<int64_t>(ALIGN_COMPLEX)) {
            per = static_cast<int64_t>(ALIGN_COMPLEX);
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
        tiling->usedCores = static_cast<uint32_t>(cores);

        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *srcShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);
        *outShape = *srcShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class ComplexAbs : public OpDef {
    public:
        explicit ComplexAbs(const char *name) : OpDef(name) {
            this->Input("src")
                .ParamType(REQUIRED)
                .DataType({ge::DT_COMPLEX64})
                .Format({ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ComplexAbs);
}  // namespace ops

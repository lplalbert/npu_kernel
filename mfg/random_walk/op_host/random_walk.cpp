// Host side tiling for RandomWalk
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/random_walk_tiling.h"
#include "../op_kernel/tiling_key_random_walk.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_rand = context->GetRequiredInputTensor(3);
        ge::DataType dtype_rand = tensor_rand->GetDataType();
        uint32_t DT_RAND = static_cast<uint32_t>(dtype_rand);
        ASCENDC_TPL_SEL_PARAM(context, DT_RAND);

        const gert::Shape rpShape = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape colShape = context->GetInputShape(1)->GetStorageShape();
        const gert::Shape seedShape = context->GetInputShape(2)->GetStorageShape();
        const gert::Shape randShape = context->GetInputShape(3)->GetStorageShape();
        int64_t rows = rpShape.GetDim(0);
        int64_t n = rows > 0 ? rows - 1 : 0;
        int64_t e = colShape.GetDim(0);
        int64_t b = seedShape.GetDim(0);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_wl = attrs->GetInt(0);
        const float *attr_p = attrs->GetFloat(1);
        const float *attr_q = attrs->GetFloat(2);
        int64_t wl = (attr_wl != nullptr) ? *attr_wl : 16;
        if (wl < 0) {
            wl = 0;
        }
        if (randShape.GetDimNum() >= 2) {
            const int64_t rwl = randShape.GetDim(1);
            if (rwl > 0) {
                wl = rwl;
            }
        }
        float p = (attr_p != nullptr) ? *attr_p : 1.0f;
        float q = (attr_q != nullptr) ? *attr_q : 1.0f;

        RandomWalkTilingData *tiling = context->GetTilingData<RandomWalkTilingData>();
        tiling->n = static_cast<uint64_t>(n);
        tiling->e = static_cast<uint64_t>(e > 0 ? e : 0);
        tiling->b = static_cast<uint64_t>(b);
        tiling->wl = static_cast<uint64_t>(wl);
        tiling->p = p;
        tiling->q = q;
        tiling->chunkW = static_cast<uint32_t>(RW_CHUNK_W);

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        if (b <= 0) {
            cores = 1;
        }
        int64_t wpc = (b + cores - 1) / (cores > 0 ? cores : 1);
        if (wpc < 1) {
            wpc = 1;
        }
        int64_t need = (b + wpc - 1) / wpc;
        if (need < cores) {
            cores = static_cast<int32_t>(need > 0 ? need : 1);
        }
        if (cores < 1) {
            cores = 1;
        }
        tiling->walkersPerCore = static_cast<uint32_t>(wpc);
        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *seedShape = context->GetInputShape(2);
        const gert::Shape *randShape = context->GetInputShape(3);
        gert::Shape *outShape = context->GetOutputShape(0);
        int64_t wl = 16;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs != nullptr) {
            const int64_t *p = attrs->GetInt(0);
            if (p != nullptr && *p >= 0) {
                wl = *p;
            }
        }
        if (randShape->GetDimNum() >= 2 && randShape->GetDim(1) > 0) {
            wl = randShape->GetDim(1);
        }
        outShape->SetDimNum(2);
        outShape->SetDim(0, seedShape->GetDim(0));
        outShape->SetDim(1, wl + 1);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_INT64);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class RandomWalk : public OpDef {
    public:
        explicit RandomWalk(const char *name) : OpDef(name) {
            this->Input("rowptr")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("col")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("seed")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("rand")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("walk_length").AttrType(OPTIONAL).Int(16);
            this->Attr("p").AttrType(OPTIONAL).Float(1.0);
            this->Attr("q").AttrType(OPTIONAL).Float(1.0);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(RandomWalk);
}  // namespace ops

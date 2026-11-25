// Host侧Tiling实现
// BallQuery: 对每个查询中心 new_xyz[b,m], 在 xyz[b] 中按索引从小到大扫描,
//            收集 d^2 < radius^2 的点的索引, 取前 nsample 个; 不足则用首个命中索引填充; 无命中全 0。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/ball_query_tiling.h"
#include "../op_kernel/tiling_key_ball_query.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_xyz = context->GetRequiredInputTensor(0);
        ge::DataType dtype_xyz = tensor_xyz->GetDataType();
        uint32_t DT_XYZ = static_cast<uint32_t>(dtype_xyz);
        ASCENDC_TPL_SEL_PARAM(context, DT_XYZ);

        const gert::Shape xyzShape = context->GetInputShape(0)->GetStorageShape();
        int64_t batch = xyzShape.GetDim(0);
        int64_t n = xyzShape.GetDim(1);
        const gert::Shape newShape = context->GetInputShape(1)->GetStorageShape();
        int64_t m = newShape.GetDim(1);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_radius = attrs->GetFloat(0);
        const int64_t *attr_nsample = attrs->GetInt(1);
        float radius = (attr_radius != nullptr) ? *attr_radius : 0.4f;
        int64_t nsample = (attr_nsample != nullptr) ? *attr_nsample : 16;
        if (nsample < 1) {
            nsample = 1;
        }

        uint32_t dtype_mode = (dtype_xyz == ge::DT_FLOAT16) ? 0u : 1u;

        // 每块 UB: raw(3*chunk) + float 拷贝(3*chunk) + x/y/z(3*chunk) + d2/tmp(2*chunk) = 11*chunk float
        uint64_t budget = (ub_size * 3) / 4;
        uint64_t chunk = budget / (11u * sizeof(float));
        if (chunk > 1024u) {
            chunk = 1024u;
        }
        chunk = (chunk / 32u) * 32u;
        if (chunk < 32u) {
            chunk = 32u;
        }
        if (n > 0 && chunk > static_cast<uint64_t>(n)) {
            chunk = static_cast<uint64_t>(n);
        }

        BallQueryTilingData *tiling = context->GetTilingData<BallQueryTilingData>();
        tiling->n = static_cast<uint64_t>(n);
        tiling->m = static_cast<uint64_t>(m);
        tiling->batch = static_cast<uint64_t>(batch);
        tiling->nsample = static_cast<uint64_t>(nsample);
        tiling->radius2 = radius * radius;
        tiling->dtypeMode = dtype_mode;
        tiling->chunk = static_cast<uint32_t>(chunk);

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        int32_t qPerCore = static_cast<int32_t>((m + cores - 1) / cores);
        if (qPerCore < 1) {
            qPerCore = 1;
        }
        if (qPerCore > BQ_MAX_Q_PER_CORE) {
            qPerCore = BQ_MAX_Q_PER_CORE;
        }
        int32_t needCores = static_cast<int32_t>((m + qPerCore - 1) / qPerCore);
        if (needCores < cores) {
            cores = needCores > 0 ? needCores : 1;
        }
        if (cores < 1) {
            cores = 1;
        }
        tiling->qPerCore = static_cast<uint32_t>(qPerCore);

        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *newShape = context->GetInputShape(1);
        gert::Shape *outShape = context->GetOutputShape(0);
        int64_t ns = 16;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs != nullptr) {
            const int64_t *p = attrs->GetInt(1);
            if (p != nullptr && *p > 0) {
                ns = *p;
            }
        }
        outShape->SetDimNum(3);
        outShape->SetDim(0, newShape->GetDim(0));
        outShape->SetDim(1, newShape->GetDim(1));
        outShape->SetDim(2, ns);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_INT32);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class BallQuery : public OpDef {
    public:
        explicit BallQuery(const char *name) : OpDef(name) {
            this->Input("xyz")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("new_xyz")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("radius").AttrType(OPTIONAL).Float(0.4);
            this->Attr("nsample").AttrType(OPTIONAL).Int(16);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(BallQuery);
}  // namespace ops

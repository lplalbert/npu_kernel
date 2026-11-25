// Host侧Tiling实现
// CylinderQuery: delta = xyz - new_xyz; [xl,yl,zl] = rot^T @ delta;
//   命中 = (yl^2+zl^2 < radius^2) && (hmin < xl < hmax), 按输入索引序取前 nsample 个;
//   不足尾部重复第一个命中, 0 命中全 0; f16 输入先转 f32 再算。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/cylinder_query_tiling.h"
#include "../op_kernel/tiling_key_cylinder_query.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;

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
        const float *attr_hmin = attrs->GetFloat(1);
        const float *attr_hmax = attrs->GetFloat(2);
        const int64_t *attr_nsample = attrs->GetInt(3);
        float radius = (attr_radius != nullptr) ? *attr_radius : 0.05f;
        float hmin = (attr_hmin != nullptr) ? *attr_hmin : -0.02f;
        float hmax = (attr_hmax != nullptr) ? *attr_hmax : 0.04f;
        int64_t nsample = (attr_nsample != nullptr) ? *attr_nsample : 32;
        if (nsample < 1) {
            nsample = 1;
        }
        if (nsample > CQ_MAX_NSAMPLE) {
            nsample = CQ_MAX_NSAMPLE;
        }
        if (nsample > n) {
            nsample = n;
        }

        uint32_t dtype_mode = (dtype_xyz == ge::DT_FLOAT16) ? 0u : 1u;

        // 每块 UB: raw(3*chunk*sizeof(T)) + f(12*chunk) + 8 个 f32 数组(32*chunk) + offset(12*chunk)
        // 固定部分: qRaw + qBuf (约 (b*m*12)*8) + oBuf + off 头
        const uint64_t fixedBytes = static_cast<uint64_t>(batch * m * 12) * 8u + 8192u;
        uint64_t budget = (ub_size * 3) / 4;
        if (budget <= fixedBytes) {
            budget = fixedBytes + 4096u;
        }
        const uint64_t perChunk = (dtype_mode == 0u) ? 62u : 68u;
        uint64_t chunk = (budget - fixedBytes) / perChunk;
        if (chunk > 1024u) {
            chunk = 1024u;
        }
        chunk = (chunk / ALIGN_ELEM) * ALIGN_ELEM;
        if (chunk < ALIGN_ELEM) {
            chunk = ALIGN_ELEM;
        }

        CylinderQueryTilingData *tiling = context->GetTilingData<CylinderQueryTilingData>();
        tiling->n = static_cast<uint64_t>(n);
        tiling->m = static_cast<uint64_t>(m);
        tiling->batch = static_cast<uint64_t>(batch);
        tiling->nsample = static_cast<uint64_t>(nsample);
        tiling->radius2 = radius * radius;
        tiling->hmin = hmin;
        tiling->hmax = hmax;
        tiling->rsvF = 0.0f;
        tiling->dtypeMode = dtype_mode;
        tiling->chunk = static_cast<uint32_t>(chunk);
        tiling->rsv = 0;

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        int32_t qPerCore = static_cast<int32_t>((m + cores - 1) / cores);
        if (qPerCore < 1) {
            qPerCore = 1;
        }
        if (qPerCore > CQ_MAX_Q_PER_CORE) {
            qPerCore = CQ_MAX_Q_PER_CORE;
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
        int64_t ns = 32;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs != nullptr) {
            const int64_t *p = attrs->GetInt(3);
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
    class CylinderQuery : public OpDef {
    public:
        explicit CylinderQuery(const char *name) : OpDef(name) {
            this->Input("xyz")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("new_xyz")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("rot")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("radius").AttrType(OPTIONAL).Float(0.05);
            this->Attr("hmin").AttrType(OPTIONAL).Float(-0.02);
            this->Attr("hmax").AttrType(OPTIONAL).Float(0.04);
            this->Attr("nsample").AttrType(OPTIONAL).Int(32);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(CylinderQuery);
}  // namespace ops

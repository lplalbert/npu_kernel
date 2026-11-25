// Host侧Tiling实现
// Radius: 对每个查询点 i, 在其所属 batch 的 x 里按 j 升序扫描,
//   d^2 <= r^2 且未被 ignore_same_index 剔除的 (i,j) 收集, 收满 max_num_neighbors 停止;
//   写入槽位 [i*K,(i+1)*K), 未命中槽位填 (-1,-1)。out 形状 (2, M*K), int64。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/radius_tiling.h"
#include "../op_kernel/tiling_key_radius.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape yShape = context->GetInputShape(1)->GetStorageShape();
        int64_t n = xShape.GetDim(0);
        int64_t dim = xShape.GetDim(1);
        int64_t m = yShape.GetDim(0);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_r = (attrs != nullptr) ? attrs->GetFloat(0) : nullptr;
        const int64_t *attr_k = (attrs != nullptr) ? attrs->GetInt(1) : nullptr;
        float radius = (attr_r != nullptr) ? *attr_r : 1.0f;
        int64_t maxNeighbors = (attr_k != nullptr) ? *attr_k : 32;
        // ignore_same_index 可能被按 bool 或 int 传入, 两种都取(类型不符时 GetAttrPointer 返回 nullptr)
        bool ignoreSame = false;
        if (attrs != nullptr) {
            const bool *b = attrs->GetBool(2);
            if (b != nullptr) {
                ignoreSame = *b;
            } else {
                const int64_t *iv = attrs->GetInt(2);
                if (iv != nullptr) {
                    ignoreSame = (*iv != 0);
                }
            }
        }
        if (maxNeighbors < 1) {
            maxNeighbors = 1;
        }

        uint32_t dtype_mode = (dtype_x == ge::DT_FLOAT16) ? 0u : 1u;

        // UB: raw chunk*D*e + f32 chunk*D*4 + out 2*K*8
        uint64_t elemBytes = static_cast<uint64_t>(dim) * (dtype_mode == 0 ? 2u : 4u) +
                             static_cast<uint64_t>(dim) * 4u;
        uint64_t budget = (ub_size * 3) / 4;
        uint64_t chunk = budget / (elemBytes + 4);
        if (chunk > 1024u) {
            chunk = 1024u;
        }
        chunk = (chunk / ALIGN_ELEM) * ALIGN_ELEM;
        if (chunk < ALIGN_ELEM) {
            chunk = ALIGN_ELEM;
        }

        RadiusTilingData *tiling = context->GetTilingData<RadiusTilingData>();

        int32_t cores = num_cores_aiv > 0 ? num_cores_aiv : 1;
        int64_t per = (m + cores - 1) / cores;
        if (per < 1) {
            per = 1;
        }
        int64_t needCores = (m + per - 1) / per;
        if (needCores < static_cast<int64_t>(cores)) {
            cores = needCores > 0 ? static_cast<int32_t>(needCores) : 1;
        }
        if (cores < 1) {
            cores = 1;
        }

        tiling->n = static_cast<uint64_t>(n);
        tiling->m = static_cast<uint64_t>(m);
        tiling->dim = static_cast<uint64_t>(dim);
        tiling->k = static_cast<uint64_t>(maxNeighbors);
        tiling->qPerCore = static_cast<uint64_t>(per);
        tiling->chunk = static_cast<uint32_t>(chunk);
        tiling->dtypeMode = dtype_mode;
        tiling->ignoreSame = ignoreSame ? 1u : 0u;
        tiling->rsv = 0;
        tiling->r2 = radius * radius;

        context->SetBlockDim(static_cast<uint32_t>(cores));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *yShape = context->GetInputShape(1);
        gert::Shape *outShape = context->GetOutputShape(0);
        const int64_t m = yShape->GetDim(0);
        int64_t kk = 32;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs != nullptr) {
            const int64_t *p = attrs->GetInt(1);
            if (p != nullptr && *p > 0) {
                kk = *p;
            }
        }
        outShape->SetDimNum(2);
        outShape->SetDim(0, 2);
        outShape->SetDim(1, m * kk);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_INT64);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Radius : public OpDef {
    public:
        explicit Radius(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("ptr_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("ptr_y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("r").AttrType(OPTIONAL).Float(1.0);
            this->Attr("max_num_neighbors").AttrType(OPTIONAL).Int(32);
            this->Attr("ignore_same_index").AttrType(OPTIONAL).Bool(false);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Radius);
}  // namespace ops

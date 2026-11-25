// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdlib>

#include "../op_kernel/spspmm_tiling.h"
#include "../op_kernel/tiling_key_spspmm.h"

namespace optiling {
    static constexpr uint64_t ALIGN_ELEM = 64;
    static constexpr uint64_t ACC_MAX = 8192;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_valueA = context->GetRequiredInputTensor(1);
        ge::DataType dtype_valueA = tensor_valueA->GetDataType();
        uint32_t DT_VALUEA = static_cast<uint32_t>(dtype_valueA);
        ASCENDC_TPL_SEL_PARAM(context, DT_VALUEA);

        const gert::Shape shA = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape shB = context->GetInputShape(2)->GetStorageShape();
        int64_t e1 = shA.GetDimNum() >= 2 ? shA.GetDim(1) : shA.GetDim(0);
        int64_t e2 = shB.GetDimNum() >= 2 ? shB.GetDim(1) : shB.GetDim(0);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_m = attrs->GetInt(0);
        const int64_t *attr_k = attrs->GetInt(1);
        const int64_t *attr_n = attrs->GetInt(2);
        int64_t m = attr_m ? *attr_m : 64;
        int64_t k = attr_k ? *attr_k : 64;
        int64_t n = attr_n ? *attr_n : 64;
        if (m < 0) m = 0;
        if (k < 0) k = 0;
        if (n < 0) n = 0;
        if (e1 < 0) e1 = 0;
        if (e2 < 0) e2 = 0;

        uint64_t U = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);

        // accN: n 方向一次驻留 UB 的列数
        uint64_t accN = n > 0 ? static_cast<uint64_t>(n) : 64;
        if (accN > ACC_MAX) accN = ACC_MAX;
        accN = ((accN + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
        if (accN < ALIGN_ELEM) accN = ALIGN_ELEM;
        uint64_t chN = n > 0 ? (static_cast<uint64_t>(n) + accN - 1) / accN : 0;

        // B 键二分迭代次数
        uint32_t bsIter = 1;
        uint64_t v = e2 > 0 ? static_cast<uint64_t>(e2) : 0;
        while (v > 0) { bsIter++; v >>= 1; }

        SpspmmTilingData *tiling = context->GetTilingData<SpspmmTilingData>();
        tiling->m = static_cast<uint32_t>(m);
        tiling->n = static_cast<uint32_t>(n);
        tiling->k = static_cast<uint32_t>(k);
        tiling->U = static_cast<uint32_t>(U > 0xFFFFFFFFull ? 0xFFFFFFFFull : U);
        tiling->e1 = static_cast<uint32_t>(e1 > 0xFFFFFFFFll ? 0xFFFFFFFFll : e1);
        tiling->e2 = static_cast<uint32_t>(e2 > 0xFFFFFFFFll ? 0xFFFFFFFFll : e2);
        tiling->coreNum = 1;
        tiling->accN = static_cast<uint32_t>(accN);
        tiling->bsIter = bsIter;
        tiling->chN = static_cast<uint32_t>(chN);

        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_m = attrs->GetInt(0);
        const int64_t *attr_n = attrs->GetInt(2);
        int64_t m = attr_m ? *attr_m : 64;
        int64_t n = attr_n ? *attr_n : 64;
        if (m < 0) m = 0;
        if (n < 0) n = 0;
        int64_t U = m * n;
        gert::Shape *outI = context->GetOutputShape(0);
        outI->SetDimNum(2);
        outI->SetDim(0, 2);
        outI->SetDim(1, U);
        gert::Shape *outV = context->GetOutputShape(1);
        outV->SetDimNum(1);
        outV->SetDim(0, U);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_INT64);
        context->SetOutputDataType(1, context->GetInputDataType(1));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Spspmm : public OpDef {
    public:
        explicit Spspmm(const char *name) : OpDef(name) {
            this->Input("indexA")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("valueA")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("indexB")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("valueB")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("indexC")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("valueC")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("m").AttrType(OPTIONAL).Int(64);
            this->Attr("k").AttrType(OPTIONAL).Int(64);
            this->Attr("n").AttrType(OPTIONAL).Int(64);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Spspmm);
}  // namespace ops

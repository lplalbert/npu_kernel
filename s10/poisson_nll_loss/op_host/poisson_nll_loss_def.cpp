/*!
 * \file poisson_nll_loss_def.cpp
 * \brief PoissonNllLoss 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class PoissonNllLoss : public OpDef {
public:
    explicit PoissonNllLoss(const char* name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("log_input")
            .AttrType(OPTIONAL)
            .Bool(1);
        this->Attr("full")
            .AttrType(OPTIONAL)
            .Bool(0);
        this->Attr("eps")
            .AttrType(OPTIONAL)
            .Float(1e-08);
        this->Attr("reduction")
            .AttrType(OPTIONAL)
            .String("mean");
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(PoissonNllLoss);
} // namespace ops

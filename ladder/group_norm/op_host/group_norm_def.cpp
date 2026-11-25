/*!
 * \file group_norm_def.cpp
 * \brief GroupNorm operator definition
 */
#include "register/op_def_registry.h"

namespace ops {
class GroupNorm : public OpDef {
public:
    explicit GroupNorm(const char* name) : OpDef(name)
    {
    // x / gamma / beta dtype combinations: (fp16, fp32, fp32), (bf16, fp32, fp32),
    // (fp32, fp32, fp32). gamma and beta are always fp32, matching the reference
    // signature; the output follows the dtype of x.
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("gamma")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("num_groups")
        .Int();
    this->Attr("epsilon")
        .AttrType(OPTIONAL)
        .Float(1e-05);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(GroupNorm);
} // namespace ops

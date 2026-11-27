/*!
 * \file quant_batch_mat_mul_v3_def.cpp
 * \brief QuantBatchMatMulV3 operator definition
 *
 * x1 / x2 are int8 and stored in ND as (m, k) and (k, n); scale is channel (n)
 * aligned and may also be a single broadcast value; pertokenScale is token (m)
 * aligned; out is bf16 of shape (m, n).  The operator has no transpose attribute.
 */
#include "register/op_def_registry.h"

namespace ops {
class QuantBatchMatMulV3 : public OpDef {
public:
    explicit QuantBatchMatMulV3(const char* name) : OpDef(name)
    {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("pertokenScale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(QuantBatchMatMulV3);
} // namespace ops

/*!
 * \file gf2k_mul_square_def.cpp
 * \brief Gf2kMulSquare op definition
 */
#include "register/op_def_registry.h"

namespace ops {
class Gf2kMulSquare : public OpDef {
public:
    explicit Gf2kMulSquare(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("k")
            .Int();
        this->Attr("operation")
            .AttrType(OPTIONAL)
            .String("mul");
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Gf2kMulSquare);
} // namespace ops

/*!
 * \file bit_plane_transform_def.cpp
 * \brief BitPlaneTransform 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class BitPlaneTransform : public OpDef {
public:
    explicit BitPlaneTransform(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("direction")
            .AttrType(OPTIONAL)
            .String("transpose");
        this->Attr("n")
            .AttrType(OPTIONAL)
            .Int(-1);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(BitPlaneTransform);
} // namespace ops


/*!
 * \file index_def.cpp
 * \brief Index
 *
 * x/y  FLOAT/FLOAT16/INT64/INT32/INT8/UINT8/BOOL kernel
 * indexed_sizes  INT64indices  INT64 / INT32
 *  DataType  x dtype  indices dtype
 */
#include "register/op_def_registry.h"

namespace ops {
class Index : public OpDef {
public:
    explicit Index(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT32,   ge::DT_INT32,
                   ge::DT_INT8,   ge::DT_INT8,   ge::DT_UINT8,   ge::DT_UINT8,
                   ge::DT_BOOL,   ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("indexed_sizes")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                   ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                   ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                   ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64,  ge::DT_INT32,  ge::DT_INT64,  ge::DT_INT32,
                   ge::DT_INT64,  ge::DT_INT32,  ge::DT_INT64,  ge::DT_INT32,
                   ge::DT_INT64,  ge::DT_INT32,  ge::DT_INT64,  ge::DT_INT32,
                   ge::DT_INT64,  ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT32,   ge::DT_INT32,
                   ge::DT_INT8,   ge::DT_INT8,   ge::DT_UINT8,   ge::DT_UINT8,
                   ge::DT_BOOL,   ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Index);
} // namespace ops

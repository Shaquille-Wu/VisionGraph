#include "scalar_compare.h"

namespace vision_graph{

const std::map<std::string, SCALAR_COMPARE_TYPE>  kCompTypeMap = {
    { ">=",  SCALAR_COMPARE_TYPE::GTE },
    { ">",   SCALAR_COMPARE_TYPE::GT },
    { "<=",  SCALAR_COMPARE_TYPE::LTE },
    { "<",   SCALAR_COMPARE_TYPE::LT },
    { "==",  SCALAR_COMPARE_TYPE::EQ },
    { "!=",  SCALAR_COMPARE_TYPE::NEQ }
};

void ReadScalarStringValue(nlohmann::json const& x_value_json, std::string const& key_name, Tensor*& const_value)  noexcept
{
    nlohmann::json    value_json = x_value_json.at(key_name);
    if(value_json.is_number_integer())
    {
        TensorInt32*  int32_value   = new TensorInt32;
        int32_value->value_ = value_json.get<int>();
        const_value         = int32_value;
    }
    else if(value_json.is_number_float())
    {
        TensorFloat32*  float_value = new TensorFloat32;
        float_value->value_ = value_json.get<float>();
        const_value         = float_value;
    }
    else if(value_json.is_string())
    {
        TensorString*  string_value  = new TensorString;
        *string_value = value_json.get<std::string>();
        const_value   = string_value;
    }   
}

void GenerateConstValue(nlohmann::json const& x_value_json, Tensor*& const_value, bool& const_flag) noexcept
{
    std::string      source_type = x_value_json.at(kCompareNodeRightSourceType);
    const_flag = false;
    if(kCompareNodeSourceTypeConst == source_type)
    {
        ReadScalarStringValue(x_value_json, kCompareNodeSourceValue, const_value);
        const_flag    = true;
    }
};

}
#ifndef SCALAR_COMPARE_H_
#define SCALAR_COMPARE_H_

#include "../../../../include/graph_tensor.h"
#include "../../../../include/json.hpp"
#include <map>

namespace vision_graph{

static const std::string kCompareNodeCompareType      = std::string("compare_type");
static const std::string kCompareNodeRightValue       = std::string("right_value");
static const std::string kCompareNodeRightSourceType  = std::string("source");
static const std::string kCompareNodeSourceTypeVar    = std::string("var");
static const std::string kCompareNodeSourceTypeConst  = std::string("constant");
static const std::string kCompareNodeSourceValue      = std::string("value");

typedef enum tag_scalar_compare_type
{
    GTE = 0,
    GT,
    LTE,
    LT,
    EQ,
    NEQ,
    COMPARE_TYPE_SUM
}SCALAR_COMPARE_TYPE;

extern const std::map<std::string, SCALAR_COMPARE_TYPE>  kCompTypeMap;

void ReadScalarStringValue(nlohmann::json const& x_value_json, std::string const& key_name, Tensor*& const_value)  noexcept ;

void GenerateConstValue(nlohmann::json const& x_value_json, Tensor*& const_value, bool& const_flag) noexcept ;

template<typename NumericType>
inline bool                  compare_gte(Tensor const& left, Tensor const& right)
{
    return (static_cast<TensorNumeric<NumericType> const&>(left)).value_ >= (static_cast<TensorNumeric<NumericType> const&>(right)).value_;
}

template<typename NumericType>
inline bool                  compare_gt(Tensor const& left, Tensor const& right)
{
    return (static_cast<TensorNumeric<NumericType> const&>(left)).value_ > (static_cast<TensorNumeric<NumericType> const&>(right)).value_;
}

template<typename NumericType>
inline bool                  compare_lte(Tensor const& left, Tensor const& right)
{
    return !(compare_gt<NumericType>(left, right));
}

template<typename NumericType>
inline bool                  compare_lt(Tensor const& left, Tensor const& right)
{
    return !(compare_gte<NumericType>(left, right));
}

template<typename NumericType, typename std::enable_if<std::is_integral<NumericType>::value, void>::type* = nullptr>
inline bool                  compare_eq(Tensor const& left, Tensor const& right)
{
    return (static_cast<TensorNumeric<NumericType> const&>(left)).value_ == (static_cast<TensorNumeric<NumericType> const&>(right)).value_;
}
template<typename NumericType, typename std::enable_if<std::is_floating_point<NumericType>::value && sizeof(NumericType) == 4, void>::type* = nullptr>
inline bool                  compare_eq(Tensor const& left, Tensor const& right)
{
    float  diff = (static_cast<TensorNumeric<NumericType> const&>(left)).value_ - (static_cast<TensorNumeric<NumericType> const&>(right)).value_;
    diff = fabsf(diff);
    if(diff < 1e-6)
        return true;
    return false;
}
template<typename NumericType, typename std::enable_if<std::is_floating_point<NumericType>::value && sizeof(NumericType) == 8, void>::type* = nullptr>
inline bool                  compare_eq(Tensor const& left, Tensor const& right)
{
    double  diff = (static_cast<TensorNumeric<NumericType> const&>(left)).value_ - (static_cast<TensorNumeric<NumericType> const&>(right)).value_;
    diff = fabs(diff);
    if(diff < 1e-15)
        return true;
    return false;
}
template<typename NumericType>
inline bool                  compare_neq(Tensor const& left, Tensor const& right)
{
    return !(compare_eq<NumericType>(left, right));
}

template<typename NumericType>
inline bool                  compare_value_gte(NumericType left, NumericType right)
{
    return left >= right;
}

template<typename NumericType>
inline bool                  compare_value_gt(NumericType left, NumericType right)
{
    return left > right;
}

template<typename NumericType>
inline bool                  compare_value_lte(NumericType left, NumericType right)
{
    return !(compare_value_gt<NumericType>(left, right));
}

template<typename NumericType>
inline bool                  compare_value_lt(NumericType left, NumericType right)
{
    return !(compare_value_gte<NumericType>(left, right));
}

template<typename NumericType, typename std::enable_if<std::is_integral<NumericType>::value, void>::type* = nullptr>
inline bool                  compare_value_eq(NumericType left, NumericType right)
{
    return left == right;
}
template<typename NumericType, typename std::enable_if<std::is_floating_point<NumericType>::value && sizeof(NumericType) == 4, void>::type* = nullptr>
inline bool                  compare_value_eq(NumericType left, NumericType right)
{
    float  diff = fabsf(left - right);
    if(diff < 1e-6)
        return true;
    return false;
}
template<typename NumericType, typename std::enable_if<std::is_floating_point<NumericType>::value && sizeof(NumericType) == 8, void>::type* = nullptr>
inline bool                  compare_value_eq(NumericType left, NumericType right)
{
    double  diff = fabs(left - right);
    if(diff < 1e-15)
        return true;
    return false;
}
template<typename NumericType>
inline bool                  compare_value_neq(NumericType left, NumericType right)
{
    return !(compare_value_eq<NumericType>(left, right));
}

template<typename StrongType, TENSOR_TYPE TensorType>
inline bool                  compare_heterogeneous_numeric(Tensor const& left, Tensor const& right, SCALAR_COMPARE_TYPE compare_type)
{
    int   strong_type = 0;
    StrongType  numeric_value_left  = 0;
    StrongType  numeric_value_right = 0;
    if(left.GetType() == TensorType)
    {
        numeric_value_left  = (static_cast<TensorNumeric<StrongType> const&>(left)).value_;
        numeric_value_right = 0;
        switch (right.GetType())
        {
            case kTensorUInt8:
                numeric_value_right =  (StrongType)((static_cast<TensorUInt8 const&>(right)).value_);
                break;
            case kTensorInt8:
                numeric_value_right =  (StrongType)((static_cast<TensorInt8 const&>(right)).value_);
                break;
            case kTensorUInt16:
                numeric_value_right =  (StrongType)((static_cast<TensorUInt16 const&>(right)).value_);
                break;
            case kTensorInt16:
                numeric_value_right =  (StrongType)((static_cast<TensorInt16 const&>(right)).value_);
                break;
            case kTensorUInt32:
                numeric_value_right =  (StrongType)((static_cast<TensorUInt32 const&>(right)).value_);
                break;
            case kTensorInt32:
                numeric_value_right =  (StrongType)((static_cast<TensorInt32 const&>(right)).value_);
                break;
            case kTensorUInt64:
                numeric_value_right =  (StrongType)((static_cast<TensorUInt64 const&>(right)).value_);
                break;
            case kTensorInt64:
                numeric_value_right =  (StrongType)((static_cast<TensorInt64 const&>(right)).value_);
                break;
            case kTensorFloat32:
                numeric_value_right =  (StrongType)((static_cast<TensorFloat32 const&>(right)).value_);
                break;
            case kTensorFloat64:
                numeric_value_right =  (StrongType)((static_cast<TensorFloat64 const&>(right)).value_);
                break;
            default:
                break;
        }
    }
    else
    {
        numeric_value_left  = 0;
        numeric_value_right = (static_cast<TensorNumeric<StrongType> const&>(right)).value_;
        switch (left.GetType())
        {
            case kTensorUInt8:
                numeric_value_left =  (StrongType)((static_cast<TensorUInt8 const&>(left)).value_);
                break;
            case kTensorInt8:
                numeric_value_left =  (StrongType)((static_cast<TensorInt8 const&>(left)).value_);
                break;
            case kTensorUInt16:
                numeric_value_left =  (StrongType)((static_cast<TensorUInt16 const&>(left)).value_);
                break;
            case kTensorInt16:
                numeric_value_left =  (StrongType)((static_cast<TensorInt16 const&>(left)).value_);
                break;
            case kTensorUInt32:
                numeric_value_left =  (StrongType)((static_cast<TensorUInt32 const&>(left)).value_);
                break;
            case kTensorInt32:
                numeric_value_left =  (StrongType)((static_cast<TensorInt32 const&>(left)).value_);
                break;
            case kTensorUInt64:
                numeric_value_left =  (StrongType)((static_cast<TensorUInt64 const&>(left)).value_);
                break;
            case kTensorInt64:
                numeric_value_left =  (StrongType)((static_cast<TensorInt64 const&>(left)).value_);
                break;
            case kTensorFloat32:
                numeric_value_left =  (StrongType)((static_cast<TensorFloat32 const&>(left)).value_);
                break;
            case kTensorFloat64:
                numeric_value_left =  (StrongType)((static_cast<TensorFloat64 const&>(left)).value_);
                break;
            default:
                break;
        }
    }
    
    bool res = false;
    switch(compare_type)
    {
        case SCALAR_COMPARE_TYPE::GTE:
            res = compare_value_gte<StrongType>(numeric_value_left, numeric_value_right);
            break;
        case SCALAR_COMPARE_TYPE::GT:
            res = compare_value_gt<StrongType>(numeric_value_left, numeric_value_right);
            break;
        case SCALAR_COMPARE_TYPE::LTE:
            res = compare_value_lte<StrongType>(numeric_value_left, numeric_value_right);
            break;
        case SCALAR_COMPARE_TYPE::LT:
            res = compare_value_lt<StrongType>(numeric_value_left, numeric_value_right);
            break;
        case SCALAR_COMPARE_TYPE::EQ:
            res = compare_value_eq<StrongType>(numeric_value_left, numeric_value_right);
            break; 
        case SCALAR_COMPARE_TYPE::NEQ:
            res = compare_value_neq<StrongType>(numeric_value_left, numeric_value_right);
            break;
        default:
            break;
    }

    return res;
};

template<typename ValueType, typename std::enable_if<std::is_arithmetic<ValueType>::value, void>::type* = nullptr>
inline bool                  compare_value(Tensor const& left, Tensor const& right, SCALAR_COMPARE_TYPE compare_type) noexcept
{
    bool res = false;
    switch(compare_type)
    {
        case SCALAR_COMPARE_TYPE::GTE:
            res = compare_gte<ValueType>(left, right);
            break;
        case SCALAR_COMPARE_TYPE::GT:
            res = compare_gt<ValueType>(left, right);
            break;
        case SCALAR_COMPARE_TYPE::LTE:
            res = compare_lte<ValueType>(left, right);
            break;
        case SCALAR_COMPARE_TYPE::LT:
            res = compare_lt<ValueType>(left, right);
            break;
        case SCALAR_COMPARE_TYPE::EQ:
            res = compare_eq<ValueType>(left, right);
            break; 
        case SCALAR_COMPARE_TYPE::NEQ:
            res = compare_neq<ValueType>(left, right);
            break;
        default:
            break;
    }

    return res;
}

template<typename ValueType, typename std::enable_if<!std::is_arithmetic<ValueType>::value, void>::type* = nullptr>
inline bool                  compare_value(Tensor const& left, Tensor const& right, SCALAR_COMPARE_TYPE compare_type) noexcept
{
    bool res = false;
    switch(compare_type)
    {
        case SCALAR_COMPARE_TYPE::GTE:
            res = (static_cast<TensorString const&>(left)) >= (static_cast<TensorString const&>(right));
            break;
        case SCALAR_COMPARE_TYPE::GT:
            res = (static_cast<TensorString const&>(left)) > (static_cast<TensorString const&>(right));
            break;
        case SCALAR_COMPARE_TYPE::LTE:
            res = (static_cast<TensorString const&>(left)) <= (static_cast<TensorString const&>(right));
            break;
        case SCALAR_COMPARE_TYPE::LT:
            res = (static_cast<TensorString const&>(left)) < (static_cast<TensorString const&>(right));
            break;
        case SCALAR_COMPARE_TYPE::EQ:
            res = (static_cast<TensorString const&>(left)) == (static_cast<TensorString const&>(right));
            break; 
        case SCALAR_COMPARE_TYPE::NEQ:
            res = (static_cast<TensorString const&>(left)) != (static_cast<TensorString const&>(right));
            break;
        default:
            break;
    }

    return res;
}

inline bool compare_scalar_tensor(Tensor const& left_value, Tensor const& right_value, SCALAR_COMPARE_TYPE compare_type) noexcept
{
    bool res = false;
    if(left_value.GetType() == right_value.GetType())
    {
        switch(left_value.GetType())
        {
            case kTensorUInt8:
                res = compare_value<unsigned char>(left_value, right_value, compare_type);
                break;
            case kTensorInt8:
                res = compare_value<char>(left_value, right_value, compare_type);
                break;
            case kTensorUInt16:
                res = compare_value<unsigned short int>(left_value, right_value, compare_type);
                break;
            case kTensorInt16:
                res = compare_value<short int>(left_value, right_value, compare_type);
                break;
            case kTensorUInt32:
                res = compare_value<unsigned int>(left_value, right_value, compare_type);
                break;
            case kTensorInt32:
                res = compare_value<int>(left_value, right_value, compare_type);
                break;
            case kTensorUInt64:
                res = compare_value<unsigned long long int>(left_value, right_value, compare_type);
                break;
            case kTensorInt64:
                res = compare_value<long long int>(left_value, right_value, compare_type);
                break;
            case kTensorFloat32:
                res = compare_value<float>(left_value, right_value, compare_type);
                break;
            case kTensorFloat64:
                res = compare_value<double>(left_value, right_value, compare_type);
                break;
            case kTensorString:
                res = compare_value<std::string>(left_value, right_value, compare_type);
                break;
            default:
                break;
        }
        
    }
    else
    {
        int   strong_type = 0;
        if(left_value.GetType() > right_value.GetType())
            strong_type = left_value.GetType();
        else
            strong_type = right_value.GetType();
        switch(strong_type)
        {
            case kTensorUInt8:
                res = compare_heterogeneous_numeric<unsigned char, kTensorUInt8>(left_value, right_value, compare_type);
                break;
            case kTensorInt8:
                res = compare_heterogeneous_numeric<char, kTensorInt8>(left_value, right_value, compare_type);
                break;
            case kTensorUInt16:
                res = compare_heterogeneous_numeric<unsigned short int, kTensorUInt16>(left_value, right_value, compare_type);
                break;
            case kTensorInt16:
                res = compare_heterogeneous_numeric<short int, kTensorInt16>(left_value, right_value, compare_type);
                break;
            case kTensorUInt32:
                res = compare_heterogeneous_numeric<unsigned int, kTensorUInt32>(left_value, right_value, compare_type);
                break;
            case kTensorInt32:
                res = compare_heterogeneous_numeric<int, kTensorInt32>(left_value, right_value, compare_type);
                break;
            case kTensorUInt64:
                res = compare_heterogeneous_numeric<long long int, kTensorUInt64>(left_value, right_value, compare_type);
                break;
            case kTensorInt64:
                res = compare_heterogeneous_numeric<unsigned long long int, kTensorInt64>(left_value, right_value, compare_type);
                break;
            case kTensorFloat32:
                res = compare_heterogeneous_numeric<float, kTensorFloat32>(left_value, right_value, compare_type);
                break;
            case kTensorFloat64:
                res = compare_heterogeneous_numeric<double, kTensorFloat64>(left_value, right_value, compare_type);
                break;
            default:
                break;
        }
    }

    return res;
}

}//namespace vision_graph

#endif
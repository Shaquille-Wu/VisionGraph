#include "../../include/graph_error.h"
#include "solver_logic.h"
#include <logging.h>
#include <math.h>

namespace vision_graph{

static const std::string kLogicNodeLogicType      = std::string("logic_type");
static const std::string kLogicNodeBitwise        = std::string("bitwise");

const std::map<std::string, SolverLogic::LOGIC_OP_TYPE>  kLogicTypeMap = {
    { "and",  SolverLogic::AND },
    { "or",   SolverLogic::OR  },
    { "xor",  SolverLogic::XOR },
    { "not",  SolverLogic::NOT }
};

SolverLogic::SolverLogic(nlohmann::json const& param) noexcept:Solver(param), bitwise_(false)
{
    std::string   logic_type = param.at(kLogicNodeLogicType).get<std::string>();
    std::map<std::string, LOGIC_OP_TYPE>::const_iterator  iter = kLogicTypeMap.find(logic_type);
    if(kLogicTypeMap.end() != iter)
        logic_type_ = iter->second;
    else
        logic_type_ = SolverLogic::AND;

    if(param.contains(kLogicNodeBitwise))
        bitwise_ = param.at(kLogicNodeBitwise);
}

SolverLogic::~SolverLogic() noexcept
{

}

Tensor* SolverLogic::CreateOutTensor(int out_tensor_idx) noexcept
{
    Tensor*  logic_result = new TensorNumeric<LOGIC_RESULT_TYPE>(0U);

    return logic_result;
}

bool SolverLogic::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(logic_type_ != SolverLogic::NOT)
    {
        if(in.size() != 2)
            return false;
        if(in[0]->GetType() < kTensorUInt8 || in[0]->GetType() >= kTensorString ||
           in[1]->GetType() < kTensorUInt8 || in[1]->GetType() >= kTensorString)
            return false;
    }
    else
    {
        if(in.size() != 1)
            return false;
        if(in[0]->GetType() < kTensorUInt8 || in[0]->GetType() >= kTensorString)
            return false;
    }
    
    if(out.size() != 1 || out[0]->GetType() != kTensorUInt32)
        return false;

    return true;
}

int SolverLogic::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    LOGIC_RESULT_TYPE    res          = 0;
    if(logic_type_ != SolverLogic::NOT)
    {
        if(false == bitwise_)
            res = logic_op2(*(in[0]), *(in[1]), logic_type_);
        else
            res = logic_bitwise_op2(*(in[0]), *(in[1]), logic_type_);
    }
    else
    {
        if(false == bitwise_)
            res = logic_op1(*(in[0]), logic_type_);
        else
            res = logic_bitwise_op1(*(in[0]), logic_type_);
    }
    
    (dynamic_cast<TensorNumeric<LOGIC_RESULT_TYPE>* >(out[0]))->value_ = res;

    return 0;
}

bool SolverLogic::get_numeric_logic_value(Tensor const& numeric) noexcept
{
    bool res = false;
    switch (numeric.GetType())
    {
    case kTensorUInt8:
        res = ((dynamic_cast<TensorUInt8 const*>(&numeric))->value_ != 0);
        break;
    case kTensorInt8:
        res = ((dynamic_cast<TensorInt8 const*>(&numeric))->value_ != 0);
        break;
    case kTensorUInt16:
        res = ((dynamic_cast<TensorUInt16 const*>(&numeric))->value_ != 0);
        break;
    case kTensorInt16:
        res = ((dynamic_cast<TensorInt16 const*>(&numeric))->value_ != 0);
        break;
    case kTensorUInt32:
        res = ((dynamic_cast<TensorUInt32 const*>(&numeric))->value_ != 0);
        break;
    case kTensorInt32:
        res = ((dynamic_cast<TensorInt32 const*>(&numeric))->value_ != 0);
        break;
    case kTensorUInt64:
        res = ((dynamic_cast<TensorUInt64 const*>(&numeric))->value_ != 0);
        break;
    case kTensorInt64:
        res = ((dynamic_cast<TensorInt64 const*>(&numeric))->value_ != 0);
        break;
    case kTensorFloat32:
        res = (fabsf((dynamic_cast<TensorFloat32 const*>(&numeric))->value_) >= 1e-6);
        break;
    case kTensorFloat64:
        res = (fabs((dynamic_cast<TensorFloat64 const*>(&numeric))->value_) >= 1e-15);
        break;
    default:
        break;
    }

    return res;
}

SolverLogic::LOGIC_RESULT_TYPE SolverLogic::get_numeric_bitwise_value(Tensor const& numeric) noexcept
{
    LOGIC_RESULT_TYPE res = 0;
    switch (numeric.GetType())
    {
    case kTensorUInt8:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorUInt8 const*>(&numeric))->value_);
        break;
    case kTensorInt8:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorInt8 const*>(&numeric))->value_);
        break;
    case kTensorUInt16:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorUInt16 const*>(&numeric))->value_);
        break;
    case kTensorInt16:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorInt16 const*>(&numeric))->value_);
        break;
    case kTensorUInt32:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorUInt32 const*>(&numeric))->value_);
        break;
    case kTensorInt32:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorInt32 const*>(&numeric))->value_);
        break;
    case kTensorUInt64:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorUInt64 const*>(&numeric))->value_);
        break;
    case kTensorInt64:
        res = (LOGIC_RESULT_TYPE)((dynamic_cast<TensorInt64 const*>(&numeric))->value_);
        break;
    case kTensorFloat32:
        res = float2Int<LOGIC_RESULT_TYPE>(numeric);
        break;
    case kTensorFloat64:
        res = double2Int<LOGIC_RESULT_TYPE>(numeric);
        break;
    default:
        break;
    }

    return res;
}

bool SolverLogic::logic_op1(Tensor const& oprand, LOGIC_OP_TYPE op_type) noexcept
{
    bool logic_oprand  = get_numeric_logic_value(oprand);
    switch(op_type)
    {
        case SolverLogic::NOT:
            logic_oprand = (!logic_oprand);
            break;
        default:
            break;
    }

    return logic_oprand;
}

bool SolverLogic::logic_op2(Tensor const& left, Tensor const& right, LOGIC_OP_TYPE op_type) noexcept
{
    bool res_left  = get_numeric_logic_value(left);
    bool res_right = get_numeric_logic_value(right);
    bool res       = false;
    switch(op_type)
    {
        case SolverLogic::AND:
            res = (res_left & res_right);
            break;
        case SolverLogic::OR:
            res = (res_left | res_right);
            break;
        case SolverLogic::XOR:
            res = (res_left ^ res_right);
            break;
        default:
            break;
    }

    return res;
}

SolverLogic::LOGIC_RESULT_TYPE SolverLogic::logic_bitwise_op1(Tensor const& oprand, LOGIC_OP_TYPE op_type) noexcept
{
    LOGIC_RESULT_TYPE value  = get_numeric_bitwise_value(oprand);
    switch(op_type)
    {
        case SolverLogic::NOT:
            value = (~value);
            break;
        default:
            break;
    }

    return value;
}

SolverLogic::LOGIC_RESULT_TYPE SolverLogic::logic_bitwise_op2(Tensor const& left, Tensor const& right, LOGIC_OP_TYPE op_type) noexcept
{
    LOGIC_RESULT_TYPE res_left  = get_numeric_bitwise_value(left);
    LOGIC_RESULT_TYPE res_right = get_numeric_bitwise_value(right);
    LOGIC_RESULT_TYPE res       = false;
    switch(op_type)
    {
        case SolverLogic::AND:
            res = (res_left & res_right);
            break;
        case SolverLogic::OR:
            res = (res_left | res_right);
            break;
        case SolverLogic::XOR:
            res = (res_left ^ res_right);
            break;
        default:
            break;
    }

    return res;
}

};//class SolverLogic
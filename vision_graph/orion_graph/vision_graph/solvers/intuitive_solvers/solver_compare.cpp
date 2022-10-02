#include "../../include/graph_error.h"
#include "solver_compare.h"
#include <logging.h>

namespace vision_graph{

SolverCompare::SolverCompare(nlohmann::json const& param) noexcept:Solver(param)
{
    std::string   compare_type = param.at(kCompareNodeCompareType).get<std::string>();
    std::map<std::string, SCALAR_COMPARE_TYPE>::const_iterator  iter = kCompTypeMap.find(compare_type);
    if(kCompTypeMap.end() != iter)
        compare_type_ = iter->second;
    else
        compare_type_ = SCALAR_COMPARE_TYPE::GTE;

    const_val_    = nullptr;
    is_constant_  = false;
    nlohmann::json   const_json  = param.at(kCompareNodeRightValue);
    GenerateConstValue(const_json,  const_val_, is_constant_);
}

SolverCompare::~SolverCompare() noexcept
{
    if(nullptr != const_val_)
        delete const_val_;
    const_val_ = nullptr;
}

Tensor* SolverCompare::CreateOutTensor(int out_tensor_idx) noexcept
{
    Tensor*   out_tensor = new TensorUInt32(0);
    return out_tensor;
}

bool SolverCompare::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() == 0 || in.size() > 2 || out.size() != 1)
        return false;
    
    if(in[0]->GetType() < kTensorUInt8 || in[0]->GetType() > kTensorString)
        return false;

    if(true == is_constant_)
    {
        if(const_val_->GetType() != in[0]->GetType())
        {
            if((const_val_->GetType() == kTensorString && in[0]->GetType() < kTensorString) ||
               (const_val_->GetType() < kTensorString && in[0]->GetType() == kTensorString))
                return false;
        }
    }
    else
    {
        if(in.size()!= 2)
            return false;
        
        if(in[0]->GetType() != in[1]->GetType())
        {
            if((in[0]->GetType() == kTensorString && in[1]->GetType() < kTensorString) ||
               (in[0]->GetType() < kTensorString && in[1]->GetType() == kTensorString))
                return false;
        }
    }

    if(kTensorUInt32 != out[0]->GetType())
        return false;

    return true;
}

int SolverCompare::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    int             forward_mask = 0;
    const Tensor*   left_value   = in[0];
    Tensor*         right_value  = 0;
    if(true == is_constant_ && nullptr != const_val_)
        right_value = const_val_;
    else
        right_value = in[1];

    bool res = compare_scalar_tensor(*left_value, *right_value, compare_type_);
    (dynamic_cast<TensorUInt32* >(out[0]))->value_ = (int)(res);

    return 0;
}

};//class SolverLogic
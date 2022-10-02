#include "../../include/graph_error.h"
#include "solver_branch.h"
#include <logging.h>

namespace vision_graph{

static const std::string kBranchNodeForwardType      = std::string("forward_type");
static const std::string kBranchNodeCompareValue     = std::string("value");
static const TENSOR_TYPE kOutTensorSupportType       = kTensorUInt32;

const std::map<std::string, SolverBranch::FORWARD_TYPE>  kForwardTypeMap = {
    { "var_num",    SolverBranch::VAR_NUM },
    { "var_mask",   SolverBranch::VAR_MASK },
    { "fixed_num",  SolverBranch::FIXED_NUM },
    { "fixed_mask", SolverBranch::FIXED_MASK },
    { "compare",    SolverBranch::COMPARE }
};


SolverBranch::SolverBranch(nlohmann::json const& param) noexcept : Solver(param), 
                                                                   forward_type_(VAR_NUM), 
                                                                   compare_val_(nullptr)
{
    if(param.contains(kBranchNodeForwardType))
    {
        std::string   forward_type  = param.at(kBranchNodeForwardType).get<std::string>();
        std::map<std::string, SolverBranch::FORWARD_TYPE>::const_iterator  iter = kForwardTypeMap.find(forward_type);
        if(iter != kForwardTypeMap.end())
            forward_type_ = (iter->second);
    }

    if(SolverBranch::FIXED_NUM == forward_type_ ||
       SolverBranch::FIXED_MASK == forward_type_)
    {
        ReadScalarStringValue(param, kBranchNodeCompareValue, compare_val_);
        if(kTensorInt32 != compare_val_->GetType())
        {
            //LOG(ERROR) << "SolverBranch, cannot accept none-integer if its forward_type is fixed type";
            ABORT();
        }
    }
    else if(SolverBranch::COMPARE == forward_type_)
    {
        std::string   compare_type = param.at(kCompareNodeCompareType).get<std::string>();
        std::map<std::string, SCALAR_COMPARE_TYPE>::const_iterator  iter = kCompTypeMap.find(compare_type);
        if(kCompTypeMap.end() != iter)
            compare_type_ = iter->second;
        else
            compare_type_ = SCALAR_COMPARE_TYPE::GTE;
        ReadScalarStringValue(param, kBranchNodeCompareValue, compare_val_);
    }
};

SolverBranch::~SolverBranch() noexcept 
{
    if(nullptr != compare_val_)
        delete compare_val_;
    compare_val_ = nullptr;
};

Tensor* SolverBranch::CreateOutTensor(int out_tensor_idx) noexcept
{
    Tensor*   out_tensor = new TensorUInt32(0);
    return out_tensor;
}

bool SolverBranch::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() < 2)
        return false;

    for(int i = 0 ; i < (int)(out.size()) ; i ++)
    {
        if(out[i]->GetType() != kOutTensorSupportType)
            return false;
    }

    if(COMPARE == forward_type_)
    {
        if(in[0]->GetType() != compare_val_->GetType())
        {
            if((compare_val_->GetType() == kTensorString && in[0]->GetType() < kTensorString) ||
               (compare_val_->GetType() < kTensorString && in[0]->GetType() == kTensorString))
                return false;
        }
    }
    else
    {
        if(in[0]->GetType() < kTensorUInt8 || in[0]->GetType() >= kTensorString)
            return false;
    }

    return true;
}

int SolverBranch::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    int           i            = 0;
    int           out_size     = (int)(out.size());
    for(i = 0 ; i < out_size ; i ++)
        *(dynamic_cast<TensorUInt32*>(out[i])) = 0;

    if(VAR_NUM == forward_type_ || FIXED_NUM == forward_type_)
    {
        unsigned int num = 0;
        if(VAR_NUM == forward_type_)
            num = GetTensorNumericValue<unsigned int>(*(in[0]));
        else
            num = GetTensorNumericValue<unsigned int>(*compare_val_);
        if(num >= (unsigned int)(out_size))
        {
            LOG(ERROR) << "CheckTensor failed";
            ABORT();
            return vision_graph::kErrCodeParamInvalid;
        }
        (dynamic_cast<TensorUInt32*>(out[num]))->value_ = 1;
    }
    else if(VAR_MASK == forward_type_ || FIXED_MASK == forward_type_)
    {
        unsigned int num = 0;
        if(VAR_NUM == forward_type_)
            num = GetTensorNumericValue<unsigned int>(*(in[0]));
        else
            num = GetTensorNumericValue<unsigned int>(*compare_val_);
        for(i = 0 ; i < out_size ; i ++)
        {
            if(0 != (num & (1 << i)))
                (dynamic_cast<TensorUInt32*>(out[i]))->value_ = 1;
        }
    }
    else
    {
        bool      res           = compare_scalar_tensor(*(in[0]), *compare_val_, compare_type_);
        Tensor*   tensor_result = nullptr;
        if(true == res)
            (dynamic_cast<TensorUInt32*>(out[0]))->value_ = 1;
        else
            (dynamic_cast<TensorUInt32*>(out[1]))->value_ = 1;
    }

    return 0;
}

}//namespace vision_graph
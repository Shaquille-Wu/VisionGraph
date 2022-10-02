#include "../../include/graph_error.h"
#include "solver_selector.h"
#include <logging.h>


namespace vision_graph{

static const std::string kSelhNodeSelectType        = std::string("select_type");
static const std::string kSelNodeSelectCompareType  = std::string("compare_type");
static const std::string kSelNodeSelectCompareValue = std::string("value");

static const int         kCompareTensorIdx          = 0;

const std::map<std::string, SolverSelector::SELECT_TYPE>  kSelectTypeMap = {
    { "var_num",      SolverSelector::VAR_NUM },
    { "fixed_num",    SolverSelector::FIXED_NUM },
    { "compare",      SolverSelector::COMPARE },
    { "pass_through", SolverSelector::PASS_THROUGH },
};


SolverSelector::SolverSelector(nlohmann::json const& param) noexcept : Solver(param),
                                                                       select_type_(VAR_NUM), 
                                                                       select_idx_(0),
                                                                       compare_type_(SCALAR_COMPARE_TYPE::GTE), 
                                                                       compare_val_(nullptr)
{
    if(param.contains(kSelhNodeSelectType))
    {
        std::string   select_type  = param.at(kSelhNodeSelectType).get<std::string>();
        std::map<std::string, SolverSelector::SELECT_TYPE>::const_iterator  iter = kSelectTypeMap.find(select_type);
        if(iter != kSelectTypeMap.end())
            select_type_ = (iter->second);
    }

    if(FIXED_NUM == select_type_)
    {
        select_idx_  = param.at(kSelNodeSelectCompareValue);
    }
    else if(PASS_THROUGH == select_type_)
    {
        ;
    }
    else
    {
        std::string   compare_type = param.at(kSelNodeSelectCompareType).get<std::string>();
        std::map<std::string, SCALAR_COMPARE_TYPE>::const_iterator  iter = kCompTypeMap.find(compare_type);
        if(kCompTypeMap.end() != iter)
            compare_type_ = iter->second;
        else
            compare_type_ = SCALAR_COMPARE_TYPE::GTE;
        ReadScalarStringValue(param, kSelNodeSelectCompareValue, compare_val_);
    }
}

SolverSelector::~SolverSelector() noexcept
{
    if(nullptr != compare_val_)
        delete compare_val_;
    compare_val_ = nullptr;
}

Tensor* SolverSelector::CreateOutTensor(int out_tensor_idx) noexcept
{
    Tensor*  ref = new TensorReference;
    return ref;
}

bool SolverSelector::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(VAR_NUM == select_type_)
    {
        if(in.size() < 2 || (in[kCompareTensorIdx]->GetType() < kTensorUInt8 || in[kCompareTensorIdx]->GetType() > kTensorString))
            return false;
    }
    else if(FIXED_NUM == select_type_)
    {
        if((int)(in.size()) <= select_idx_)
            return false;
    }
    else if(PASS_THROUGH == select_type_)
    {
        if(in.size() < 1)
            return false;
    }
    else
    {
        if(in.size() < 2)
            return false;
        
        if(compare_val_->GetType() != in[kCompareTensorIdx]->GetType())
        {
            if((compare_val_->GetType() == kTensorString && in[kCompareTensorIdx]->GetType() < kTensorString) ||
               (compare_val_->GetType() < kTensorString && in[kCompareTensorIdx]->GetType() == kTensorString))
                return false;
        }
    }

    if(out.size() < 1 || out[0]->GetType() != kTensorReference)
        return false;

    return true;
}

int SolverSelector::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
        
    
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    //LOG(ERROR) << "SolverSelector start";
    TensorReference*   out_tensor = dynamic_cast<TensorReference*>(out[0]);
    int                select_idx = select_idx_;
    if(VAR_NUM == select_type_)
    {
        select_idx             = GetTensorNumericValue<int>(*(in[kCompareTensorIdx]));
        if(select_idx >= (int)(in.size() - 1))
        {
            LOG(ERROR) << "select_idx beyond limit";
            ABORT();
        }
        out_tensor->reference_ = in[1 + select_idx];
    }
    else if(FIXED_NUM == select_type_)
    {
        out_tensor->reference_ = in[select_idx];
    }
    else if(PASS_THROUGH == select_type_)
    {
        out_tensor->reference_ = in[0];
    }
    else
    {
        const Tensor*   left_value   = in[kCompareTensorIdx];
        bool res = compare_scalar_tensor(*left_value, *compare_val_, compare_type_);
        if(false == res)
            out_tensor->reference_ = in[1];
        else
            out_tensor->reference_ = in[2];
    }
        //LOG(ERROR) << "SolverSelector end";

    return 0;
}

}
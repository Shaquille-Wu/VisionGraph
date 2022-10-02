#include "../../include/graph_error.h"
#include "solver_counter.h"

namespace vision_graph{

SolverCounter::SolverCounter(nlohmann::json const& param) noexcept : Solver(param), counter_(0LL), cycle_(0)
{
    if(param.contains("counter_type"))
    {
        std::string   counter_type = param.at("counter_type");
        if(counter_type == "increase")
            counter_type_ = COUNTER_TYPE::INCREASE;
        else if(counter_type == "decrease")
            counter_type_ = COUNTER_TYPE::DESCREASE;
    }
    if(param.contains("cycle"))
        cycle_ = param.at("cycle");
}

SolverCounter::~SolverCounter() noexcept
{

}

Tensor* SolverCounter::CreateOutTensor(int out_tensor_idx) noexcept
{
    Tensor*  tensor_counter = new TensorInt64(0LL);
    return tensor_counter;
}

bool SolverCounter::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(out.size() < 1 || out[0]->GetType() != kTensorInt64)
        return false;

    return true;
}

int SolverCounter::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
        return vision_graph::kErrCodeParamInvalid;
    
    if(cycle_ != 0)
        counter_ = counter_ % cycle_;

    if(INCREASE == counter_type_)
        counter_ ++;
    else
        counter_ --;

    TensorInt64* tensor_count = dynamic_cast<TensorInt64*>(out[0]);
    tensor_count->value_ = counter_;

    return 0;
}

}
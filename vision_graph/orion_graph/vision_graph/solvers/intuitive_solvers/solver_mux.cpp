#include "../../include/graph_error.h"
#include "solver_mux.h"
#include <logging.h>

namespace vision_graph{

SolverMux::SolverMux(nlohmann::json const& param) noexcept : Solver(param)
{
}

SolverMux::~SolverMux() noexcept
{
}

    
Tensor* SolverMux::CreateOutTensor(int out_tensor_idx) noexcept
{
    TensorPtrVector*  tensor_vec = new TensorPtrVector;
    return tensor_vec;
}

bool SolverMux::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(out.size() <= 0 || out[0]->GetType() != kTensorPtrVector)
        return false;
    
    return true;
}

int SolverMux::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    TensorPtrVector* tensor_vec = dynamic_cast<TensorPtrVector*>(out[0]);
    *tensor_vec = in;

    return 0;
}

}// namespace vision_graph
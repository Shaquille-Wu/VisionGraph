#include "../../include/graph_error.h"
#include "solver_demux.h"
#include <logging.h>

namespace vision_graph{

SolverDeMux::SolverDeMux(nlohmann::json const& param) noexcept : Solver(param)
{
}

SolverDeMux::~SolverDeMux() noexcept
{
}

    
Tensor* SolverDeMux::CreateOutTensor(int out_tensor_idx) noexcept
{
    TensorReference*  tensor_ref = new TensorReference;
    return tensor_ref;
}

bool SolverDeMux::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() <= 0 || in[0]->GetType() != kTensorPtrVector)
        return false;
    
    TensorPtrVector const*  tensor_vec = dynamic_cast<TensorPtrVector const*>(in[0]);

    if(tensor_vec->size() != out.size())
        return false;

    return true;
}

int SolverDeMux::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    TensorPtrVector const* tensor_vec = dynamic_cast<TensorPtrVector const*>(in[0]);
    int                    tensor_cnt = (int)(tensor_vec->size());
    int                 i          = 0;
    for(i = 0 ; i < tensor_cnt ; i ++)
        out[i] = (*tensor_vec)[i];

    return 0;
}

}// namespace vision_graph
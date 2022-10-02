#include "../../include/graph_error.h"
#include "solver_kpts_box.h"
#include <logging.h>

namespace vision_graph{

SolverKptsBox::SolverKptsBox(nlohmann::json const& param) noexcept : Solver(param)
{
}

SolverKptsBox::~SolverKptsBox() noexcept
{
}

Tensor* SolverKptsBox::CreateOutTensor(int out_tensor_idx) noexcept
{
    TensorKeypoints*  kpts = new TensorKeypoints;
    return kpts;
}

bool SolverKptsBox::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 && out.size() != 1)
        return false;

    if(in[0]->GetType() != kTensorKeyPoints || out[0]->GetType() != kTensorBox)
        return false;
    
    return true;
}

int SolverKptsBox::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    
    vision_graph::Tensor*                  input_tensor         = in[0];
    vision_graph::TensorKeypoints const*   input_tensor_kpts    = dynamic_cast<vision_graph::TensorKeypoints const*>(input_tensor);
    vision_graph::Tensor*                  output_tensor        = out[0];
    vision_graph::TensorBox*               output_tensor_box    = dynamic_cast<vision_graph::TensorBox*>(output_tensor);
    output_tensor_box->clear();
    assert(nullptr != input_tensor_kpts);
    assert(nullptr != output_tensor_box);
    if(nullptr == input_tensor_kpts)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_box)
        return vision_graph::kErrCodeParamInvalid;

    if(input_tensor_kpts->size() <= 0)
        return 0;

    float l = (*input_tensor_kpts)[0].x;
    float r = (*input_tensor_kpts)[0].x;
    float t = (*input_tensor_kpts)[0].y;
    float b = (*input_tensor_kpts)[0].y;

    int   i          = 0 ;
    int   kpts_count = (int)(input_tensor_kpts->size());
    for(i = 1 ; i < kpts_count ; i ++)
    {
        if(l > (*input_tensor_kpts)[i].x)
            l = (*input_tensor_kpts)[i].x;
        if(r < (*input_tensor_kpts)[i].x)
            r = (*input_tensor_kpts)[i].x;
        if(t > (*input_tensor_kpts)[i].y)
            t = (*input_tensor_kpts)[i].y;
        if(b < (*input_tensor_kpts)[i].y)
            b = (*input_tensor_kpts)[i].y;
    }
    output_tensor_box->x1 = l;
    output_tensor_box->x2 = r;
    output_tensor_box->y1 = t;
    output_tensor_box->y2 = b;

    return 0;
}

};//namespace vision_graph
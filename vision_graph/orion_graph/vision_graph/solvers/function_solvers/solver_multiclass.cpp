#include "../../include/graph_error.h"
#include "solver_multiclass.h"

namespace vision_graph{

static const TENSOR_TYPE  kInTensorTypeSupported  = kTensorImageVector;
static const TENSOR_TYPE  kOutTensorTypeSupported = kTensorAttributesVector;

SolverMultiClass::SolverMultiClass(nlohmann::json const& param) noexcept : Solver(param), multiclass_(nullptr)
{
    if(false == param.contains("dlcv_json"))
        return ;
    std::string   dlcv_json_file = param.at("dlcv_json");
    multiclass_ = new vision::Multiclass(dlcv_json_file);
}

SolverMultiClass::~SolverMultiClass() noexcept
{
    if(nullptr != multiclass_)
        delete multiclass_;
    multiclass_ = nullptr;
}

Tensor* SolverMultiClass::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    (void)out_tensor_idx;
    Tensor*  tensor = new TensorAttributesVector;
    return tensor;
}

bool SolverMultiClass::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported)
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported)
        return false;

    return true;
}

int SolverMultiClass::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
        return vision_graph::kErrCodeParamInvalid;
    
    vision_graph::Tensor*            input_tensor         = const_cast<vision_graph::Tensor*>(in[0]);
    vision_graph::TensorImageVector*       input_tensor_image   = dynamic_cast<vision_graph::TensorImageVector*>(input_tensor);
    vision_graph::TensorAttributesVector*  output_tensor_attr   = dynamic_cast<vision_graph::TensorAttributesVector*>(out[0]);

    assert(nullptr != input_tensor_image);
    assert(nullptr != output_tensor_attr);
    if(nullptr == input_tensor_image)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_attr)
        return vision_graph::kErrCodeParamInvalid;
    output_tensor_attr->clear();
    for (cv::Mat& mat : *input_tensor_image) {
        std::vector<std::vector<float>> attr;
        int res = multiclass_->run(mat, attr);
        output_tensor_attr->push_back(attr);
    }

    return 0;
}

};//namespace vision_graph
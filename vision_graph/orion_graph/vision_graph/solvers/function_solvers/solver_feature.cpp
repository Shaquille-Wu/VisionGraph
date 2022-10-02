#include "../../include/graph_error.h"
#include "solver_feature.h"

namespace vision_graph{

static const TENSOR_TYPE  kInTensorTypeSupported  = kTensorImage;
static const TENSOR_TYPE  kOutTensorTypeSupported = kTensorFeature;

SolverFeature::SolverFeature(nlohmann::json const& param) noexcept : Solver(param), feature_(nullptr), is_norm_(false)
{
    if(false == param.contains("dlcv_json"))
        return ;
    std::string   dlcv_json_file = param.at("dlcv_json");
    feature_ = new vision::Feature(dlcv_json_file);
    if(param.contains("norm"))
        is_norm_ = param.at("norm");
}

SolverFeature::~SolverFeature() noexcept
{
    if(nullptr != feature_)
        delete feature_;
    feature_ = nullptr;
}

Tensor* SolverFeature::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    (void)out_tensor_idx;
    Tensor*   tensor = new TensorFeature;
    return tensor;
}

bool SolverFeature::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported)
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported)
        return false;

    return true;
}

int SolverFeature::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
        return vision_graph::kErrCodeParamInvalid;
    
    vision_graph::Tensor*        input_tensor          = const_cast<vision_graph::Tensor*>(in[0]);
    vision_graph::TensorImage*   input_tensor_image    = dynamic_cast<vision_graph::TensorImage*>(input_tensor);
    vision_graph::TensorFeature* output_tensor_feature = dynamic_cast<vision_graph::TensorFeature*>(out[0]);

    assert(nullptr != input_tensor_image);
    assert(nullptr != output_tensor_feature);
    if(nullptr == input_tensor_image)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_feature)
        return vision_graph::kErrCodeParamInvalid;

    output_tensor_feature->clear();
    if(input_tensor_image->cols <= 1 ||
       input_tensor_image->rows <= 1)
       return vision_graph::kErrCodeParamInvalid;
    int res = feature_->run(*(input_tensor_image), *output_tensor_feature);
    if(res >= 0)
    {
        if(true == is_norm_)
        {
            double sum        = 0.0;
            int    i          = 0;
            int    data_count = (int)((*output_tensor_feature).size());
            for(i = 0 ; i < data_count ; i ++)
                sum += (*output_tensor_feature)[i];
            sum = sqrt(sum);
            if(fabs(sum) < 1e-6)
            {
                for(i = 0 ; i < data_count ; i ++)
                    (*output_tensor_feature)[i] = 0.0f;
            }
            else
            {
                for(i = 0 ; i < data_count ; i ++)
                    (*output_tensor_feature)[i] = (float)((double)((*output_tensor_feature)[i]) / sum);
            }
        }
    }

    return 0;
}

};//namespace vision_graph
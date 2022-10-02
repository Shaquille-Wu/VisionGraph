#include "../../include/graph_error.h"
#include "solver_featuremap.h"

namespace vision_graph{

static const TENSOR_TYPE  kInTensorTypeSupported  = kTensorImage;
static const TENSOR_TYPE  kOutTensorTypeSupported = kTensorFeatureMaps;

SolverFeatureMap::SolverFeatureMap(nlohmann::json const& param) noexcept : Solver(param), feature_map_(nullptr)
{
    if(false == param.contains("dlcv_json"))
        return ;
    std::string   dlcv_json_file = param.at("dlcv_json");
    feature_map_ = new vision::FeaturemapRunner(dlcv_json_file);
}

SolverFeatureMap::~SolverFeatureMap() noexcept
{
    if(nullptr != feature_map_)
        delete feature_map_;
    feature_map_ = nullptr;
}

Tensor* SolverFeatureMap::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    (void)out_tensor_idx;
    Tensor*   tensor = new TensorFeatureMaps;
    return tensor;
}

bool SolverFeatureMap::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported)
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported)
        return false;

    return true;
}

int SolverFeatureMap::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
        return vision_graph::kErrCodeParamInvalid;
    
    vision_graph::Tensor*            input_tensor           = const_cast<vision_graph::Tensor*>(in[0]);
    vision_graph::TensorImage*       input_tensor_image     = dynamic_cast<vision_graph::TensorImage*>(input_tensor);
    vision_graph::TensorFeatureMaps* output_tensor_feat_map = dynamic_cast<vision_graph::TensorFeatureMaps*>(out[0]);

    assert(nullptr != input_tensor_image);
    assert(nullptr != output_tensor_feat_map);
    if(nullptr == input_tensor_image)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_feat_map)
        return vision_graph::kErrCodeParamInvalid;

    int res = feature_map_->run(*(input_tensor_image), *output_tensor_feat_map);

    return res;
}

};//namespace vision_graph
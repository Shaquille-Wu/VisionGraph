#include "../../include/graph_error.h"
#include "solver_keypoints.h"
#include <logging.h>

namespace vision_graph{

static const TENSOR_TYPE               kInTensorTypeSupported  = kTensorImageVector;
static const std::vector<TENSOR_TYPE>  kOutTensorTypeSupported = {
    kTensorKeypointsVector,
    kTensorAttributes
};

static const std::string               kJSONNodeReoderKpts     = "reorder_kpts";
static const std::string               kJSONNodeAngelScale     = "angel_scale";

SolverKeypoints::SolverKeypoints(nlohmann::json const& param) noexcept : Solver(param), 
                                                                         keypoints_(nullptr), 
                                                                         reorder_kpts_(false), 
                                                                         angle_scale_(1.0f)
{
    if(false == param.contains("dlcv_json"))
        return ;
    std::string   dlcv_json_file = param.at("dlcv_json");
    keypoints_ = new vision::KeypointsAttributes(dlcv_json_file);

    if(param.contains(kJSONNodeReoderKpts))
        reorder_kpts_ = param.at(kJSONNodeReoderKpts);
    if(param.contains(kJSONNodeAngelScale))
        angle_scale_ = param.at(kJSONNodeAngelScale);
}

SolverKeypoints::~SolverKeypoints() noexcept
{
    if(nullptr != keypoints_)
        delete keypoints_;
    keypoints_ = nullptr;
}

Tensor* SolverKeypoints::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*   tensor = nullptr;
    if(0 == out_tensor_idx)
        tensor = new TensorKeypointsVector;
    else if(1 == out_tensor_idx)
        tensor = new TensorAttributes;
    return tensor;
}

bool SolverKeypoints::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if (in.size() != 1 || out.size() != 2) {
        return false; }
 

    if (in[0]->GetType() != kInTensorTypeSupported) {
        return false;}
     

    if (out[0]->GetType() != kOutTensorTypeSupported[0] ||
        out[1]->GetType() != kOutTensorTypeSupported[1]) {
        return false; }
 
    
    return true;
}
 
int SolverKeypoints::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{


    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
        
    
    vision_graph::Tensor*                  input_tensor         = const_cast<vision_graph::Tensor*>(in[0]);
    vision_graph::TensorImageVector*       input_tensor_image   = dynamic_cast<vision_graph::TensorImageVector*>(input_tensor);
    vision_graph::TensorKeypointsVector*   output_tensor_kpts   = dynamic_cast<vision_graph::TensorKeypointsVector*>(out[0]);
    vision_graph::TensorAttributes*        output_tensor_attrs  = dynamic_cast<vision_graph::TensorAttributes*>(out[1]);
    output_tensor_kpts->clear();
    output_tensor_attrs->clear();
    assert(nullptr != input_tensor_image);
    assert(nullptr != output_tensor_kpts);
    assert(nullptr != output_tensor_attrs);
    if(nullptr == input_tensor_image)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_kpts)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_attrs)
        return vision_graph::kErrCodeParamInvalid;

    output_tensor_kpts->clear();
    output_tensor_attrs->clear();
    int res;
    for (const cv::Mat& mat : *input_tensor_image) {
        if (mat.cols <= 1 || mat.rows <= 1) {
            return vision_graph::kErrCodeParamInvalid;
        }
        std::vector<cv::Point2f> fkp2f;
        std::vector<cv::Point> fkp;
        std::vector<vector<float>> attributes;
        res = keypoints_->run((cv::Mat &)mat, fkp, attributes);
        for (cv::Point p : fkp) {
            cv::Point2f p2f(p.x, p.y);
            fkp2f.push_back(p2f);
        }
        // output_tensor_kpts->push_back(fkp);
        output_tensor_kpts->push_back(fkp2f);
        output_tensor_attrs->push_back(attributes[0]);
    }

    return res;
}

};//namespace vision_graph
#include "../../include/graph_error.h"
#include "solver_detect.h"

#include "logging.h"

namespace vision_graph{

static const TENSOR_TYPE  kInTensorTypeSupported  = kTensorImage;
static const TENSOR_TYPE  kOutTensorTypeSupported = kTensorBoxesMap;

SolverDetect::SolverDetect(nlohmann::json const& param) noexcept : Solver(param), detector_(nullptr)
{
    if(false == param.contains("dlcv_json"))
        return ;
    std::string   dlcv_json_file = param.at("dlcv_json");
    // LOG(ERROR) << "wyb debug dlcv_json_file " << dlcv_json_file;
    // if (detector_!=nullptr) {
    //     LOG(ERROR) << "wyb debug detector_!=nullptr";
        
    // }
    if(!detector_)
        detector_ = new vision::Detector(dlcv_json_file);
    // LOG(ERROR) << "wyb debug dlcv_json_file end";

}

SolverDetect::~SolverDetect() noexcept
{
    if(nullptr != detector_)
        delete detector_;
    detector_ = nullptr;
}

Tensor* SolverDetect::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    (void)out_tensor_idx;
    Tensor*   tensor = new TensorBoxesMap;
    return tensor;
}

bool SolverDetect::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported)
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported)
        return false;
    return true;
}

int SolverDetect::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    //LOG(ERROR) << "SolverDetect start";

    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
        

    vision_graph::Tensor*         input_tensor        = const_cast<vision_graph::Tensor*>(in[0]);
    vision_graph::TensorImage*    input_tensor_image  = dynamic_cast<vision_graph::TensorImage*>(input_tensor);
    vision_graph::TensorBoxesMap* output_tensor_boxes = dynamic_cast<vision_graph::TensorBoxesMap*>(out[0]);
    output_tensor_boxes->clear();
    assert(nullptr != input_tensor_image);
    assert(nullptr != output_tensor_boxes);
    if(nullptr == input_tensor_image)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_boxes)
        return vision_graph::kErrCodeParamInvalid;

    int res = detector_->run(*(input_tensor_image), *output_tensor_boxes);

    return res;
}

};//namespace vision_graph
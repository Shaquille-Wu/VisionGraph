#include "../../include/graph_error.h"
#include "solver_select_boxes.h"
#include <logging.h>
namespace vision_graph{

static const std::string kFilterNodeFilters  = std::string("filters");
static const std::string kScoreThresholdNode = std::string("score_threshold");

static const std::vector<TENSOR_TYPE>  kInTensorTypeSupported  = {
    kTensorBoxesMap
};

static const std::vector<TENSOR_TYPE>  kOutTensorTypeSupported = {
    kTensorBoxVector
};

SolverSelectBoxes::SolverSelectBoxes(nlohmann::json const& param) noexcept : Solver(param), score_threshold_(0.0f), area_threshold_(0.0f)
{
    if(true == param.contains(kFilterNodeFilters))
        filters_ = param.at(kFilterNodeFilters).get<std::vector<std::string> >();
    if(true == param.contains(kScoreThresholdNode))
        score_threshold_ = param.at(kScoreThresholdNode).get<float>();
}

SolverSelectBoxes::~SolverSelectBoxes() noexcept
{
}

Tensor* SolverSelectBoxes::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*   tensor = new TensorBoxVector;
    return tensor;
}

bool SolverSelectBoxes::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported[0])
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported[0])
        return false;

    return true;
}

int SolverSelectBoxes::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    
    vision_graph::TensorBoxesMap const* input_tensor_boxes  = dynamic_cast<vision_graph::TensorBoxesMap const*>(in[0]);
    vision_graph::TensorBoxVector*       output_tensor_box   = dynamic_cast<vision_graph::TensorBoxVector*>(out[0]);
    output_tensor_box->clear();
    assert(nullptr != input_tensor_boxes);
    assert(nullptr != output_tensor_box);
    if(nullptr == input_tensor_boxes)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_box)
        return vision_graph::kErrCodeParamInvalid;

    if(input_tensor_boxes->size() > 0)
    {
        bool          found      = false;
        int           i          = 0;
        int           j          = 0;
        float         max_score  = FLT_MIN;
        std::vector<vision::Box>   boxes;
        
        if(filters_.size() > 0)
        {
            for(i = 0 ; i < (int)(filters_.size()) ; i ++)
            {
                boxes.clear();
                const std::string&                             cur_filter = filters_[i];
                vision_graph::TensorBoxesMap::const_iterator   iter       = input_tensor_boxes->find(cur_filter);
                if(input_tensor_boxes->end() != iter)
                {
                    for (const vision::Box& b : iter->second) {
                        if (b.score > score_threshold_) {
                            vision::Box box;
                            box.tid = boxes.size();
                            box.x1 = b.x1;
                            box.y1 = b.y1;
                            box.x2 = b.x2;
                            box.y2 = b.y2;
                            boxes.push_back(box);
                        }
                            
                    }
                }
                TensorBoxesMap::NormalizeBox(boxes);
                *output_tensor_box = boxes;
            }
        }

    }

    return 0;
}

};//namespace vision_graph
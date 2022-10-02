#include "../../include/graph_error.h"
#include "solver_select_box.h"
#include <logging.h>
namespace vision_graph{

static const std::string kFilterNodeFilters  = std::string("filters");
static const std::string kScoreThresholdNode = std::string("score_threshold");
static const std::string kAreaThresholdNode  = std::string("area_threshold");

static const std::vector<TENSOR_TYPE>  kInTensorTypeSupported  = {
    kTensorBoxesMap
};

static const std::vector<TENSOR_TYPE>  kOutTensorTypeSupported = {
    kTensorBox
};

SolverSelectBox::SolverSelectBox(nlohmann::json const& param) noexcept : Solver(param), score_threshold_(0.0f), area_threshold_(0.0f)
{
    if(true == param.contains(kFilterNodeFilters))
        filters_ = param.at(kFilterNodeFilters).get<std::vector<std::string> >();
    if(true == param.contains(kScoreThresholdNode))
        score_threshold_ = param.at(kScoreThresholdNode).get<float>();
    if(true == param.contains(kAreaThresholdNode))
        area_threshold_  = param.at(kAreaThresholdNode).get<float>();
}

SolverSelectBox::~SolverSelectBox() noexcept
{
}

Tensor* SolverSelectBox::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*   tensor = new TensorBox;
    return tensor;
}

bool SolverSelectBox::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 1 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported[0])
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported[0])
        return false;

    return true;
}

int SolverSelectBox::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    
    vision_graph::TensorBoxesMap const* input_tensor_boxes  = dynamic_cast<vision_graph::TensorBoxesMap const*>(in[0]);
    vision_graph::TensorBox*            output_tensor_box   = dynamic_cast<vision_graph::TensorBox*>(out[0]);

    assert(nullptr != input_tensor_boxes);
    assert(nullptr != output_tensor_box);
    if(nullptr == input_tensor_boxes)
        return vision_graph::kErrCodeParamInvalid;
    if(nullptr == output_tensor_box)
        return vision_graph::kErrCodeParamInvalid;

    output_tensor_box->clear();
    if(input_tensor_boxes->size() > 0)
    {
        bool          found      = false;
        int           i          = 0;
        int           j          = 0;
        float         max_score  = FLT_MIN;
        vision::Box   max_box;
        max_box.clear();
        if(filters_.size() > 0)
        {
            for(i = 0 ; i < (int)(filters_.size()) ; i ++)
            {
                const std::string&                             cur_filter = filters_[i];
                vision_graph::TensorBoxesMap::const_iterator   iter       = input_tensor_boxes->find(cur_filter);
                if(input_tensor_boxes->end() != iter)
                {
                    const std::vector<vision::Box>&   boxes = iter->second;
                    //we assume the box_map has been sorted, so, the first box is the "max" box
                    for(j = 0 ; j < ((int)(boxes.size())) ; j ++)
                    {
                        bool score_valid = (score_threshold_ > 1e-6 ? (boxes[j].score >= score_threshold_) : true);
                        bool area_valid  = (area_threshold_ > 1e-6 ?  (boxes[j].area() >= area_threshold_) : true);
                        if((boxes[j].score > max_score) && (true == area_valid && true == score_valid))
                        {
                            max_box   = boxes[j];
                            max_score = boxes[j].score;
                            found     = true;
                        }
                    }
                }
            }
        }
        else
        {
            vision_graph::TensorBoxesMap::const_iterator   iter = input_tensor_boxes->begin();
            while(input_tensor_boxes->end() != iter)
            {
                const std::vector<vision::Box>&   boxes = iter->second;
                //we assume the box_map has been sorted, so, the first box is the "max" box
                for(j = 0 ; j < ((int)(boxes.size())) ; j ++)
                {
                    bool score_valid = (score_threshold_ > 1e-6 ? (boxes[j].score >= score_threshold_) : true);
                    bool area_valid  = (area_threshold_ > 1e-6 ? (boxes[j].area() >= area_threshold_) : true);
                    if((boxes[j].score > max_score) && (true == area_valid && true == score_valid))
                    {
                        max_box   = boxes[j];
                        max_score = boxes[j].score;
                        found     = true;
                    }
                }
                iter ++;
            }
        }

        if(true == found)
        {
            TensorBox::NormalizeBox(max_box);
            (*output_tensor_box)  = max_box;
        }
    }

    return 0;
}

};//namespace vision_graph
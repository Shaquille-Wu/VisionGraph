#include "../../include/graph_error.h"
#include "solver_get_good_boxes.h"
#include <logging.h>

namespace vision_graph{

static const std::vector<TENSOR_TYPE> kInputSrcTypes = {
    kTensorKeypointsVector,
    kTensorAttributes,
    kTensorBoxVector,
    kTensorFloat32Vector
};

static const std::vector<TENSOR_TYPE> kInputOffsetTypes = {
    kTensorKeypointsVector,
    kTensorAttributes,
    kTensorBoxVector,
    kTensorFloat32Vector,
    kTensorUInt32
};

SolverGetGoodBoxes::SolverGetGoodBoxes(nlohmann::json const& param) noexcept : Solver(param)
{
    value_  = param.at("value");
}

SolverGetGoodBoxes::~SolverGetGoodBoxes() noexcept
{
}

Tensor* SolverGetGoodBoxes::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*   tensor = nullptr;
    if(0 == out_tensor_idx)
        tensor = new TensorKeypointsVector;
    else if(1 == out_tensor_idx)
        tensor = new TensorAttributes;
    else if(2 == out_tensor_idx)
        tensor = new TensorBoxVector;
    else if(3 == out_tensor_idx)
        tensor = new TensorFloat32Vector;
    else if(4 == out_tensor_idx)
        tensor = new TensorUInt32;
    return tensor;
}

bool SolverGetGoodBoxes::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    size_t i = 0 ;
    if(in.size() != 4 || out.size() != 5)
        return false;
    
    for(i = 0 ; i < kInputSrcTypes.size(); i ++)
    {
        if(in[0]->GetType() == kInputSrcTypes[i])
            break;
    }
    if(i >= kInputSrcTypes.size())
        return false;
    for(i = 0 ; i < kInputOffsetTypes.size(); i ++)
    {
        if(in[1]->GetType() == kInputOffsetTypes[i])
            break;
    }
    if(i >= kInputOffsetTypes.size())
        return false;
    
    if(out[0]->GetType() != in[0]->GetType())
        return false;

    return true;
}

int SolverGetGoodBoxes::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    

    vision_graph::TensorKeypointsVector const*  input_tensor_kpts        = dynamic_cast<vision_graph::TensorKeypointsVector*>(in[0]);
    vision_graph::TensorAttributes const*       input_tensor_att         = dynamic_cast<vision_graph::TensorAttributes*>(in[1]);
    vision_graph::TensorBoxVector const*        input_tensor_boxes       = dynamic_cast<vision_graph::TensorBoxVector*>(in[2]);
    vision_graph::TensorFloat32Vector const*    input_tensor_hog         = dynamic_cast<vision_graph::TensorFloat32Vector*>(in[3]);
    // LOG(ERROR) << "SolverGetGoodBoxes "<< input_tensor_kpts->size()<<" "<<input_tensor_att->size()<<" "<<input_tensor_boxes->size()<<" "<<input_tensor_hog->size();

    vision_graph::TensorKeypointsVector*   out_tensor_kpts          = dynamic_cast<vision_graph::TensorKeypointsVector*>(out[0]);
    vision_graph::TensorAttributes*        out_tensor_att           = dynamic_cast<vision_graph::TensorAttributes*>(out[1]);
    vision_graph::TensorBoxVector*         out_tensor_boxes         = dynamic_cast<vision_graph::TensorBoxVector*>(out[2]);
    vision_graph::TensorFloat32Vector*     out_tensor_hog           = dynamic_cast<vision_graph::TensorFloat32Vector*>(out[3]);
    out_tensor_kpts->clear();
    out_tensor_att->clear();
    out_tensor_boxes->clear();
    out_tensor_hog->clear();
    (dynamic_cast<TensorUInt32*>(out[4]))->value_ = 0;

    for (int i = 0; i < input_tensor_hog->size(); i++) {
        if (*(input_tensor_hog->begin() + i) > value_) {
            out_tensor_kpts->push_back(*(input_tensor_kpts->begin() + i));
            out_tensor_att->push_back(*(input_tensor_att->begin() + i));
            out_tensor_boxes->push_back(*(input_tensor_boxes->begin() + i));
            out_tensor_hog->push_back(*(input_tensor_hog->begin() + i));
        }
        else
        {
            (dynamic_cast<TensorUInt32* >(out[4]))->value_ = 1;
        }
        
    }

    return 0;
}

};//namespace vision_graph
#include "../../include/graph_error.h"
#include "solver_geometry_offset.h"
#include <logging.h>

namespace vision_graph{

static const std::vector<TENSOR_TYPE> kInputSrcTypes = {
    kTensorPoint,
    kTensorBox,
    kTensorKeyPoints,
    kTensorBoxesMap,
    kTensorKeypointsVector,
    kTensorBoxVector
};

static const std::vector<TENSOR_TYPE> kInputOffsetTypes = {
    kTensorKeypointsVector,
    kTensorBoxVector
};

SolverGeometryOffset::SolverGeometryOffset(nlohmann::json const& param) noexcept : Solver(param)
{
}

SolverGeometryOffset::~SolverGeometryOffset() noexcept
{
}

Tensor* SolverGeometryOffset::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*   tensor = new TensorKeypointsVector;
    return tensor;
}

bool SolverGeometryOffset::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    size_t i = 0 ;
    if(in.size() != 2 || out.size() != 1)
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

int SolverGeometryOffset::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    
    vision_graph::TensorPoint const*                    input_tensor_point          = nullptr;
    vision_graph::TensorBox const*                      input_tensor_box            = nullptr;
    vision_graph::TensorKeypoints const*                input_tensor_kpts           = nullptr;
    vision_graph::TensorBoxesMap const*                 input_tensor_boxesmap       = nullptr;
    vision_graph::TensorPoint const*                    input_tensor_pt_offset      = nullptr;
    vision_graph::TensorBox const*                      input_tensor_box_offset     = nullptr;
    vision_graph::TensorKeypointsVector const*          input_tensor_ptvec_offset   = nullptr;
    vision_graph::TensorBoxVector const*                input_tensor_boxvec_offset  = nullptr;




    vision_graph::TensorPoint*                          output_tensor_pt             = nullptr;
    vision_graph::TensorBox*                            output_tensor_box            = nullptr;
    vision_graph::TensorKeypoints*                      output_tensor_kpts           = nullptr;
    vision_graph::TensorBoxesMap*                       output_tensor_boxesmap       = nullptr;
    vision_graph::TensorKeypointsVector*                output_tensor_ptvec_offset   = nullptr;
    vision_graph::TensorBoxVector*                      output_tensor_boxvec_offset  = nullptr;



    if(kTensorKeypointsVector == in[0]->GetType()&&kTensorBoxVector == in[1]->GetType())
    {
        input_tensor_ptvec_offset   = dynamic_cast<vision_graph::TensorKeypointsVector const*>(in[0]);
        input_tensor_boxvec_offset  = dynamic_cast<vision_graph::TensorBoxVector const*>(in[1]);
        output_tensor_ptvec_offset  = dynamic_cast<vision_graph::TensorKeypointsVector*>(out[0]);
        output_tensor_ptvec_offset->clear();
        for (int i = 0; i < input_tensor_boxvec_offset->size(); i++) {
            vision::Box b = *(input_tensor_boxvec_offset->begin() + i);   
            float offset_x = b.x1;
            float offset_y = b.y1;
            std::vector<cv::Point2f> fkp = *(input_tensor_ptvec_offset->begin() + i);
            for (int j = 0; j < fkp.size(); j++) {
                fkp[j].x += offset_x;
                fkp[j].y += offset_y;
            }
            output_tensor_ptvec_offset->push_back(fkp);
        }
    }

    // if(nullptr == in[0] || nullptr == in[1])
    //     return vision_graph::kErrCodeParamInvalid;

    // float  offset_x = 0.0f, offset_y = 0.0f;
    // if(kTensorPoint == in[1]->GetType())
    // {
    //     input_tensor_pt_offset  = dynamic_cast<vision_graph::TensorPoint const*>(in[1]);
    //     offset_x                = input_tensor_pt_offset->x;
    //     offset_y                = input_tensor_pt_offset->y;
    // }
    // else if(kTensorBox == in[1]->GetType())
    // {
    //     input_tensor_box_offset = dynamic_cast<vision_graph::TensorBox const*>(in[1]);
    //     offset_x                = input_tensor_box_offset->x1;
    //     offset_y                = input_tensor_box_offset->y1;
    // }

    // size_t i = 0 ;
    // if(kTensorPoint == in[0]->GetType())
    // {
    //     input_tensor_point   = dynamic_cast<vision_graph::TensorPoint const*>(in[0]);
    //     output_tensor_pt     = dynamic_cast<vision_graph::TensorPoint*>(out[0]);
    //     output_tensor_pt->x  = input_tensor_point->x + offset_x;
    //     output_tensor_pt->y  = input_tensor_point->y + offset_y;
    // }
    // else if(kTensorBox == in[0]->GetType())
    // {
    //     input_tensor_box       = dynamic_cast<vision_graph::TensorBox const*>(in[0]);
    //     output_tensor_box      = dynamic_cast<vision_graph::TensorBox*>(out[0]);
    //     output_tensor_box->x1  = input_tensor_box->x1 + offset_x;
    //     output_tensor_box->y1  = input_tensor_box->y1 + offset_y;
    //     output_tensor_box->x2  = input_tensor_box->x2 + offset_x;
    //     output_tensor_box->y2  = input_tensor_box->y2 + offset_y;
    // }
    // else if(kTensorKeyPoints == in[0]->GetType())
    // {
    //     input_tensor_kpts        = dynamic_cast<vision_graph::TensorKeypoints const*>(in[0]);
    //     output_tensor_kpts       = dynamic_cast<vision_graph::TensorKeypoints*>(out[0]);
    //     output_tensor_kpts->resize(input_tensor_kpts->size());
    //     for(i = 0 ; i < input_tensor_kpts->size() ; i ++)
    //     {
    //         (*output_tensor_kpts)[i].x = (*input_tensor_kpts)[i].x + offset_x;
    //         (*output_tensor_kpts)[i].y = (*input_tensor_kpts)[i].y + offset_y;
    //     }
    // }
    // else if(kTensorBoxesMap == in[0]->GetType())
    // {
    //     input_tensor_boxesmap    = dynamic_cast<vision_graph::TensorBoxesMap const*>(in[0]);
    //     output_tensor_boxesmap   = dynamic_cast<vision_graph::TensorBoxesMap*>(out[0]);
    //     *output_tensor_boxesmap  = *input_tensor_boxesmap;
    //     vision_graph::TensorBoxesMap::const_iterator  input_iter = input_tensor_boxesmap->begin();
    //     while(input_tensor_boxesmap->end() != input_iter)
    //     {
    //         std::string const&               box_name     = input_iter->first;
    //         std::vector<vision::Box> const&  input_boxes  = input_iter->second;
    //         std::vector<vision::Box>&        output_boxes = (output_tensor_boxesmap->find(box_name))->second;
    //         for(i = 0 ; i < output_boxes.size() ; i ++)
    //         {
    //             output_boxes[i].x1 = input_boxes[i].x1 + offset_x;
    //             output_boxes[i].y1 = input_boxes[i].y1 + offset_y;
    //             output_boxes[i].x2 = input_boxes[i].x2 + offset_x;
    //             output_boxes[i].y2 = input_boxes[i].y2 + offset_y;
    //         }
    //         input_iter ++;
    //     }
    // }

    return 0;
}

};//namespace vision_graph
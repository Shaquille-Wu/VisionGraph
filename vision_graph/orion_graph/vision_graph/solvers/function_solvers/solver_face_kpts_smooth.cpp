#include <string.h>
#include "../../include/graph_error.h"
#include "../../common/utils.h"
#include "solver_face_kpts_smooth.h"
#include <logging.h>

namespace vision_graph{

SolverFaceKptsSmooth::SolverFaceKptsSmooth(nlohmann::json const& param) noexcept : Solver(param)
{
    face_smooth_map_.clear();
    for(int i = 0 ; i < FACE_SMOOTH_MAX ; i ++)
        face_list_[i] = -1;
    face_head_    = 0;
    multi_target_ = false;
}

SolverFaceKptsSmooth::~SolverFaceKptsSmooth() noexcept
{
    std::map<int, FaceKptsSmooth*>::iterator iter = face_smooth_map_.begin();
    while(face_smooth_map_.end() != iter)
    {
        if(nullptr != iter->second)
            delete iter->second;
        iter ++;
    }
    face_smooth_map_.clear();
}

Tensor* SolverFaceKptsSmooth::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*  out_tensor = nullptr;
    if(0 == out_tensor_idx)
        out_tensor = new TensorKeypointsVector;
    else if(1 == out_tensor_idx)
        out_tensor = new TensorBoxVector;
    return out_tensor;
}

bool SolverFaceKptsSmooth::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 2 || out.size() != 2)
        return false;
    if(in[0]->GetType() != kTensorKeypointsVector ||
        in[1]->GetType() != kTensorBoxVector ||
        out[0]->GetType() != kTensorKeypointsVector ||
        out[1]->GetType() != kTensorBoxVector)
        return false;
    
    return true;
}

int SolverFaceKptsSmooth::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    int target_id = 0;
    int                     i               = 0;
    Tensor const*           tensor_in0      = in[0];
    Tensor const*           tensor_in1      = in[1];
    Tensor*                 tensor_out0     = out[0];
    Tensor*                 tensor_out1     = out[1];
    TensorKeypointsVector const*  tensor_kpts_in  = dynamic_cast<TensorKeypointsVector const*>(tensor_in0);
    TensorBoxVector const*        tensor_box_in   = dynamic_cast<TensorBoxVector const*>(tensor_in1);
    TensorKeypointsVector*        tensor_kpts_out = dynamic_cast<TensorKeypointsVector*>(tensor_out0);
    TensorBoxVector*              tensor_box_out  = dynamic_cast<TensorBoxVector*>(tensor_out1);
    assert(nullptr != tensor_kpts_in);
    assert(nullptr != tensor_box_in);
    assert(nullptr != tensor_kpts_out);
    assert(nullptr != tensor_box_out);
    tensor_kpts_out->clear();
    tensor_box_out->clear();

    solve_target(tensor_kpts_in, tensor_box_in, tensor_kpts_out, tensor_box_out);

    return 0;
}

int SolverFaceKptsSmooth::solve_target(TensorKeypointsVector const*  tensor_kpts_in,
                                       TensorBoxVector const*        tensor_box_in,
                                       TensorKeypointsVector*        tensor_kpts_out,
                                       TensorBoxVector*              tensor_box_out) noexcept
{
    for (int i = 0; i < tensor_box_in->size(); i++) {
        int box_id = (tensor_box_in->begin() + i)->tid;
        auto it = face_smooth_map_.find(box_id);
        std::vector<cv::Point2f> fkp = *(tensor_kpts_in->begin() + i);
        std::vector<cv::Point2f> smooth_fkp = fkp;
        if (it != face_smooth_map_.end()) {
            it->second->Update(fkp);
            smooth_fkp = it->second->GetProcKpts();
        }else
        {
            FaceKptsSmooth* face_smooth = new FaceKptsSmooth();
            face_smooth_map_.insert(std::make_pair(box_id, face_smooth));
            face_smooth->Update(fkp);
            smooth_fkp = face_smooth->GetProcKpts();

        }
        tensor_kpts_out->push_back(smooth_fkp);
        vision::Box out_box = get_bounding_box(smooth_fkp);
        out_box.tid = box_id;
        tensor_box_out->push_back(out_box);
    }
        return 0;
}

vision::Box SolverFaceKptsSmooth::get_bounding_box(std::vector<cv::Point2f> const& points) noexcept
{
    float x1 = points[0].x, y1 = points[0].y;
    float x2 = points[0].x, y2 = points[0].y;
    for (int ix = 1; ix < (int)(points.size()); ++ix) 
    {
        if (x1 > points[ix].x) 
            x1 = points[ix].x;
        if (x2 < points[ix].x) 
            x2 = points[ix].x;
        if (y1 > points[ix].y) 
            y1 = points[ix].y;
        if (y2 < points[ix].y) 
            y2 = points[ix].y;
    }
    vision::Box b;
    b.x1 = x1;
    b.y1 = y1;
    b.x2 = x2;
    b.y2 = y2;
    return b;
}

};//namespace vision_graph
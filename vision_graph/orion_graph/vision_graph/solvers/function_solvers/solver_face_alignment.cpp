#include "../../include/graph_error.h"
#include "../../common/utils.h"
#include "solver_face_alignment.h"
#include <logging.h>
#include "implement/face_alignment/face_alignment.h"


namespace vision_graph{

SolverFaceAlignment::SolverFaceAlignment(nlohmann::json const& param) noexcept : Solver(param)
{
}

SolverFaceAlignment::~SolverFaceAlignment() noexcept
{
}

Tensor* SolverFaceAlignment::CreateOutTensor(int out_tensor_idx) noexcept
{
    TensorKeypoints*  kpts = new TensorKeypoints;
    return kpts;
}

bool SolverFaceAlignment::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 3 && out.size() != 1)
        return false;

    if(in[0]->GetType() != kTensorImage || 
       in[1]->GetType() != kTensorBox || 
       in[2]->GetType() != kTensorKeyPoints ||
       out[0]->GetType() != kTensorImage)
        return false;

    return true;
}

int SolverFaceAlignment::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    int                     i                = 0;
    Tensor const*           tensor_in0       = in[0];
    Tensor const*           tensor_in1       = in[1];
    Tensor const*           tensor_in2       = in[2];
    Tensor*                 tensor_out       = out[0];
    TensorImage const*      tensor_in_image  = dynamic_cast<TensorImage const*>(tensor_in0);
    TensorBox const*        tensor_in_box    = dynamic_cast<TensorBox const*>(tensor_in1);
    TensorKeypoints const*  tensor_in_kpts   = dynamic_cast<TensorKeypoints const*>(tensor_in2);
    TensorImage*            tensor_out_image = dynamic_cast<TensorImage*>(tensor_out);
    assert(nullptr != tensor_in_image);
    assert(nullptr != tensor_in_box);
    assert(nullptr != tensor_in_kpts);
    assert(nullptr != tensor_out_image);

    *tensor_out_image = (*tensor_in_image)(cv::Rect(0, 0, 0, 0));
    int x = FLT2INT32(tensor_in_box->x1);
    int y = FLT2INT32(tensor_in_box->y1);
    int w = FLT2INT32((tensor_in_box->x2 - tensor_in_box->x1));
    int h = FLT2INT32((tensor_in_box->y2 - tensor_in_box->y1));

    if(x < 0)                       x = 0;
    if(y < 0)                       y = 0;
    if(w > tensor_in_image->cols)   w = tensor_in_image->cols;
    if(h > tensor_in_image->rows)   w = tensor_in_image->rows;
    cv::Rect    face_rect(x, y, w, h);
    face_align_compact(*tensor_in_image, *tensor_in_kpts, face_rect, *tensor_out_image);

    return 0;
}

};//namespace vision_graph
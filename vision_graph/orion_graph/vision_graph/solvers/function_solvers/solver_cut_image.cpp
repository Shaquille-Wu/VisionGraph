#include "../../include/graph_error.h"
#include "solver_cut_image.h"
#include <logging.h>

namespace vision_graph{

static const std::string kFilterNodePadding = std::string("padding");

static const std::vector<TENSOR_TYPE>  kInTensorTypeSupported  = {
    kTensorImage,
    kTensorBoxVector
};

static const std::vector<TENSOR_TYPE>  kOutTensorTypeSupported = {
    kTensorImageVector
};

SolverCutImage::SolverCutImage(nlohmann::json const& param) noexcept : Solver(param), padding_(false)
{
    if(true == param.contains(kFilterNodePadding))
        padding_ = param.at(kFilterNodePadding);
}

SolverCutImage::~SolverCutImage() noexcept
{
}

Tensor* SolverCutImage::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*   tensor = new TensorImageVector;
    return tensor;
}

bool SolverCutImage::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 2 || out.size() != 1)
        return false;
    if(in[0]->GetType() != kInTensorTypeSupported[0] ||
       in[1]->GetType() != kInTensorTypeSupported[1])
        return false;
    if(out[0]->GetType() != kOutTensorTypeSupported[0])
        return false;

    return true;
}

int SolverCutImage::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    //LOG(ERROR) << "SolverCutImage start";

    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
    
    vision_graph::TensorImage const*        input_tensor_image  = dynamic_cast<vision_graph::TensorImage const*>(in[0]);
    vision_graph::TensorBoxVector const*    input_tensor_boxes  = dynamic_cast<vision_graph::TensorBoxVector const*>(in[1]);
    vision_graph::TensorImageVector*        output_tensor_image = dynamic_cast<vision_graph::TensorImageVector*>(out[0]);
    output_tensor_image->clear();
    assert(nullptr != input_tensor_image);
    assert(nullptr != input_tensor_boxes);
    assert(nullptr != output_tensor_image);
    if (nullptr == input_tensor_image) {
    //LOG(ERROR) << "SolverCutImage 1";

        return vision_graph::kErrCodeParamInvalid;
    
    }
    if (nullptr == input_tensor_boxes) {
    //LOG(ERROR) << "SolverCutImage 2";

        return vision_graph::kErrCodeParamInvalid;

    }
    if (nullptr == output_tensor_image) {
    //LOG(ERROR) << "SolverCutImage 3";

        return vision_graph::kErrCodeParamInvalid;
    
    }
    
    if(false == input_tensor_boxes->empty())
    {
        TensorBoxVector norm_input_box = *input_tensor_boxes;
        if(true == norm_input_box.empty())
            return vision_graph::kErrCodeParamInvalid;

        // TensorBoxesMap max_box = norm_input_box;
        for (vision::Box& box : norm_input_box) {
            int x0 = box.x1;
            int y0 = box.y1;
            int x1 = box.x2;
            int y1 = box.y2;
            int w = x1 - x0;
            int h = y1 - y0;
            if(true == padding_)
            {
          
                int l_pad = 0, t_pad = 0, r_pad = 0, b_pad = 0;
                int  l_new = x0, t_new = y0, r_new = x1, b_new = y1;
                if(x0 < 0)
                {
                    l_pad = -x0;
                    l_new = l_pad;
                }
                if(y0 < 0)
                {
                    t_pad = -y0;
                    t_new = t_pad;
                }
                if(x1 > input_tensor_image->cols)
                {
                    r_pad = x1 - input_tensor_image->cols;
                    r_new = x1 + r_pad;
                }
                if(y1 > input_tensor_image->rows)
                {
                    t_pad = y1 - input_tensor_image->rows;
                    b_new = y1 + t_pad;
                }
                cv::Mat  padding_image = (*input_tensor_image);
                cv::copyMakeBorder(*input_tensor_image, padding_image, 
                                    l_pad, t_pad, r_pad, t_pad,
                                    cv::BORDER_CONSTANT);
                cv::Mat cut_image;
                cut_image = padding_image(cv::Rect(l_new, t_new, r_new - l_new, b_new - t_new));
                output_tensor_image->push_back(cut_image);
                //***************************
                // (*output_tensor_image) = padding_image(cv::Rect(l_new, t_new, r_new - l_new, b_new - t_new));

                //***************************
            }
            else
            {
                if(x0 < 0)
                    x0 = 0;
                if(y0 < 0)
                    y0 = 0;
                if(x1 > input_tensor_image->cols)
                    x1 = input_tensor_image->cols;
                if(y1 > input_tensor_image->rows)
                    y1 = input_tensor_image->rows;
                w = x1 - x0;
                h = y1 - y0;
                if(w <= 1 || h <= 1)
                {
                    *output_tensor_image = (*input_tensor_image)(cv::Rect(0, 0, 0, 0));
                    return vision_graph::kErrCodeParamInvalid;
                }
                cv::Mat cut_image;
                cut_image = (*input_tensor_image)(cv::Rect(x0, y0, x1 - x0, y1 - y0));
                output_tensor_image->push_back(cut_image);
               
                // *output_tensor_image = (*input_tensor_image)(cv::Rect(x0, y0, x1 - x0, y1 - y0));
            }
        }

        // int  x0 = (max_box.x1 >= 0.0f ? ((int)(max_box.x1 + 0.5f)) : ((int)(max_box.x1 - 0.5f)));
        // int  y0 = (max_box.y1 >= 0.0f ? ((int)(max_box.y1 + 0.5f)) : ((int)(max_box.y1 - 0.5f)));
        // int  x1 = (max_box.x2 >= 0.0f ? ((int)(max_box.x2 + 0.5f)) : ((int)(max_box.x2 - 0.5f)));
        // int  y1 = (max_box.y2 >= 0.0f ? ((int)(max_box.y2 + 0.5f)) : ((int)(max_box.y2 - 0.5f)));
        // int  w  = x1 - x0;
        // int  h  = y1 - y0;
        // if(x0 >= 0 && 
        //     y0 >= 0 && 
        //     x1 <= input_tensor_image->cols && 
        //     y1 <= input_tensor_image->rows)
        // {
        //     if(x0 >= input_tensor_image->cols || y0 >= input_tensor_image->rows || w <= 1 || h <= 1)
        //     {
        //         return vision_graph::kErrCodeParamInvalid;
        //     }
        //     *output_tensor_image = (*input_tensor_image)(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        // }
        // else
        // {
        //     if(true == padding_)
        //     {
        //         int  l_pad = 0, t_pad = 0, r_pad = 0, b_pad = 0;
        //         int  l_new = x0, t_new = y0, r_new = x1, b_new = y1;
        //         if(x0 < 0)
        //         {
        //             l_pad = -x0;
        //             l_new = l_pad;
        //         }
        //         if(y0 < 0)
        //         {
        //             t_pad = -y0;
        //             t_new = t_pad;
        //         }
        //         if(x1 > input_tensor_image->cols)
        //         {
        //             r_pad = x1 - input_tensor_image->cols;
        //             r_new = x1 + r_pad;
        //         }
        //         if(y1 > input_tensor_image->rows)
        //         {
        //             t_pad = y1 - input_tensor_image->rows;
        //             b_new = y1 + t_pad;
        //         }
        //         cv::Mat  padding_image = (*input_tensor_image);
        //         cv::copyMakeBorder(*input_tensor_image, padding_image, 
        //                             l_pad, t_pad, r_pad, t_pad,
        //                             cv::BORDER_CONSTANT);
        //         (*output_tensor_image) = padding_image(cv::Rect(l_new, t_new, r_new - l_new, b_new - t_new));
        //     }
        //     else
        //     {
        //         if(x0 < 0)
        //             x0 = 0;
        //         if(y0 < 0)
        //             y0 = 0;
        //         if(x1 > input_tensor_image->cols)
        //             x1 = input_tensor_image->cols;
        //         if(y1 > input_tensor_image->rows)
        //             y1 = input_tensor_image->rows;
        //         w = x1 - x0;
        //         h = y1 - y0;
        //         if(w <= 1 || h <= 1)
        //         {
        //             *output_tensor_image = (*input_tensor_image)(cv::Rect(0, 0, 0, 0));
        //             return vision_graph::kErrCodeParamInvalid;
        //         }
                
        //         *output_tensor_image = (*input_tensor_image)(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        //     }
        // }
    }
    return 0;
}

};//namespace vision_graph
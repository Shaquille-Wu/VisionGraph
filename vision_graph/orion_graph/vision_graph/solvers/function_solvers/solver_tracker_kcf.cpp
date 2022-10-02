#include <string.h>
#include "../../include/graph_error.h"
#include "../../common/utils.h"
#include "solver_tracker_kcf.h"
#include <logging.h>

namespace vision_graph{

SolverTrackerKCF::SolverTrackerKCF(nlohmann::json const& param) noexcept : Solver(param),                                
                                                                           has_target_(false),
                                                                           zoom_scale_(1.0f),
                                                                           track_side_(70),
                                                                           max_tracker_(4)
{
    track_side_                           = param.at("track_side");
    max_tracker_                          = param.at("max_tracker");
    for (int i = 0; i < max_tracker_; i++) {
        kcf_track::KCF_Tracker*      tracker_ = new kcf_track::KCF_Tracker();
        tracker_->m_use_scale                 = false;
        tracker_->m_use_color                 = false;
        tracker_->m_use_subpixel_localization = false;
        tracker_->m_use_subgrid_scale         = false;
        tracker_->m_use_multithreading        = false;
        tracker_->m_use_cnfeat                = false;
        tracker_->m_use_linearkernel          = false;
        tracker_->m_gray_feature              = true;
        tracker_->m_pca_feature               = false;
        tracker_->m_hog_feature               = false;
        pool.push_back(tracker_);
    }

    target_box_.cx = target_box_.cy = target_box_.w = target_box_.h = 0.0;
}

SolverTrackerKCF::~SolverTrackerKCF() noexcept
{
    for (kcf_track::KCF_Tracker* tracker_ : pool) {
        if(nullptr != tracker_)
            delete tracker_;
        tracker_ = nullptr;
    }
    
}

Tensor* SolverTrackerKCF::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    TensorBoxVector*  track_box = new TensorBoxVector;
    return track_box;
}

bool SolverTrackerKCF::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(in.size() != 2 && in.size() != 3)
        return false;
    if(in[0]->GetType() != kTensorInt32 && in[0]->GetType() != kTensorUInt32)
        return false;
    if(in[1]->GetType() != kTensorImage)
        return false;
    if(3 == in.size() && in[2]->GetType() != kTensorBoxVector)
        return false;
    if(out.size() <= 0 || out[0]->GetType() != kTensorBoxVector)
        return false;
    
    return true;
}

int SolverTrackerKCF::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{

    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }   
    const Tensor*        tensor_in0   = in[0];
    int                  value        = GetTensorNumericValue<int>(*tensor_in0);


    const Tensor*        tensor_in1   = in[1];
    const TensorImage*   tensor_image = dynamic_cast<const TensorImage*>(tensor_in1);
    bool                 is_init      = true;
    TensorBoxVector*     tensor_out   = dynamic_cast<TensorBoxVector*>(out[0]);
    TensorBoxVector      in_box ;
    bool                 image_empty  = false;
    bool                 box_empty    = false;
    bool                 update_track = true;
    tensor_out->clear();
    target_box_ = kcf_track::BBox_c::from_rect(cv::Rect(0, 0, 0, 0));
    if(tensor_image->cols <= 1 || tensor_image->rows <= 1)
    {
        image_empty = true;
        box_empty   = true;
    }

    // LOG(ERROR) << "SolverTrackerKCF start " << value;
    
    if(0 == value)
    {
        is_init = false;
    }
    else if(1 == value)
    {
        TensorBoxVector const* tensor_box = dynamic_cast<TensorBoxVector const*>(in[2]);
        if(true == tensor_box->empty())
            box_empty = true;
        else
            in_box  = *tensor_box;
    }
    std::vector<vision::Box> boxes;

    if(true == is_init)
    {
        // if(false == image_empty || false == box_empty)
        // {

            for (auto it = current.begin(); it != current.end();) {
                pool.push_back(it->second);
                it = current.erase(it);
            }

            for (vision::Box& b : in_box) {
                
                if (current.size() < max_tracker_) {
                    b.tid = current.size();
                    boxes.push_back(b);

                    kcf_track::KCF_Tracker* tracker = pool[0];
                    cv::Rect   init_box(0, 0, 0, 0);
                    float scale      = b.expandLongMaxToPixel(track_side_, tensor_image->cols, tensor_image->rows);
                    init_box.x       = FLT2INT32(b.x1);
                    init_box.y       = FLT2INT32(b.y1);
                    init_box.width   = FLT2INT32(b.x2 - b.x1);
                    init_box.height  = FLT2INT32(b.y2 - b.y1);
                    tracker->init(*tensor_image, init_box);
                    current.insert(std::make_pair(b.tid, tracker));
                    pool.erase(pool.begin());
                    zoom_scale_      = 1.0f / scale;
                    has_target_      = true;
                }
            }  
            // LOG(ERROR) << "SolverTrackerKCF kcf  current " << current.size()<<" pool "<<pool.size();

        // }
        // else
        // {
        //     has_target_  = false;
        //     update_track = false;
        // }
    }
    else
    {
        if(false == image_empty)
        {
            // if (true == has_target_) {
                for (auto it = current.begin(); it != current.end(); ++it) {
                    it->second->track(*tensor_image);
                    target_box_ = it->second->getBBox();
                    vision::Box b;
                    b.x1 = (float)(target_box_.cx - 0.5 * target_box_.w);
                    b.y1 = (float)(target_box_.cy - 0.5 * target_box_.h);
                    b.x2 = (float)(b.x1 + target_box_.w);
                    b.y2 = (float)(b.y1 + target_box_.h);
                    b.tid = it->first;
                    b.expandByLongSide(zoom_scale_, tensor_image->cols, tensor_image->rows);
                    boxes.push_back(b);
                }
            // }
            // else
            //     update_track = false;
        }
        else
            update_track = false;
    }

    // if (true == has_target_ && true == update_track) {
        // for (auto it = current.begin(); it != current.end(); ++it) {
        //     target_box_ = it->second->getBBox();
        //     vision::Box b;
        //     b.x1 = (float)(target_box_.cx - 0.5 * target_box_.w);
        //     b.y1 = (float)(target_box_.cy - 0.5 * target_box_.h);
        //     b.x2 = (float)(target_box_.cx + target_box_.w);
        //     b.y2 = (float)(target_box_.cy + target_box_.h);
        //     b.expandByLongSide(zoom_scale_, tensor_image->cols, tensor_image->rows);
        //     boxes.push_back(b);
        // }
        // LOG(ERROR) << "SolverTrackerKCF  boxes "<<boxes.size();

        (*tensor_out) = boxes; 
    // }
    // target_box_ = tracker_->getBBox();
    
    // (*tensor_out).x1 = (float)(target_box_.cx - 0.5 * target_box_.w);
    // (*tensor_out).y1 = (float)(target_box_.cy - 0.5 * target_box_.h);
    // (*tensor_out).x2 = (float)((*tensor_out).x1 + target_box_.w);
    // (*tensor_out).y2 = (float)((*tensor_out).y1 + target_box_.h);
    // (*tensor_out).expandByLongSide(zoom_scale_, tensor_image->cols, tensor_image->rows);
    //LOG(ERROR) << "SolverTrackerKCF end";

    return 0;
}

};//namespace vision_graph
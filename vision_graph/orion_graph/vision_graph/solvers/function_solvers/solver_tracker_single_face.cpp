#include "solver_tracker_single_face.h"

#include <logging.h>
#include <string.h>

#include "../../common/utils.h"
#include "../../include/graph_error.h"

#define DESTROY(ptr)          \
    {                         \
        if (ptr != nullptr) { \
            delete ptr;       \
            ptr = nullptr;    \
        }                     \
    }
namespace vision_graph {

SolverTrackerSingleFace::SolverTrackerSingleFace(nlohmann::json const& param) noexcept : Solver(param) {
    if (true == param.contains("eco_dlcv_json")) {
        std::string eco_dlcv_json_file = param.at("eco_dlcv_json");
        if (feat_factory == nullptr) {
            feat_factory = create_feat_factory_cnn(eco_dlcv_json_file, std::vector<int>{10});
        }
    }

    if (true == param.contains("reid_dlcv_json")) {
        std::string reid_dlcv_json_file = param.at("reid_dlcv_json");
        if (face_verifier == nullptr) {
            face_verifier = get_face_verifier(reid_dlcv_json_file);
        }
    }
}

SolverTrackerSingleFace::~SolverTrackerSingleFace() noexcept {
    release_feat_factory(feat_factory);
    DESTROY(fp_detector);
    DESTROY(face_verifier);
    DESTROY(face_keypoints);
}

Tensor* SolverTrackerSingleFace::CreateOutTensor(int out_tensor_idx) noexcept {
    Tensor* tensor = nullptr;
    if (0 == out_tensor_idx)
        tensor = new TensorBox;
    else if (1 == out_tensor_idx)
        tensor = new TensorKeypoints;
    return tensor;
}

bool SolverTrackerSingleFace::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept {
    if (in.size() != 2 && in.size() != 3)
        return false;
    if (in[0]->GetType() != kTensorInt32)
        return false;
    if (in[1]->GetType() != kTensorImage)
        return false;
    if (3 == in.size() && in[2]->GetType() != kTensorBox)
        return false;
    if (out.size() <= 1 || out[0]->GetType() != kTensorBox || out[1]->GetType() != kTensorKeyPoints)
        return false;

    return true;
}

cv::Rect getCopyMakeBorderMat(const cv::Mat& in, Box box, cv::Mat& out, int& left, int& top) {
    cv::Rect box_rect(static_cast<int>(box.x1), static_cast<int>(box.y1),
                      static_cast<int>(box.x2 - box.x1),
                      static_cast<int>(box.y2 - box.y1));

    cv::Mat bgr_copy = in.clone();
    if (box.y2 > in.rows)
        cv::copyMakeBorder(bgr_copy, bgr_copy, 0, (int)(box.y2 - in.rows) + 1, 0, 0,
                           cv::BORDER_CONSTANT);
    if (box.x2 > in.cols)
        cv::copyMakeBorder(bgr_copy, bgr_copy, 0, 0, 0, (int)(box.x2 - in.cols) + 1,
                           cv::BORDER_CONSTANT);

    if (box.x1 < 0) {
        left = abs(box.x1) + 1;
        cv::copyMakeBorder(bgr_copy, bgr_copy, 0, 0, abs(box.x1) + 1, 0,
                           cv::BORDER_CONSTANT);
        box_rect.x = 0;
    }

    if (box.y1 < 0) {
        top = abs(box.y1) + 1;
        cv::copyMakeBorder(bgr_copy, bgr_copy, abs(box.y1) + 1, 0, 0, 0,
                           cv::BORDER_CONSTANT);
        box_rect.y = 0;
    }

    out = bgr_copy(box_rect);
    return box_rect;
}

int SolverTrackerSingleFace::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept {
    if (false == CheckTensor(in, out)) {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }

    std::vector<Solver*> friends_ = Solver::friends_;
    Solver* detect_solver = friends_[0];
    Solver* fkp_solver = friends_[1];
    if (fp_detector == nullptr) {
        fp_detector = get_fp_detector(detect_solver, 320, 320);
    }

    if (face_keypoints == nullptr) {
        face_keypoints = get_face_keypoints_regressor(fkp_solver);
    }
    // LOG(ERROR) << "wyb debug face_tracker 0";

    if (face_tracker == nullptr) {
        const float face_reid_thresh = 1.0f;
        const float face_reid_conf_thresh = 1.0f;
        // LOG(ERROR) << "wyb debug face_tracker 1";
        face_tracker = new FaceEcoTrackStrategy(feat_factory, fp_detector,
                                                face_verifier, face_keypoints, 1.f,
                                                face_reid_thresh,
                                                face_reid_conf_thresh, 10, 5);
        // LOG(ERROR) << "wyb debug face_tracker 2";
    }
    // LOG(ERROR) << "wyb debug face_tracker 3";

    const Tensor* tensor_in0 = in[0];
    const TensorInt32* tensor_Int32 = dynamic_cast<const TensorInt32*>(tensor_in0);
    int value = tensor_Int32->CastValue<int>();

    const Tensor* tensor_in1 = in[1];
    const TensorImage* tensor_image = dynamic_cast<const TensorImage*>(tensor_in1);

    vision_graph::TensorBox* output_tensor_box = dynamic_cast<vision_graph::TensorBox*>(out[0]);
    vision_graph::TensorKeypoints* output_tensor_fkp = dynamic_cast<vision_graph::TensorKeypoints*>(out[1]);

    output_tensor_box->clear();
    if (value == 0) {
        TensorBox const* tensor_box = dynamic_cast<TensorBox const*>(in[2]);
        cv::Rect rect_face(tensor_box->x1, tensor_box->y1, tensor_box->width(), tensor_box->height());
        face_tracker->reset();
        face_tracker->init(*tensor_image, &rect_face);
    }
    if (value == 1) {
        face_tracker->track(*tensor_image);

        TrackingStatus status_face = face_tracker->get_target_status();

        if (status_face == TRACKING_SUCCESS) {
            cv::Rect rec = face_tracker->get_target_pos();
            vision::Box face_Box;
            face_Box.x1 = rec.x;
            face_Box.y1 = rec.y;
            face_Box.x2 = rec.x + rec.width;
            face_Box.y2 = rec.y + rec.height;

            cv::Mat keypoint_in;
            int left = 0;
            int top = 0;
            rec = getCopyMakeBorderMat(*tensor_image, face_Box, keypoint_in, left, top);
            float mask_score = 0.f;
            std::vector<cv::Point2f> points;
            RPYPose pose;
            face_keypoints->compute_keypoints_pose(keypoint_in, points, pose);
            for (int i = 0; i < 106; i++) {
                points[i].x = (points[i].x + rec.x - left);
                points[i].y = (points[i].y + rec.y - top);
            }
            (*output_tensor_fkp) = points;
            (*output_tensor_box) = face_Box;
        }
    }

    return 0;
}

};  //namespace vision_graph
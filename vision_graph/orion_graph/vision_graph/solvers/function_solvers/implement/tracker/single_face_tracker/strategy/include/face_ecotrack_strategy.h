//
//
//

#ifndef GRAPH_ROBOT_TRACKING_FACE_ECOTRACK_STRATEGY_H
#define GRAPH_ROBOT_TRACKING_FACE_ECOTRACK_STRATEGY_H

#include "FacialLandmarks.h"
#include "Histogram.h"
#include "base_strategy.h"
#include "eco_tracker.h"
#include "infer_factories.h"
#include <timer.h>

class FaceEcoTrackStrategy : public BaseStrategy {
public:
    FaceEcoTrackStrategy();

    FaceEcoTrackStrategy(FeatureFactory *feat_factory,
                         BaseDetector *fp_detector,
                         BaseVerifier *face_verifier,
                         BaseKeypoints *face_keypoints,
                         float track_input_sz_scale = 1.f,
                         float face_reg_thresh = 0.8f,
                         float face_conf_thresh = 0.8f,
                         int reid_frame_intv = 10,
                         int retry_thresh = 5);

    virtual ~FaceEcoTrackStrategy();

    int init(const cv::Mat &img, cv::Rect *face_rect, cv::Rect *body_rect = NULL);

    void reset();

    void track(const cv::Mat &img);

    RPYPose get_facepose();

private:
    cv::Mat get_tracker_input(const cv::Mat &img);

    void init_tracker(const cv::Mat &img, cv::Rect rect);

    void create_tracker();

    void destroy_tracker();

    cv::Rect get_tracker_rect();

    void compute_detface_keypoints(const cv::Mat &img);

    void face_det_reid(const cv::Mat &img);

    FaceDirection check_face_direction(const std::vector<cv::Point2f> &kpts, const RPYPose &pose);

    FaceDirection check_sideface_extra(const cv::Mat &face_img, const std::vector<cv::Point2f> &kpts);

    void combine_reid_with(const cv::Mat &img, ObjStatus eco_status, bool eco_conf_high);

    void combine_eco_with_rule(const cv::Mat &img, cv::Rect track_rect);

    void reid_check(const cv::Mat &img, cv::Rect track_rect);

    void set_facepose(const RPYPose &pose);

    void set_facepose(float roll, float pitch, float yaw);

public:
    FeatureFactory *ptr_feat_factory;
    EcoTracker *ptr_eco_tracker;
    BaseDetector *ptr_fp_detector;
    BaseVerifier *ptr_face_verifier;
    BaseKeypoints *ptr_face_keypoints;
    std::vector<vision::Box> face_boxs;
    float face_thresh;
    float face_reliable_thresh;
    std::vector<std::vector<float>> face_templates;

    TrackingStatus face_status;
    BBox face_bbox;
    std::vector<BBox> det_face_bboxes;
    std::vector<std::vector<cv::Point2f>> kpts_vecs;
    std::vector<RPYPose> pose_vec;

    RPYPose face_pose;
    bool face_pose_checked;
    CFacialLandmarks extra_sideface_cls;

    // tracker_input_scale = tracker_input / ori_img
    float tracker_input_scale;

    int frame_count;
    int reid_interval;
    int reid_retry_thresh;
    int reid_retry_count;
    int frames_since_reid;

    TrackingAction tracking_action;
    HighClock clock;
};

#endif  //ROBOT_TRACKING_FACE_TRACK_STRATEGY_H

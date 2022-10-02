/////////////////////////////////////////////////
////created by anwenju 20180206
/////////////////////////////////////////////////
#ifndef GRAPH_ROBOT_TRACKING_LEAD_STRATEGY_H
#define GRAPH_ROBOT_TRACKING_LEAD_STRATEGY_H

#include"base_strategy.h"
#include"feature_factory.h"
#include"eco_tracker.h"
#include "infer_factories.h"

class LeadStrategy: public BaseStrategy
{
public:
    LeadStrategy();
    
    LeadStrategy(FeatureFactory *feat_factory,
                 BaseDetector *fp_detector,
                 BaseVerifier *face_verifier,
                 BaseVerifier *body_verifier,
                 BaseKeypoints *face_keypointer,
                 float tracker_input_sz_scale = 1.f,
                 float face_reg_thresh = 0.6,
                 float face_conf_thresh = 0.6,
                 float body_reg_thresh = 0.9,
                 float body_conf_thresh = 0.9,
                 int reid_frame_intv = 10);
   
     virtual ~LeadStrategy();
 
     int init(const cv::Mat &img, cv::Rect *face_rect, cv::Rect *body_rect);

     void reset();

     void track(const cv::Mat &img);

private:     
     cv::Mat get_tracker_input(const cv::Mat &img);
    
     void init_tracker(const cv::Mat & img, cv::Rect rect);
    
     void destroy_tracker();
   
     cv::Rect get_tracker_rect();
     
public:
    FeatureFactory *ptr_feat_factory;
    EcoTracker *ptr_eco_tracker;

    BaseDetector *ptr_fp_detector;
    BaseVerifier *ptr_face_verifier;
    BaseVerifier *ptr_body_verifier;
    BaseKeypoints *ptr_face_keypointer;

    TrackingStatus face_status;
    TrackingStatus body_status;

    std::vector<std::vector<float>> face_templates;
    std::vector<std::vector<float>> body_templates;

    BBox face_bbox;
    BBox body_bbox;
    std::vector<BBox> det_face_bboxes;
    std::vector<BBox> det_body_bboxes;
  
    float face_thresh;
    float face_reliable_thresh;
    float body_thresh;
    float body_reliable_thresh;

    float tracker_input_scale;
    
    int frame_count;
    int reid_interval;
    
    TrackingAction tracking_action;
};

#endif 

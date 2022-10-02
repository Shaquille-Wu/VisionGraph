#ifndef GRAPH_FACEKEYPOINT_H
#define GRAPH_FACEKEYPOINT_H

#include <CComputSim.h>
#include <FacialLandmarks.h>

#include <opencv2/opencv.hpp>
#include <vector>

#include "base_keypoints.h"
#include "keypoints_attributes.h"
#include "vision_graph.h"

#include "logging.h"
using namespace vision;

class InferenceKeypoint : public BaseKeypoints {
public:
    InferenceKeypoint();
    InferenceKeypoint(vision_graph::Solver* solver);
    ~InferenceKeypoint();
    void compute_keypoints(const cv::Mat& img, std::vector<cv::Point2f>& kpts);
    void compute_keypoints_pose(const cv::Mat& img, std::vector<cv::Point2f>& kpts, RPYPose& pose);
    void compute_keypoints_pose_hog(const cv::Mat& img, std::vector<cv::Point2f>& kpts, RPYPose& pose, float& hog_score);

private:
    // vision::KeypointsAttributes* keypoint_ptr = NULL;
    vision_graph::Solver* keypoint_ptr = NULL;
    vision_graph::TensorImageVector*        tensor_image = NULL;
    vision_graph::TensorKeypointsVector*     tensor_fkp  = NULL;
    vision_graph::TensorAttributes*     tensor_att       = NULL;
};

#endif  //GRAPH_FACEKEYPOINT_H

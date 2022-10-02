//
// Created by yuan on 18-4-13.
//

#ifndef ROBOT_TRACKING_BASE_KEYPOINTS_H
#define ROBOT_TRACKING_BASE_KEYPOINTS_H

#include <opencv2/opencv.hpp>

#include "rpy_pose.h"

class BaseKeypoints {
public:
    BaseKeypoints() {}

    virtual ~BaseKeypoints() {}

    virtual void compute_keypoints(const cv::Mat& img, std::vector<cv::Point2f>& kpts) = 0;

    virtual void compute_keypoints_pose(const cv::Mat& img, std::vector<cv::Point2f>& kpts, RPYPose& pose) {}

    virtual void compute_keypoints_pose_hog(const cv::Mat& img, std::vector<cv::Point2f>& kpts, RPYPose& pose, float& hog_score) {}
};

#endif  //ROBOT_TRACKING_BASE_KEYPOINTS_H

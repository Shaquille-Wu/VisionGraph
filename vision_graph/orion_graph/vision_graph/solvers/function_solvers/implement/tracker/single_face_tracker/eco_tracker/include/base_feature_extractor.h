//
// Created by yuan on 17-9-19.
//

#ifndef ROBOT_TRACKING_BASE_FEATURE_EXTRACTOR_H
#define ROBOT_TRACKING_BASE_FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "data_type.h"


typedef Vec1dMat Feature;
typedef Vec2dMat Sample;

class BaseFeatureExtractor
{
public:
    BaseFeatureExtractor() {}
    virtual ~BaseFeatureExtractor() {}
    virtual void extract_feature(const cv::Mat& img, Sample& out_feature) = 0;
    virtual cv::Size get_image_support_sz(cv::Size2f new_sample_sz, float scale) = 0;

public:
    cv::Size input_size;
    int output_blocks;
};



#endif //ROBOT_TRACKING_BASE_FEATURE_EXTRACTOR_H

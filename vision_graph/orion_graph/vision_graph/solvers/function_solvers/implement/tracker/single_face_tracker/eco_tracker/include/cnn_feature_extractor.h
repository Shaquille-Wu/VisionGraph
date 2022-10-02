//
// Created by yuan on 17-9-19.
//

#ifndef ROBOT_TRACKING_CNN_FEATURE_EXTRACTOR_H
#define ROBOT_TRACKING_CNN_FEATURE_EXTRACTOR_H

#include <string>

#include "base_feature_extractor.h"
#include "featuremap_runner.h"
using namespace vision;

class CnnFeatureExtractor : public BaseFeatureExtractor {
public:
    CnnFeatureExtractor();

    CnnFeatureExtractor(const std::string &pFile);

    ~CnnFeatureExtractor();

    void extract_feature(const cv::Mat &img, Sample &out_feature);

    // deprecated
    cv::Size get_image_support_sz(cv::Size2f new_sample_sz, float scale);

private:
    const int out_channel = 64;
    const int out_H = 40;
    const int out_W = 40;

    vision::FeaturemapRunner *cnn_featuremap = NULL;
};

#endif  //ROBOT_TRACKING_CNN_FEATURE_EXTRACTOR_H

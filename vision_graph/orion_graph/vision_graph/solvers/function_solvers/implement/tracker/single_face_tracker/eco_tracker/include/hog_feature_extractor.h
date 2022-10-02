//
// Created by yuan on 17-9-19.
//

#ifndef ROBOT_TRACKING_HOG_FEATURE_EXTRACTOR_H
#define ROBOT_TRACKING_HOG_FEATURE_EXTRACTOR_H

#include "base_feature_extractor.h"
#include "fhog.h"

class HogFeatureExtractor : public BaseFeatureExtractor
{
public:
    HogFeatureExtractor(int bin=4, int orients_num=9, float clip_value = 0.2f, bool crop = false);

    ~HogFeatureExtractor();

    void extract_feature(const cv::Mat& img, Sample& out_feature);
    void extract_feature(const cv::Mat& img, Feature& out_feature);

    cv::Size get_image_support_sz(cv::Size2f new_sample_sz, float scale);

public:
    int bin_size;
    int orients;
    float clip;
    bool is_crop;
};

#endif //ROBOT_TRACKING_HOG_FEATURE_EXTRACTOR_H

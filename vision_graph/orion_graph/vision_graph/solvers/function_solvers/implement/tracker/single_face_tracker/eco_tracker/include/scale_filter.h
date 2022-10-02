//
// Created by yuan on 17-11-8.
//

#ifndef ROBOT_TRACKING_SCALE_FILTER_H
#define ROBOT_TRACKING_SCALE_FILTER_H

#include "data_type.h"
#include "hog_feature_extractor.h"

class ScaleFilter
{
public:
    ScaleFilter(float sigma = 1 / 16.f, float lrt = 0.025f, int max_patch_area = 32 * 16, int n_filters = 17,
                int n_interp_scales = 33, float scale_mdl_coef = 1.f, float scale_step = 1.02f, float reg_coef = 0.01f);

    virtual ~ScaleFilter();
    ScaleFilter& operator=(const ScaleFilter &src_filter);

    void init(cv::Size init_target_sz);

    float track(const cv::Mat& img, cv::Point2f pos, cv::Size2f base_target_sz, float current_scale_factor);

    void update(const cv::Mat& img, cv::Point2f pos, cv::Size2f base_target_sz, float current_scale_factor);

private:
    cv::Mat extract_sample(const cv::Mat& img, cv::Point2f pos, cv::Size2f base_target_sz, const cv::Mat& scale_factors);

    void feature_proj_scale(const cv::Mat& src, const cv::Mat& proj_mat, cv::Mat& dst);

public:
    float sigma_factor; // 1 / 16
    float learning_rate; // 0.025

    int max_sample_area; // 32 * 16;
    int num_filters; // 17
    int num_interp_scales; //33
//    int num_compressed_dim;
    float scale_model_factor; // 1.0
    cv::Size scale_model_size;
    float filter_step; // 1.02
    float reg_factor; // 1e-2

    cv::Mat scale_size_factors;
    cv::Mat interp_scale_factors;
    cv::Mat yf;
    cv::Mat cos_window;
    cv::Mat basis;
    cv::Mat sf_num;
    cv::Mat sf_den;
    cv::Mat ss_num;

    HogFeatureExtractor hog_extractor;
};

#endif //ROBOT_TRACKING_SCALE_FILTER_H

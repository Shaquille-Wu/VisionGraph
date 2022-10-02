//
// Created by yuan on 17-9-21.
//

#ifndef ROBOT_TRACKING_FEATURE_COMPRESS_H
#define ROBOT_TRACKING_FEATURE_COMPRESS_H

#include "base_feature_extractor.h"
#include "data_type.h"

class FeatureFactory
{
public:
    FeatureFactory(const std::vector<BaseFeatureExtractor*>& ptr_feat_makers,
                   const std::vector<int>& compressed_channels);

    ~FeatureFactory();

    void make_id_convert_table();

    void extract(const cv::Mat& img, Sample& out_feats);

    void normalize_feature(const Feature& in_feats, Feature& out_feats);

    void normalize_feature(const Sample& in_feats, Sample& out_feats);

    cv::Mat compute_project_mat(const Feature& feat, int target_channel);

    void compute_project_mats(const Sample& sample);

    void compute_project_mats_each(const Sample& sample, int extractor_id);

    void modify_project_mats(const Vec1dMat& proj_mat_inc);

    void modify_project_mats_each(const Vec1dMat &proj_mat_inc, int extractor_id);

    void modify_project_mat(const cv::Mat& proj_mat_inc, int feat_id);

    void compress(const Feature& in_feat, const cv::Mat& proj_mat, int target_channel, Feature& out_feat);

    void compress_all(const Sample& in_feats, Sample& out_feats);

    void compress_each(const Feature& in_feat, int feat_id, Feature& out_feat);

    void compress_each(const Sample& in_feats, int extractor_id, Sample& out_feats);

    void flatten_feature(const Feature& feat,
                         cv::Mat& feat_flat,
                         int& height, int& width, int& channels);

    void restore_flatten_feature(const cv::Mat& feat_flat,
                                 int height, int width, int channels,
                                 Feature& out_feat);

    cv::Size get_image_support_sz(cv::Size2f new_sample_sz, float scale);

public:
    std::vector<BaseFeatureExtractor*> ptr_extractors;

    // global feature parameters
    int normalize_size;
    int normalize_dim;

    // feature dims reduction
    std::vector<int> target_channels;
    std::vector<cv::Mat> project_mats;
    std::map<int, std::vector<int>> id_convert_table;
    int num_feat_blocks;
};

#endif //ROBOT_TRACKING_FEATURE_COMPRESS_H

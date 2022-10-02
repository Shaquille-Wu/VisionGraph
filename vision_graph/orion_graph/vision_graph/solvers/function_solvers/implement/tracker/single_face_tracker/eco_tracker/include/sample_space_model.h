//
// Created by yuan on 17-9-28.
//

#ifndef ROBOT_TRACKING_SAMPLE_SPACE_MODEL_H
#define ROBOT_TRACKING_SAMPLE_SPACE_MODEL_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "base_feature_extractor.h"


class SampleSpaceModel
{
public:
    SampleSpaceModel();

    SampleSpaceModel(int capacity, std::vector<cv::Size>& filter_sz, std::vector<int>& feat_channels, float lrt = 0.009f);

    SampleSpaceModel(const SampleSpaceModel& src_model);

    SampleSpaceModel(const SampleSpaceModel&& src_model);

    SampleSpaceModel& operator=(const SampleSpaceModel& src_model);

    SampleSpaceModel& operator=(const SampleSpaceModel&& src_model);

    virtual ~SampleSpaceModel();

    void update(Sample& new_sample, std::vector<int>& changed_ids);

    void force_correct(Sample& new_sample, float new_weight, std::vector<int>& changed_ids);

    void apply_changes(const SampleSpaceModel& src, const std::vector<int>& changed_ids);

    void init();

    void get_sample(int ind, Sample& sample);

    void insert_sample(Sample& sample, int ind);

private:

    void find_gram_vector(const Sample& new_sample, cv::Mat& gram_vec);

    void update_distance_matrix(const cv::Mat& gram_vec, float new_sample_norm, int id1, int id2, float weight1, float weight2);

    void merge_samples(const Sample& s1, float weight1, const Sample& s2, float weight2, Sample& out_s);

    void decay_sample_weights();

    void normalize_sample_weights();

    void save_changed_ids(int id, std::vector<int>& changed_ids);

public:
    int sample_capacity;
    int sample_num;
    int feat_block_num;
    std::vector<cv::Size> filter_size;
    std::vector<int> feat_channels;
    // dim1: feature index; dim2: sample index; dim3: feature channel index.
    Vec3dMat sample_space;
    cv::Mat distance_mat;
    cv::Mat gram_mat;
    std::vector<float> sample_weights;
    double min_weight_thresh;
    float learning_rate; // 0.009
};

#endif //ROBOT_TRACKING_SAMPLE_SPACE_MODEL_H

//
// Created by yuan on 17-9-29.
//

#include "sample_space_model.h"
#include "utils.h"
#include "complexmat.h"
#include <numeric>

SampleSpaceModel::SampleSpaceModel() {}


SampleSpaceModel::SampleSpaceModel(int capacity, std::vector<cv::Size>& filter_sz,
                                   std::vector<int>& feat_channels, float lrt)
{
    this->sample_capacity = capacity;
    this->learning_rate = lrt;
    this->filter_size = filter_sz;
    this->feat_channels = feat_channels;
    this->init();
}


SampleSpaceModel::~SampleSpaceModel() {}


void SampleSpaceModel::init()
{
    this->gram_mat = cv::Mat(sample_capacity, sample_capacity, CV_32FC1, cv::Scalar(FLOAT_INF));
    this->distance_mat = cv::Mat(sample_capacity, sample_capacity, CV_32FC1, cv::Scalar(FLOAT_INF));

    this->feat_block_num = this->filter_size.size();
    this->sample_space.resize(feat_block_num);
    for (int i = 0; i < feat_block_num; ++i)
    {
        this->sample_space[i].resize(sample_capacity);
        cv::Size sample_sz((filter_size[i].width + 1) / 2, filter_size[i].height);
        for (int j = 0; j < sample_capacity; ++j)
        {
            init_vec1dmat(feat_channels[i], sample_sz, CV_32FC2, this->sample_space[i][j]);
        }
    }

    this->sample_num = 0;
    this->sample_weights = std::vector<float>(sample_capacity, 0);
    // Find the minimum allowed sample weight. Samples are discarded if their weights become lower
    this->min_weight_thresh = learning_rate * pow(1 - learning_rate, 2 * sample_capacity);
}

SampleSpaceModel::SampleSpaceModel(const SampleSpaceModel &src_model)
{
    *this = src_model;
}

SampleSpaceModel::SampleSpaceModel(const SampleSpaceModel &&src_model)
{
    *this = src_model;
}

SampleSpaceModel& SampleSpaceModel::operator=(const SampleSpaceModel &src_model)
{
    if(this != &src_model)
    {
        this->sample_capacity = src_model.sample_capacity;
        this->sample_weights = src_model.sample_weights;
        this->sample_num = src_model.sample_num;
        this->min_weight_thresh = src_model.min_weight_thresh;
        this->learning_rate = src_model.learning_rate;

        this->feat_block_num = src_model.feat_block_num;
        this->filter_size = src_model.filter_size;
        this->feat_channels = src_model.feat_channels;

        this->distance_mat = src_model.distance_mat.clone();
        this->gram_mat = src_model.gram_mat.clone();

        auto mat_copy = [](const cv::Mat& src, cv::Mat& dst){dst = src.clone();};
        proc_batch_vec3dmat(mat_copy, src_model.sample_space, this->sample_space);
    }
    return *this;
}


SampleSpaceModel& SampleSpaceModel::operator=(const SampleSpaceModel && src_model)
{
    this->sample_capacity = src_model.sample_capacity;
    this->sample_weights = src_model.sample_weights;
    this->sample_num = src_model.sample_num;
    this->min_weight_thresh = src_model.min_weight_thresh;
    this->learning_rate = src_model.learning_rate;

    this->feat_block_num = src_model.feat_block_num;
    this->filter_size = src_model.filter_size;
    this->feat_channels = src_model.feat_channels;

    this->distance_mat = src_model.distance_mat;
    this->gram_mat = src_model.gram_mat;
    this->sample_space = src_model.sample_space;

    return *this;
}


void SampleSpaceModel::insert_sample(Sample &sample, int ind)
{
    for (int i = 0; i < feat_block_num; ++i)
    {
        for (int j = 0; j < sample[i].size(); ++j)
        {
            sample_space[i][ind][j] = sample[i][j];
        }
    }
}


void SampleSpaceModel::get_sample(int ind, Sample &sample)
{
    sample.resize(feat_block_num);
    for (int i = 0; i < feat_block_num; ++i)
    {
        auto channels = sample_space[i][ind].size();
        sample[i].resize(channels);
        for (int j = 0; j < channels; ++j)
        {
            sample[i][j] = sample_space[i][ind][j];
        }
    }
}


/*
 * Find the inner product of the new sample with the existing samples.
 * To be used for distance calculation.
 * */
void SampleSpaceModel::find_gram_vector(const Sample& new_sample, cv::Mat& gram_vec)
{
    gram_vec = cv::Mat(sample_capacity, 1, CV_32FC1, FLOAT_INF); 
//    Sample exist_sample;
    auto feat_block_num = new_sample.size();
    std::vector<std::vector<ComplexMat>> new_sample_cmats;
    new_sample_cmats.resize(feat_block_num);
    for (int i = 0; i < feat_block_num; ++i)
    {
        auto channels = new_sample[i].size();
        new_sample_cmats[i].resize(channels);
        for (int j = 0; j < channels; ++j)
        {
            ComplexMat feat_cmat(new_sample[i][j]);
            new_sample_cmats[i][j] = feat_cmat.conj();
        }
    }

#pragma omp parallel for
    for (int i = 0; i < sample_num; ++i)
    {
        Sample exist_sample;
        //get_sample do not change 
        get_sample(i, exist_sample);

        float gram = 0;
        for (int j = 0; j < feat_block_num; ++j)
        {
            const Vec1dMat& exist_feat = exist_sample[j];
            auto channels = exist_feat.size();
            assert(channels == new_sample_cmats[j].size());
            for (int k = 0; k < channels; ++k)
            {
                ComplexMat exist_cmat(exist_feat[k]);
                gram += 2 * exist_cmat.dot(new_sample_cmats[j][k]).real();
            }
        }
        gram_vec.at<float>(i, 0) = gram;
    }
}


void SampleSpaceModel::merge_samples(const Sample &s1, float weight1, const Sample &s2, float weight2, Sample &out_s)
{
    auto alpha1 = weight1 / (weight1 + weight2);
    auto alpha2 = 1.f - alpha1;
    auto feat_block_num = s1.size();
    Sample s_mg;
    s_mg.resize(feat_block_num);
    for (int i = 0; i < feat_block_num; ++i)
    {
        auto channels = s1[i].size();
        s_mg[i].resize(channels);
        for (int j = 0; j < channels; ++j)
        {
            s_mg[i][j] = s1[i][j] * alpha1 + s2[i][j] * alpha2;
        }
    }
    out_s = std::move(s_mg);
//    out_s = s_mg;
}


void SampleSpaceModel::update_distance_matrix(const cv::Mat &new_gram_vec, float new_sample_norm,
                                              int id1, int id2, float weight1, float weight2)
{
    cv::Mat gram_vec = new_gram_vec.clone();
    auto alpha1 = weight1 / (weight1 + weight2);
    auto alpha2 = 1.f - alpha1;

    cv::Mat gram_col1 = gram_mat.col(id1); // slicing
    cv::Mat gram_row1 = gram_mat.row(id1); // slicing

    cv::Mat dist_col1 = distance_mat.col(id1); //slicing
    cv::Mat dist_row1 = distance_mat.row(id1); //slicing

    if(id2 < 0)
    {
        float norm_id1 = gram_mat.at<float>(id1, id1);
        // update the gram matrix
        if(alpha1 == 0.f)
        {
            // the new sample replaces an existing sample
            gram_vec.copyTo(gram_col1);
            cv::Mat gram_col1_t = gram_col1.t();
            gram_col1_t.copyTo(gram_row1);
            gram_mat.at<float>(id1, id1) = new_sample_norm;
        }
        else if(alpha2 == 0.f)
        {
            // the new sample is discarded
        }
        else
        {
            // the new sample is merged with an existing sample
            cv::Mat gram_merged = gram_col1 * alpha1 + gram_vec * alpha2;
            gram_merged.copyTo(gram_col1);
            cv::Mat gram_col1_t = gram_col1.t();
            gram_col1_t.copyTo(gram_row1);
            gram_mat.at<float>(id1, id1) = alpha1 * alpha1 * norm_id1 + alpha2 * alpha2 * new_sample_norm +\
                                            2 * alpha1 * alpha2 * gram_vec.at<float>(id1, 0);

        }

        // update distance matrix
        cv::Mat dist_vec(sample_capacity, 1, CV_32FC1, cv::Scalar(FLOAT_INF));
        for (int i = 0; i < sample_capacity; ++i)
        {
            dist_vec.at<float>(i, 0) = std::max(0.f, gram_mat.at<float>(id1, id1) + gram_mat.at<float>(i, i) - 2 * gram_col1.at<float>(i, 0));
        }

        dist_vec.copyTo(dist_col1);
        cv::Mat dist_col1_t = dist_col1.t();
        dist_col1_t.copyTo(dist_row1);
        distance_mat.at<float>(id1, id1) = FLOAT_INF;
    }
    else
    {
        assert(alpha1 > 0.f && alpha2 > 0.f);

        cv::Mat gram_col2 = gram_mat.col(id2); // slicing
        cv::Mat gram_row2 = gram_mat.row(id2); // slicing

        cv::Mat dist_col2 = distance_mat.col(id2); //slicing
        cv::Mat dist_row2 = distance_mat.row(id2); //slicing

        //  Two existing samples are merged and the new sample fills the empty slot
        float norm_id1 = gram_mat.at<float>(id1, id1);
        float norm_id2 = gram_mat.at<float>(id2, id2);
        float ip_id1_id2 = gram_mat.at<float>(id1, id2);

        // Handle the merge of existing samples
        cv::Mat gram_merged = gram_col1 * alpha1 + gram_col2 * alpha2;
        gram_merged.copyTo(gram_col1);
        cv::Mat gram_col1_t = gram_col1.t();
        gram_col1_t.copyTo(gram_row1);
        gram_mat.at<float>(id1, id1) = alpha1 * alpha1 * norm_id1 + alpha2 * alpha2 * norm_id2 \
                                        + 2 * alpha1 * alpha2 * ip_id1_id2;
        gram_vec.at<float>(id1, 0) = gram_vec.at<float>(id1, 0) * alpha1 + gram_vec.at<float>(id2, 0) * alpha2;

        // Handle the new sample
        gram_vec.copyTo(gram_col2);
        cv::Mat gram_col2_t = gram_col2.t();
        gram_col2_t.copyTo(gram_row2);
        gram_mat.at<float>(id2, id2) = new_sample_norm;

        // update the distance matrix
        cv::Mat dist_vec1(sample_capacity, 1, CV_32FC1, cv::Scalar(FLOAT_INF));
        for (int i = 0; i < sample_capacity; ++i)
        {
            dist_vec1.at<float>(i, 0) = std::max(0.f, gram_mat.at<float>(id1, id1) + gram_mat.at<float>(i, i) - \
                                             2 * gram_col1.at<float>(i, 0));
        }

        dist_vec1.copyTo(dist_col1);
        cv::Mat dist_col1_t = dist_col1.t();
        dist_col1_t.copyTo(dist_row1);
        distance_mat.at<float>(id1, id1) = FLOAT_INF;

        cv::Mat dist_vec2(sample_capacity, 1, CV_32FC1, cv::Scalar(FLOAT_INF));
        for (int i = 0; i < sample_capacity; ++i)
        {
            dist_vec2.at<float>(i, 0) = std::max(0.f, gram_mat.at<float>(id2, id2) + \
                    gram_mat.at<float>(i, i) - 2 * gram_col2.at<float>(i, 0));
        }

        dist_vec2.copyTo(dist_col2);
        cv::Mat dist_col2_t = dist_col2.t();
        dist_col2_t.copyTo(dist_row2);
        distance_mat.at<float>(id2, id2) = FLOAT_INF;
    }
}


void SampleSpaceModel::decay_sample_weights()
{
    for (int i = 0; i < sample_weights.size(); ++i)
    {
        sample_weights[i] *= (1.f - learning_rate);
    }
}


void SampleSpaceModel::normalize_sample_weights()
{
    float weights_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), 0.f);
    for (int i = 0; i < sample_weights.size(); ++i)
    {
        sample_weights[i] /= weights_sum;
    }
}


void SampleSpaceModel::force_correct(Sample &new_sample, float new_weight, std::vector<int>& changed_ids)
{
    cv::Mat gram_vec;
    // find the inner product of the new sample with existing samples
    find_gram_vector(new_sample, gram_vec);
    // find the distance of the new sample with existing samples
    auto feat_block_num = new_sample.size();
    float new_sample_norm = 0;
    for (int i = 0; i < feat_block_num; ++i)
    {
        auto& feat = new_sample[i];
        auto channels = feat.size();
        for (int j = 0; j < channels; ++j)
        {
            cv::Mat feat_vec = feat[j].reshape(0, feat[j].rows * feat[j].cols);
            assert(feat_vec.channels() == 2);
            ComplexMat feat_vec_cmat(feat_vec);
            cv::Mat mul_mat = (feat_vec_cmat.conj_t() * feat_vec_cmat).to_cv_mat();
            new_sample_norm += 2 * mul_mat.ptr<float>(0)[0]; // real
        }
    }

    cv::Mat dist_vec(sample_capacity, 1, CV_32FC1, cv::Scalar(FLOAT_INF));
    for (int i = 0; i < sample_num; ++i)
    {
        dist_vec.at<float>(i, 0) = std::max(0.f, new_sample_norm + gram_mat.at<float>(i, i) - 2 * gram_vec.at<float>(i, 0));
    }

    if(sample_num == sample_capacity)
    {
        std::vector<float>::iterator min_iter = std::min_element(sample_weights.begin(), sample_weights.end());
        auto min_ind = min_iter - sample_weights.begin();

        // replace the min weight sample with the new sample
        update_distance_matrix(gram_vec, new_sample_norm, min_ind, -1, 0, 1);
        sample_weights[min_ind] = new_weight;
        insert_sample(new_sample, min_ind);
        save_changed_ids(min_ind, changed_ids);
            
        //  Normalise the sample weights
        normalize_sample_weights();
    }
    else
    {
        // If the memory is not full, insert the correct sample in the next empty location
        int sample_pos = sample_num;
        update_distance_matrix(gram_vec, new_sample_norm, sample_pos, -1, 0, 1);
        if(sample_pos == 0)
        {
            sample_weights[sample_pos] = 1;
        }
        else
        {
            sample_weights[sample_pos] = new_weight;
            //  Normalise the sample weights
            normalize_sample_weights();
        }

        insert_sample(new_sample, sample_pos);
        save_changed_ids(sample_pos, changed_ids);
    }

    assert(fabs(std::accumulate(sample_weights.begin(), sample_weights.end(), 0.f) - 1.f) <= 1e-4);

    if(sample_num < sample_capacity)
        sample_num++;
}


void SampleSpaceModel::save_changed_ids(int id, std::vector<int>& changed_ids)
{
    std::vector<int>::iterator it = std::find(changed_ids.begin(), changed_ids.end(), id);
    if(it == changed_ids.end())
        changed_ids.push_back(id);
}


void SampleSpaceModel::update(Sample &new_sample, std::vector<int>& changed_ids)
{
    cv::Mat gram_vec;
    // find the inner product of the new sample with existing samples
    find_gram_vector(new_sample, gram_vec);

    // find the distance of the new sample with existing samples
    auto feat_block_num = new_sample.size();
    float new_sample_norm = 0;
    for (int i = 0; i < feat_block_num; ++i)
    {
        auto& feat = new_sample[i];
        auto channels = feat.size();
        for (int j = 0; j < channels; ++j)
        {
            cv::Mat feat_vec = feat[j].reshape(0, feat[j].rows * feat[j].cols);
            assert(feat_vec.channels() == 2);
            ComplexMat feat_vec_cmat(feat_vec);
            cv::Mat mul_mat = (feat_vec_cmat.conj_t() * feat_vec_cmat).to_cv_mat();
            new_sample_norm += 2 * mul_mat.ptr<float>(0)[0]; // real
        }
    }

    cv::Mat dist_vec(sample_capacity, 1, CV_32FC1, cv::Scalar(FLOAT_INF));
    for (int i = 0; i < sample_num; ++i)
    {
        dist_vec.at<float>(i, 0) = std::max(0.f, new_sample_norm + gram_mat.at<float>(i, i) - 2 * gram_vec.at<float>(i, 0));
    }

    if(sample_num == sample_capacity)
    {
        std::vector<float>::iterator min_iter = std::min_element(sample_weights.begin(), sample_weights.end());
        auto min_weight = *min_iter;
        auto min_ind = min_iter - sample_weights.begin();

        // If any prior weight is less than the minimum allowed weight,
        // replace that sample with the new sample
        if(min_weight < min_weight_thresh)
        {
            update_distance_matrix(gram_vec, new_sample_norm, min_ind, -1, 0, 1);
            //  Normalise the prior weights so that the new sample gets weight as the learning rate
            sample_weights[min_ind] = 0;
            float weights_sum = std::accumulate(sample_weights.begin(), sample_weights.end(), 0.f);
            for (int i = 0; i < sample_weights.size(); ++i)
            {
                sample_weights[i] = sample_weights[i] * (1.f - learning_rate) / weights_sum;
            }
            sample_weights[min_ind] = learning_rate;
            insert_sample(new_sample, min_ind);
            save_changed_ids(min_ind, changed_ids);
        }
        // If no sample has low enough prior weight, then we either merge
        // the new sample with an existing sample, or merge two of the
        // existing samples and insert the new sample in the vacated position
        else
        {
            // Find sample closest to the new sample
            double min_dist_to_new, max_dist_to_new;
            cv::Point min_pt;
            cv::minMaxLoc(dist_vec, &min_dist_to_new, &max_dist_to_new, &min_pt);
            int closest_to_new_ind = min_pt.y;

            // Find the closest pair amongst existing samples
            double min_dist_in_exist, max_dist_in_exist;
            cv::Point min_pt_in_exist;
            cv::minMaxLoc(distance_mat, &min_dist_in_exist, &max_dist_in_exist, &min_pt_in_exist);
            int closest_in_exist_s1 = min_pt_in_exist.y;
            int closest_in_exist_s2 = min_pt_in_exist.x;
            assert(closest_in_exist_s1 != closest_in_exist_s2);

            // If the min distance of the new sample to the existing samples
            // is less than the min distance amongst any of the existing samples,
            // we merge the new sample with the nearest existing sample
            if(min_dist_to_new < min_dist_in_exist)
            {
                decay_sample_weights();

                int merged_sample_ind = closest_to_new_ind;
                Sample sample_to_merge, sample_merged;
                //get_sample do not change
                get_sample(merged_sample_ind, sample_to_merge);
                merge_samples(sample_to_merge, sample_weights[merged_sample_ind], new_sample, learning_rate, sample_merged);
                insert_sample(sample_merged, merged_sample_ind);
                save_changed_ids(merged_sample_ind, changed_ids);

                update_distance_matrix(gram_vec, new_sample_norm, merged_sample_ind, -1, sample_weights[merged_sample_ind], learning_rate);
                sample_weights[closest_to_new_ind] += learning_rate;
            }
            //If the min distance amongst any of the existing samples
            // is less than the min distance of the new sample to the existing samples,
            // we merge the nearest existing samples and insert the new sample in the vacated position
            else
            {
                decay_sample_weights();

                if(sample_weights[closest_in_exist_s2] > sample_weights[closest_in_exist_s1])
                {
                    std::swap(closest_in_exist_s1, closest_in_exist_s2);
                }

                Sample sample_to_merge1, sample_to_merge2, sample_merged;
                //get_sample do not change
                get_sample(closest_in_exist_s1, sample_to_merge1);
                get_sample(closest_in_exist_s2, sample_to_merge2);

                merge_samples(sample_to_merge1, sample_weights[closest_in_exist_s1],
                                sample_to_merge2, sample_weights[closest_in_exist_s2], sample_merged);

                insert_sample(sample_merged, closest_in_exist_s1);
                insert_sample(new_sample, closest_in_exist_s2);
                save_changed_ids(closest_in_exist_s2, changed_ids);
                save_changed_ids(closest_in_exist_s1, changed_ids);

                update_distance_matrix(gram_vec, new_sample_norm, closest_in_exist_s1, closest_in_exist_s2,
                                        sample_weights[closest_in_exist_s1], sample_weights[closest_in_exist_s2]);

                sample_weights[closest_in_exist_s1] += sample_weights[closest_in_exist_s2];
                sample_weights[closest_in_exist_s2] = learning_rate;
            }
        }
    }
    else
    {
        // If the memory is not full, insert the new sample in the next empty location
        int sample_pos = sample_num;
        update_distance_matrix(gram_vec, new_sample_norm, sample_pos, -1, 0, 1);
        if(sample_pos == 0)
        {
            sample_weights[sample_pos] = 1;
        }
        else
        {
            decay_sample_weights();
            sample_weights[sample_pos] = learning_rate;
        }

        insert_sample(new_sample, sample_pos);
        save_changed_ids(sample_pos, changed_ids);
    }

    assert(fabs(std::accumulate(sample_weights.begin(), sample_weights.end(), 0.f) - 1.f) <= 1e-4);

    if(sample_num < sample_capacity)
        sample_num++;
}


void SampleSpaceModel::apply_changes(const SampleSpaceModel& src, const std::vector<int>& changed_ids)
{
    if(changed_ids.empty())
        return;

    sample_capacity = src.sample_capacity;
    sample_weights = src.sample_weights;
    sample_num = src.sample_num;
    min_weight_thresh = src.min_weight_thresh;
    learning_rate = src.learning_rate;

    feat_block_num = src.feat_block_num;
    filter_size = src.filter_size;
    feat_channels = src.feat_channels;

    distance_mat = src.distance_mat.clone();
    gram_mat = src.gram_mat.clone();

    for(int i = 0; i < feat_block_num; i++)
    {
        for(int j = 0; j < changed_ids.size(); j++)
        {
            int id = changed_ids[j];
            auto channels = src.sample_space[i][id].size();
            for(int k = 0; k < channels; k++)
            {
                sample_space[i][id][k] = src.sample_space[i][id][k].clone();
            }
        }
    }
}


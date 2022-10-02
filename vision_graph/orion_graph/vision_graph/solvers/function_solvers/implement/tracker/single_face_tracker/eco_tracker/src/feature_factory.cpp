//
// Created by yuan on 17-9-21.
//

#include "feature_factory.h"
#include "cnn_feature_extractor.h"
#include "eigen_utils.h"

FeatureFactory::FeatureFactory(const std::vector<BaseFeatureExtractor *> &ptr_feat_makers,
                               const std::vector<int> &compressed_channels)
{
    normalize_size = 1;
    normalize_dim = 1;

    ptr_extractors = ptr_feat_makers;
    target_channels = compressed_channels;
    num_feat_blocks = (int)compressed_channels.size();
    project_mats.resize(num_feat_blocks);

    make_id_convert_table();
}


FeatureFactory::~FeatureFactory()
{

}


void FeatureFactory::make_id_convert_table()
{
    int num_blocks = 0;
    for (int i = 0; i < ptr_extractors.size(); ++i)
    {
        int output_blocks = ptr_extractors[i]->output_blocks;
        std::vector<int> feat_ids(output_blocks);
        for (int j = 0; j < output_blocks; ++j)
        {
            feat_ids[j] = j + num_blocks;
        }
        id_convert_table.insert(std::pair<int, std::vector<int>>(i, feat_ids));
        num_blocks += output_blocks;
    }
    assert(num_blocks == num_feat_blocks);
}


void FeatureFactory::extract(const cv::Mat &img, Sample &out_feats)
{
    for (int i = 0; i < ptr_extractors.size(); ++i)
    {
        BaseFeatureExtractor *ptr = ptr_extractors[i];
        ptr->extract_feature(img, out_feats);
        normalize_feature(out_feats, out_feats);
    }
}


void FeatureFactory::normalize_feature(const Feature &in_feat, Feature &out_feat)
{
    for(size_t j=0; j<in_feat.size(); j++)
        in_feat[j] *= 1000;

    auto channels = in_feat.size();
    out_feat.resize(channels);
    double len_sqr = 0;
    for (int i = 0; i < channels; ++i)
    {
        len_sqr += pow(cv::norm(in_feat[i]), 2);
    }
    double coef = sqrt(pow((in_feat[0].rows * in_feat[0].cols), normalize_size) * pow(channels, normalize_dim) / (len_sqr + FLT_EPSILON));
    for (int i = 0; i < channels; ++i)
    {
         out_feat[i] = in_feat[i] * (float)coef;
    }
}


void FeatureFactory::normalize_feature(const Sample &in_feats, Sample &out_feats)
{
    for(size_t i=0; i<in_feats.size(); i++)
    {
       for(size_t j=0; j<in_feats[i].size(); j++)
           in_feats[i][j] *= 1000;
    } 
    int feat_cnt = in_feats.size();
    Sample ret_feats(feat_cnt);
    for (int i = 0; i < feat_cnt; ++i)
    {
        normalize_feature(in_feats[i], ret_feats[i]);
    }
    out_feats = ret_feats;
}


void FeatureFactory::flatten_feature(const Feature &feat, cv::Mat &feat_flat,
                                     int &height, int &width, int &channels)
{
    height = feat[0].rows;
    width = feat[0].cols;
    channels = feat.size();

    // flattened feature matrix
    feat_flat = cv::Mat(channels, height * width, feat[0].type());
    for (int i = 0; i < channels; ++i)
    {
        cv::Mat feat_mat_t = feat[i].t();
        // cv::Mat is row-major, operation on rows are much faster
        feat_mat_t.reshape(0, 1).copyTo(feat_flat.row(i));
    }
    feat_flat = feat_flat.t();
}


void FeatureFactory::restore_flatten_feature(const cv::Mat &feat_flat, int height,
                                             int width, int channels, Feature &out_feat)
{
    out_feat.resize(channels);
    cv::Mat feat_flat_t = feat_flat.t();
    for (int i = 0; i < channels; ++i)
    {
        // cv::Mat is row-major, operation on rows are much faster
        cv::Mat feat_flat_row = feat_flat_t.row(i).clone();
        out_feat[i] = feat_flat_row.reshape(0, width).t();
    }
}


cv::Mat FeatureFactory::compute_project_mat(const Feature &feat, int target_channel)
{
    assert(feat[0].channels() == 1);

    // (height * width) x channels
    cv::Mat feat_flat;
    int height, width, channels;
    flatten_feature(feat, feat_flat, height, width, channels);

    assert(target_channel < channels);

    // remove mean
    cv::Mat mean_mat = cv::Mat::zeros(1, channels, feat_flat.type());
    for (int j = 0; j < feat_flat.rows; ++j)
    {
        mean_mat += feat_flat.row(j);
    }
    mean_mat /= feat_flat.rows;

    for (int j = 0; j < feat_flat.rows; ++j)
    {
        feat_flat.row(j) -= mean_mat;
    }

    // channels x channels
    cv::Mat feat_mul = Ei_mat_multiply(feat_flat.t(),  feat_flat);

    cv::Mat u;
    u = Ei_svd_decomp(feat_mul);
    // shape: channels x compressed_channels
    cv::Mat project_mat = u.colRange(0, target_channel).clone();
    return project_mat;
}


void FeatureFactory::compute_project_mats(const Sample& sample)
{
    auto num_blocks = sample.size();
    assert(sample.size() == num_feat_blocks);
    for (int i = 0 ; i < num_blocks; i++)
    {
        const Feature& feat = sample[i];
        int compressed_channels = this->target_channels[i];
        project_mats[i]= compute_project_mat(feat, compressed_channels);
    }
}


void FeatureFactory::compute_project_mats_each(const Sample &sample, int extractor_id)
{
    auto& feat_ids = id_convert_table[extractor_id];
    auto num_blocks = feat_ids.size();
    assert(sample.size() == num_blocks);
    for (int i = 0; i < num_blocks; ++i)
    {
        int id = feat_ids[i];
        const Feature& feat = sample[i];
        int compressed_channels = target_channels[id];
        project_mats[id] = compute_project_mat(feat, compressed_channels);
    }
}


void FeatureFactory::modify_project_mats(const Vec1dMat &proj_mat_inc)
{
    for (int i = 0; i < project_mats.size(); ++i)
    {
        project_mats[i] += proj_mat_inc[i];
    }
}


void FeatureFactory::modify_project_mats_each(const Vec1dMat &proj_mat_inc, int extractor_id)
{
    auto& feat_ids = id_convert_table[extractor_id];
    auto num_blocks = feat_ids.size();
    for (int i = 0; i < num_blocks; ++i)
    {
        auto id = feat_ids[i];
        project_mats[id] += proj_mat_inc[i];
    }
}


void FeatureFactory::modify_project_mat(const cv::Mat &proj_mat_inc, int feat_id)
{
    project_mats[feat_id] += proj_mat_inc;
}


void FeatureFactory::compress(const Feature &in_feat, const cv::Mat &proj_mat, int target_channel, Feature &out_feat)
{
    // shape: (height * width)  x channels
    cv::Mat feat_flat;
    int height, width, channels;
    flatten_feature(in_feat, feat_flat, height, width, channels);

    assert(feat_flat.channels() < 3);
    // shape: (height * width)  x compressed_channels
    cv::Mat compressed_feat;
    // real
    if(feat_flat.channels() == 1)
    {
        compressed_feat = feat_flat * proj_mat;
    }
        // complex
    else if(feat_flat.channels() == 2)
    {
        std::vector<cv::Mat> mat_vec;
        cv::split(feat_flat, mat_vec);
        for (int j = 0; j < mat_vec.size(); ++j)
        {
            mat_vec[j] *= proj_mat;
        }
        cv::merge(mat_vec, compressed_feat);
    }
    restore_flatten_feature(compressed_feat, height, width, target_channel, out_feat);
}


void FeatureFactory::compress_all(const Sample& in_feats, Sample& out_feats)
{
    auto num_blocks = in_feats.size();
    assert(num_blocks == num_feat_blocks);
    Sample ret_feats(num_blocks);
    for (int i = 0; i < num_feat_blocks; i++)
    {
        compress(in_feats[i], project_mats[i], target_channels[i], ret_feats[i]);
    }
    out_feats = std::move(ret_feats);
}


void FeatureFactory::compress_each(const Feature &in_feat, int feat_id, Feature &out_feat)
{
    compress(in_feat, project_mats[feat_id], target_channels[feat_id], out_feat);
}


void FeatureFactory::compress_each(const Sample &in_feats, int extractor_id, Sample &out_feats)
{
    auto& feat_ids = id_convert_table[extractor_id];
    auto num_blocks = feat_ids.size();
    Sample ret_feats(num_blocks);
    for (int i = 0; i < num_blocks; ++i)
    {
        int id = feat_ids[i];
        compress(in_feats[i], project_mats[id], target_channels[id], ret_feats[i]);
    }
    out_feats = std::move(ret_feats);
}


cv::Size FeatureFactory::get_image_support_sz(cv::Size2f new_sample_sz, float scale)
{
    bool has_cnn_feature = false;
    int cnn_feature_id = -1;
    int extractor_num = ptr_extractors.size();
    for(int i=0; i < extractor_num; i++)
    {
        CnnFeatureExtractor* ptr_cnn_extractor = dynamic_cast<CnnFeatureExtractor*>(ptr_extractors[i]);
        if(ptr_cnn_extractor != NULL)
        {
            cnn_feature_id = i;
            has_cnn_feature = true;
            break;
        }
    }

    cv::Size img_support_sz;
    for (int i = 0; i < extractor_num; ++i)
    {
        if(i != cnn_feature_id)
            img_support_sz = ptr_extractors[i]->get_image_support_sz(new_sample_sz, scale);
    }

    if(has_cnn_feature)
    {
        img_support_sz = ptr_extractors[cnn_feature_id]->get_image_support_sz(new_sample_sz, scale);
    }
    return img_support_sz;
}


//
// Created by yuan on 18-1-31.
//

#ifndef ROBOT_TRACKING_FEAT_FACTORY_CREATOR_H
#define ROBOT_TRACKING_FEAT_FACTORY_CREATOR_H

#include "cnn_feature_extractor.h"
#include "hog_feature_extractor.h"
#include "feature_factory.h"

FeatureFactory* create_feat_factory_hog(const std::vector<int>& target_channels);

#ifdef PLATFORM_CAFFE

FeatureFactory* create_feat_factory_cnn(const std::string & net_def,
                                        const std::string & net_weights,
                                        const std::string & input_mean_file,
                                        const std::vector<std::string> output_layers,
                                        const std::vector<int>& downsample_factors,
                                        const std::vector<int>& target_channels,
                                        float scale_coef = 1.f,
                                        bool is_bgr_seq = true);

FeatureFactory* create_feat_factory_cnn(const std::string & net_def,
                                        const std::string & net_weights,
                                        const std::vector<float>& bgr_mean,
                                        const std::vector<std::string> output_layers,
                                        const std::vector<int>& downsample_factors,
                                        const std::vector<int>& target_channels,
                                        float scale_coef = 1.f,
                                        bool is_bgr_seq = true);

FeatureFactory* create_feat_factory_cnn_hog(const std::string & net_def,
                                            const std::string & net_weights,
                                            const std::string & input_mean_file,
                                            const std::vector<std::string> output_layers,
                                            const std::vector<int>& downsample_factors,
                                            const std::vector<int>& target_channels,
                                            float scale_coef = 1.f,
                                            bool is_bgr_seq = true);

FeatureFactory* create_feat_factory_cnn_hog(const std::string & net_def,
                                            const std::string & net_weights,
                                            const std::vector<float>& bgr_mean,
                                            const std::vector<std::string> output_layers,
                                            const std::vector<int>& downsample_factors,
                                            const std::vector<int>& target_channels,
                                            float scale_coef = 1.f,
                                            bool is_bgr_seq = true);


#else

FeatureFactory* create_feat_factory_cnn(const std::string& model,
                                        const std::vector<int>& target_channels);

FeatureFactory* create_feat_factory_cnn_hog(const std::string& model,
                                            const std::vector<int>& target_channels);
#endif

void release_feat_factory(FeatureFactory *ptr);

#endif //ROBOT_TRACKING_FEAT_FACTORY_CREATOR_H

//
// Created by yuan on 18-1-31.
//

#include <logging.h>
#include "feat_factory_creator.h"

FeatureFactory *create_feat_factory_hog(const std::vector<int> &target_channels) {
    std::vector<BaseFeatureExtractor *> ptr_extractors(1);
    ptr_extractors[0] = new HogFeatureExtractor();
    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}

#ifdef PLATFORM_CAFFE

FeatureFactory* create_feat_factory_cnn(const std::string & net_def,
                                        const std::string & net_weights,
                                        const std::string & input_mean_file,
                                        const std::vector<std::string> output_layers,
                                        const std::vector<int>& downsample_factors,
                                        const std::vector<int>& target_channels,
                                        float scale_coef,
                                        bool is_bgr_seq)
{
    std::vector<BaseFeatureExtractor*> ptr_extractors(1);
    ptr_extractors[0] = new CnnFeatureExtractor(net_def, net_weights, input_mean_file, output_layers,
                                                downsample_factors, scale_coef, is_bgr_seq);
    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}


FeatureFactory* create_feat_factory_cnn(const std::string & net_def,
                                        const std::string & net_weights,
                                        const std::vector<float>& bgr_mean,
                                        const std::vector<std::string> output_layers,
                                        const std::vector<int>& downsample_factors,
                                        const std::vector<int>& target_channels,
                                        float scale_coef,
                                        bool is_bgr_seq)
{
    std::vector<BaseFeatureExtractor*> ptr_extractors(1);
    ptr_extractors[0] = new CnnFeatureExtractor(net_def, net_weights, bgr_mean, output_layers,
                                                downsample_factors, scale_coef, is_bgr_seq);
    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}


FeatureFactory* create_feat_factory_cnn_hog(const std::string & net_def,
                                            const std::string & net_weights,
                                            const std::string & input_mean_file,
                                            const std::vector<std::string> output_layers,
                                            const std::vector<int>& downsample_factors,
                                            const std::vector<int>& target_channels,
                                            float scale_coef,
                                            bool is_bgr_seq)
{
    std::vector<BaseFeatureExtractor*> ptr_extractors(2);
    ptr_extractors[0] = new CnnFeatureExtractor(net_def, net_weights, input_mean_file, output_layers,
                                                downsample_factors, scale_coef, is_bgr_seq);
    ptr_extractors[1] = new HogFeatureExtractor();

    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}


FeatureFactory* create_feat_factory_cnn_hog(const std::string & net_def,
                                            const std::string & net_weights,
                                            const std::vector<float>& bgr_mean,
                                            const std::vector<std::string> output_layers,
                                            const std::vector<int>& downsample_factors,
                                            const std::vector<int>& target_channels,
                                            float scale_coef,
                                            bool is_bgr_seq)
{
    std::vector<BaseFeatureExtractor*> ptr_extractors(2);
    ptr_extractors[0] = new CnnFeatureExtractor(net_def, net_weights, bgr_mean, output_layers,
                                                downsample_factors, scale_coef, is_bgr_seq);
    ptr_extractors[1] = new HogFeatureExtractor();

    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}

#else

FeatureFactory *create_feat_factory_cnn(const std::string &model,
                                        const std::vector<int> &target_channels) {
    std::vector<BaseFeatureExtractor *> ptr_extractors(1);
    ptr_extractors[0] = new CnnFeatureExtractor(model);
    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}


FeatureFactory *create_feat_factory_cnn_hog(const std::string &model,
                                            const std::vector<int> &target_channels) {
    std::vector<BaseFeatureExtractor *> ptr_extractors(2);
    ptr_extractors[0] = new CnnFeatureExtractor(model);
    ptr_extractors[1] = new HogFeatureExtractor();

    FeatureFactory *ptr_feat_factory = new FeatureFactory(ptr_extractors, target_channels);
    return ptr_feat_factory;
}

#endif


void release_feat_factory(FeatureFactory *ptr) {
    assert(ptr != NULL);
    for (int i = 0; i < ptr->ptr_extractors.size(); ++i) {
        delete ptr->ptr_extractors[i];
    }
    delete ptr;
    ptr = NULL;

}

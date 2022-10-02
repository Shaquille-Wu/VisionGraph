//
// Created by yuan on 17-9-19.
//

#include "cnn_feature_extractor.h"

#include <logging.h>
#include <timer.h>

CnnFeatureExtractor::CnnFeatureExtractor() {
    output_blocks = 1;
}

CnnFeatureExtractor::CnnFeatureExtractor(const std::string &pFile) {
    cnn_featuremap = new vision::FeaturemapRunner(pFile);
    output_blocks = 1;
}

CnnFeatureExtractor::~CnnFeatureExtractor() {
    if (cnn_featuremap != nullptr) {
        delete cnn_featuremap;
        cnn_featuremap = nullptr;
    }
}

void CnnFeatureExtractor::extract_feature(const cv::Mat &img, Sample &out_feature) {
    // HighClock c;
    // c.Start();
    output_blocks = 1;

    std::vector<FeatureMap> featuremap;
    cnn_featuremap->run((cv::Mat &)img, featuremap);

    // cv::Size m_size(out_channel, out_H * out_W);
    // cv::Mat m(m_size, CV_32FC1, featuremap[0].data.get());
    // cv::transpose(m, m);
    // int size = out_channel * out_H * out_W;
    // float buf[size];
    // memset(buf, 0, size * sizeof(float));
    // int buf_size = out_channel * out_H * out_W;
    // for (int i = 0; i < buf_size; ++i) {
    //     memcpy((uchar *)(&buf[i]), m.data + i * 4, sizeof(float));
    // }

    // Feature feat(out_channel);
    // cv::Size feat_sz(out_W, out_H);

    // for (int i = 0; i < out_channel; ++i) {
    //     float *ptr = buf + i * out_W * out_H;
    //     cv::Mat feat_chn(feat_sz, CV_32FC1, ptr);
    //     feat_chn.copyTo(feat[i]);
    // }
    // out_feature.push_back(feat);

    Feature feat(out_channel);
    cv::Size feat_sz(out_H, out_W);
    float *buf = (float *)featuremap.front().data.lock().get();

    for (int i = 0; i < out_channel; ++i) {
        float *ptr = buf + i * out_W * out_H;

        cv::Mat feat_chn(feat_sz, CV_32FC1, ptr);
        feat_chn.copyTo(feat[i]);
    }

    out_feature.push_back(feat);

}

// deprecated
cv::Size CnnFeatureExtractor::get_image_support_sz(cv::Size2f new_sample_sz, float scale) {
    return cv::Size(0, 0);
}

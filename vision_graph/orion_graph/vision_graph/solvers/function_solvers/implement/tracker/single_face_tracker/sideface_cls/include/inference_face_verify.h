#ifndef GRAPH_INFERENCE_VERIFY_H_
#define GRAPH_INFERENCE_VERIFY_H_

#include <vector>

#include "base_verifier.h"
#include "norm.h"
#include "reidfeature.h"
using namespace vision;

class InferenceVerifier : public BaseVerifier {
public:
    InferenceVerifier();
    InferenceVerifier(char* pcModel);
    ~InferenceVerifier();
    void compute_feature(cv::Mat& img, std::vector<float>& output);
    void compute_feature(std::vector<cv::Mat>& img, std::vector<std::vector<float>>& output);
    float compute_feature_diff(std::vector<float>& query, std::vector<float>& feat);
    float compute_feature_diff(std::vector<float>& query, std::vector<std::vector<float>>& feats);

private:
    vision::Feature* verify_ptr;
};

#endif

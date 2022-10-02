#ifndef BASE_VERIFIER_H
#define BASE_VERIFIER_H

#include <opencv2/opencv.hpp>

class BaseVerifier {
public:
    BaseVerifier() {}
    virtual ~BaseVerifier() {}
    virtual void compute_feature(cv::Mat & img, std::vector<float> & output) = 0;
    virtual void compute_feature(std::vector<cv::Mat> & img, std::vector< std::vector<float> > & output) = 0;
    virtual float compute_feature_diff(std::vector<float> & query, std::vector<float> & feat) = 0;
    virtual float compute_feature_diff(std::vector<float> & query, std::vector<std::vector<float>> & feats) = 0;
};

#endif

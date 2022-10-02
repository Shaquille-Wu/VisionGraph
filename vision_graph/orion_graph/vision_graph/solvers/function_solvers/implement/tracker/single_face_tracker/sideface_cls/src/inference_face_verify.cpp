#include "inference_face_verify.h"

#include <logging.h>

#include "timer.h"

InferenceVerifier::InferenceVerifier() {
    verify_ptr = NULL;
}

InferenceVerifier::InferenceVerifier(char* pcModel) {
    verify_ptr = new vision::Feature(pcModel);
    if (!verify_ptr) {
        //		DLOG(INFO)<<"create verify_ptr failed!";
        return;
    }
}

InferenceVerifier::~InferenceVerifier() {
    if (verify_ptr)
        delete verify_ptr;
}

void InferenceVerifier::compute_feature(cv::Mat& img, std::vector<float>& output) {

    output.clear();
    verify_ptr->run(img, output);
    norm(output);

}

void InferenceVerifier::compute_feature(std::vector<cv::Mat>& img, std::vector<std::vector<float>>& output) {
    HighClock clock;
    clock.Start();
    int num = img.size();
    if (num <= 0)
        return;
    output.clear();
    output.reserve(num);
    for (int i = 0; i < img.size(); ++i) {
        output.push_back(std::vector<float>());
        compute_feature(img[i], output.back());
    }
    clock.Stop();
#ifdef PRINT_TIME_COST
//	DLOG(INFO)<<"verify time cost is "<<clock.GetTime()/1000<<" ms";
#endif
}

float InferenceVerifier::compute_feature_diff(std::vector<float>& query, std::vector<float>& feat) {
    if (query.empty() || feat.empty()) {
        return -1.0f;
    }

    assert(query.size() == feat.size());

    int iLen = query.size();

    float delta = 0.0f, dist = 0.0f;
    for (int ix = 0; ix < iLen; ++ix) {
        delta = query[ix] - feat[ix];
        dist += delta * delta;
    }

    return dist;
}

float InferenceVerifier::compute_feature_diff(std::vector<float>& query, std::vector<std::vector<float>>& feats) {
    float min_dist = 100000;
    for (size_t i = 0; i < feats.size(); i++) {
        float dist = compute_feature_diff(query, feats[i]);
        if (dist < min_dist) {
            min_dist = dist;
        }
    }
    return min_dist;
}

//
// Created by yuan on 17-11-13.
//

#include "apce_conf.h"
#include <logging.h>

ApceConfidence::ApceConfidence(int win_len, float score_thresh_coef, float apce_thresh_coef) : slide_win_len(win_len),
                                                                                               score_thresh_factor(score_thresh_coef),
                                                                                               apce_thresh_factor(apce_thresh_coef),
                                                                                               max_score(0),
                                                                                               apce(0),
                                                                                               avg_max_score(0),
                                                                                               avg_apce(0) {}

ApceConfidence::~ApceConfidence() {}

void ApceConfidence::reset() {
    max_score = 0;
    apce = 0;
    max_score_his.clear();
    apce_his.clear();
    avg_max_score = 0;
    avg_apce = 0;
}

void ApceConfidence::compute(const cv::Mat &scores_map) {
    double min_value = 0.0, max_value = 0.0;
    cv::minMaxLoc(scores_map, &min_value, &max_value);
    cv::Mat scores_offset     = scores_map - min_value;
    cv::Mat scores_offset_sqr = scores_offset.mul(scores_offset);
    cv::Scalar sum_sqr = cv::sum(scores_offset_sqr);
    float mean_amp = (float)sum_sqr[0] / (float)scores_offset.total();

    max_score = (float)max_value;
    apce = (float)(pow(max_value - min_value, 2) / (mean_amp + 1e-7));

    if (judge()) {
        int count = (int)(max_score_his.size());
        if (count < slide_win_len) {
            if (count == 0) {
                avg_max_score = max_score;
                avg_apce = apce;
            } else {
                avg_max_score = (avg_max_score * count + max_score) / (float)(count + 1);
                avg_apce = (avg_apce * count + apce) / (float)(count + 1);
            }
        } else {
            float front_max_score = max_score_his.front();
            float front_apce = apce_his.front();

            avg_max_score = (avg_max_score * slide_win_len - front_max_score + max_score) / (float)slide_win_len;
            avg_apce = (avg_apce * slide_win_len - front_apce + apce) / (float)slide_win_len;

            max_score_his.pop_front();
            apce_his.pop_front();
        }
        max_score_his.push_back(max_score);
        apce_his.push_back(apce);
    }
}

bool ApceConfidence::judge() {
    float max_score_thresh = score_thresh_factor * avg_max_score;
    float apce_thresh = apce_thresh_factor * avg_apce;
    bool score_result = (max_score >= max_score_thresh);
    bool apce_result = (apce >= apce_thresh);
    bool result = (score_result && apce_result);
    return result;
}

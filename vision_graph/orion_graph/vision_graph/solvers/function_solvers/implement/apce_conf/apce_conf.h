//
// Created by yuan on 17-11-13.
//

#ifndef ROBOT_TRACKING_APCE_CONF_H
#define ROBOT_TRACKING_APCE_CONF_H

#include <deque>
#include <opencv2/opencv.hpp>

class ApceConfidence {
public:
    ApceConfidence(int win_len = 500, float score_thresh_coef = 0.4f, float apce_thresh_coef = 0.3f);

    ~ApceConfidence();

    void compute(const cv::Mat& scores_map);

    void reset();

    bool judge();

public:
    int slide_win_len;
    float score_thresh_factor;
    float apce_thresh_factor;
    float max_score;
    float apce;
    std::deque<float> max_score_his;
    std::deque<float> apce_his;
    float avg_max_score;
    float avg_apce;
};

#endif  //ROBOT_TRACKING_APCE_CONF_H

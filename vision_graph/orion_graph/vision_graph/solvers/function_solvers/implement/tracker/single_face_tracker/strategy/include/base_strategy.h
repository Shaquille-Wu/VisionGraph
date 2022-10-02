//
// Created by yuan on 18-1-30.
//

#ifndef GRAPH_ROBOT_TRACKING_BASE_STRATEGY_H
#define GRAPH_ROBOT_TRACKING_BASE_STRATEGY_H

#include <opencv2/opencv.hpp>
#include "base_track_def.h"


class BaseStrategy
{
public:
    BaseStrategy()
    {
        target_status = TRACKING_FAILED;
        target_pos = cv::Rect(-1, -1, -1, -1);
    }

    virtual ~BaseStrategy() {}

    virtual int init(const cv::Mat & img, cv::Rect * face_rect, cv::Rect * body_rect) = 0;

    virtual void reset() = 0;

    virtual void track(const cv::Mat & img) = 0;

    virtual cv::Rect get_target_pos() { return target_pos;}

    virtual TrackingStatus get_target_status() { return target_status;}

public:
    TrackingStatus target_status;
    cv::Rect target_pos;
};

#endif //ROBOT_TRACKING_BASE_STRATEGY_H

//
// Created by yuan on 17-10-12.
//

#ifndef ROBOT_TRACKING_DATA_TYPE_H
#define ROBOT_TRACKING_DATA_TYPE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <omp.h>

#ifndef FLOAT_INF
#define FLOAT_INF (std::numeric_limits<float>::max())
#endif

#ifndef FLOAT_EPS
#define FLOAT_EPS (std::numeric_limits<float>::epsilon())
#endif

typedef std::vector<cv::Mat> Vec1dMat;
typedef std::vector<Vec1dMat> Vec2dMat;
typedef std::vector<Vec2dMat> Vec3dMat;
typedef std::vector<Vec3dMat> Vec4dMat;


#define TEST_TIME_COST(STR_INFO, X) {auto start_ticks = cv::getTickCount(); \
                                    {X}\
                                    double time_cost = (cv::getTickCount() - start_ticks) * 1000 / cv::getTickFrequency();\
                                    DLOG(INFO) << STR_INFO << ": " << time_cost << " ms" ;}

#endif //ROBOT_TRACKING_DATA_TYPE_H

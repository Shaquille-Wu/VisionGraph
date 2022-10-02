#ifndef GRAPH_HISTOGRAM_H
#define GRAPH_HISTOGRAM_H 

#include <opencv2/opencv.hpp>


class Histogram
{
private:
    int histSize[1]; // 项的数量
    float hranges[2]; // 统计像素的最大值和最小值
    const float* ranges[1];
    int channels[1]; // 仅计算一个通道

public:
    Histogram();

    cv::MatND getHistogram(const cv::Mat &image);
    
    cv::Mat getHistogramImage(const cv::Mat &image);

    bool isLightEnough(const cv::Mat& img);

    bool isTooBright(const cv::Mat& img);

    int calcLightLevel(const cv::Mat& img);

    int calcAvgLight(const cv::Mat& img);
};

#endif



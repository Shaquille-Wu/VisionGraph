#include "Histogram.h"

Histogram::Histogram()
{
    // 准备1D直方图的参数
    histSize[0] = 256;
    hranges[0] = 0;
    hranges[1] = 256;
    ranges[0] = hranges;
    channels[0] = 0;
}
 
cv::MatND Histogram::getHistogram(const cv::Mat &image)
{
    cv::MatND hist;
    // 计算直方图
    calcHist(&image ,  // 要计算的图像
            1,         // 只计算一幅图像的直方图
            channels,  // 通道数量
            cv::Mat(), // 不使用掩码
            hist,      // 存放直方图
            1,         // 1D直方图
            histSize,  // 统计的灰度的个数
            ranges);   // 灰度值的范围
    return hist;
}
   
int Histogram::calcLightLevel(const cv::Mat& img)
{
#if 0
    cv::Mat reduceImg = img.clone();
#else 
    int reduceHeight = img.rows / 10;
    int reduceWidth = img.cols / 10;
    cv::Rect reduceRect(reduceWidth, reduceHeight, img.cols - 2*reduceWidth, img.rows - 2*reduceHeight);
    cv::Mat reduceImg = img(reduceRect);
#endif
    cv::MatND hist = getHistogram(reduceImg);
    int light_thresh = 0;
    float allPix = reduceImg.rows * reduceImg.cols;
    float darkPix = 0, allGray = 0;
    
    for(int i = 0; i < histSize[0]; i++)
    {
        allGray += i * hist.at<float>(i);
    }

    float avg = allGray / allPix;
    if(avg > 50)
        light_thresh++;

    for(int i = 0; i < 50 ; i++)
    {
        darkPix += hist.at<float>(i);
    }
    //DLOG(INFO)<<"darkPix = "<<darkPix <<", all pix is "<<allPix;
    float darkRate = darkPix / allPix;
    if(darkRate < 0.6)
        light_thresh++;
 
    cv::Mat gray;
	  cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    cv::Scalar scalar = cv::mean(gray);
    double grayAvg=scalar.val[0];
    if(grayAvg > 43)
        light_thresh++;
    if(grayAvg > 65)
        light_thresh++;

//    DLOG(INFO)<<"avg is "<< avg<<", darkRate is "<<darkRate<<", grayAvg is "<<grayAvg;

    return light_thresh;
}

bool Histogram::isLightEnough(const cv::Mat& img)
{
    return calcLightLevel(img) > 1;
}

bool Histogram::isTooBright(const cv::Mat& img)
{
#if 0
    cv::Mat reduceImg = img.clone();
#else 
    int reduceHeight = img.rows / 10;
    int reduceWidth = img.cols / 10;
    cv::Rect reduceRect(reduceWidth, reduceHeight, img.cols - 2*reduceWidth, img.rows - 2*reduceHeight);
    cv::Mat reduceImg = img(reduceRect);
#endif
    cv::MatND hist = getHistogram(reduceImg);
    int bad_thresh = 0;
    float allPix = reduceImg.rows * reduceImg.cols;
    float bright_pix = 0, allGray = 0;
    
    for(int i = 0; i < histSize[0]; i++)
    {
        allGray += i * hist.at<float>(i);
    }

    float avg = allGray / allPix;
    if(avg > 205)
        bad_thresh++;

    for(int i = 200; i < 255 ; i++)
    {
        bright_pix += hist.at<float>(i);
    }
    float bright_rate = bright_pix / allPix;
    if(bright_rate > 0.6)
        bad_thresh ++ ;


    cv::Mat gray;
	  cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    cv::Scalar scalar = cv::mean(gray);
    double grayAvg=scalar.val[0];
    if(grayAvg > 195)
        bad_thresh ++;

//    DLOG(INFO)<<"avg is "<< avg<<",bright_rate is "<<bright_rate<<", grayAvg is "<<grayAvg;

    return bad_thresh > 1;
}

cv::Mat Histogram::getHistogramImage(const cv::Mat &img)
{
    cv::MatND hist = getHistogram(img);
    double maxVal = 0.0f;
    double minVal = 0.0f;

    minMaxLoc(hist, &minVal, &maxVal);

    //显示直方图的图像
    cv::Mat histImg(histSize[0], histSize[0], CV_8U, cv::Scalar(255));

    // 设置最高点为nbins的90%
    float hpt = 0.9 * histSize[0];
    //每个条目绘制一条垂直线
    for (int h = 0; h < histSize[0]; h++)
    {
        float binVal = hist.at<float>(h);
        float intensity = 0;
        if(maxVal>0)
            intensity = binVal * hpt / maxVal;
        cv::line(histImg, cv::Point(h, 255), cv::Point(h, 255 - intensity), cv::Scalar::all(0));
    }
    return histImg;
}

int Histogram::calcAvgLight(const cv::Mat& img)
{
    cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar scalar = cv::mean(gray);
    double grayAvg=scalar.val[0];
    return (int)grayAvg;
}

#ifndef GRAPH_FACIAL_LANDMARKS_H
#define GRAPH_FACIAL_LANDMARKS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "string.h"

class CFacialLandmarks
{
public:
	CFacialLandmarks(void);
	virtual ~CFacialLandmarks(void);

	float compute_similarity(unsigned char* pdata , int w, int h, const std::vector<cv::Point2f>& keypts);
private:
	void* m_ccsim;
};

#endif
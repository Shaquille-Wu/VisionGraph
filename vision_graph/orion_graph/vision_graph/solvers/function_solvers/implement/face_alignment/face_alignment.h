#ifndef FACE_ALIGNMENT_H_
#define FACE_ALIGNMENT_H_

#include <opencv2/opencv.hpp>

// face alignment using 106 face keypoints
cv::Mat face_align(const cv::Mat& src,
                   const std::vector<cv::Point2f>& kpts_vec,
                   cv::Rect face_rect,
                   cv::Size dst_sz = cv::Size(256, 256),
                   int eye_h = 96);

// face alignment using 106 face keypoints, return compact face
int face_align_compact(const cv::Mat& src,
                       const std::vector<cv::Point2f>& kpts_vec,
                       cv::Rect face_rect, cv::Mat &out_img);

#endif
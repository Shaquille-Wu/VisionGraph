//
// Created by yuan on 17-9-19.
//

#ifndef ROBOT_TRACKING_UTILS_H
#define ROBOT_TRACKING_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
//#include <cuda_runtime.h>
#include "data_type.h"

const static std::function<void(const cv::Mat&, cv::Mat&)> lamda_copy_mat = \
                            [](const cv::Mat& src, cv::Mat& dst){dst = src.clone();};

// rotate of Mat by 90, 180, 270
enum RotateFlags
{
    ROTATE_0 = 0,
    ROTATE_90_CLOCKWISE = 1, //Rotate 90 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
    ROTATE_180 = 3 //Rotate 180 degrees clockwise/counter-clockwise
};

enum ConvolutionType
{
    /* Return the full convolution, including border */
    CONV_FULL = 0,

    /* Return only the part that corresponds to the original image */
    CONV_SAME,

    /* Return only the submatrix containing elements that were not influenced by the border */
    CONV_VALID
};

void normalize_image(const cv::Mat& src, const cv::Scalar& mean_values, const cv::Scalar& std_values, cv::Mat& dst);

float* split_bgrmat_channels(const cv::Mat& image);

void rot90(const cv::Mat& src, cv::Mat& dst, int rotflag);

cv::Point2f get_rect_center(const cv::Rect& box);

cv::Rect scale_rect(cv::Rect src, float scale);

cv::Rect_<float> scale_rect(cv::Rect_<float> src, float scale);

cv::Rect clip_rect(cv::Rect src, cv::Size img_sz);

void conv2(const cv::Mat &src, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dst);

void conv2_complex(const cv::Mat &src, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dst);

// src1 * alpha + src2 * beta
void vec3dmat_linear_opt(Vec3dMat& dst, const Vec3dMat& src1, const Vec3dMat& src2, float alpha = 1.f, float beta = 1.f);

// src1 * alpha + src2 * beta
void vec2dmat_linear_opt(Vec2dMat& dst, const Vec2dMat& src1, const Vec2dMat& src2, float alpha = 1.f, float beta = 1.f);

// src1 * alpha + src2 * beta
void vec1dmat_linear_opt(Vec1dMat& dst, const Vec1dMat& src1, const Vec1dMat& src2, float alpha = 1.f, float beta = 1.f);

void proc_batch_vec1dmat(const std::function<void(const cv::Mat&, cv::Mat&)>& fun_handle,
                         const Vec1dMat& src, Vec1dMat& dst);

void proc_batch_vec2dmat(const std::function<void(const cv::Mat&, cv::Mat&)>& fun_handle,
                         const Vec2dMat& src, Vec2dMat& dst);

void proc_batch_vec3dmat(const std::function<void(const cv::Mat&, cv::Mat&)>& fun_handle,
                         const Vec3dMat& src, Vec3dMat& dst);

void init_vec1dmat(int capacity, cv::Size sz, int type, Vec1dMat& dst);


int mod(int num, int base);

cv::Mat fcv_mul(const cv::Mat& src1,const cv::Mat& src2);

#endif //ROBOT_TRACKING_UTILS_H

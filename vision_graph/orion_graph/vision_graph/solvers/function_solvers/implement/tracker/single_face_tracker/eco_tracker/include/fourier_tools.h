//
// Created by An Wenju and yuan on 17-9-26.
//

#ifndef ROBOT_TRACKING_FOURIER_TOOLS_H
#define ROBOT_TRACKING_FOURIER_TOOLS_H

#include "opencv2/opencv.hpp"
#include "complexmat.h"
#include "data_type.h"

cv::Mat circshift(cv::Mat x, int x_rot, int y_rot);

cv::Mat fftshift(const cv::Mat& in, int dim);

cv::Mat fftshift(const cv::Mat& src);

cv::Mat ifftshift(const cv::Mat& x, int dim);

cv::Mat ifftshift(const cv::Mat& x);

void cfft2(const cv::Mat& x, cv::Mat& xf);

void cifft2(const cv::Mat& xf, cv::Mat& x);

void full_fourier_coef(const cv::Mat& xf, cv::Mat& xf_full);

void cubic_spline_fourier(const cv::Mat& f, float a, cv::Mat& df);

void interpolate_dft(const Vec2dMat& xf, Vec1dMat& interp1_fs, Vec1dMat& interp2_fs, Vec2dMat& xf_out);
void interpolate_dft(const Vec1dMat& xf, Vec1dMat& interp1_fs, Vec1dMat& interp2_fs, Vec1dMat& xf_out, int i);

void compact_fourier_coef(const cv::Mat& xf, cv::Mat& xf_compact);

void resize_dft(const cv::Mat& input_dft, int desired_len, cv::Mat& output_dft);

void shift_sample(const Vec1dMat& xf, cv::Point2f shift, const cv::Mat& kx,
                  const cv::Mat& ky, Vec1dMat& xf_sample);

void sample_fs(const cv::Mat& fs, cv::Mat& fs_dst, int* grid_sz=NULL);

void symmetrize_filter(const cv::Mat& hf, cv::Mat& hf_symm);

#endif //ROBOT_TRACKING_FOURIER_TOOLS_H

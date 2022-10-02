//
// Created by yuan on 17-9-21.
//

#ifndef ROBOT_TRACKING_EIGEN_H
#define ROBOT_TRACKING_EIGEN_H

//#define EIGEN_USE_LAPACKE


#include "../../../../../../../common/eigen3/Eigen/Dense"
#include <vector>
#include <opencv2/opencv.hpp>

// Eigen use col-major as default
// to keep consistent with the definition of feature, we use row-major
//#define EIGEN_ROW_MAJOR


#ifdef EIGEN_ROW_MAJOR
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EiMatrixXf;
#else
typedef Eigen::MatrixXf EiMatrixXf;
#endif

// Eigen only support 2d matrix operations
// use vector to contain 3d matrix
typedef std::vector<EiMatrixXf> EiMatXf3d;


void cvmat2d_to_eimat(const cv::Mat &cv_mat, EiMatrixXf &ei_mat);

void eimat_to_cvmat2d(const EiMatrixXf &ei_mat, cv::Mat &cv_mat);

void qr_decomp(const cv::Mat &src, cv::Mat &Q);

cv::Mat Ei_mat_multiply(const cv::Mat &src1, const cv::Mat &src2);

cv::Mat Ei_svd_decomp(const cv::Mat &src);

#endif //ROBOT_TRACKING_EIGEN_H

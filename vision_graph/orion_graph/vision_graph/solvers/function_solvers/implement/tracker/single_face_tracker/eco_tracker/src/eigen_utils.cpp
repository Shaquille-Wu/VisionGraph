//
// Created by yuan on 17-10-20.
//

#include "eigen_utils.h"
//#include <opencv/cxeigen.hpp>
#include<opencv2/core/eigen.hpp>
void cvmat2d_to_eimat(const cv::Mat& cv_mat, EiMatrixXf& ei_mat)
{
    assert(cv_mat.channels() == 1);
    EiMatrixXf res(cv_mat.rows, cv_mat.cols);
    for (int i = 0; i < cv_mat.rows; ++i)
    {
        const float *ptr = cv_mat.ptr<float>(i);
        for (int j = 0; j < cv_mat.cols; ++j)
        {
            res(i, j) = ptr[j];
        }
    }
    ei_mat = res;
}

void eimat_to_cvmat2d(const EiMatrixXf& ei_mat, cv::Mat& cv_mat)
{
    cv_mat = cv::Mat(ei_mat.rows(), ei_mat.cols(), CV_32FC1);
    for (int i = 0; i < ei_mat.rows(); ++i)
    {
        float *ptr = cv_mat.ptr<float>(i);
        for (int j = 0; j < ei_mat.cols(); ++j)
        {
            ptr[j] = ei_mat(i, j);
        }
    }
}

void qr_decomp(const cv::Mat& src, cv::Mat& Q)
{
    assert(src.channels() == 1);

    EiMatrixXf src_eimat;
    cv::cv2eigen(src, src_eimat);
    EiMatrixXf thin_q(EiMatrixXf::Identity(src_eimat.rows(), src_eimat.cols()));
    Eigen::HouseholderQR<EiMatrixXf> qr(src_eimat);
    EiMatrixXf q_eimat = qr.householderQ() * thin_q;
    cv::eigen2cv(q_eimat, Q);
}

cv::Mat Ei_mat_multiply(const cv::Mat& src1, const cv::Mat& src2)
{
    EiMatrixXf src1_eimat, src2_eimat;
    cv::cv2eigen(src1, src1_eimat);
    cv::cv2eigen(src2, src2_eimat);

    EiMatrixXf dst_eimat = src1_eimat * src2_eimat;
    cv::Mat dst;
    cv::eigen2cv(dst_eimat, dst);
    return dst;
}

cv::Mat Ei_svd_decomp(const cv::Mat& src)
{
    EiMatrixXf src_eimat;
    cv::cv2eigen(src, src_eimat);
    //Eigen::BDCSVD<EiMatrixXf> svd(src_eimat, Eigen::ComputeFullV|Eigen::ComputeFullU);
    Eigen::BDCSVD<EiMatrixXf> svd(src_eimat, Eigen::ComputeFullU);
    EiMatrixXf u_eimat = svd.matrixU();
    cv::Mat u;
    cv::eigen2cv(u_eimat, u);
    return u;
}

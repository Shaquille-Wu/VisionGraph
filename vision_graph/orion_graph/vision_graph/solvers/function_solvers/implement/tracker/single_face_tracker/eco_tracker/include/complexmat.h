//
// Created by yuan on 17-10-31.
//

#ifndef ROBOT_TRACKING_COMPLEXMAT_H
#define ROBOT_TRACKING_COMPLEXMAT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <logging.h>
#include "eigen_utils.h"
#include "fastcv.h"
#include "logging.h"

class ComplexMat
{

    typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EiMatrixXc;

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EiMatrixX;

public:
    ComplexMat() : cols(0), rows(0) {}

    //assuming that mat has 2 channels (real, img)
    ComplexMat(const cv::Mat & mat)
    {
        assert(mat.channels() <= 2);
        if (mat.channels() == 2)
        {
            // ensure data is continuous
            data = mat.isContinuous() ? mat : mat.clone();
        }
        else
        {
            std::vector<cv::Mat> src_vec;
            src_vec.resize(2);
            src_vec[0] = mat; // real
            src_vec[1] = cv::Mat::zeros(mat.size(), mat.type());
            cv::merge(src_vec, data);
        }

        assert(data.isContinuous());
        cols = mat.cols;
        rows = mat.rows;
    }

    //return 2 channels (real, imag)
    cv::Mat to_cv_mat() const
    {
        return data;
    }


    ComplexMat conj() const
    {
        cv::Mat dst(data.size(), data.type());
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_src = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

        dst_eimat = src_eimat.conjugate();
        return ComplexMat(dst);
    }


    ComplexMat conj_t() const
    {
        cv::Mat dst(cols, rows, data.type());
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, cols, rows);

        std::complex<float> *ptr_src = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

        dst_eimat = src_eimat.adjoint();
        return ComplexMat(dst);
    }

    ComplexMat abs() const
    {
        cv::Mat dst(data.size(), data.type());
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_src = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

        dst_eimat = src_eimat.cwiseAbs().template cast<std::complex<float>>();
        return ComplexMat(dst);
    }

    cv::Mat abs2() const
    {
        cv::Mat_<float> dst(rows, cols);
        float *ptr_dst = (float*)dst.data;
        Eigen::Map<EiMatrixX> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_src = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

        dst_eimat = src_eimat.cwiseAbs();
        return dst;
    }


    cv::Mat real() const
    {
        cv::Mat_<float> dst(rows, cols);
        float *ptr_dst = (float*)dst.data;
        Eigen::Map<EiMatrixX> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_src = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

        dst_eimat = src_eimat.real();
        return dst;
    }


    //matrix multiplication
    ComplexMat operator*(const ComplexMat & rhs) const
    {
#ifdef FASTCV_X
        assert(cols == rhs.rows);
        cv::Mat dst(rows, rhs.cols, data.type());
        fcvMatrixMultiplyf32((float*)data.data, data.cols, data.rows, 2*data.cols*sizeof(float), (float*)rhs.data.data,
                             rhs.data.cols, 2*rhs.data.cols*sizeof(float), (float*)dst.data, 2*dst.cols*sizeof(float));
        return ComplexMat(dst);
#else
        return rhs;
#endif
    }


    //element-wise multiplication
    ComplexMat mul(const ComplexMat & rhs) const
    {
        assert(data.size() == rhs.data.size());
        cv::Mat dst(data.size(), data.type());
#ifdef FASTCV_X
        fcvElementMultiplyf32((float*)data.data, data.channels()*data.cols, data.rows, data.channels()*data.cols*sizeof(float), (float*)rhs.data.data,
                              rhs.data.channels()*rhs.data.cols*sizeof(float), (float*)dst.data, dst.channels()*dst.cols*sizeof(float));
#else
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

        std::complex<float> *ptr_rhs = (std::complex<float> *)rhs.data.data;
        Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

        dst_eimat = lhs_eimat.cwiseProduct(rhs_eimat);
#endif
        return ComplexMat(dst);
    }


    //element-wise division
    ComplexMat operator/(const ComplexMat & rhs) const
    {
        assert(data.size() == rhs.data.size());

        cv::Mat dst(data.size(), data.type());
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

        std::complex<float> *ptr_rhs = (std::complex<float> *)rhs.data.data;
        Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

        dst_eimat = lhs_eimat.cwiseProduct(rhs_eimat.cwiseInverse());
        return ComplexMat(dst);
    }


    ComplexMat operator+(const ComplexMat & rhs) const
    {
        assert(data.size() == rhs.data.size());
//        return ComplexMat_(data + rhs.data);

        cv::Mat dst(data.size(), data.type());

#ifdef FASTCV_X
        LOG(ERROR) << "wyb debug use fastcv";
        fcvAddf32((float *)data.data, data.cols * data.channels(), data.rows, data.channels() * data.cols * sizeof(float), (float *)rhs.data.data,
                  rhs.data.channels() * rhs.data.cols * sizeof(float), (float *)dst.data, dst.channels() * dst.cols * sizeof(float));
#else
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

        std::complex<float> *ptr_rhs = (std::complex<float> *)rhs.data.data;
        Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

        dst_eimat = lhs_eimat + rhs_eimat;
#endif
        return ComplexMat(dst);
    }


    ComplexMat operator-(const ComplexMat & rhs) const
    {
        assert(data.size() == rhs.data.size());
//        return ComplexMat_(data - rhs.data);
        cv::Mat dst(data.size(), data.type());

        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

        std::complex<float> *ptr_rhs = (std::complex<float> *)rhs.data.data;
        Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

        dst_eimat = lhs_eimat - rhs_eimat;

        return ComplexMat(dst);
    }


    std::complex<float> dot(const ComplexMat & rhs) const
    {
        assert(data.size() == rhs.data.size());

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, 1, rows * cols);

        std::complex<float> *ptr_rhs = (std::complex<float> *)rhs.data.data;
        Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows * cols, 1);

        auto dot_pd = lhs_eimat * rhs_eimat;
        return dot_pd(0, 0);
    }


    ComplexMat operator*(const float & rhs) const
    {
        cv::Mat dst(data.size(), data.type());
#ifdef FASTCV_X
        LOG(ERROR) << "wyb debug use fastcv";

        //LOGE("Debug ComplexMat * start");
        fcvMultiplyScalarf32((float*)data.data, data.channels()*data.cols, data.rows, data.channels()*data.cols*sizeof(float),
                             rhs, (float*)dst.data, dst.channels()*dst.cols*sizeof(float));
        //LOGE("Debug ComplexMat * end");

#else
        LOG(ERROR) << "wyb debug not use fastcv";

        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

        dst_eimat = rhs * lhs_eimat;
#endif
        return ComplexMat(dst);
    }
    ComplexMat operator+(const float & rhs) const
    {
        cv::Mat dst(data.size(), data.type());

#ifdef FASTCV_X
        //LOGE("Debug ComplexMat + start");
        fcvAddScalarf32((float*)data.data, data.cols*data.channels(), data.rows, data.channels()*data.cols*sizeof(float),
                        rhs, (float*)dst.data, dst.channels()*dst.cols*sizeof(float));
        //LOGE("Debug ComplexMat + end");

#else
        std::complex<float> *ptr_dst = (std::complex<float> *)dst.data;
        Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

        std::complex<float> *ptr_lhs = (std::complex<float> *)data.data;
        Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

        std::complex<float> temp_rhs = std::complex<float>(rhs, 0);
        Eigen::Map<EiMatrixXc> rhs_eimat(&temp_rhs, rows, cols);

        dst_eimat = lhs_eimat + rhs_eimat;
#endif
        return ComplexMat(dst);


    }

private:
    cv::Mat data;
    int cols;
    int rows;
};

//typedef ComplexMat_<float> ComplexMat;


#endif //ROBOT_TRACKING_COMPLEXMAT_H

#ifndef COMPLEX_MAT_HPP_213123048309482094
#define COMPLEX_MAT_HPP_213123048309482094

//#define EIGEN_USE_LAPACKE

#include "../../../../../common/eigen3/Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include <complex>


template<typename T> class ComplexMatt_
{
    typedef Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EiMatrixXc;

public:
    ComplexMatt_() : cols(0), rows(0), n_channels(0) {}

    ComplexMatt_(int _rows, int _cols, int _n_channels) : cols(_cols), rows(_rows), n_channels(_n_channels)
    {
        p_data.resize(n_channels);
    }

    //assuming that mat has 2 channels (real, img)
    ComplexMatt_(const cv::Mat & mat)
    {
        assert(mat.channels() <= 2);
        cv::Mat new_mat;
        if (mat.channels() == 2)
        {
            // ensure data is continuous
            new_mat = mat.isContinuous() ? mat : mat.clone();
        }
        else
        {
            std::vector<cv::Mat> src_vec;
            src_vec.resize(2);
            src_vec[0] = mat; // real
            src_vec[1] = cv::Mat::zeros(mat.size(), mat.type());
            cv::merge(src_vec, new_mat);
        }
        p_data.push_back(new_mat);

        rows = new_mat.rows;
        cols = new_mat.cols;
        n_channels = 1;
    }

    //assuming that mat has 2 channels (real, imag)
    void set_channel(int idx, const cv::Mat & mat)
    {
        assert(idx >= 0 && idx < n_channels);
        assert(mat.rows == rows && mat.cols == cols);
        p_data[idx] = mat.isContinuous() ? mat : mat.clone();
    }

    T sqr_norm() const
    {
        T sum_sqr_norm = 0;
//        for (int i = 0; i < n_channels; ++i)
//            for (auto lhs = p_data[i].begin(); lhs != p_data[i].end(); ++lhs)
//                sum_sqr_norm += lhs->real()*lhs->real() + lhs->imag()*lhs->imag();
//        return sum_sqr_norm / static_cast<T>(cols*rows);
        for (int i = 0; i < n_channels; ++i)
        {
            std::complex<T> *ptr_src = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

            sum_sqr_norm += src_eimat.cwiseAbs2().sum();
        }
        return sum_sqr_norm / static_cast<T>(cols*rows);
    }

    ComplexMatt_<T> sqr_mag() const
    {
//        return mat_const_operator( [](std::complex<T> & c) { c = c.real()*c.real() + c.imag()*c.imag(); } );
        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(rows, cols, p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_src = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

            dst_eimat = src_eimat.cwiseAbs2();
            res_cmat.set_channel(i, dst);
        }
        return  res_cmat;
    }

    ComplexMatt_<T> conj() const
    {
        //return mat_const_operator( [](std::complex<T> & c) { c = std::complex<T>(c.real(), -c.imag()); } );
        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(rows, cols, p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_src = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> src_eimat(ptr_src, rows, cols);

            dst_eimat = src_eimat.conjugate();
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }

    ComplexMatt_<T> sum_over_channels() const
    {
        assert(p_data.size() > 1);
//        ComplexMatt_<T> result(this->rows, this->cols, 1);
//        result.p_data[0] = p_data[0];
//        for (int i = 1; i < n_channels; ++i) {
//            std::transform(result.p_data[0].begin(), result.p_data[0].end(), p_data[i].begin(), result.p_data[0].begin(), std::plus<std::complex<T>>());
//        }
//        return result;

        cv::Mat sum_mat = cv::Mat::zeros(p_data[0].size(), p_data[0].type());
        for (int i = 0; i < n_channels; ++i)
        {
            sum_mat += p_data[i];
        }
        return ComplexMatt_(sum_mat);
    }

    //return 2 channels (real, imag) for first complex channel
    cv::Mat to_cv_mat() const
    {
        assert(p_data.size() >= 1);
        return p_data[0];
    }
    //return a vector of 2 channels (real, imag) per one complex channel
    std::vector<cv::Mat> to_cv_mat_vector() const
    {
        std::vector<cv::Mat> result;
        result.reserve(n_channels);

        for (int i = 0; i < n_channels; ++i)
            result.push_back(p_data[i]);

        return result;
    }

    //element-wise per channel multiplication, division and addition
    ComplexMatt_<T> operator*(const ComplexMatt_<T> & rhs) const
    {
//        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs) { c_lhs *= c_rhs; }, rhs);
        assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);
        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(p_data[i].size(), p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_lhs = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

            std::complex<T> *ptr_rhs = (std::complex<T> *)rhs.p_data[i].data;
            Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

            dst_eimat = lhs_eimat.cwiseProduct(rhs_eimat);
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }

    ComplexMatt_<T> operator/(const ComplexMatt_<T> & rhs) const
    {
//        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs) { c_lhs /= c_rhs; }, rhs);
        assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);
        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(p_data[i].size(), p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_lhs = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

            std::complex<T> *ptr_rhs = (std::complex<T> *)rhs.p_data[i].data;
            Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

            dst_eimat = lhs_eimat.cwiseProduct(rhs_eimat.cwiseInverse());
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }

    ComplexMatt_<T> operator+(const ComplexMatt_<T> & rhs) const
    {
//        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs)  { c_lhs += c_rhs; }, rhs);
        assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);
        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(p_data[i].size(), p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_lhs = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

            std::complex<T> *ptr_rhs = (std::complex<T> *)rhs.p_data[i].data;
            Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

            dst_eimat = lhs_eimat + rhs_eimat;
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }

    //multiplying or adding constant
    ComplexMatt_<T> operator*(const T & rhs) const
    {
        //return mat_const_operator( [&rhs](std::complex<T> & c) { c *= rhs; });

        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(p_data[i].size(), p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_lhs = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

            dst_eimat = rhs * lhs_eimat;
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }


    ComplexMatt_<T> operator+(const T & rhs) const
    {
//        return mat_const_operator( [&rhs](std::complex<T> & c) { c += rhs; });
        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(p_data[i].size(), p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_lhs = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

            dst_eimat = lhs_eimat + EiMatrixXc::Constant(rows, cols, std::complex<T>(rhs, 0));
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }

    //multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
    ComplexMatt_<T> mul(const ComplexMatt_<T> & rhs) const
    {
//        return matn_mat1_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs) { c_lhs *= c_rhs; }, rhs);
        assert(rhs.n_channels == 1 && rhs.cols == cols && rhs.rows == rows);

        std::complex<T> *ptr_rhs = (std::complex<T> *)rhs.p_data[0].data;
        Eigen::Map<EiMatrixXc> rhs_eimat(ptr_rhs, rows, cols);

        ComplexMatt_<T> res_cmat(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i)
        {
            cv::Mat dst(p_data[i].size(), p_data[i].type());
            std::complex<T> *ptr_dst = (std::complex<T> *)dst.data;
            Eigen::Map<EiMatrixXc> dst_eimat(ptr_dst, rows, cols);

            std::complex<T> *ptr_lhs = (std::complex<T> *)p_data[i].data;
            Eigen::Map<EiMatrixXc> lhs_eimat(ptr_lhs, rows, cols);

            dst_eimat = lhs_eimat.cwiseProduct(rhs_eimat);
            res_cmat.set_channel(i, dst);
        }
        return res_cmat;
    }

    //text output
//    friend std::ostream & operator<<(std::ostream & os, const ComplexMatt_<T> & mat)
//    {
//        //for (int i = 0; i < mat.n_channels; ++i){
//        for (int i = 0; i < 1; ++i){
//            os << "Channel " << i << std::endl;
//            for (int j = 0; j < mat.rows; ++j) {
//                for (int k = 0; k < mat.cols-1; ++k)
//                    os << mat.p_data[i][j*mat.cols + k] << ", ";
//                os << mat.p_data[i][j*mat.cols + mat.cols-1] << std::endl;
//            }
//        }
//        return os;
//    }

public:
    int cols;
    int rows;
    int n_channels;

private:
    std::vector<cv::Mat> p_data;

    //convert 2 channel mat (real, imag) to vector row-by-row
//    std::vector<std::complex<T>> convert(const cv::Mat & mat)
//    {
//        std::vector<std::complex<T>> result;
//        result.reserve(mat.cols*mat.rows);
//        for (int y = 0; y < mat.rows; ++y) {
//            const T * row_ptr = mat.ptr<T>(y);
//            for (int x = 0; x < 2*mat.cols; x += 2){
//                result.push_back(std::complex<T>(row_ptr[x], row_ptr[x+1]));
//            }
//        }
//        return result;
//    }
//
//    ComplexMatt_<T> mat_mat_operator(void (*op)(std::complex<T> & c_lhs, const std::complex<T> & c_rhs), const ComplexMatt_<T> & mat_rhs) const
//    {
//        assert(mat_rhs.n_channels == n_channels && mat_rhs.cols == cols && mat_rhs.rows == rows);
//
//        ComplexMatt_<T> result = *this;
//        for (int i = 0; i < n_channels; ++i) {
//            auto lhs = result.p_data[i].begin();
//            auto rhs = mat_rhs.p_data[i].begin();
//            for ( ; lhs != result.p_data[i].end(); ++lhs, ++rhs)
//                op(*lhs, *rhs);
//        }
//
//        return result;
//    }
//    ComplexMatt_<T> matn_mat1_operator(void (*op)(std::complex<T> & c_lhs, const std::complex<T> & c_rhs), const ComplexMatt_<T> & mat_rhs) const
//    {
//        assert(mat_rhs.n_channels == 1 && mat_rhs.cols == cols && mat_rhs.rows == rows);
//
//        ComplexMatt_<T> result = *this;
//        for (int i = 0; i < n_channels; ++i) {
//            auto lhs = result.p_data[i].begin();
//            auto rhs = mat_rhs.p_data[0].begin();
//            for ( ; lhs != result.p_data[i].end(); ++lhs, ++rhs)
//                op(*lhs, *rhs);
//        }
//
//        return result;
//    }
//    ComplexMatt_<T> mat_const_operator(const std::function<void(std::complex<T> & c_rhs)> & op) const
//    {
//        ComplexMatt_<T> result = *this;
//        for (int i = 0; i < n_channels; ++i)
//            for (auto lhs = result.p_data[i].begin(); lhs != result.p_data[i].end(); ++lhs)
//                op(*lhs);
//        return result;
//    }

//    cv::Mat channel_to_cv_mat(int channel_id) const
//    {
////        cv::Mat result(rows, cols, CV_32FC2);
////        int data_id = 0;
////        for (int y = 0; y < rows; ++y) {
////            T * row_ptr = result.ptr<T>(y);
////            for (int x = 0; x < 2*cols; x += 2){
////                row_ptr[x] = p_data[channel_id][data_id].real();
////                row_ptr[x+1] = p_data[channel_id][data_id++].imag();
////            }
////        }
////        return result;
//        return p_data[channel_id];
//    }

};

typedef ComplexMatt_<float> ComplexMatt;


#endif //COMPLEX_MAT_HPP_213123048309482094
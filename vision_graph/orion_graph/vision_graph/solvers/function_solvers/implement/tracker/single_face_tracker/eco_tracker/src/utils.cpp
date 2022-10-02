//
// Created by yuan on 17-10-17.
//

#include <logging.h>
#include "utils.h"
#include "complexmat.h"


void normalize_image(const cv::Mat& src, const cv::Scalar& mean_values, const cv::Scalar& std_values, cv::Mat& dst)
{
    int channels = src.channels();
    cv::Mat out = cv::Mat(src.rows, src.cols, CV_32FC3);
    for(int i = 0; i < src.rows; i++)
    {
        float* out_ptr = out.ptr<float>(i);
        const uchar* in_ptr = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++)
        {
            for (int k = 0; k < channels; k++)
            {
                out_ptr[j * channels + k] = (in_ptr[j * channels + k] / 256.f - mean_values[k]) / std_values[k];
            }
        }
    }
    dst = std::move(out);
//    dst = out;
}


float* split_bgrmat_channels(const cv::Mat& image)
{
    int row = image.rows;
    int col = image.cols;
    float *ptr_data = new float[row * col * 3];
    cv::Mat B(row, col, CV_32FC1, ptr_data);
    cv::Mat G(row, col, CV_32FC1, ptr_data + row * col);
    cv::Mat R(row, col, CV_32FC1, ptr_data + 2 * row * col);
    std::vector<cv::Mat> bgr;
    bgr.push_back(B);
    bgr.push_back(G);
    bgr.push_back(R);
    cv::split(image, bgr);
    return ptr_data;
}

void rot90(const cv::Mat& src, cv::Mat& dst, int rotflag)
{
    //1=CW, 2=CCW, 3=180

    if (rotflag == ROTATE_0)
    {
        dst = src.clone();
    }
    else if (rotflag == ROTATE_90_CLOCKWISE)
    {
        cv::Mat temp;
        transpose(src, temp);
        flip(temp, dst, 1); //transpose+flip(1)=CW
    }
    else if (rotflag == ROTATE_90_COUNTERCLOCKWISE)
    {
        cv::Mat temp;
        transpose(src, temp);
        flip(temp, dst,0); //transpose+flip(0)=CCW
    }
    else if (rotflag == ROTATE_180)
    {
        flip(src, dst, -1); //flip(-1)=180
    }
//    else
//    { //if not 0,1,2,3:
//       DLOG(INFO)  << "Unknown rotation flag(" << rotflag << ")" ;
//    }
}


cv::Point2f get_rect_center(const cv::Rect& box)
{
    return cv::Point2f(box.x + 0.5f * box.width, box.y + 0.5 * box.height);
}


cv::Rect scale_rect(cv::Rect src, float scale)
{
    cv::Rect dst;
    dst.x = (int)round(src.x * scale);
    dst.y = (int)round(src.y * scale);
    dst.width = (int)round(src.width * scale);
    dst.height = (int)round(src.height * scale);
    return dst;
}


cv::Rect_<float> scale_rect(cv::Rect_<float> src, float scale)
{
    cv::Rect_<float> dst;
    dst.x = src.x * scale;
    dst.y = src.y * scale;
    dst.width = src.width * scale;
    dst.height = src.height * scale;
    return dst;
}


cv::Rect clip_rect(cv::Rect src, cv::Size img_sz)
{
    cv::Rect dst;
    dst.x = std::min(std::max(0, src.x), img_sz.width - 1);
    dst.y = std::min(std::max(0, src.y), img_sz.height - 1);
    dst.width = std::max(std::min(src.width, img_sz.width - dst.x), 0);
    dst.height = std::max(std::min(src.height, img_sz.height - dst.y), 0);
    return dst;
}


void conv2(const cv::Mat &src, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dst)
{
    cv::Mat source = src.clone();
    cv::Mat kernel_flip;
    flip(kernel, kernel_flip, -1);
    if(type == CONV_FULL)
    {
        source = cv::Mat();
        const int additionalRows = kernel_flip.rows-1, additionalCols = kernel_flip.cols-1;
        cv::copyMakeBorder(src, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    cv::Point anchor(kernel_flip.cols - kernel_flip.cols/2 - 1, kernel_flip.rows - kernel_flip.rows/2 - 1);
    int borderMode = cv::BORDER_CONSTANT;
    cv::filter2D(source, dst, src.depth(), kernel_flip, anchor, 0, borderMode);

    if(type == CONV_VALID)
    {
        dst = dst.colRange((kernel_flip.cols-1)/2, dst.cols - kernel_flip.cols/2)
                .rowRange((kernel_flip.rows-1)/2, dst.rows - kernel_flip.rows/2);
    }
}

void conv2_complex(const cv::Mat &src, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dst)
{
    assert(src.channels() == 2 && kernel.channels() < 3);
    if (kernel.channels() == 1)
    {
        std::vector<cv::Mat> src_vec;
        cv::split(src, src_vec);
        conv2(src_vec[0], kernel, type, src_vec[0]);
        conv2(src_vec[1], kernel, type, src_vec[1]);
        cv::merge(src_vec, dst);
    }
    else
    {
        std::vector<cv::Mat> src_vec;
        cv::split(src, src_vec);

        std::vector<cv::Mat> kernel_vec;
        cv::split(kernel, kernel_vec);

        std::vector<cv::Mat> res_vec;
        res_vec.resize(4);
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                conv2(src_vec[i], kernel_vec[j], type, res_vec[2 * i + j]);
            }
        }

        std::vector<cv::Mat> dst_vec;
        dst_vec.resize(2);
        // real
        dst_vec[0] = res_vec[0] - res_vec[3];
        // imag
        dst_vec[1] = res_vec[1] + res_vec[2];
        cv::merge(dst_vec, dst);
    }
}

// src1 * alpha + src2 * beta
void vec3dmat_linear_opt(Vec3dMat& dst, const Vec3dMat& src1, const Vec3dMat& src2, float alpha, float beta)
{
    Vec3dMat out_res;
    out_res.resize(src1.size());
    for (int i = 0; i < src1.size(); ++i)
    {
        out_res[i].resize(src1[i].size());
        for (int j = 0; j < src1[i].size(); ++j)
        {
            out_res[i][j].resize(src1[i][j].size());
            for (int k = 0; k < src1[i][j].size(); ++k)
            {
                out_res[i][j][k] = src1[i][j][k] * alpha + src2[i][j][k] * beta;
            }
        }
    }
    dst = std::move(out_res);
//    dst = out_res;
}

// src1 * alpha + src2 * beta
void vec2dmat_linear_opt(Vec2dMat& dst, const Vec2dMat& src1, const Vec2dMat& src2, float alpha, float beta)
{
    Vec2dMat out;
    out.resize(src1.size());
    for (int i = 0; i < src1.size(); ++i)
    {
        out[i].resize(src1[i].size());
        for (int j = 0; j < src1[i].size(); ++j)
        {
            out[i][j] = src1[i][j] * alpha + src2[i][j] * beta;
        }
    }
    dst = std::move(out);
//    dst = out;
}

// src1 * alpha + src2 * beta
void vec1dmat_linear_opt(Vec1dMat& dst, const Vec1dMat& src1, const Vec1dMat& src2, float alpha, float beta)
{
    Vec1dMat out(src1.size());
    for (int i = 0; i < src1.size(); ++i)
    {
        out[i] = src1[i] * alpha + src2[i] * beta;
    }
    dst = std::move(out);
}


void proc_batch_vec1dmat(const std::function<void(const cv::Mat&, cv::Mat&)>& fun_handle,
                         const Vec1dMat& src, Vec1dMat& dst)
{
    Vec1dMat res_out;
    res_out.resize(src.size());
    for (int i = 0; i < src.size(); ++i)
    {
        fun_handle(src[i], res_out[i]);
    }
    dst = std::move(res_out);
//    dst = res_out;
}

void proc_batch_vec2dmat(const std::function<void(const cv::Mat&, cv::Mat&)>& fun_handle,
                         const Vec2dMat& src, Vec2dMat& dst)
{
    Vec2dMat res_out;
    res_out.resize(src.size());
    for (int i = 0; i < src.size(); ++i)
    {
        res_out[i].resize(src[i].size());
        for (int j = 0; j < src[i].size(); ++j)
        {
            fun_handle(src[i][j], res_out[i][j]);
        }
    }
    dst = std::move(res_out);
//    dst = res_out;
}

void proc_batch_vec3dmat(const std::function<void(const cv::Mat&, cv::Mat&)>& fun_handle,
                         const Vec3dMat& src, Vec3dMat& dst)
{
    Vec3dMat res_out;
    res_out.resize(src.size());
    for (int i = 0; i < src.size(); ++i)
    {
        res_out[i].resize(src[i].size());
        for (int j = 0; j < src[i].size(); ++j)
        {
            res_out[i][j].resize(src[i][j].size());
            for (int k = 0; k < src[i][j].size(); ++k)
            {
                fun_handle(src[i][j][k], res_out[i][j][k]);
            }
        }
    }
    dst = std::move(res_out);
//    dst = res_out;
}

void init_vec1dmat(int capacity, cv::Size sz, int type, Vec1dMat& dst)
{
    dst.resize(capacity);
    for (int i = 0; i < capacity; ++i)
    {
        dst[i] = cv::Mat::zeros(sz, type);
    }
}


int mod(int num, int base)
{
    int res = num - base * (int)floor(num / (float)base);
    return res;
}

cv::Mat fcv_mul(const cv::Mat& src1,const cv::Mat& src2)
{
    assert(src1.size() == src2.size() && src1.channels()==src2.channels());
    cv::Mat dst(src1.rows, src1.cols, src1.type());
    //LOGE("Debug fcv_mul start");
#ifdef FASTCV_X
    fcvElementMultiplyf32((float*)src1.data, src1.cols*src1.channels(), src1.rows, src1.channels()*src1.cols*sizeof(float), (float*)src2.data,
                          src2.channels()*src2.cols*sizeof(float), (float*)dst.data, src1.channels()*dst.cols*sizeof(float));
#else
    dst = src1 * src2;
#endif
    //LOGE("Debug fcv_mul end");

    return dst;
}
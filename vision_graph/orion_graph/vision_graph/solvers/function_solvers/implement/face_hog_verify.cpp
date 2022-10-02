#include <stdio.h>
#include <string.h>
#include "face_hog_verify.h"
#include <logging.h>
#include "face_hog_const.h"
#include "../../../common/matrix.h"

namespace vision_graph{

static const int kKptsEyeIdx[]     = { 52, 53, 72, 54, 56, 57, 73, 74, 55, 104, 58, 59, 60, 61, 62, 63, 75, 76, 105, 77 } ;
static const int kKptsEyeBrowIdx[] = { 33, 34, 35, 36, 37, 38, 39, 40, 41, 64, 65, 66, 67, 68, 69, 70, 71, 42 } ;
static const int kKptsNoseIdx[]    = { 43, 44, 45, 46, 47, 48, 49, 50, 51, 78, 79, 80, 81, 82, 83 } ;
static const int kKptsMouthIdx[]   = { 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103 } ;

static const int kKptsSelectIdx[FaceHOGVerify::kKptsSelCnt] = {
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,
    34,35,36,37,38,39,40,41,
    44,45,46,
    49,
    52,53,54,55,56,57,58,59,
    60,61,62,63,
    72,75,
    80,81,82,83,84,85,87,89,
    90,91,93,95,97,98,99,
    101,102,103
};

static const float    kPI     = 3.141592653590f ;
static const float    kPI2    = 6.283185307180f ;
static const float    kPIInv  = 0.318309886184f ;
static const float    kPIInv4 = 1.273239544736f ;

FaceHOGVerify::FaceHOGVerify() noexcept : affined_image_(nullptr), work_buf_(nullptr), work_buf_len_(0)
{

}

FaceHOGVerify::~FaceHOGVerify() noexcept
{
    if(nullptr != affined_image_)
        delete affined_image_;
    affined_image_ = nullptr;

    if(nullptr != work_buf_)
        delete work_buf_;
    work_buf_ = nullptr;  
}

int FaceHOGVerify::SetTargetDefault() noexcept
{
    int i = 0;
    for(i = 0 ; i < kKptsSelCnt ; i ++)
    {
        target_kpts_[i].x = mean_face_kpts_61[2 * i];
        target_kpts_[i].y = mean_face_kpts_61[2 * i + 1];
    }
    memcpy(target_face_hog_, mean_face_hog_feat, kHOGFeatureLen * sizeof(float));

    return 0;
}

int FaceHOGVerify::SetTargetFromFile(std::string const& kpts_file, std::string const& hog_desc_file) noexcept
{
    FILE*   kpts_data_file = fopen(kpts_file.c_str(), "rb");
    FILE*   hog_data_file  = fopen(hog_desc_file.c_str(), "rb");
    if(nullptr == kpts_data_file || nullptr == hog_data_file)
    {
        LOG(ERROR) << "Error: \"kpts\" or \"hog\" file is invalid" << kpts_file << ", " << hog_desc_file;
        ABORT();
    }
    fseek (kpts_data_file, 0, SEEK_END);
    int    kpts_data_len  = ftell (kpts_data_file); 
    fseek (kpts_data_file, 0, SEEK_SET);
    fseek (hog_data_file, 0, SEEK_END);
    int    hog_data_len   = ftell (hog_data_file); 
    fseek (hog_data_file, 0, SEEK_SET);

    
    if(kpts_data_len < (int)(kKptsSelCnt * sizeof(KEYPOINT_2D)) ||
        hog_data_len < (int)(kHOGFeatureLen * sizeof(float)))
    {
        LOG(ERROR) << "Error: \"kpts\" or \"hog\" data len is invalid" << kpts_data_len << ", " << hog_data_file;
        ABORT();
    }
    size_t file_size = 0;
    file_size = fread(target_kpts_,     kKptsSelCnt * sizeof(KEYPOINT_2D), 1, kpts_data_file);
    file_size = fread(target_face_hog_, kHOGFeatureLen * sizeof(float),    1, hog_data_file);
    (void)file_size;
    fclose(kpts_data_file);
    fclose(hog_data_file);

    return 0;
}

int FaceHOGVerify::SetTargetData(TensorKeypoints const& kpts, TensorFeature const& hog_desc_feature) noexcept
{
    int i = 0 ;
    for(i = 0 ; i < kKptsSelCnt ; i ++)
    {
        target_kpts_[i].x = kpts[i].x;
        target_kpts_[i].y = kpts[i].y;
    }
    memcpy(target_face_hog_, &(hog_desc_feature[0]), kHOGFeatureLen * sizeof(float));
    norm_target_kpts(target_kpts_);

    return 0;
}

int FaceHOGVerify::Init() noexcept
{
    norm_target_kpts(target_kpts_);
    affined_image_      = new unsigned char[kKptsNormRegressionSize * kKptsNormRegressionSize];
    alloc_work_buf();
    generate_hog_gaussian(hog_gaus_, kHOGFeatureWndSize, 0.5f * kHOGFeatureWndSize);

    return 0;
}

float FaceHOGVerify::Solve(cv::Mat const& image, TensorKeypoints const& kpts) noexcept
{
    int i = 0 ;

    KEYPOINT_2D offset = { 0.0f, 0.0f };
    select_kpts(kpts, offset, real_kpts_, true, work_buf_);

    AFFINE_PARAM_2D   affine_result, affine_inv;
    init_affine_param(affine_result);
    init_affine_param(affine_inv);

    const unsigned char* image_buf = image.data;
    if(3 == image.channels())
    {
        cv::cvtColor(image, gray_image_, cv::COLOR_BGR2GRAY);
        image_buf = gray_image_.data;
    } 
    else if(4 == image.channels())
    {
        cv::cvtColor(image, gray_image_, cv::COLOR_BGRA2GRAY);
        image_buf = gray_image_.data;
    }

    //memset(real_kpts_, 0, kKptsSelCnt * sizeof(KEYPOINT_2D));
    bool res = cal_affine_transform_param(target_kpts_, real_kpts_, kKptsSelCnt, &affine_result, work_buf_);
    if(false == res)
        return 0.0f ;
    affine_transform_8bit_gray(image_buf, 
                               image.cols, 
                               image.rows, 
                               affined_image_, 
                               kKptsNormRegressionSize, 
                               kKptsNormRegressionSize, 
                               &affine_result,
                               work_buf_);
    //cv::Mat affine_mat(kKptsNormRegressionSize, kKptsNormRegressionSize, CV_8UC1, affined_image_);
    //cv::imwrite(std::string("/home/shaquille/WorkSpace/orion_workspace/graph_workspace/build/Debug/x86_64-linux/vision_graph/demos/affine.png"), affined_image_)
    res = cal_affine_inv(&affine_result, &affine_inv);
    if(false == res)
        return 0.0f ;
    for(i = 0; i < kKptsSelCnt; ++i)
	{
		real_kpts_norm_[i].x = real_kpts_[i].x * affine_inv.rot_x - real_kpts_[i].y * affine_inv.rot_y + affine_inv.x;
		real_kpts_norm_[i].y = real_kpts_[i].y * affine_inv.rot_x + real_kpts_[i].x * affine_inv.rot_y + affine_inv.y;
	}
    memset(work_buf_, 0, work_buf_len_);
    assign_buf();
    sobel_image(affined_image_, 
                kKptsNormRegressionSize, 
                kKptsNormRegressionSize, 
                grad_x_image_, 
                grad_y_image_, 
                magnitude_image_, 
                ort_image_);
    extract_hog(real_kpts_norm_, kKptsSelCnt);

    float similarity = compare_hog(hog_feature_, target_face_hog_, kHOGFeatureLen);

    return similarity;
}

void FaceHOGVerify::alloc_work_buf() noexcept
{
    int grad_buf_len    = kKptsNormRegressionSize * kKptsNormRegressionSize * sizeof(unsigned short) +              //grad_x_image_
                          kKptsNormRegressionSize * kKptsNormRegressionSize * sizeof(unsigned short) +              //grad_y_image_
                          kKptsNormRegressionSize * kKptsNormRegressionSize * sizeof(unsigned short) +              //magnitude_image_
                          kKptsNormRegressionSize * kKptsNormRegressionSize * sizeof(int);                          //ort_image_
    int hog_buf_len     = 2 * kKptsSelCnt * sizeof(float) +                                                         //kpts_left_up_
                          2 * kKptsSelCnt * sizeof(float) +                                                         //kpts_right_up_
                          2 * kKptsSelCnt * sizeof(float) +                                                         //kpts_left_down_
                          2 * kKptsSelCnt * sizeof(float) +                                                         //kpts_right_down_
                          kHOGFeatureLen * sizeof(float) +                                                          //hog_left_up_
                          kHOGFeatureLen * sizeof(float) +                                                          //hog_right_up_
                          kHOGFeatureLen * sizeof(float) +                                                          //hog_left_down_
                          kHOGFeatureLen * sizeof(float) +                                                          //hog_right_down_
                          kKptsNormRegressionSize * kKptsNormRegressionSize * kHOGDescriptorCount * sizeof(float) + //img_hog_feature_
                          kKptsNormRegressionSize * kKptsNormRegressionSize ;                                       //feature_proc_flag_
    work_buf_len_       = grad_buf_len + hog_buf_len;
    work_buf_           = new unsigned char[work_buf_len_];
    memset(work_buf_, 0, work_buf_len_);
}

void FaceHOGVerify::assign_buf() noexcept
{
    grad_x_image_      = (unsigned short int*)work_buf_;                                                //size: kKptsNormRegressionSize * kKptsNormRegressionSize
    grad_y_image_      = grad_x_image_ + kKptsNormRegressionSize * kKptsNormRegressionSize;             //size: kKptsNormRegressionSize * kKptsNormRegressionSize
    magnitude_image_   = grad_y_image_ + kKptsNormRegressionSize * kKptsNormRegressionSize;             //size: kKptsNormRegressionSize * kKptsNormRegressionSize
    ort_image_         = (int*)(magnitude_image_ + kKptsNormRegressionSize * kKptsNormRegressionSize);  //size: kKptsNormRegressionSize * kKptsNormRegressionSize
    kpts_left_up_      = (float*)(ort_image_ + kKptsNormRegressionSize * kKptsNormRegressionSize);      //size: 2 * kKptsSelCnt
    kpts_right_up_     = kpts_left_up_    + 2 * kKptsSelCnt;                                            //size: 2 * kKptsSelCnt
    kpts_left_down_    = kpts_right_up_   + 2 * kKptsSelCnt;                                            //size: 2 * kKptsSelCnt
    kpts_right_down_   = kpts_left_down_  + 2 * kKptsSelCnt;                                            //size: 2 * kKptsSelCnt
    hog_left_up_       = kpts_right_down_ + 2 * kKptsSelCnt;                                            //size: kHOGFeatureLen
    hog_right_up_      = hog_left_up_     + kHOGFeatureLen;                                             //size: kHOGFeatureLen
    hog_left_down_     = hog_right_up_    + kHOGFeatureLen;                                             //size: kHOGFeatureLen
    hog_right_down_    = hog_left_down_   + kHOGFeatureLen;                                             //size: kHOGFeatureLen
    img_hog_feature_   = hog_right_down_  + kHOGFeatureLen;                                             //size: kKptsNormRegressionSize * kKptsNormRegressionSize * kHOGDescriptorCount
    feature_proc_flag_ = (unsigned char*)(hog_right_down_  + kHOGFeatureLen);                           //size: kKptsNormRegressionSize * kKptsNormRegressionSize
}

void FaceHOGVerify::generate_hog_gaussian(unsigned short int* g, int kernel, float sigma)
{
    int    i = 0, j = 0;
    float  half_kernel = kernel / 2 - 0.5;
    float *g_value     = new float[kernel * kernel];
    float  temp        = 0.0f;
    float  x = 0.0f, y = 0.0f;
    static const double kPI64 = 3.1415926535897932384626433832795;				
    for (i = 0; i < kernel; i++) 
    {
        for (j = 0; j < kernel; j++) 
        {
            x                       = j - half_kernel;
            y                       = i - half_kernel;
            g_value[i * kernel + j] = 1.0 / (2 * kPI64 * sigma * sigma) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));
        }
    }

    for (i = 0; i < kernel; i++) 
    {
        for (j = 0; j < kernel; j++)
        {
            g[i * kernel + j] = int(g_value[i * kernel + j] * 1024 * 1024 + 0.5);
        }
    }
    delete[] g_value;
}

void FaceHOGVerify::norm_target_kpts(KEYPOINT_2D* kpts)
{
    for(int i = 0 ; i < kKptsSelCnt ; i ++)
    {
        kpts[i].x = kpts[i].x + (kKptsNormRegressionSize >> 1) - (kKptsNormTextureSize >> 1);
	    kpts[i].y = kpts[i].y + (kKptsNormRegressionSize >> 1) - (kKptsNormTextureSize >> 1);
    }
}

void FaceHOGVerify::select_kpts(TensorKeypoints const& src_kpt, KEYPOINT_2D const& offset, KEYPOINT_2D* selected_kpts, bool re_order, unsigned char* work_buf) noexcept
{
    int    i = 0 ;
    float* outx       = (float*)work_buf;
    float* outy       = outx + 106;
    float* re_order_x = outy + 106;
    float* re_order_y = re_order_x + 106;

    for (i = 33; i < 53; i++) 
    {
        outx[kKptsEyeIdx[i - 33]] = src_kpt[i].x;
        outy[kKptsEyeIdx[i - 33]] = src_kpt[i].y;
    }

    for (i = 53; i < 71; i++) 
    {
        outx[kKptsEyeBrowIdx[i - 53]] = src_kpt[i].x;
        outy[kKptsEyeBrowIdx[i - 53]] = src_kpt[i].y;
    }

    for (i = 71; i < 86; i++) 
    {
        outx[kKptsNoseIdx[i - 71]] = src_kpt[i].x;
        outy[kKptsNoseIdx[i - 71]] = src_kpt[i].y;
    }

    for (i = 86; i < 106; i++) 
    {
        outx[kKptsMouthIdx[i - 86]] = src_kpt[i].x;
        outy[kKptsMouthIdx[i - 86]] = src_kpt[i].y;
    }

    if(false == re_order)
    {
        for(i = 0 ; i < kKptsSelCnt ; i ++)
        {
            selected_kpts[i].x = src_kpt[kKptsSelectIdx[i]].x - offset.x;
            selected_kpts[i].y = src_kpt[kKptsSelectIdx[i]].y - offset.y;
        }
    }
    else
    {
        for(i = 0 ; i < 33 ; i ++)
        {
            re_order_x[i] = src_kpt[i].x;
            re_order_y[i] = src_kpt[i].y;
        }
        for (i = 33; i < 106; i++) 
        {
            re_order_x[i] = outx[i];
            re_order_y[i] = outy[i];
        }

        for(i = 0 ; i < kKptsSelCnt ; i ++)
        {
            selected_kpts[i].x = re_order_x[kKptsSelectIdx[i]] - offset.x;
            selected_kpts[i].y = re_order_y[kKptsSelectIdx[i]] - offset.y;
        }
    }
    
}

bool FaceHOGVerify::cal_affine_transform_param(const KEYPOINT_2D* target_kpts, 
                                               const KEYPOINT_2D* real_kpts, 
                                               int                point_count, 
                                               AFFINE_PARAM_2D*   result,
                                               unsigned char*     work_buf) noexcept
{
    float *X = nullptr, *A = nullptr, *B = nullptr;
    float *temp = nullptr, *TA = nullptr, *inv = nullptr;
    int    solution_count = 4, rows = point_count * 2;
    int    i  = 0, ii = 0;
    int    n1 = 0, n2 = 0;
    float* new_buf = (float*)work_buf;

    X    = new_buf ;                               //size: solution_count
    A    = X  + solution_count;                    //size: point_count * solution_count * 2
    TA   = A  + point_count * solution_count * 2;  //size: point_count * solution_count * 2
    B    = TA + point_count * solution_count * 2;  //size: point_count * 2
    temp = B  + point_count * 2;                   //size: solution_count * solution_count
    inv  = temp + solution_count * solution_count; //size: solution_count * solution_count

    for (i = 0; i < point_count; ++i) 
    {
        ii        = (i << 1);
        n1        = ii * solution_count;
        n2        = (ii + 1) * solution_count;
        B[ii]     = real_kpts[i].x;
        B[ii + 1] = real_kpts[i].y;
        A[n1]     = target_kpts[i].x;
        A[n1 + 1] = -target_kpts[i].y;
        A[n1 + 2] = 1.0f;
        A[n1 + 3] = 0.0f;
        A[n2]     = target_kpts[i].y;
        A[n2 + 1] = target_kpts[i].x;
        A[n2 + 2] = 0.0f;
        A[n2 + 3] = 1.0f;
    }

    matrix_transpose(A, rows, solution_count, TA);
    matrix_multiply(TA, solution_count, rows, A, solution_count, temp);
    bool res = matrix_inv(temp, solution_count, inv);
    if(true == res)
    {
        matrix_multiply(TA, solution_count, rows, B, 1, A);
        matrix_multiply(inv, solution_count, solution_count, A, 1, X);
        result->rot_x = X[0];
        result->rot_y = X[1];
        result->x     = X[2];
        result->y     = X[3];
    }

    return res;
}

void FaceHOGVerify::affine_transform_8bit_gray(const unsigned char*    src_image, 
                                               int                     src_w, 
                                               int                     src_h, 
                                               unsigned char*          dst_image,
                                               int                     dst_w,
                                               int                     dst_h,
                                               const AFFINE_PARAM_2D*  affine_param,
                                               unsigned char*          work_buf) noexcept
{
    int    i  = 0,    j  = 0;
    float  x1 = 0.0f, y1 = 0.0f;
    int    dst_w_h_max   = (dst_w > dst_h ? dst_w : dst_h);
    float *rx = (float*)work_buf;
    float *ry = rx + dst_w_h_max;
    float  tx1 = 0.0f, ty1 = 0.0f;
    for (i = 0; i < dst_w_h_max; ++i)
        rx[i] = affine_param->rot_x * i ;
    for (i = 0; i < dst_w_h_max; ++i)
        ry[i] = affine_param->rot_y * i ;

    //todo can be optimaized.
    for (i = 0; i < dst_h; ++i) 
    {
        tx1 = -ry[i] + affine_param->x;
        ty1 =  rx[i] + affine_param->y;
        for (j = 0; j < dst_w; ++j) 
        {
            x1                       = rx[j] + tx1;
            y1                       = ry[j] + ty1;
            dst_image[i * dst_w + j] = 0;
            if (x1 < 0 || y1 < 0 || x1 >= src_w - 1 || y1 >= src_h - 1) {
                continue;
            }
            int x_int    = int(x1), y_int = int(y1);
            float x_tail = x1 - x_int, y_tail = y1 - y_int;
            int x_round  = x_int + 1, y_round = y_int + 1;
            float pixel1 = src_image[y_int * src_w + x_int] * (1.0f - x_tail) +
                           src_image[y_int * src_w + x_round] * x_tail;
            float pixel2 = src_image[y_round * src_w + x_int] * (1.0f - x_tail) +
                           src_image[y_round * src_w + x_round] * x_tail;
            dst_image[i * dst_w + j] = int(pixel1 * (1.0f - y_tail) + pixel2 * y_tail + 0.5);
        }
    }
}

bool FaceHOGVerify::cal_affine_inv(AFFINE_PARAM_2D const* src_affine_param, AFFINE_PARAM_2D* inv_affine_param) noexcept
{
    float inv  = src_affine_param->rot_x * src_affine_param->rot_x + src_affine_param->rot_y * src_affine_param->rot_y;
    if(fabsf(inv) < 1e-6)
        return false;
    float temp = 1.0f / inv;
    inv_affine_param->rot_x =  temp * src_affine_param->rot_x;
    inv_affine_param->rot_y = -temp * src_affine_param->rot_y;
    inv_affine_param->x     = 1.0f - (inv_affine_param->rot_x * (src_affine_param->rot_x + src_affine_param->x) - inv_affine_param->rot_y * (src_affine_param->rot_y + src_affine_param->y));
    inv_affine_param->y     =       -(inv_affine_param->rot_y * (src_affine_param->rot_x + src_affine_param->x) + inv_affine_param->rot_x * (src_affine_param->rot_y + src_affine_param->y));

    return true;
}

void FaceHOGVerify::sobel_image(unsigned char const* src_image, 
                                int                  w, 
                                int                  h, 
                                unsigned short int*  grad_x,
                                unsigned short int*  grad_y,
                                unsigned short int*  magnitutde_image, 
                                int*                 ort_image) noexcept
{
    int i = 0;
    sobel_image_x(src_image, w, h, grad_x);
    sobel_image_y(src_image, w, h, grad_y);
    for (i = 0; i < w * h; i++) 
    {
        short dx = grad_x[i];
        short dy = grad_y[i];
        magnitutde_image[i] = int(sqrtf(float(dx * dx + dy * dy)) + 0.5f);
        float theta         = atan2f(float(dy), float(dx));
        //cout<<theta<<endl;
        if (theta < 0.0f)
            theta += kPI2;
        ort_image[i] = int(theta * kPIInv4);
    }
}

void FaceHOGVerify::sobel_image_x(unsigned char const* src_image, int w, int h, unsigned short int* grad_image) noexcept
{
    static int matrix_x[25] = {2, 1, 0, -1, -2,
                               3, 2, 0, -2, -3,
                               4, 3, 0, -3, -4,
                               3, 2, 0, -2, -3,
                               2, 1, 0, -1, -2};
    // int sobel_wd = 5, half_sobel_wd = 2;
    int i, j, k, l, n;
    int x, y;
    static const int kRadius = 2;
    for (i = kRadius; i < h - kRadius; ++i)
    {
        for (j = kRadius; j < w - kRadius; ++j) 
        {
            n = 0;
            for (k = -kRadius; k <= kRadius; ++k)
            {
                for (l = -kRadius; l <= kRadius; ++l) 
                {
                    x = j + l;
                    y = i + k;
                    grad_image[i * w + j] += -src_image[y * w + x] * matrix_x[n];
                    n++;
                }
            }
        }
    }
}

void FaceHOGVerify::sobel_image_y(unsigned char const* src_image, int w, int h, unsigned short int* grad_image) noexcept
{
    static int matrix_y[25] = {2, 3, 4, 3, 2,
                               1, 2, 3, 2, 1,
                               0, 0, 0, 0, 0,
                               -1, -2, -3, -2, -1,
                               -2, -3, -4, -3, -2};
    // int sobel_wd = 5, half_sobel_wd = 2;
    int i, j, k, l, n;
    int x, y;
    static const int kRadius = 2;

    for (i = kRadius; i < h - kRadius; ++i)
    {
        for (j = kRadius; j < w - kRadius; ++j) 
        {
            n = 0;
            for (k = -kRadius; k <= kRadius; ++k)
            {
                for (l = -kRadius; l <= kRadius; ++l) 
                {
                    x = j + l;
                    y = i + k;
                    grad_image[i * w + j] += -src_image[y * w + x] * matrix_y[n];
                    n++;
                }
            }
        }
    }
}

void FaceHOGVerify::extract_hog(KEYPOINT_2D const* src_kpts, int point_count) noexcept
{
    static const float  diff_eps = 1e-5;
    float               temp, b1, b2, b3, b4;
    int                 i = 0, j = 0, int_x = 0, int_y = 0;
    int                 w = kKptsNormRegressionSize;
    int                 h = kKptsNormRegressionSize;

    for (i = 0; i < point_count; i++) 
    {
        int_x                  = (int)(src_kpts[i].x);
        kpts_left_up_[2 * i]   = int_x;
        kpts_left_down_[2 * i] = int_x;
        temp                   = src_kpts[i].x - (float)(int_x);
        if (temp < diff_eps)
        {
            kpts_right_up_[2 * i]   = int_x;
            kpts_right_down_[2 * i] = int_x;
        } 
        else 
        {
            kpts_right_up_[2 * i]   = int_x + 1;
            kpts_right_down_[2 * i] = int_x + 1;
        }

        int_y                     = (int)(src_kpts[i].y);
        kpts_left_up_[2 * i + 1]  = int_y;
        kpts_right_up_[2 * i + 1] = int_y;
        temp                      = src_kpts[i].y - (float)(int_y);
        if (temp < diff_eps) 
        {
            kpts_left_down_[2 * i + 1]  = int_y;
            kpts_right_down_[2 * i + 1] = int_y;
        } 
        else 
        {
            kpts_left_down_[2 * i + 1]  = int_y + 1;
            kpts_right_down_[2 * i + 1] = int_y + 1;
        }
    }
    extract_kpt_desc(w, h, kpts_left_up_,    point_count, hog_gaus_, magnitude_image_, ort_image_, img_hog_feature_, feature_proc_flag_, hog_left_up_) ;
    extract_kpt_desc(w, h, kpts_right_up_,   point_count, hog_gaus_, magnitude_image_, ort_image_, img_hog_feature_, feature_proc_flag_, hog_right_up_) ;
    extract_kpt_desc(w, h, kpts_left_down_,  point_count, hog_gaus_, magnitude_image_, ort_image_, img_hog_feature_, feature_proc_flag_, hog_left_down_) ;
    extract_kpt_desc(w, h, kpts_right_down_, point_count, hog_gaus_, magnitude_image_, ort_image_, img_hog_feature_, feature_proc_flag_, hog_right_down_) ;
    //todo
    for (i = 0; i < point_count; i++) 
    {
        float x = src_kpts[i].x - (int) (src_kpts[i].x);
        float y = src_kpts[i].y - (int) (src_kpts[i].y);
        for (j = 0; j < kHOGDescriptorCount; j++) 
        {
            b1                                        = hog_left_up_[i * kHOGDescriptorCount + j];
            b2                                        = hog_right_up_[i * kHOGDescriptorCount + j] - b1;
            b3                                        = hog_left_down_[i * kHOGDescriptorCount + j] - b1;
            b4                                        = hog_right_down_[i * kHOGDescriptorCount + j] - b2 - b3 - b1;
            hog_feature_[i * kHOGDescriptorCount + j] = b1 + b2 * x + b3 * y + b4 * x * y;
        }
    }
}

void FaceHOGVerify::extract_kpt_desc(int                      w, 
                                     int                      h, 
                                     float const*             kpts_xy, 
                                     int                      point_count, 
                                     unsigned short const*    gaus_weight,
                                     unsigned short const*    img_grad,
                                     int const*               img_ort,
                                     float*                   img_hog_feature,
                                     unsigned char*           feature_cal_flag,
                                     float*                   hog) noexcept
{
    int                 i = 0, j = 0, k = 0, x = 0, y = 0, ipt = 0, n = 0;
    int                 dis_x = 0, dis_y = 0 ;
    int                 pt_x  = 0, pt_y  = 0 ;
    static const int    kSearchBlockCount       = 4;
    static const int    kSearchBlockSize        = 4;
    static const int    kHalfWndSize            = 8;
    static const float  kFeatureThreshold       = 0.2f;
    int                 filter_sum[kHOGFeatureWndSize*kHOGFeatureWndSize];
    float               fv[kHOGDescriptorCount];
    unsigned int        hist[kHOGHistBinCount];
    int                 feat_index              = 0;
    for (ipt = 0; ipt < point_count; ipt++) 
    {
        pt_x = kpts_xy[ipt * 2];
        pt_y = kpts_xy[ipt * 2 + 1];
        if (pt_x < kHalfWndSize - 1)      pt_x = kHalfWndSize - 1;
        if (pt_x >= w - kHalfWndSize)     pt_x = w - kHalfWndSize - 1;
        if (pt_y < kHalfWndSize - 1)      pt_y = kHalfWndSize - 1;
        if (pt_y >= h - kHalfWndSize)     pt_y = h - kHalfWndSize - 1;

        if (feature_cal_flag[pt_y * w + pt_x]) 
        {
            memcpy(hog + ipt * kHOGDescriptorCount, 
                   img_hog_feature + (pt_y * w + pt_x) * kHOGDescriptorCount,
                   sizeof(float) * kHOGDescriptorCount);
            continue;
        }
        n     = 0;
        dis_y = pt_y + 1 - kHalfWndSize;
        dis_x = pt_x + 1 - kHalfWndSize;
        for (y = 0; y < kHOGFeatureWndSize; y++) 
        {
            int temp_pos = (dis_y + y) * w + dis_x;
            for (x = 0; x < kHOGFeatureWndSize; x++) 
            {
                filter_sum[n] = gaus_weight[n] * img_grad[temp_pos + x];
                n++;
            }
        }

        int n = 0;
        for (i = 0; i < kSearchBlockCount; i++)            // 4x4 thingy
        {
            for (j = 0; j < kSearchBlockCount; j++) 
            {
                // Clear the histograms
                memset(hist, 0, sizeof(unsigned int) * kHOGHistBinCount);

                // Calculate the coordinates of the 4x4 block
                int startx = pt_x - kHalfWndSize + 1 + kSearchBlockSize * i;
                int limitx = startx + kSearchBlockSize;
                int starty = pt_y - kHalfWndSize + 1 + kSearchBlockSize * j;
                int limity = starty + kSearchBlockSize;

                // Go though this 4x4 block and do the thingy :D
                for (y = starty; y < limity; y++) 
                {
                    int temp_pos = (y - dis_y) * kHOGFeatureWndSize - dis_x;
                    for (x = startx; x < limitx; x++) 
                        hist[img_ort[y * w + x]] += filter_sum[temp_pos + x];
                }

                for (k = 0; k < kHOGHistBinCount; ++k) 
                {
                    fv[n] = hist[k];
                    n++;
                }
            }
        }

        float norm       = 0.0f;
        float sum_norm2  = 0.0f;
        float norm_thres = 0.0f, norm_thres_2 = 0.0f;
        for (i = 0; i < kHOGDescriptorCount; i++)
            sum_norm2 += fv[i] * fv[i];
        if (sum_norm2 < 1e-5) 
            sum_norm2 = 1e-5;

        norm = sqrtf(sum_norm2);

        norm_thres   = kFeatureThreshold * norm;
        norm_thres_2 = norm_thres * norm_thres;
        // Now, threshold the vector
        for (i = 0; i < kHOGDescriptorCount; i++) 
        {
            if (fv[i] > norm_thres) 
            {
                sum_norm2 += norm_thres_2 - fv[i] * fv[i];
                fv[i]      = norm_thres;
            }
        }

        if (sum_norm2 < 1e-5) 
            sum_norm2 = 1e-5;
        norm = 1.0f / sqrtf(sum_norm2);


        feat_index = ipt * kHOGDescriptorCount;
        for (i = 0; i < kHOGDescriptorCount; i++) 
        {
            hog[feat_index] = fv[i] * norm;
            feat_index++;
        }
        feature_cal_flag[pt_y * w + pt_x] = 1;
        memcpy(img_hog_feature + (pt_y * w + pt_x) * kHOGDescriptorCount, 
               hog + ipt * kHOGDescriptorCount,
               sizeof(float) * kHOGDescriptorCount);
    }
}

float FaceHOGVerify::compare_hog(float const* hog_a, float const* hog_b, int hog_len) noexcept
{
    int   i = 0;
	float result = 1.0f;
	float aa = 0.0f, bb = 0.0f, ab = 0.0f;

    for(i = 0;i < hog_len; ++i)
	{
		ab  += hog_a[i]*hog_b[i];
		aa  += hog_a[i]*hog_a[i];
		bb  += hog_b[i]*hog_b[i];
	}

	aa      = sqrt(aa);
	bb      = sqrt(bb);
	result  = ab/(aa*bb+ 0.000000001f);

    return result;
}

} //namespace vision_graph


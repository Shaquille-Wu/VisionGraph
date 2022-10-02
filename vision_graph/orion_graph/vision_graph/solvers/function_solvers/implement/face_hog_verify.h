#ifndef FACE_HOG_VERIFY_
#define FACE_HOG_VERIFY_

#include <opencv2/opencv.hpp>
#include "../../../include/graph_tensor.h"

namespace vision_graph{

class FaceHOGVerify
{
public:
    static const int    kKptsSelCnt              = 61;
    static const int    kKptsNormTextureSize     = 64;
    static const int    kKptsNormRegressionSize  = 84;
    static const int    kHOGHistBinCount         = 8;
    static const int    kHOGDescriptorCount      = 128;
    static const int    kHOGFeatureWndSize       = 16;
    static const int    kHOGFeatureLen           = kHOGDescriptorCount * kKptsSelCnt;
    typedef struct   tag_KeyPoint_2D
    {
        float x;
        float y;
    }KEYPOINT_2D, *PKEYPOINT_2D;

    typedef struct tag_affine_param_2d{
        float   rot_x;
        float   rot_y;
        float   x;
        float   y;
    }AFFINE_PARAM_2D, *PAFFINE_PARAM_2D;

public:
    FaceHOGVerify() noexcept;
    virtual ~FaceHOGVerify() noexcept;

    int                          SetTargetDefault() noexcept ;
    int                          SetTargetFromFile(std::string const& kpts_file, std::string const& hog_desc_file) noexcept ;
    int                          SetTargetData(TensorKeypoints const& kpts, TensorFeature const& hog_desc_feature) noexcept;
    int                          Init() noexcept ;
    float                        Solve(cv::Mat const& image, TensorKeypoints const& kpts) noexcept ;

private:
    void                         alloc_work_buf() noexcept;
    void                         assign_buf() noexcept;
    void                         extract_hog(KEYPOINT_2D const*  src_kpts, int point_count) noexcept;

    static void                  generate_hog_gaussian(unsigned short int* g, int kernel, float sigma);
    static void                  norm_target_kpts(KEYPOINT_2D* kpts);
    static void                  select_kpts(TensorKeypoints const& src_kpt, KEYPOINT_2D const& offset, KEYPOINT_2D* selected_kpts, bool re_order, unsigned char* work_buf) noexcept;
    static bool                  cal_affine_transform_param(const KEYPOINT_2D* target_kpts, 
                                                            const KEYPOINT_2D* real_kpts, 
                                                            int                point_count, 
                                                            AFFINE_PARAM_2D*   result,
                                                            unsigned char*     work_buf) noexcept;

    static void                  affine_transform_8bit_gray(const unsigned char*    src_image, 
                                                            int                     src_w, 
                                                            int                     src_h, 
                                                            unsigned char*          dst_image,
                                                            int                     dst_w,
                                                            int                     dst_h,
                                                            const AFFINE_PARAM_2D*  affine_param,
                                                            unsigned char*          work_buf) noexcept;
    static bool                  cal_affine_inv(AFFINE_PARAM_2D const* src_affine_param, AFFINE_PARAM_2D* inv_affine_param) noexcept;

    static void                  init_affine_param(AFFINE_PARAM_2D& affine) noexcept
    {
        affine.rot_x = 1.0; affine.rot_y = 0.0; affine.x = 0.0; affine.y = 0.0;
    }

    static void                  sobel_image(unsigned char const* src_image, 
                                             int                  w, 
                                             int                  h, 
                                             unsigned short int*  grad_x,
                                             unsigned short int*  grad_y,
                                             unsigned short int*  magnitutde_image, 
                                             int*                 ort_image) noexcept;
    static void                  sobel_image_x(unsigned char const* src_image, int w, int h, unsigned short int* grad_image) noexcept;
    static void                  sobel_image_y(unsigned char const* src_image, int w, int h, unsigned short int* grad_image) noexcept;
    static void                  extract_kpt_desc(int                     w, 
                                                  int                     h, 
                                                  float const*            kpts_xy, 
                                                  int                     point_count, 
                                                  unsigned short const*   gaus_weight,
                                                  unsigned short const*   img_grad,
                                                  int const*              img_ort,
                                                  float*                  img_hog_feature,
                                                  unsigned char*          feature_cal_flag, 
                                                  float*                  hog) noexcept ;
    static float                 compare_hog(float const* hog_a, float const* hog_b, int hog_len) noexcept;

protected:
    cv::Mat                      gray_image_;
    KEYPOINT_2D                  target_kpts_[kKptsSelCnt];
    float                        target_face_hog_[kHOGFeatureLen];
    KEYPOINT_2D                  real_kpts_[kKptsSelCnt];
    KEYPOINT_2D                  real_kpts_norm_[kKptsSelCnt];
    unsigned char*               affined_image_;
    float                        hog_feature_[kHOGFeatureLen];
    unsigned short               hog_gaus_[kHOGFeatureWndSize*kHOGFeatureWndSize];

    unsigned char*               work_buf_;
    int                          work_buf_len_;

    unsigned short*              grad_x_image_;
    unsigned short*              grad_y_image_;
    unsigned short*              magnitude_image_;
    int*                         ort_image_;
    
    unsigned char*               feature_proc_flag_;
    float*                       kpts_left_up_;
	float*                       kpts_right_up_;
	float*                       kpts_left_down_;
	float*                       kpts_right_down_;
	float*                       hog_left_up_;
	float*                       hog_right_up_;
	float*                       hog_left_down_;
	float*                       hog_right_down_;
    float*                       img_hog_feature_;

} ; //class FaceHOGVerify

} //namespace vision_graph

#endif
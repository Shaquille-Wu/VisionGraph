#ifndef KFC_H_
#define KFC_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "piotr_fhog/fhog.hpp"
#include "../../apce_conf/apce_conf.h"
#include "complexmat.hpp"
#include "cn/cnfeat.hpp"

namespace kcf_track 
{

struct BBox_c {
    double cx, cy, w, h;

    inline void scale(double factor) {
        cx *= factor;
        cy *= factor;
        w *= factor;
        h *= factor;
    }

    inline cv::Rect get_rect() {
        return cv::Rect(cx - w / 2., cy - h / 2., w, h);
    }

    static BBox_c from_rect(const cv::Rect &rect) {
        BBox_c bbox;
        bbox.cx = rect.x + 0.5 * rect.width;
        bbox.cy = rect.y + 0.5 * rect.height;
        bbox.w = rect.width;
        bbox.h = rect.height;
        return bbox;
    }
};

class KCF_Tracker {
public:
    bool m_use_scale;
    bool m_use_color;
    bool m_use_subpixel_localization;
    bool m_use_subgrid_scale;
    bool m_use_multithreading;
    bool m_use_cnfeat;
    bool m_use_linearkernel;
//new tracker updata
    bool m_gray_feature;
    bool m_pca_feature;
    bool m_hog_feature;
    //************

    /*
    padding             ... extra area surrounding the target           (1.5)
    kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
    lambda              ... regularization                              (1e-4)
    interp_factor       ... linear interpolation factor for adaptation  (0.02)
    output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
    cell_size           ... hog cell size                               (4)
    */
//new tracker updata
    KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor,
                double output_sigma_factor, int cell_size,int target_channel);
    //*******

    KCF_Tracker();

    ~KCF_Tracker();

    // Init/re-init methods
    void init(const cv::Mat &img, const cv::Rect &bbox);

    void setTrackerPose(BBox_c &bbox, const cv::Mat &img);

    void updateTrackerPosition(BBox_c &bbox);

    // frame-to-frame object tracking
    void track(const cv::Mat &img);

    BBox_c getBBox();

    float getMaxResponse();

    bool is_conf_high(){return conf_high;}

private:
    BBox_c p_pose;
    bool p_resize_image = false;

    double p_padding;
    double p_output_sigma_factor;
    double p_output_sigma;
    double p_kernel_sigma;    //def = 0.5
    double p_lambda;         //regularization in learning step
    double p_interp_factor;  //def = 0.02, linear interpolation factor for adaptation
    int p_cell_size;            //4 for hog (= bin_size)
    int p_windows_size[2];
    cv::Mat p_cos_window;
    int p_num_scales;
    double p_scale_step;
    double p_current_scale;
    double p_min_max_scale[2];
    std::vector<double> p_scales;
    //pca  new tracker updata
    int p_target_channel;//目标特征维数
    cv::Mat p_proj_mat;//pca投影矩阵
    bool p_cal_proj_mat;//是否是第一次初始化,计算pca投影矩阵
    //
    float p_max_respose;

    //model
    ComplexMatt p_yf;
    ComplexMatt p_model_alphaf;
    ComplexMatt p_model_alphaf_num;
    ComplexMatt p_model_alphaf_den;
    ComplexMatt p_model_xf;

    //helping functions
    cv::Mat get_subwindow(const cv::Mat &input, int cx, int cy, int size_x, int size_y);

    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);

    ComplexMatt gaussian_correlation(const ComplexMatt &xf, const ComplexMatt &yf, double sigma, bool auto_correlation = false);

    cv::Mat circshift(const cv::Mat &patch, int x_rot, int y_rot);

    cv::Mat cosine_window_function(int dim1, int dim2);

    ComplexMatt fft2(const cv::Mat &input);

    ComplexMatt fft2(const std::vector<cv::Mat> &input, const cv::Mat &cos_window);

    cv::Mat ifft2(const ComplexMatt &inputf);
    //new tracker updata
    void flatten_feature(const std::vector<cv::Mat> &feat, cv::Mat &feat_flat,int &height, int &width, int &channels);

    void restore_flatten_feature(const cv::Mat &feat_flat, int height,
                                int width, int channels, std::vector<cv::Mat> &out_feat);
    cv::Mat compute_project_mat(const std::vector<cv::Mat> &feat);

    void compress(const std::vector<cv::Mat> &in_feat, const cv::Mat &proj_mat, std::vector<cv::Mat> &out_feat);
    //****************

    std::vector<cv::Mat> get_features(cv::Mat &input_rgb, cv::Mat &input_gray, int cx, int cy, int size_x, int size_y,
                double scale = 1.);

    cv::Point2f sub_pixel_peak(cv::Point &max_loc, cv::Mat &response);

    double sub_grid_scale(std::vector<double> &responses, int index = -1);

    cv::Mat CalcResponse(std::vector<cv::Mat>& path_feat);

    void    init_response(cv::Mat& response_map);

    ApceConfidence* apce_conf;

    bool conf_high;
};

}//namespace kcf_track

#endif //KFC_H_
#include "kcf.h"
#include <numeric>
#include <thread>
#include <future>

namespace kcf_track 
{

KCF_Tracker::KCF_Tracker(double padding, double kernel_sigma, double lambda,
                            double interp_factor,
                            double output_sigma_factor, int cell_size, int target_channel) :
        p_padding(padding), p_output_sigma_factor(output_sigma_factor),
        p_kernel_sigma(kernel_sigma),
        p_lambda(lambda), p_interp_factor(interp_factor), p_cell_size(cell_size),
        p_target_channel(target_channel) {
    p_cal_proj_mat = false;
    m_use_scale = false;
    m_use_color = false;
    m_use_subpixel_localization = true;
    m_use_subgrid_scale = true;
    m_use_multithreading = false;
    m_use_cnfeat = false;
    m_use_linearkernel = true;

}


KCF_Tracker::KCF_Tracker() {
    p_padding = 1.5;
    p_output_sigma_factor = 0.1;
    p_kernel_sigma = 0.5;    //def = 0.5
    p_lambda = 1e-4;         //regularization in learning step
    p_interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    p_cell_size = 4;            //4 for hog (= bin_size)
    p_num_scales = 7;

    p_scale_step = 1.02;
    p_current_scale = 1.;

    p_max_respose = -1;

    m_use_scale = false;
    m_use_color = false;
    m_use_subpixel_localization = true;
    m_use_subgrid_scale = true;
    m_use_multithreading = false;
    m_use_cnfeat = false;
    m_gray_feature = false;
    m_hog_feature = false;
    m_use_linearkernel = true;
    m_pca_feature = false;
    p_target_channel = 10;
    p_cal_proj_mat = false;

    apce_conf = new ApceConfidence(100, 0.1f, 0.1f);


}


void KCF_Tracker::init(const cv::Mat &img, const cv::Rect &bbox) {
    //check boundary, enforce min size
    double x1 = bbox.x, x2 = bbox.x + bbox.width, y1 = bbox.y, y2 = bbox.y + bbox.height;
    if (x1 < 0) x1 = 0.;
    if (x2 > img.cols - 1) x2 = img.cols - 1;
    if (y1 < 0) y1 = 0;
    if (y2 > img.rows - 1) y2 = img.rows - 1;

    if (x2 - x1 < 2 * p_cell_size) {
        double diff = (2 * p_cell_size - x2 + x1) / 2.;
        if (x1 - diff >= 0 && x2 + diff < img.cols) {
            x1 -= diff;
            x2 += diff;
        } else if (x1 - 2 * diff >= 0) {
            x1 -= 2 * diff;
        } else {
            x2 += 2 * diff;
        }
    }
    if (y2 - y1 < 2 * p_cell_size) {
        double diff = (2 * p_cell_size - y2 + y1) / 2.;
        if (y1 - diff >= 0 && y2 + diff < img.rows) {
            y1 -= diff;
            y2 += diff;
        } else if (y1 - 2 * diff >= 0) {
            y1 -= 2 * diff;
        } else {
            y2 += 2 * diff;
        }
    }

    p_pose.w = x2 - x1;
    p_pose.h = y2 - y1;
    p_pose.cx = x1 + p_pose.w / 2.;
    p_pose.cy = y1 + p_pose.h / 2.;

    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, input_gray, cv::COLOR_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    } else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
//        if (p_pose.w * p_pose.h > 100. * 100.) {
//            std::cout << "resizing image by factor of 2" << std::endl;
//            p_resize_image = true;
//            p_pose.scale(0.5);
//            cv::resize(input_gray, input_gray, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
//            cv::resize(input_rgb, input_rgb, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
//        }
    p_resize_image = false;

    //compute win size + fit to fhog cell size
    p_windows_size[0] = round(p_pose.w * (1. + p_padding) / p_cell_size) * p_cell_size;
    p_windows_size[1] = round(p_pose.h * (1. + p_padding) / p_cell_size) * p_cell_size;

    p_scales.clear();
    if (m_use_scale) {
//            std::cout << "num of scales: " << p_num_scales << std::endl;
        for (int i = -p_num_scales / 2; i <= p_num_scales / 2; ++i) {
            std::cout << i << std::endl;
            p_scales.push_back(std::pow(p_scale_step, i));
        }

    } else
        p_scales.push_back(1.);

    p_current_scale = 1.;

    double min_size_ratio = std::max(5. * p_cell_size / p_windows_size[0],
                                        5. * p_cell_size / p_windows_size[1]);
    double max_size_ratio = std::min(
            floor((img.cols + p_windows_size[0] / 3) / p_cell_size) * p_cell_size /
            p_windows_size[0],
            floor((img.rows + p_windows_size[1] / 3) / p_cell_size) * p_cell_size /
            p_windows_size[1]);
    //scale尺寸的最大最小值
    p_min_max_scale[0] = std::pow(p_scale_step, std::ceil(
            std::log(min_size_ratio) / log(p_scale_step)));//ceil向上取整
    p_min_max_scale[1] = std::pow(p_scale_step,
                                    std::floor(std::log(max_size_ratio) /
                                                log(p_scale_step)));//floor向下取整

//        std::cout << "init: img size " << img.cols << " " << img.rows << std::endl;
//        std::cout << "init: win size. " << p_windows_size[0] << " " << p_windows_size[1] << std::endl;
//        std::cout << "init: min max scales factors: " << p_min_max_scale[0] << " " << p_min_max_scale[1] << std::endl;

    p_output_sigma = std::sqrt(p_pose.w * p_pose.h) * p_output_sigma_factor /
                        static_cast<double>(p_cell_size);

    //window weights, i.e. labels
    p_yf = fft2(gaussian_shaped_labels(p_output_sigma, p_windows_size[0] / p_cell_size,
                                        p_windows_size[1] / p_cell_size));

//        std::cout << "begin cos_win" << std::endl;
//        std::cout << p_yf.cols << " " << p_yf.rows << std::endl;


    p_cos_window = cosine_window_function(p_yf.cols, p_yf.rows);

//        std::cout << "end cos_win" << std::endl;

//        std::cout << "begin get features" << std::endl;
    p_cal_proj_mat = true;
    //obtain a sub-window for training initial model
    std::vector<cv::Mat> path_feat = get_features(input_rgb, input_gray, p_pose.cx, p_pose.cy,
                                                    p_windows_size[0],
                                                    p_windows_size[1]);

//        std::cout << "end get features" << std::endl;

    p_model_xf = fft2(path_feat, p_cos_window);

    if (m_use_linearkernel) {
        ComplexMatt xfconj = p_model_xf.conj();
        p_model_alphaf_num = xfconj.mul(p_yf);
        p_model_alphaf_den = (p_model_xf * xfconj);
    } else {
        //Kernel Ridge Regression, calculate alphas (in Fourier domain)
        ComplexMatt kf = gaussian_correlation(p_model_xf, p_model_xf, p_kernel_sigma, true);
        p_model_alphaf_num = p_yf * kf;
        p_model_alphaf_den = kf * (kf + p_lambda);
    }

    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
//        p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training

    apce_conf->reset();
    cv::Mat max_response_map = CalcResponse(path_feat);
    apce_conf->compute(max_response_map);
}

void KCF_Tracker::init_response(cv::Mat& response_map)
{
    int i = 0, j = 0;
    for(i = 0 ; i < response_map.rows ; i ++)
        for(j = 0 ; j < response_map.cols ; j ++)
            ((float*)(response_map.data))[i * response_map.cols + j] = -1.0f;
}

cv::Mat KCF_Tracker::CalcResponse(std::vector<cv::Mat> &path_feat) {
    ComplexMatt zf = fft2(path_feat, p_cos_window);
    cv::Mat response;
    if (m_use_linearkernel)
        response = ifft2((p_model_alphaf * zf).sum_over_channels());
    else {
        ComplexMatt kzf = gaussian_correlation(zf, p_model_xf, p_kernel_sigma);
        response = ifft2(p_model_alphaf * kzf);
    }
    return response;
}

void KCF_Tracker::setTrackerPose(BBox_c &bbox, const cv::Mat &img) {
    init(img, bbox.get_rect());
}

void KCF_Tracker::updateTrackerPosition(BBox_c &bbox) {
    if (p_resize_image) {
        BBox_c tmp = bbox;
        tmp.scale(0.5);
//            p_pose.cx = tmp.cx;
//            p_pose.cy = tmp.cy;
        p_pose = tmp;
    } else {
//            p_pose.cx = bbox.cx;
//            p_pose.cy = bbox.cy;
        p_pose = bbox;
    }
}

float KCF_Tracker::getMaxResponse() {
    return p_max_respose;
}

BBox_c KCF_Tracker::getBBox() {
    BBox_c tmp = p_pose;
    tmp.w *= p_current_scale;
    tmp.h *= p_current_scale;

    if (p_resize_image)
        tmp.scale(2);

    return tmp;
}

void KCF_Tracker::track(const cv::Mat &img) {
    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, input_gray, cv::COLOR_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    } else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
    if (p_resize_image) {
        cv::resize(input_gray, input_gray, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
    }

    std::vector<cv::Mat> patch_feat;
    double max_response = -1.;
    cv::Mat      max_response_map(p_cos_window.rows, p_cos_window.cols, p_cos_window.type());
    init_response(max_response_map);
    //cv::Mat      max_response_map;
    cv::Point2i  max_response_pt;
    int scale_index = 0;
    std::vector<double> scale_responses;
    if (m_use_multithreading) {
        std::vector<std::future<cv::Mat>> async_res(p_scales.size());
        for (size_t i = 0; i < p_scales.size(); ++i) {
            async_res[i] = std::async(std::launch::async,
                                        [this, &input_gray, &input_rgb, i]() -> cv::Mat {
                                            std::vector<cv::Mat> patch_feat_async = get_features(
                                                    input_rgb,
                                                    input_gray,
                                                    this->p_pose.cx,
                                                    this->p_pose.cy,
                                                    this->p_windows_size[0],
                                                    this->p_windows_size[1],
                                                    this->p_current_scale *
                                                    this->p_scales[i]);
                                            ComplexMatt zf = fft2(patch_feat_async,
                                                                this->p_cos_window);
                                            if (m_use_linearkernel)
                                                return ifft2((p_model_alphaf *
                                                            zf).sum_over_channels());
                                            else {
                                                ComplexMatt kzf = gaussian_correlation(zf,
                                                                                        this->p_model_xf,
                                                                                        this->p_kernel_sigma);
                                                return ifft2(this->p_model_alphaf * kzf);
                                            }
                                        });
        }

        for (size_t i = 0; i < p_scales.size(); ++i) {
            // wait for result
            async_res[i].wait();
            cv::Mat response = async_res[i].get();

            double min_val, max_val;
            cv::Point2i min_loc, max_loc;
            cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

            double weight = p_scales[i] < 1. ? p_scales[i] : 1. / p_scales[i];
            if (max_val * weight > max_response) {
                max_response = max_val * weight;
                max_response_map = response;
                max_response_pt = max_loc;
                scale_index = i;
            }
            scale_responses.push_back(max_val * weight);
        }
    } else {
        for (size_t i = 0; i < p_scales.size(); ++i) {
            patch_feat = get_features(input_rgb, input_gray, p_pose.cx, p_pose.cy,
                                        p_windows_size[0],
                                        p_windows_size[1], p_current_scale * p_scales[i]);
            ComplexMatt zf = fft2(patch_feat, p_cos_window);
            cv::Mat response;
            if (m_use_linearkernel)
                response = ifft2((p_model_alphaf * zf).sum_over_channels());
            else {
                ComplexMatt kzf = gaussian_correlation(zf, p_model_xf, p_kernel_sigma);
                response = ifft2(p_model_alphaf * kzf);
            }

            /* target location is at the maximum response. we must take into
            account the fact that, if the target doesn't move, the peak
            will appear at the top-left corner, not at the center (this is
            discussed in the paper). the responses wrap around cyclically. */
            double min_val = 1.0, max_val = -1.0;
            cv::Point2i min_loc, max_loc;
            cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

            double weight = p_scales[i] < 1. ? p_scales[i] : 1. / p_scales[i];
            if (max_val * weight > max_response) {
                max_response = max_val * weight;
                max_response_map = response;
                max_response_pt = max_loc;
                scale_index = i;
                p_max_respose = max_val * weight;
            }
            scale_responses.push_back(max_val * weight);
        }
    }
    apce_conf->compute(max_response_map);
    conf_high = apce_conf->judge();
    if (conf_high) {
        //sub pixel quadratic interpolation from neighbours
        if (max_response_pt.y >
            max_response_map.rows / 2) //wrap around to negative half-space of vertical axis
            max_response_pt.y = max_response_pt.y - max_response_map.rows;
        if (max_response_pt.x > max_response_map.cols / 2) //same for horizontal axis
            max_response_pt.x = max_response_pt.x - max_response_map.cols;

        cv::Point2f new_location(max_response_pt.x, max_response_pt.y);

        if (m_use_subpixel_localization)
            new_location = sub_pixel_peak(max_response_pt, max_response_map);

        p_pose.cx += p_current_scale * p_cell_size * new_location.x;
        p_pose.cy += p_current_scale * p_cell_size * new_location.y;
        if (p_pose.cx < 0) p_pose.cx = 0;
        if (p_pose.cx > img.cols - 1) p_pose.cx = img.cols - 1;
        if (p_pose.cy < 0) p_pose.cy = 0;
        if (p_pose.cy > img.rows - 1) p_pose.cy = img.rows - 1;

        //sub grid scale interpolation
        double new_scale = p_scales[scale_index];
        if (m_use_subgrid_scale)
            new_scale = sub_grid_scale(scale_responses, scale_index);

        p_current_scale *= new_scale;

        if (p_current_scale < p_min_max_scale[0])
            p_current_scale = p_min_max_scale[0];
        if (p_current_scale > p_min_max_scale[1])
            p_current_scale = p_min_max_scale[1];

        //obtain a subwindow for training at newly estimated target position
        patch_feat = get_features(input_rgb, input_gray, p_pose.cx, p_pose.cy,
                                    p_windows_size[0], p_windows_size[1],
                                    p_current_scale);
        ComplexMatt xf = fft2(patch_feat, p_cos_window);

        //subsequent frames, interpolate model
        p_model_xf = p_model_xf * (1. - p_interp_factor) + xf * p_interp_factor;

        ComplexMatt alphaf_num, alphaf_den;

        if (m_use_linearkernel) {
            ComplexMatt xfconj = xf.conj();
            alphaf_num = xfconj.mul(p_yf);
            alphaf_den = (xf * xfconj);
        } else {
            //Kernel Ridge Regression, calculate alphas (in Fourier domain)
            ComplexMatt kf = gaussian_correlation(xf, xf, p_kernel_sigma, true);
//        ComplexMatt alphaf = p_yf / (kf + p_lambda); //equation for fast training
//        p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) + alphaf * p_interp_factor;
            alphaf_num = p_yf * kf;
            alphaf_den = kf * (kf + p_lambda);
        }

        p_model_alphaf_num =
                p_model_alphaf_num * (1. - p_interp_factor) + alphaf_num * p_interp_factor;
        p_model_alphaf_den =
                p_model_alphaf_den * (1. - p_interp_factor) + alphaf_den * p_interp_factor;
        p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
    }
}

// ****************************************************************************
void
KCF_Tracker::flatten_feature(const std::vector<cv::Mat> &feat, cv::Mat &feat_flat, int &height,
                                int &width,
                                int &channels) {
    height = feat[0].rows;
    width = feat[0].cols;
    channels = feat.size();

    // flattened feature matrix
    feat_flat = cv::Mat(channels, height * width, feat[0].type());
    for (int i = 0; i < channels; ++i) {
        cv::Mat feat_mat_t = feat[i].t();
        // cv::Mat is row-major, operation on rows are much faster
        feat_mat_t.reshape(0, 1).copyTo(feat_flat.row(i));
    }
    feat_flat = feat_flat.t();
}

void KCF_Tracker::restore_flatten_feature(const cv::Mat &feat_flat, int height,
                                            int width, int channels,
                                            std::vector<cv::Mat> &out_feat) {
    out_feat.resize(channels);
    cv::Mat feat_flat_t = feat_flat.t();
    for (int i = 0; i < channels; ++i) {
        // cv::Mat is row-major, operation on rows are much faster
        cv::Mat feat_flat_row = feat_flat_t.row(i).clone();
        out_feat[i] = feat_flat_row.reshape(0, width).t();
    }
}

KCF_Tracker::~KCF_Tracker() {
    if (apce_conf)
        delete apce_conf;
}

cv::Mat KCF_Tracker::compute_project_mat(const std::vector<cv::Mat> &feat) {
    assert(feat[0].channels() == 1);

    // (height * width) x channels
    cv::Mat feat_flat;
    int height, width, channels;
    flatten_feature(feat, feat_flat, height, width, channels);

    assert(p_target_channel < channels);

    // remove mean
    cv::Mat mean_mat = cv::Mat::zeros(1, channels, feat_flat.type());
    for (int j = 0; j < feat_flat.rows; ++j) {
        mean_mat += feat_flat.row(j);
    }
    mean_mat /= feat_flat.rows;

    for (int j = 0; j < feat_flat.rows; ++j) {
        feat_flat.row(j) -= mean_mat;
    }

    // channels x channels
    cv::Mat feat_mul = feat_flat.t() * feat_flat;

    cv::Mat w, u, vt;
    cv::SVD::compute(feat_mul, w, u, vt, cv::SVD::MODIFY_A);
    // shape: channels x compressed_channels
    cv::Mat project_mat = u.colRange(0, p_target_channel).clone();
    return project_mat;
}

void KCF_Tracker::compress(const std::vector<cv::Mat> &in_feat, const cv::Mat &proj_mat,
                            std::vector<cv::Mat> &out_feat) {
    // shape: (height * width)  x channels
    cv::Mat feat_flat;
    int height, width, channels;
    flatten_feature(in_feat, feat_flat, height, width, channels);

    assert(feat_flat.channels() < 3);
    // shape: (height * width)  x compressed_channels
    cv::Mat compressed_feat;
    // real
    if (feat_flat.channels() == 1) {
        compressed_feat = feat_flat * proj_mat;
    }
        // complex
    else if (feat_flat.channels() == 2) {
        std::vector<cv::Mat> mat_vec;
        cv::split(feat_flat, mat_vec);
        for (size_t j = 0; j < mat_vec.size(); ++j) {
            mat_vec[j] *= proj_mat;
        }
        cv::merge(mat_vec, compressed_feat);
    }
    restore_flatten_feature(compressed_feat, height, width, p_target_channel, out_feat);
}

std::vector<cv::Mat>
KCF_Tracker::get_features(cv::Mat &input_rgb, cv::Mat &input_gray, int cx, int cy, int size_x,
                            int size_y,
                            double scale) {
    //input_rgb, input_gray, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1],p_current_scale
    int size_x_scaled = floor(size_x * scale);
    int size_y_scaled = floor(size_y * scale);

    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy, size_x_scaled, size_y_scaled);
    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy, size_x_scaled, size_y_scaled);

    //resize to default size
    if (scale > 1.) {
        //if we downsample use  INTER_AREA interpolation
        cv::resize(patch_gray, patch_gray, cv::Size(size_x, size_y), 0., 0., cv::INTER_AREA);
    } else {
        cv::resize(patch_gray, patch_gray, cv::Size(size_x, size_y), 0., 0., cv::INTER_LINEAR);
    }
    std::vector<cv::Mat> all_feat;
    if (m_hog_feature) {
        // get hog features
        std::vector<cv::Mat> hog_feat = FHoG::extract(patch_gray, 2, p_cell_size, 9);
        //moyang,dimensionality reduction
        if (m_pca_feature) {
            std::vector<cv::Mat> pca_feat;
            if (p_cal_proj_mat == true) {
                p_proj_mat = compute_project_mat(hog_feat);
                p_cal_proj_mat = false;
            }
            compress(hog_feat, p_proj_mat, pca_feat);
            all_feat.insert(all_feat.end(), pca_feat.begin(), pca_feat.end());
        } else {
            all_feat.insert(all_feat.end(), hog_feat.begin(), hog_feat.end());
        }
    }

    if (m_gray_feature) {
        //moyang,pixel
        // int hb = size_x / p_cell_size, wb = size_y / p_cell_size;
        // int hb = size_x / p_cell_size, wb = size_y / p_cell_size;
        std::vector<cv::Mat> pixel_fea;
        //resize to default size
        if (scale > 1.) {
            //if we downsample use  INTER_AREA interpolation
            cv::resize(patch_gray, patch_gray,
                        cv::Size(size_x / p_cell_size, size_y / p_cell_size), 0., 0.,
                        cv::INTER_AREA);
        } else {
            cv::resize(patch_gray, patch_gray,
                        cv::Size(size_x / p_cell_size, size_y / p_cell_size), 0., 0.,
                        cv::INTER_LINEAR);
        }
        cv::Mat patch_gray_norm;
        patch_gray.convertTo(patch_gray_norm, CV_32F, 1. / 255., -0.5);
        pixel_fea.push_back(patch_gray_norm);
        all_feat.insert(all_feat.end(), pixel_fea.begin(), pixel_fea.end());
    }
    //get color rgb features (simple r,g,b channels)
    std::vector<cv::Mat> color_feat;
    if ((m_use_color || m_use_cnfeat) && input_rgb.channels() == 3) {
        //resize to default size
        if (scale > 1.) {
            //if we downsample use  INTER_AREA interpolation
            cv::resize(patch_rgb, patch_rgb,
                        cv::Size(size_x / p_cell_size, size_y / p_cell_size), 0., 0.,
                        cv::INTER_AREA);
        } else {
            cv::resize(patch_rgb, patch_rgb,
                        cv::Size(size_x / p_cell_size, size_y / p_cell_size), 0., 0.,
                        cv::INTER_LINEAR);
        }
    }

    if (m_use_color && input_rgb.channels() == 3) {
        //use rgb color space
        cv::Mat patch_rgb_norm;
        patch_rgb.convertTo(patch_rgb_norm, CV_32F, 1. / 255., -0.5);
        cv::Mat ch1(patch_rgb_norm.size(), CV_32FC1);
        cv::Mat ch2(patch_rgb_norm.size(), CV_32FC1);
        cv::Mat ch3(patch_rgb_norm.size(), CV_32FC1);
        std::vector<cv::Mat> rgb = {ch1, ch2, ch3};
        cv::split(patch_rgb_norm, rgb);
        color_feat.insert(color_feat.end(), rgb.begin(), rgb.end());
    }

    if (m_use_cnfeat && input_rgb.channels() == 3) {
        std::vector<cv::Mat> cn_feat = CNFeat::extract(patch_rgb);
        color_feat.insert(color_feat.end(), cn_feat.begin(), cn_feat.end());
    }
    all_feat.insert(all_feat.end(), color_feat.begin(), color_feat.end());
    return all_feat;
}

/*
static void show_float_image(cv::Mat const& image, std::string const& image_path)
{
int  w = image.cols;
int  h = image.rows;
int  i = 0;
int  j = 0;

cv::Mat  gray_image(w, h, CV_8UC1);
for(i = 0 ; i < h ; i ++)
{
    for(j = 0 ; j < w ; j ++)
    {
        int value = ((float*)(image.data))[i * w + j] * 255.0f;
        if(value > 255)
            value = 255;
        else if(value < 0)
            value = 0;
        gray_image.data[i * w + j] = (unsigned char)value;
    }
}
cv::imwrite(image_path, gray_image);
}
*/
cv::Mat KCF_Tracker::gaussian_shaped_labels(double sigma, int dim1, int dim2) {
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma * sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
        float *row_ptr = labels.ptr<float>(j);
        double y_s = y * y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
            row_ptr[i] = std::exp(-0.5 * (y_s + x * x) / sigma_s);
        }
    }

    //rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
    //sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0, 0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF_Tracker::circshift(const cv::Mat &patch, int x_rot, int y_rot) {
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

    //circular rotate x-axis
    if (x_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    } else if (x_rot > 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    } else {    //zero rotation
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }

    //circular rotate y-axis
    if (y_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    } else if (y_rot > 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    } else { //zero rotation
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

ComplexMatt KCF_Tracker::fft2(const cv::Mat &input) {
    cv::Mat complex_result;
//    cv::Mat padded;                            //expand input image to optimal size
//    int m = cv::getOptimalDFTSize( input.rows );
//    int n = cv::getOptimalDFTSize( input.cols ); // on the border add zero pixels
//    copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//    cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//    return ComplexMatt(complex_result(cv::Range(0, input.rows), cv::Range(0, input.cols)));

    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMatt(complex_result);
}

ComplexMatt KCF_Tracker::fft2(const std::vector<cv::Mat> &input, const cv::Mat &cos_window) {
    int n_channels = input.size();
    ComplexMatt result(input[0].rows, input[0].cols, n_channels);
    for (int i = 0; i < n_channels; ++i) {
        cv::Mat complex_result;
//        cv::Mat padded;                            //expand input image to optimal size
//        int m = cv::getOptimalDFTSize( input[0].rows );
//        int n = cv::getOptimalDFTSize( input[0].cols ); // on the border add zero pixels

//        copyMakeBorder(input[i].mul(cos_window), padded, 0, m - input[0].rows, 0, n - input[0].cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//        cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//        result.set_channel(i, complex_result(cv::Range(0, input[0].rows), cv::Range(0, input[0].cols)));

        cv::dft(input[i].mul(cos_window), complex_result, cv::DFT_COMPLEX_OUTPUT);
        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat KCF_Tracker::ifft2(const ComplexMatt &inputf) {

    cv::Mat real_result;
    if (inputf.n_channels == 1) {
        cv::dft(inputf.to_cv_mat(), real_result,
                cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(inputf.n_channels);
        for (int i = 0; i < inputf.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i],
                    cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return real_result;
}

//hann window actually (Power-of-cosine windows)
cv::Mat KCF_Tracker::cosine_window_function(int dim1, int dim2) {
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1. / (static_cast<double>(dim1) - 1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5 * (1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    N_inv = 1. / (static_cast<double>(dim2) - 1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = 0.5 * (1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    cv::Mat ret = m2 * m1;
    return ret;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat
KCF_Tracker::get_subwindow(const cv::Mat &input, int cx, int cy, int width, int height) {
    cv::Mat patch;

    int x1 = cx - width / 2;
    int y1 = cy - height / 2;
    int x2 = cx + width / 2;
    int y2 = cy + height / 2;

    //out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, input.type());
        patch.setTo(0.f);
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    //fit to image coordinates, set border extensions;
    if (x1 < 0) {
        left = -x1;
        x1 = 0;
    }
    if (y1 < 0) {
        top = -y1;
        y1 = 0;
    }
    if (x2 >= input.cols) {
        right = x2 - input.cols + width % 2;
        x2 = input.cols;
    } else
        x2 += width % 2;

    if (y2 >= input.rows) {
        bottom = y2 - input.rows + height % 2;
        y2 = input.rows;
    } else
        y2 += height % 2;

    if (x2 - x1 == 0 || y2 - y1 == 0)
        patch = cv::Mat::zeros(height, width, CV_32FC1);
    else
        cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom,
                            left, right,
                            cv::BORDER_REPLICATE);

    //sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

ComplexMatt
KCF_Tracker::gaussian_correlation(const ComplexMatt &xf, const ComplexMatt &yf, double sigma,
                                    bool auto_correlation) {
    float xf_sqr_norm = xf.sqr_norm();
    float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();

    ComplexMatt xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj();

    //ifft2 and sum over 3rd dimension, we dont care about individual channels
    cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
    xy_sum.setTo(0);
    cv::Mat ifft2_res = ifft2(xyf);
    for (int y = 0; y < xf.rows; ++y) {
        float *row_ptr = ifft2_res.ptr<float>(y);
        float *row_ptr_sum = xy_sum.ptr<float>(y);
        for (int x = 0; x < xf.cols; ++x) {
            row_ptr_sum[x] = std::accumulate((row_ptr + x * ifft2_res.channels()),
                                                (row_ptr + x * ifft2_res.channels() +
                                                ifft2_res.channels()), 0.f);
        }
    }

    float numel_xf_inv = 1.f / (xf.cols * xf.rows * xf.n_channels);
    cv::Mat tmp;
    cv::exp(-1.f / (sigma * sigma) *
            cv::max((xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0), tmp);

    return fft2(tmp);
}

float get_response_circular(cv::Point2i &pt, cv::Mat &response) {
    int x = pt.x;
    int y = pt.y;
    if (x < 0)
        x = response.cols + x;
    if (y < 0)
        y = response.rows + y;
    if (x >= response.cols)
        x = x - response.cols;
    if (y >= response.rows)
        y = y - response.rows;

    return response.at<float>(y, x);
}

cv::Point2f KCF_Tracker::sub_pixel_peak(cv::Point &max_loc, cv::Mat &response) {
    //find neighbourhood of max_loc (response is circular)
    // 1 2 3
    // 4   5
    // 6 7 8
    cv::Point2i p1(max_loc.x - 1, max_loc.y - 1), p2(max_loc.x, max_loc.y - 1), p3(
            max_loc.x + 1, max_loc.y - 1);
    cv::Point2i p4(max_loc.x - 1, max_loc.y), p5(max_loc.x + 1, max_loc.y);
    cv::Point2i p6(max_loc.x - 1, max_loc.y + 1), p7(max_loc.x, max_loc.y + 1), p8(
            max_loc.x + 1, max_loc.y + 1);

    // fit 2d quadratic function f(x, y) = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
    cv::Mat A = (cv::Mat_<float>(9, 6) <<
                                        p1.x * p1.x, p1.x * p1.y, p1.y * p1.y, p1.x, p1.y, 1.f,
            p2.x * p2.x, p2.x * p2.y, p2.y * p2.y, p2.x, p2.y, 1.f,
            p3.x * p3.x, p3.x * p3.y, p3.y * p3.y, p3.x, p3.y, 1.f,
            p4.x * p4.x, p4.x * p4.y, p4.y * p4.y, p4.x, p4.y, 1.f,
            p5.x * p5.x, p5.x * p5.y, p5.y * p5.y, p5.x, p5.y, 1.f,
            p6.x * p6.x, p6.x * p6.y, p6.y * p6.y, p6.x, p6.y, 1.f,
            p7.x * p7.x, p7.x * p7.y, p7.y * p7.y, p7.x, p7.y, 1.f,
            p8.x * p8.x, p8.x * p8.y, p8.y * p8.y, p8.x, p8.y, 1.f,
            max_loc.x * max_loc.x, max_loc.x * max_loc.y, max_loc.y *
                                                            max_loc.y, max_loc.x, max_loc.y, 1.f);
    cv::Mat fval = (cv::Mat_<float>(9, 1) <<
                                            get_response_circular(p1, response),
            get_response_circular(p2, response),
            get_response_circular(p3, response),
            get_response_circular(p4, response),
            get_response_circular(p5, response),
            get_response_circular(p6, response),
            get_response_circular(p7, response),
            get_response_circular(p8, response),
            get_response_circular(max_loc, response));
    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);

    double a = x.at<float>(0), b = x.at<float>(1), c = x.at<float>(2),
            d = x.at<float>(3), e = x.at<float>(4);

    cv::Point2f sub_peak(max_loc.x, max_loc.y);
    if (b > 0 || b < 0) {
        sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
        sub_peak.x = (-2 * c * sub_peak.y - e) / b;
    }

    return sub_peak;
}

double KCF_Tracker::sub_grid_scale(std::vector<double> &responses, int index) {
    cv::Mat A, fval;
    if (index < 0 || index > (int) p_scales.size() - 1) {
        // interpolate from all values
        // fit 1d quadratic function f(x) = a*x^2 + b*x + c
        A.create(p_scales.size(), 3, CV_32FC1);
        fval.create(p_scales.size(), 1, CV_32FC1);
        for (size_t i = 0; i < p_scales.size(); ++i) {
            A.at<float>(i, 0) = p_scales[i] * p_scales[i];
            A.at<float>(i, 1) = p_scales[i];
            A.at<float>(i, 2) = 1;
            fval.at<float>(i) = responses[i];
        }
    } else {
        //only from neighbours
        if (index == 0 || index == (int) p_scales.size() - 1)
            return p_scales[index];

        A = (cv::Mat_<float>(3, 3) <<
                                    p_scales[index - 1] * p_scales[index - 1], p_scales[index -
                                                                                        1], 1,
                p_scales[index] * p_scales[index], p_scales[index], 1,
                p_scales[index + 1] * p_scales[index + 1], p_scales[index + 1], 1);
        fval = (cv::Mat_<float>(3, 1) << responses[index - 1], responses[index], responses[
                index + 1]);
    }

    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);
    double a = x.at<float>(0), b = x.at<float>(1);
    double scale = p_scales[index];
    if (a > 0 || a < 0)
        scale = -b / (2 * a);
    return scale;
}

}//namesapce kfc_track
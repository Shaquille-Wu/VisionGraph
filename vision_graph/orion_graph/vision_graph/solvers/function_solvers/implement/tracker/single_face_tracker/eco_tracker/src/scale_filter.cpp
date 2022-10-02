//
// Created by yuan on 17-11-8.
//

#include "scale_filter.h"
#include "fourier_tools.h"
#include "complexmat.h"
#include "eigen_utils.h"
#include "utils.h"

ScaleFilter::ScaleFilter(float sigma, float lrt, int max_patch_area, int n_filters, int n_interp_scales,
                         float scale_mdl_coef, float scale_step, float reg_coef)
{
    sigma_factor = sigma;
    learning_rate = lrt;
    max_sample_area = max_patch_area;
    num_filters = n_filters;
    num_interp_scales = n_interp_scales;
    scale_model_factor = scale_mdl_coef;
    filter_step = scale_step;
    reg_factor = reg_coef;
}

ScaleFilter::~ScaleFilter() {}

ScaleFilter& ScaleFilter::operator=(const ScaleFilter &src_filter)
{
    if(this != &src_filter)
    {
        this->sigma_factor = src_filter.sigma_factor;
        this->learning_rate = src_filter.learning_rate;
        this->max_sample_area = src_filter.max_sample_area;
        this->num_filters = src_filter.num_filters;
        this->num_interp_scales = src_filter.num_interp_scales;
        this->scale_model_factor = src_filter.scale_model_factor;
        this->scale_model_size = src_filter.scale_model_size;
        this->filter_step = src_filter.filter_step;
        this->reg_factor = src_filter.reg_factor;

        this->scale_size_factors = src_filter.scale_size_factors.clone();
        this->interp_scale_factors = src_filter.interp_scale_factors.clone();
        this->yf = src_filter.yf.clone();
        this->cos_window = src_filter.cos_window.clone();
        this->basis = src_filter.basis.clone();
        this->sf_num = src_filter.sf_num.clone();
        this->sf_den = src_filter.sf_den.clone();
        this->ss_num = src_filter.ss_num.clone();
     
        this->hog_extractor = src_filter.hog_extractor;
    }
    return *this;
}

void ScaleFilter::init(cv::Size init_target_sz)
{
    int start_ind = (int) -floor(0.5f * (num_filters - 1));
    int end_ind = (int)ceil(0.5f * (num_filters - 1));
    cv::Mat scale_exp(1, end_ind - start_ind + 1, CV_32FC1);
    scale_size_factors = cv::Mat(scale_exp.size(), CV_32FC1);
    float coef = num_interp_scales / (float) num_filters;
    for (int i = start_ind; i <= end_ind ; ++i)
    {
        scale_exp.at<float>(0, i - start_ind) = i * coef;
        scale_size_factors.at<float>(0, i - start_ind) = (float)pow(filter_step, i * coef);
    }

    int x_rot = (int) -floor(0.5f * (num_filters - 1)), y_rot = 0;
    cv::Mat scale_exp_shift = circshift(scale_exp, x_rot, y_rot);

    start_ind = (int) - floor(0.5f * (num_interp_scales - 1));
    end_ind = (int) ceil(0.5f * (num_interp_scales - 1));
    cv::Mat interp_scale_exp(1, end_ind - start_ind + 1, CV_32FC1);
    for (int i = start_ind; i <= end_ind ; ++i)
    {
        interp_scale_exp.at<float>(0, i - start_ind) = i;
    }

    x_rot = (int) - floor(0.5f * (num_interp_scales - 1));
    cv::Mat interp_scale_exp_shift = circshift(interp_scale_exp, x_rot, y_rot);
    interp_scale_factors = cv::Mat(interp_scale_exp_shift.size(), CV_32FC1);
    for (int i = 0; i < interp_scale_factors.cols; ++i)
    {
        float temp = interp_scale_exp_shift.at<float>(0, i);
        interp_scale_factors.at<float>(0, i) = (float)pow(filter_step, temp);
    }

    float scale_sigma = num_interp_scales * sigma_factor;
    cv::Mat ys(scale_exp_shift.size(), CV_32FC1);
    for (int i = 0; i < ys.cols; ++i)
    {
        ys.at<float>(0, i) = (float)exp(-0.5f * pow(scale_exp_shift.at<float>(0, i), 2) / (scale_sigma * scale_sigma));
    }
    // FFT
    cv::dft(ys, yf, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    yf = ComplexMat(yf).real();

    int N = ys.cols;
    cos_window = cv::Mat(1, N, CV_32FC1);
    float N_inv = 1.f / (float)(N - 1);
    for (int i = 0; i < N; ++i)
    {
        cos_window.at<float>(0, i) = (float)(0.5f * (1 - cos(2 * CV_PI * i * N_inv)));
    }

    float target_area = init_target_sz.width * init_target_sz.height;
    float scaled_area = (float)(pow(scale_model_factor, 2) * target_area);
    if(scaled_area > max_sample_area)
    {
        scale_model_factor = (float)sqrt(max_sample_area / target_area);
    }

    scale_model_size.width = std::max(8, (int)floor(init_target_sz.width * scale_model_factor));
    scale_model_size.height = std::max(8, (int)floor(init_target_sz.height * scale_model_factor));
    hog_extractor.input_size = scale_model_size;
}

cv::Mat ScaleFilter::extract_sample(const cv::Mat &img, cv::Point2f pos, cv::Size2f base_target_sz,
                                    const cv::Mat& scale_factors)
{
    assert(scale_factors.rows == 1);
    int nscales = scale_factors.cols;
    double min_value;
    cv::minMaxLoc(scale_factors, &min_value);
    int df = (int)floor(min_value);
    cv::Mat img_sample = img;
    cv::Mat scale_factors_mat = scale_factors;
    if(df > 1)
    {
        int new_height = (img.rows - 1) / df + 1;
        int new_width = (img.cols - 1) / df + 1;
        img_sample = cv::Mat(new_height, new_width, img.type());
        int channels = img.channels();
        int row_ind = 0, col_ind = 0;
        for (int i = 0; i < img.rows; i += df)
        {
            const uchar *ptr = img.ptr<uchar>(i);
            uchar *ptr_new = img_sample.ptr<uchar>(row_ind);
            for (int j = 0; j < img.cols; j += df)
            {
                for (int k = 0; k < channels; ++k)
                {
                    ptr_new[channels * col_ind + k] = ptr[channels * j + k];
                }
                col_ind++;
            }
            row_ind++;
            col_ind = 0;
        }

        pos.x = (pos.x - 1) / (float)df;
        pos.y = (pos.y - 1) / (float)df;
        scale_factors_mat = scale_factors / (float)df;
    }

    cv::Mat scale_sample;
    for (int i = 0; i < nscales; ++i)
    {
        cv::Size patch_sz;
        float factor = scale_factors_mat.at<float>(0, i);
        patch_sz.width = (int)floor(base_target_sz.width * factor);
        patch_sz.height = (int)floor(base_target_sz.height * factor);

        int half_width = (int)floor(0.5f * patch_sz.width);
        int half_height = (int)floor(0.5f * patch_sz.height);

        cv::Rect roi;
        int pad_left = 0, pad_right = 0, pad_up = 0, pad_down = 0;
        int x_start = (int)(pos.x - half_width);
        int x_end = (int)(pos.x + patch_sz.width - half_width);
        bool is_set_x = false;
        for (int x = x_start; x < x_end; ++x)
        {
            if(x < 0)
                pad_left++;
            else
            {
                if(!is_set_x)
                {
                    roi.x = x;
                    is_set_x = true;
                }
            }

            if(x >= img_sample.cols)
                pad_right++;
            else
            {
                if(is_set_x)
                    roi.width = x - roi.x + 1;
            }
        }

        int y_start = (int)(pos.y - half_height);
        int y_end = (int)(pos.y + patch_sz.height - half_height);
        bool is_set_y = false;
        for (int y = y_start; y < y_end; ++y)
        {
            if(y < 0)
                pad_up++;
            else
            {
                if(!is_set_y)
                {
                    roi.y = y;
                    is_set_y = true;
                }
            }

            if(y >= img_sample.rows)
                pad_down++;
            else
            {
                if(is_set_y)
                    roi.height = y - roi.y + 1;
            }
        }

        cv::Mat roi_img  = cv::Mat(img_sample, roi);

        assert(roi.width > 0 && roi.height > 0);

        cv::Mat img_patch;
        cv::copyMakeBorder(roi_img, img_patch, pad_up, pad_down, pad_left, pad_right,
                           cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        cv::resize(img_patch, img_patch, scale_model_size, cv::INTER_LINEAR);

        Sample hog_feat;
        hog_extractor.extract_feature(img_patch, hog_feat);

        auto channels = hog_feat[0].size();
        auto numel = hog_feat[0][0].total();
        // optimized cv::Mat col operation
        if(i == 0)
        {
            auto dim_scale = numel * channels;
            scale_sample = cv::Mat(nscales, dim_scale, CV_32FC1);
        }
        for (int j = 0; j < channels; ++j)
        {
            cv::Mat feat_t = hog_feat[0][j].t();
            cv::Mat feat_flat = feat_t.reshape(0, 1);
            feat_flat.copyTo(scale_sample.row(i).colRange(j * numel, (j + 1) * numel));
        }
    }

    // optimized cv::Mat col operation
    scale_sample = scale_sample.t();
    return scale_sample;
}

/*cv::Mat ScaleFilter::extract_sample(const cv::Mat &img, cv::Point2f pos, cv::Size2f base_target_sz,
                                    const cv::Mat& scale_factors)
{
    assert(scale_factors.rows == 1);
    int nscales = scale_factors.cols;
    double min_value;
    cv::minMaxLoc(scale_factors, &min_value);
    int df = (int)floor(min_value);
    cv::Mat img_sample = img;
    cv::Mat scale_factors_mat = scale_factors;
    if(df > 1)
    {
        int new_height = (img.rows - 1) / df + 1;
        int new_width = (img.cols - 1) / df + 1;
        img_sample = cv::Mat(new_height, new_width, img.type());
        int channels = img.channels();
        int row_ind = 0, col_ind = 0;
        for (int i = 0; i < img.rows; i += df)
        {
            const uchar *ptr = img.ptr<uchar>(i);
            uchar *ptr_new = img_sample.ptr<uchar>(row_ind);
            for (int j = 0; j < img.cols; j += df)
            {
                for (int k = 0; k < channels; ++k)
                {
                    ptr_new[channels * col_ind + k] = ptr[channels * j + k];
                }
                col_ind++;
            }
            row_ind++;
            col_ind = 0;
        }

        pos.x = (pos.x - 1) / (float)df;
        pos.y = (pos.y - 1) / (float)df;
        scale_factors_mat = scale_factors / (float)df;
    }

    cv::Mat scale_sample;
    for (int i = 0; i < 1; ++i)
    {
        cv::Size patch_sz;
        float factor = scale_factors_mat.at<float>(0, i);
        patch_sz.width = (int)floor(base_target_sz.width * factor);
        patch_sz.height = (int)floor(base_target_sz.height * factor);

        int half_width = (int)floor(0.5f * patch_sz.width);
        int half_height = (int)floor(0.5f * patch_sz.height);

        cv::Rect roi;
        int pad_left = 0, pad_right = 0, pad_up = 0, pad_down = 0;
        int x_start = (int)(pos.x - half_width);
        int x_end = (int)(pos.x + patch_sz.width - half_width);
        bool is_set_x = false;
        for (int x = x_start; x < x_end; ++x)
        {
            if(x < 0)
                pad_left++;
            else
            {
                if(!is_set_x)
                {
                    roi.x = x;
                    is_set_x = true;
                }
            }

            if(x >= img_sample.cols)
                pad_right++;
            else
            {
                if(is_set_x)
                    roi.width = x - roi.x + 1;
            }
        }

        int y_start = (int)(pos.y - half_height);
        int y_end = (int)(pos.y + patch_sz.height - half_height);
        bool is_set_y = false;
        for (int y = y_start; y < y_end; ++y)
        {
            if(y < 0)
                pad_up++;
            else
            {
                if(!is_set_y)
                {
                    roi.y = y;
                    is_set_y = true;
                }
            }

            if(y >= img_sample.rows)
                pad_down++;
            else
            {
                if(is_set_y)
                    roi.height = y - roi.y + 1;
            }
        }

        cv::Mat roi_img  = cv::Mat(img_sample, roi);

        assert(roi.width > 0 && roi.height > 0);

        cv::Mat img_patch;
        cv::copyMakeBorder(roi_img, img_patch, pad_up, pad_down, pad_left, pad_right,
                           cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        cv::resize(img_patch, img_patch, scale_model_size, cv::INTER_LINEAR);

        Sample hog_feat;
        hog_extractor.extract_feature(img_patch, hog_feat);

        auto channels = hog_feat[0].size();
        auto numel = hog_feat[0][0].total();
        if(i == 0)
        {
            auto dim_scale = numel * channels;
            scale_sample = cv::Mat(dim_scale, nscales, CV_32FC1);
        }
        for (int j = 0; j < channels; ++j)
        {
            cv::Mat feat_t = hog_feat[0][j].t();
            cv::Mat feat_flat = feat_t.reshape(0, numel);
            feat_flat.copyTo(scale_sample.col(i).rowRange(j * numel, (j + 1) * numel));
        }
    }

#pragma omp parallel for
    for (int i = 1; i < nscales; ++i)
    {
        cv::Size patch_sz;
        float factor = scale_factors_mat.at<float>(0, i);
        patch_sz.width = (int)floor(base_target_sz.width * factor);
        patch_sz.height = (int)floor(base_target_sz.height * factor);

        int half_width = (int)floor(0.5f * patch_sz.width);
        int half_height = (int)floor(0.5f * patch_sz.height);

        cv::Rect roi;
        int pad_left = 0, pad_right = 0, pad_up = 0, pad_down = 0;
        int x_start = (int)(pos.x - half_width);
        int x_end = (int)(pos.x + patch_sz.width - half_width);
        bool is_set_x = false;
        for (int x = x_start; x < x_end; ++x)
        {
            if(x < 0)
                pad_left++;
            else
            {
                if(!is_set_x)
                {
                    roi.x = x;
                    is_set_x = true;
                }
            }

            if(x >= img_sample.cols)
                pad_right++;
            else
            {
                if(is_set_x)
                    roi.width = x - roi.x + 1;
            }
        }

        int y_start = (int)(pos.y - half_height);
        int y_end = (int)(pos.y + patch_sz.height - half_height);
        bool is_set_y = false;
        for (int y = y_start; y < y_end; ++y)
        {
            if(y < 0)
                pad_up++;
            else
            {
                if(!is_set_y)
                {
                    roi.y = y;
                    is_set_y = true;
                }
            }

            if(y >= img_sample.rows)
                pad_down++;
            else
            {
                if(is_set_y)
                    roi.height = y - roi.y + 1;
            }
        }

        cv::Mat roi_img  = cv::Mat(img_sample, roi);

        assert(roi.width > 0 && roi.height > 0);

        cv::Mat img_patch;
        cv::copyMakeBorder(roi_img, img_patch, pad_up, pad_down, pad_left, pad_right,
                           cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        cv::resize(img_patch, img_patch, scale_model_size, cv::INTER_LINEAR);

        Sample hog_feat;
        hog_extractor.extract_feature(img_patch, hog_feat);

        auto channels = hog_feat[0].size();
        auto numel = hog_feat[0][0].total();
        for (int j = 0; j < channels; ++j)
        {
            cv::Mat feat_t = hog_feat[0][j].t();
            cv::Mat feat_flat = feat_t.reshape(0, numel);
            feat_flat.copyTo(scale_sample.col(i).rowRange(j * numel, (j + 1) * numel));
        }
    }
    return scale_sample;
}*/

void ScaleFilter::feature_proj_scale(const cv::Mat &src, const cv::Mat &proj_mat, cv::Mat &dst)
{
    cv::Mat src_projected = proj_mat * src;

    assert(src_projected.cols == cos_window.cols);

    cv::Mat cos_win_rep;
    cv::copyMakeBorder(cos_window, cos_win_rep, 0, src_projected.rows - 1, 0, 0, cv::BORDER_REPLICATE);
    dst = cos_win_rep.mul(src_projected);
}

float ScaleFilter::track(const cv::Mat &img, cv::Point2f pos, cv::Size2f base_target_sz, float current_scale_factor)
{
    cv::Mat scales = scale_size_factors * current_scale_factor;
    // get features

    cv::Mat feats = extract_sample(img, pos, base_target_sz, scales);
    feature_proj_scale(feats, basis, feats);
    // get scores
    cv::Mat feats_dft; // complex
    cv::dft(feats, feats_dft, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    cv::Mat mul_res = ComplexMat(feats_dft).mul(ComplexMat(sf_num)).to_cv_mat();
    cv::Mat row_sum = cv::Mat::zeros(1, mul_res.cols, CV_32FC2);
    for (int i = 0; i < mul_res.rows; ++i)
    {
        row_sum += mul_res.row(i);
    }
    ComplexMat den_cmat(sf_den + reg_factor);
    cv::Mat scale_responsef = (ComplexMat(row_sum) / den_cmat).to_cv_mat();
    cv::Mat scale_resp_rsz_dft;
    resize_dft(scale_responsef, num_interp_scales, scale_resp_rsz_dft);

    cv::Mat interp_scale_resp;
    cv::dft(scale_resp_rsz_dft, interp_scale_resp, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::Point min_resp_pt, max_resp_pt;
    double min_resp_val, max_resp_val;
    cv::minMaxLoc(interp_scale_resp, &min_resp_val, &max_resp_val, &min_resp_pt, &max_resp_pt);
    int recovered_scale_ind = max_resp_pt.x;

    // fit a quadratic polynomial to get a refined scale
    int id1 = mod(recovered_scale_ind - 1,  num_interp_scales);
    assert(id1 >=0 && id1 < num_interp_scales);
    int id2 = mod(recovered_scale_ind + 1, num_interp_scales);
    assert(id2 >=0 && id2 < num_interp_scales);

    cv::Matx33f poly_A_mat;
    poly_A_mat(0, 0) = (float)pow(interp_scale_factors.at<float>(0, id1), 2);
    poly_A_mat(0, 1) = interp_scale_factors.at<float>(0, id1);
    poly_A_mat(0, 2) = 1;

    poly_A_mat(1, 0) = (float)pow(interp_scale_factors.at<float>(0, recovered_scale_ind), 2);
    poly_A_mat(1, 1) = interp_scale_factors.at<float>(0, recovered_scale_ind);
    poly_A_mat(1, 2) = 1;

    poly_A_mat(2, 0) = (float)pow(interp_scale_factors.at<float>(0, id2), 2);
    poly_A_mat(2, 1) = interp_scale_factors.at<float>(0, id2);
    poly_A_mat(2, 2) = 1;

    cv::Vec3f poly_y;
    poly_y[0] = interp_scale_resp.at<float>(0, id1);
    poly_y[1] = interp_scale_resp.at<float>(0, recovered_scale_ind);
    poly_y[2] = interp_scale_resp.at<float>(0, id2);

    cv::Vec3f poly;
    cv::solve(poly_A_mat, poly_y, poly, cv::DECOMP_SVD);

    float scale_change_factor = - poly[1] / (2 * poly[0]);
    return scale_change_factor;
}

void ScaleFilter::update(const cv::Mat &img, cv::Point2f pos, cv::Size2f base_target_sz, float current_scale_factor)
{
    cv::Mat scales = scale_size_factors * current_scale_factor;
    // get features
    cv::Mat feats = extract_sample(img, pos, base_target_sz, scales);

    bool is_first_frame = ss_num.empty();
    if(is_first_frame)
        ss_num = feats.clone();
    else
        ss_num = (1.f - learning_rate) * ss_num + learning_rate * feats;

    // compute projection basis
    cv::Mat big_y = ss_num.clone();
    cv::Mat proj_basis;
    qr_decomp(big_y, proj_basis);
    basis = proj_basis.t();

    cv::Mat big_y_den = feats.clone();
    cv::Mat scale_basis_den;
    qr_decomp(big_y_den, scale_basis_den);

    cv::Mat ss_num_proj;
    feature_proj_scale(ss_num, basis, ss_num_proj);
    cv::Mat sf_proj;
    cv::dft(ss_num_proj, sf_proj, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

    cv::Mat yf_rep;
    cv::copyMakeBorder(yf, yf_rep, 0, sf_proj.rows - 1, 0, 0, cv::BORDER_REPLICATE);
    ComplexMat sf_proj_cmat(sf_proj);
    sf_num = ComplexMat(yf_rep).mul(sf_proj_cmat.conj()).to_cv_mat();

    // update denominator
    feature_proj_scale(feats, scale_basis_den.t(), feats);
    cv::Mat feats_f;
    cv::dft(feats, feats_f, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

    ComplexMat feats_f_cmat(feats_f);
    cv::Mat dot_pd = feats_f_cmat.mul(feats_f_cmat.conj()).real();
    cv::Mat new_sf_den = cv::Mat::zeros(1, dot_pd.cols, dot_pd.type());
    for (int i = 0; i < dot_pd.rows; ++i)
    {
        new_sf_den += dot_pd.row(i);
    }

    if(is_first_frame)
        sf_den = new_sf_den;
    else
        sf_den = (1.f - learning_rate) * sf_den + learning_rate * new_sf_den;
}

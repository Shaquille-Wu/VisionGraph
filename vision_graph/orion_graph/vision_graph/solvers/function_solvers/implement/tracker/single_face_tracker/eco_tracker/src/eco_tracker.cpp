//
// Created by yuan on 17-9-22.
//

#include <timer.h>
#include <logging.h>
#include "eco_tracker.h"
#include "complexmat.h"
#include "fourier_tools.h"
#include "utils.h"
#include "unistd.h"
//#include <gperftools/profiler.h>


EcoTracker::EcoTracker(): ptr_train_thread(NULL), is_conf_high(true) {}


EcoTracker::EcoTracker(FeatureFactory *ptr_feat_handler, LearningParas lr_paras, ScaleMode scale_pred_mode,
                       InitMode init_pred_mode, bool use_conf, float area_scale, int min_img_sample_sz, int max_img_sample_sz,
                       ApceConfidence apce, float interp_coef, RegParas reg_filter_paras, CGParas train_cg_paras)
{
    ptr_feat_factory = ptr_feat_handler;
    frames_since_last_train = 0;
    scale_mode = scale_pred_mode;
    use_confidence = use_conf;
    apce_conf = apce;
    is_conf_high = true;
    search_area_scale = area_scale;
    min_image_sample_size = min_img_sample_sz;
    max_image_sample_size = max_img_sample_sz;
    learn_paras = lr_paras;
    interp_bicubic_a = interp_coef;
    reg_paras = reg_filter_paras;
    cg_paras = train_cg_paras;
    cg_paras.compute_init_forget_factor(learn_paras.learning_rate);
    ptr_train_thread = NULL;
    init_mode = init_pred_mode;
}


EcoTracker::~EcoTracker()
{
    stop_train_thread();
    stop_feat_threads();
}


void EcoTracker::compute_cosin_window(int height, int width, cv::Mat& cos_win)
{
    cv::Mat hann_vec(height + 2, 1, CV_32FC1);
    cv::Mat hann_hor(1, width + 2, CV_32FC1);
    int M = hann_vec.rows;
    int N = hann_hor.cols;
    double M_inv = 1. / (double)(M - 1);
    double N_inv = 1. / (double)(N - 1);
    for (int i = 0; i < M; ++i)
    {
        hann_vec.at<float>(i, 0) = 0.5 * (1 - cos(2 * CV_PI * i * M_inv));
    }
    for (int j = 0; j < N; ++j)
    {
        hann_hor.at<float>(0, j) = 0.5 * (1 - cos(2 * CV_PI * j * N_inv));
    }
    cv::Mat hann_mat = hann_vec * hann_hor;
    cos_win = hann_mat(cv::Range(1, height+1), cv::Range(1, width+1));
}

void EcoTracker::compute_guass_window(int height, int width, cv::Mat& guass_win)
{
    cv::Mat labels(height, width, CV_32FC1);
    int range_y[2] = {-height / 2, height - height / 2};
    int range_x[2] = {-width / 2, width - width / 2};

    double sigma1 = height * 0.1;
    double sigma2 = width * 0.1;
    double sigma_s = sigma1 * sigma2;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j)
    {
        float *row_ptr = labels.ptr<float>(j);
        double y_s = y * y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i)
        {
            row_ptr[i] = std::exp(-0.5 * (y_s + x * x) / sigma_s);
        }
    }
    guass_win = std::move(labels);
}

void EcoTracker::apply_cosin_window(const Vec2dMat& in_feat_mats, Vec2dMat& out_feat_mats)
{
    Vec2dMat ret_feat_mats;
    auto feat_num = in_feat_mats.size();
    ret_feat_mats.resize(feat_num);
    for (int i = 0; i < feat_num; ++i)
    {
        auto apply_cos_win = [&](const cv::Mat& src, cv::Mat& dst){ dst = src.mul(cos_window[i]); };
        // auto apply_cos_win = [&](const cv::Mat& src, cv::Mat& dst){ dst = fcv_mul(src,cos_window[i]); };
        proc_batch_vec1dmat(apply_cos_win, in_feat_mats[i], ret_feat_mats[i]);
    }
    out_feat_mats = std::move(ret_feat_mats);
}

void EcoTracker::apply_guass_window(const Vec2dMat& in_feat_mats, Vec2dMat& out_feat_mats)
{
    Vec2dMat ret_feat_mats;
    auto feat_num = in_feat_mats.size();
    ret_feat_mats.resize(feat_num);
    for (int i = 0; i < feat_num; ++i)
    {
        auto apply_guass_win = [&](const cv::Mat& src, cv::Mat& dst){ dst = src.mul(guass_window[i]); };
        // auto apply_guass_win = [&](const cv::Mat& src, cv::Mat& dst){ dst = fcv_mul(src,guass_window[i]); };
        proc_batch_vec1dmat(apply_guass_win, in_feat_mats[i], ret_feat_mats[i]);
    }
    out_feat_mats = std::move(ret_feat_mats);
}

void EcoTracker::apply_cosin_window(const Vec1dMat& in_feat_mat, int feat_ind, Vec1dMat& out_feat_mat)
{
    auto apply_cos_win = [&](const cv::Mat& src, cv::Mat& dst){ dst = src.mul(cos_window[feat_ind]); };
    // auto apply_cos_win = [&](const cv::Mat& src, cv::Mat& dst){ dst = fcv_mul(src,cos_window[feat_ind]); };
    proc_batch_vec1dmat(apply_cos_win, in_feat_mat, out_feat_mat);
}


void EcoTracker::get_interp_fourier(int len, cv::Mat &interp_mat, bool is_col_vec)
{
    assert(len % 2 == 1);
    int fh = (len - 1) / 2;
    cv::Mat fh_mat(len, 1, CV_32FC1);
    for (int i = -fh; i <= fh; ++i)
    {
        fh_mat.at<float>(i + fh, 0) = i / (float)len;
    }

    cv::Mat csf_mat;
    cubic_spline_fourier(fh_mat, interp_bicubic_a, csf_mat);
    csf_mat *= 1 / (float)len;

    // Center the feature grids by shifting the interpolated features
    // Multiply Fourier coeff with e^(-i*pi*k/N)
    cv::Mat ret(csf_mat.rows, 1, CV_32FC2);
    std::complex<float> imag_unit(0, 1);
    for (int i = 0; i < ret.rows; ++i)
    {
        float *ptr = ret.ptr<float>(i);
        float *ptr_csf = csf_mat.ptr<float>(i);
        std::complex<float> temp = ptr_csf[0] * exp(-imag_unit * (float)CV_PI / (float)len * (float)(-fh + i));
        ptr[0] = temp.real();
        ptr[1] = temp.imag();
    }
    if(is_col_vec)
        interp_mat = ret;
    else
        interp_mat = ret.t();
}


void EcoTracker::get_reg_filter(cv::Mat& sp_reg_filter)
{
    // normalization factor
    cv::Size2f reg_scale = base_target_size * 0.5f;

    // construct the regukarization window
    float wrg = 0.5 * (image_support_size.height - 1);
    float wcg = 0.5 * (image_support_size.width - 1);
    cv::Mat wrs(image_support_size, CV_32FC1);
    cv::Mat wcs(image_support_size, CV_32FC1);
    cv::Mat reg_win(image_support_size, CV_32FC1);
    float coef1 = reg_paras.reg_window_edge - reg_paras.reg_window_min;
    for (int i = 0; i < image_support_size.height; ++i)
    {
        float *wrs_ptr = wrs.ptr<float>(i);
        float *wcs_ptr = wcs.ptr<float>(i);
        float *reg_ptr = reg_win.ptr<float>(i);
        for (int j = 0; j < image_support_size.width; ++j)
        {
            // all the columns are the same
            wrs_ptr[j] = - wrg + i;
            // all the rows are the same
            wcs_ptr[j] = - wcg + j;

            reg_ptr[j] = coef1 * (pow(fabs(wrs_ptr[j] / reg_scale.height), reg_paras.reg_window_power) +
                                  pow(fabs(wcs_ptr[j] / reg_scale.width), reg_paras.reg_window_power)) + reg_paras.reg_window_min;
        }
    }

    // compute the DFT and enforce sparsity
    cv::Mat reg_win_dft;
    // FFT2
    cv::dft(reg_win, reg_win_dft, cv::DFT_COMPLEX_OUTPUT);
    float sz_area = image_support_size.height * image_support_size.width;
    reg_win_dft = reg_win_dft / sz_area;

    float max_mag = 0;
    cv::Mat reg_dft_mag(reg_win_dft.size(), CV_32FC1);
    for (int i = 0; i < reg_win_dft.rows; ++i)
    {
        float *dft_ptr = reg_win_dft.ptr<float>(i);
        float *mag_ptr = reg_dft_mag.ptr<float>(i);
        for (int j = 0; j < reg_win_dft.cols; ++j)
        {
            std::complex<float> complex_value(dft_ptr[2*j], dft_ptr[2*j+1]);
            float mag = abs(complex_value);
            max_mag = std::max(mag, max_mag);
            mag_ptr[j] = mag;
        }
    }

    float sparse_thresh = reg_paras.reg_sparsity_thresh * max_mag;
    for (int i = 0; i < reg_win_dft.rows; ++i)
    {
        float *dft_ptr = reg_win_dft.ptr<float>(i);
        float *mag_ptr = reg_dft_mag.ptr<float>(i);
        for (int j = 0; j < reg_win_dft.cols; ++j)
        {
            if(mag_ptr[j] < sparse_thresh)
            {
                dft_ptr[2*j] = 0;
                dft_ptr[2*j + 1] = 0;
            }
        }
    }

    // do the inverse transform, correct window minimum
    cv::Mat reg_win_sparse;
    // inverse fft
    cv::dft(reg_win_dft, reg_win_sparse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    double min_sparse_value;
    cv::minMaxLoc(reg_win_sparse, &min_sparse_value);
    std::complex<float> first_value(reg_win_dft.ptr<float>(0)[0], reg_win_dft.ptr<float>(0)[1]);
    first_value = first_value - sz_area * (float)min_sparse_value + reg_paras.reg_window_min;
    reg_win_dft.ptr<float>(0)[0] = first_value.real();
    reg_win_dft.ptr<float>(0)[1] = first_value.imag();

    cv::Mat reg_win_dft_shift = fftshift(reg_win_dft);

    // find the regularization filter by removing the zeros
    std::vector<bool> is_row_allzero(reg_win_dft_shift.rows, true);
    std::vector<bool> is_col_allzero(reg_win_dft_shift.cols, true);
    int nonzero_rows_count = 0, nonzero_cols_cout = 0;
    for (int i = 0; i < reg_win_dft_shift.rows; ++i)
    {
        float *ptr_reg_shift = reg_win_dft_shift.ptr<float>(i);
        for (int j = 0; j < reg_win_dft_shift.cols; ++j)
        {
            std::complex<float> ele(ptr_reg_shift[2*j], ptr_reg_shift[2*j + 1]);
            is_row_allzero[i] = is_row_allzero[i] && (ele == 0.f);
            is_col_allzero[j] = is_col_allzero[j] && (ele == 0.f);
        }
        if(!is_row_allzero[i])
            nonzero_rows_count++;
    }

    nonzero_cols_cout = (int)std::count(is_col_allzero.begin(), is_col_allzero.end(), false);

    assert(nonzero_rows_count > 0 && nonzero_cols_cout > 0);
    sp_reg_filter = cv::Mat(nonzero_rows_count, nonzero_cols_cout, CV_32FC1);
    int row_ind = 0, col_ind = 0;
    for (int i = 0; i < reg_win_dft_shift.rows; ++i)
    {
        float *ptr_reg_shift = reg_win_dft_shift.ptr<float>(i);
        if(is_row_allzero[i]) continue;
        for (int j = 0; j < reg_win_dft_shift.cols; ++j)
        {
            if(!is_col_allzero[j])
            {
                sp_reg_filter.at<float>(row_ind, col_ind) = ptr_reg_shift[2*j]; // real
                col_ind++;
                if(col_ind == nonzero_cols_cout)
                {
                    row_ind++;
                    col_ind = 0;
                }
            }
        }
    }
}


void EcoTracker::init_params(const Sample &sample)
{
    filter_size.clear();
    feat_pad_size.clear();
    fouri_kx.clear();
    fouri_ky.clear();
    guass_yf.clear();
    interp1_fs.clear();
    interp2_fs.clear();
    reg_filter.clear();
    reg_energy.clear();
    cg_states.clear();

    feat_output_size.height = 0;
    feat_output_size.width = 0;

    auto numel = sample.size();
    for (int i = 0; i < numel; i++)
    {
        const Feature& feat = sample[i];
        int feat_width = feat[0].cols;
        int feat_height = feat[0].rows;

        cv::Mat cos_win;
        this->compute_cosin_window(feat_height, feat_width, cos_win);
        this->cos_window.push_back(cos_win);

        cv::Mat guass_win;
        this->compute_guass_window(feat_height, feat_width, guass_win);
        this->guass_window.push_back(guass_win);

        cv::Size filter_sz;
        filter_sz.width = feat_width + (feat_width + 1) % 2;
        filter_sz.height = feat_height + (feat_height + 1) % 2;
        this->filter_size.push_back(filter_sz);

        feat_output_size.width = std::max(feat_output_size.width, filter_sz.width);
        if(feat_output_size.height < filter_sz.height)
        {
            feat_output_size.height = filter_sz.height;
            max_filter_key = i;
        }

        int ky_low = - (int)ceil(0.5 * (filter_sz.height - 1));
        int ky_high = (int)floor(0.5 * (filter_sz.height - 1));
        cv::Mat ky_vec(ky_high - ky_low + 1, 1, CV_32FC1);
        for (int j = ky_low; j <= ky_high ; ++j)
        {
            ky_vec.at<float>(j - ky_low, 0) = j;
        }
        fouri_ky.push_back(ky_vec);

        int kx_low = - (int)ceil(0.5 * (filter_sz.width - 1));
        int kx_high = 0;
        cv::Mat kx_vec(1, kx_high - kx_low + 1, CV_32FC1);
        for (int k = kx_low; k <= kx_high; ++k)
        {
            kx_vec.at<float>(0, k - kx_low) = k;
        }
        fouri_kx.push_back(kx_vec);

        cv::Mat fs1;
        get_interp_fourier(filter_sz.height, fs1);
        interp1_fs.push_back(fs1);

        cv::Mat fs2;
        get_interp_fourier(filter_sz.width, fs2, false);
        interp2_fs.push_back(fs2);
    }

    float sig_y[2];
    float base_len = sqrt(floor(base_target_size.width) * floor(base_target_size.height));
    sig_y[0] = base_len * learn_paras.output_sigma_factor * feat_output_size.height / image_support_size.height;
    sig_y[1] = base_len * learn_paras.output_sigma_factor * feat_output_size.width / image_support_size.width;

    for (int i = 0; i < numel; i++)
    {
        const cv::Size& filter_sz = this->filter_size[i];
        cv::Size pad_sz;
        pad_sz.width = (this->feat_output_size.width - filter_sz.width) / 2;
        pad_sz.height = (this->feat_output_size.height - filter_sz.height) / 2;
        this->feat_pad_size.push_back(pad_sz);

        cv::Mat& ky = fouri_ky[i];
        cv::Mat yf_y(ky.rows, 1, CV_32FC1);
        for (int j = 0; j < ky.rows; ++j)
        {
            yf_y.at<float>(j, 0) = sqrt(2 * CV_PI) * sig_y[0] / feat_output_size.height * \
                                exp(-2 * pow(((CV_PI * sig_y[0] * ky.at<float>(j, 0)) / (float)feat_output_size.height), 2));
        }

        cv::Mat& kx = fouri_kx[i];
        cv::Mat yf_x(1, kx.cols, CV_32FC1);
        for (int k = 0; k < kx.cols; ++k)
        {
            yf_x.at<float>(0, k) = sqrt(2 * CV_PI) * sig_y[1] / feat_output_size.width * \
                                exp(-2 * pow(((CV_PI * sig_y[1] * kx.at<float>(0, k)) / (float)feat_output_size.width), 2));
        }

        cv::Mat yf = yf_y * yf_x;
        guass_yf.push_back(yf);
    }

    cv::Mat sp_reg_filter;
    get_reg_filter(sp_reg_filter);
    cv::Mat sp_reg_filter_flat = sp_reg_filter.reshape(0, sp_reg_filter.total());
    cv::Mat sp_reg_energy = sp_reg_filter_flat.t() * sp_reg_filter_flat;
    float energy_value = sp_reg_energy.at<float>(0, 0);
    for (int i = 0; i < numel; ++i)
    {
        reg_filter.push_back(sp_reg_filter);
        reg_energy.push_back(energy_value);
    }

    cg_states.resize(numel);
}


void EcoTracker::sample_patch(const cv::Mat &img, cv::Point center_pos, cv::Size sample_sz,
                              cv::Size output_size, cv::Mat& img_patch)
{
    assert(img.channels() == 3);
    // downsample factor
    float resize_factor = std::min(sample_sz.width / (float)output_size.width,
                                    sample_sz.height / (float)output_size.height);
    int df = (int)fmax(floor(resize_factor - 0.1f), 1.);
    cv::Size2f new_sample_sz(sample_sz.width, sample_sz.height);
    cv::Point2f new_pos(center_pos.x, center_pos.y);
    cv::Mat img_ = img.clone();
    
    cv::Point os;
    if(df > 1)
    {
        // compute offset and new center position
        os.x = center_pos.x % df;
        os.y = center_pos.y % df;

        new_pos.x = (center_pos.x - os.x) / (float)df;
        new_pos.y = (center_pos.y - os.y) / (float)df;

        new_sample_sz.height /= (float)df;
        new_sample_sz.width /= (float)df;

        int new_height = (img.rows - os.y - 1) / df + 1;
        int new_width = (img.cols - os.x - 1) / df + 1;
        img_ = cv::Mat(new_height, new_width, img.type());
        int row_ind = 0, col_ind = 0;
        for (int i = os.y; i < img.rows; i += df)
        {
            const uchar *ptr = img.ptr<uchar>(i);
            uchar *ptr_new = img_.ptr<uchar>(row_ind);
            for (int j = os.x; j < img.cols; j += df)
            {
                for (int k = 0; k < 3; ++k)
                {
                    ptr_new[3 * col_ind + k] = ptr[3 * j + k];
                }
                col_ind++;
            }
            row_ind++;
            col_ind = 0;
        }
    }

    //make sure the size is not too small and round it
    new_sample_sz.width = std::max((int)round(new_sample_sz.width), 2);
    new_sample_sz.height = std::max((int)round(new_sample_sz.height), 2);
    int half_width = (int)floor(0.5f * (new_sample_sz.width + 1));
    int half_height = (int)floor(0.5f * (new_sample_sz.height + 1));

    cv::Rect roi;
    int pad_left = 0, pad_right = 0, pad_up = 0, pad_down = 0;
    int x_start = (int)(new_pos.x - half_width);
    int x_end = (int)(new_pos.x + new_sample_sz.width - half_width);
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

        if(x >= img_.cols)
            pad_right++;
        else
        {
            if(is_set_x)
                roi.width = x - roi.x + 1;
        }
    }

    int y_start = (int)(new_pos.y - half_height);
    int y_end = (int)(new_pos.y + new_sample_sz.height - half_height);
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

        if(y >= img_.rows)
            pad_down++;
        else
        {
            if(is_set_y)
                roi.height = y - roi.y + 1;
        }
    }

    cv::Mat roi_img  = cv::Mat(img_, roi);
    cv::copyMakeBorder(roi_img, img_patch, pad_up, pad_down, pad_left, pad_right,
                       cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    assert(img_patch.size() == cv::Size((int)new_sample_sz.width, (int)new_sample_sz.height));

    if(img_patch.size() != output_size)
        cv::resize(img_patch, img_patch, output_size);
}


void EcoTracker::extract_features(const cv::Mat &img, cv::Point center_pos, float sample_scale, Sample& feats)
{
    auto extractor_cnt = ptr_feat_factory->ptr_extractors.size();
    for (int i = 0; i < extractor_cnt; ++i)
    {
        BaseFeatureExtractor *ptr = ptr_feat_factory->ptr_extractors[i];
        cv::Size img_input_sz = ptr->input_size;
        cv::Size scaled_sample_sz;
        scaled_sample_sz.width = (int)floor(sample_scale * img_input_sz.width);
        scaled_sample_sz.height = (int)floor(sample_scale * img_input_sz.height);

        cv::Mat img_patch;
        sample_patch(img, center_pos, scaled_sample_sz, img_input_sz, img_patch);

        ptr->extract_feature(img_patch, feats);
    }
    ptr_feat_factory->normalize_feature(feats, feats);
}


void EcoTracker::extract_features_each(int extractor_id, const cv::Mat& img, cv::Point center_pos,
                                      float sample_scale, Sample& feats)
{
    BaseFeatureExtractor *ptr = ptr_feat_factory->ptr_extractors[extractor_id];
    cv::Size img_input_sz = ptr->input_size;
    cv::Size scaled_sample_sz;
    scaled_sample_sz.width = (int)floor(sample_scale * img_input_sz.width);
    scaled_sample_sz.height = (int)floor(sample_scale * img_input_sz.height);

    cv::Mat img_patch;
    sample_patch(img, center_pos, scaled_sample_sz, img_input_sz, img_patch);

    ptr->extract_feature(img_patch, feats);
    ptr_feat_factory->normalize_feature(feats, feats);
}


void EcoTracker::compute_response(const Vec2dMat &feat_mats, const Vec2dMat& filters)
{
    auto feat_cnt = feat_mats.size();
    scores_fs_feat.clear();
    scores_fs_feat.resize(feat_cnt);
    const Vec1dMat& max_size_feat = feat_mats[max_filter_key];
    scores_fs_feat[max_filter_key] = cv::Mat::zeros(max_size_feat[0].size(), CV_32FC2);
    for (int k = 0; k < max_size_feat.size(); ++k)
    {
        ComplexMat hf_cmat(filters[max_filter_key][k]);
        ComplexMat max_sz_cmat(max_size_feat[k]);
        scores_fs_feat[max_filter_key] += hf_cmat.mul(max_sz_cmat).to_cv_mat();
    }

    scores_fs_sum = scores_fs_feat[max_filter_key].clone();
    for (int i = 0; i < feat_cnt; ++i)
    {
        if(i == max_filter_key) continue;

        auto channels = feat_mats[i].size();
        scores_fs_feat[i] = cv::Mat::zeros(feat_mats[i][0].size(), CV_32FC2);
        for (int j = 0; j < channels; ++j)
        {
            ComplexMat hf_cmat(filters[i][j]);
            ComplexMat feat_cmat(feat_mats[i][j]);
            // each channel of a single feature
            scores_fs_feat[i] += hf_cmat.mul(feat_cmat).to_cv_mat();
        }

        auto sum_slice = scores_fs_sum(cv::Range(feat_pad_size[i].height, scores_fs_sum.rows - feat_pad_size[i].height),
                                       cv::Range(feat_pad_size[i].width, scores_fs_sum.cols - feat_pad_size[i].width));
        sum_slice += scores_fs_feat[i];
    }
}

// only implemented grid search for coarse translation vector
// TODO: newton iteration for higher precision
cv::Point2f EcoTracker::optimize_scores(int iter_cnt)
{
    sample_fs(scores_fs_sum, sampled_scores);
    double min_value, max_value;
    cv::Point min_pt, max_pt;
    cv::minMaxLoc(sampled_scores, &min_value, &max_value, &min_pt, &max_pt);
    max_score = (float)max_value;

    cv::Point2f trans_vec;
    int half_height = (int)floor(0.5 * (sampled_scores.rows - 1));
    int half_width = (int)floor(0.5 * (sampled_scores.cols - 1));
    trans_vec.y = (max_pt.y + half_height) % sampled_scores.rows - half_height;
    trans_vec.x = (max_pt.x + half_width) % sampled_scores.cols - half_width;

    return trans_vec;
}


void EcoTracker::do_shift_samples(const Vec2dMat &in_mats, cv::Point2f shift_vec, Vec2dMat &out_mats)
{
    Vec2dMat ret_mats;
    ret_mats.resize(in_mats.size());
    for (int i = 0; i < in_mats.size(); ++i)
    {
        shift_sample(in_mats[i], shift_vec, fouri_kx[i], fouri_ky[i], ret_mats[i]);
    }
    out_mats = std::move(ret_mats);
}


void EcoTracker::compute_sample_energy(const Vec2dMat &sample, Vec2dMat &energy)
{
    Vec2dMat out_energy;
    out_energy.resize(sample.size());
    for (int i = 0; i < sample.size(); ++i)
    {
        out_energy[i].resize(sample[i].size());
        for (int j = 0; j < sample[i].size(); ++j)
        {
            ComplexMat cplx_cmat(sample[i][j]);
            ComplexMat cplx_cmat_conj = cplx_cmat.conj();
            out_energy[i][j] = cplx_cmat.mul(cplx_cmat_conj).abs2();
        }
    }
    energy = std::move(out_energy);
}


void EcoTracker::compute_proj_energy(Vec1dMat &proj_energy)
{
    const Vec1dMat& proj_mats = ptr_feat_factory->project_mats;
    float dims_sum = 0;
    for (int i = 0; i < proj_mats.size(); ++i)
    {
        dims_sum += proj_mats[i].rows;
    }

    proj_energy.resize(guass_yf.size());
    for (int i = 0; i < guass_yf.size(); ++i)
    {
        cv::Mat& yf = guass_yf[i];
        assert(yf.channels() == 1);
        cv::Mat sqr = yf.mul(yf); // yf is real, abs() is not necessary
        cv::Scalar yf_sum_scalar = cv::sum(sqr);
        auto yf_sum = yf_sum_scalar[0];

        proj_energy[i] = cv::Mat::ones(proj_mats[i].size(), CV_32FC1) * (2 * yf_sum / dims_sum);
    }
}


void EcoTracker::lhs_operation_joint_each(int feat_ind, const Vec2dMat& hf, const Vec1dMat& samplesf,
                                          const Vec1dMat& init_samplef, const cv::Mat& init_sample_H,
                                          const Vec1dMat& init_hf, Vec2dMat& hf_out)
{
    Vec2dMat hf_dst(hf.size());
    for (int i = 0; i < hf.size(); ++i)
    {
        hf_dst[i].resize(hf[i].size());
    }

    auto& hf_filter = hf[0];
    auto& hf_proj = hf[1][0];

    cv::Mat P;
    if(hf_proj.channels() == 2)
        P = ComplexMat(hf_proj).real();
    else
        P = hf_proj;

    // Compute the operation corresponding to the data term in the optimization (blockwise matrix multiplication)
    // implements: A' diag(sample_weights) A f

    // sum over all features and feature blocks, assume the feature with the highest resolution is first
    cv::Mat sh = cv::Mat::zeros(hf_filter[0].size(), CV_32FC2);
    for (int i = 0; i < hf_filter.size(); ++i)
    {
        ComplexMat s_cmat(samplesf[i]);
        ComplexMat h_cmat(hf_filter[i]);
        sh += s_cmat.mul(h_cmat).to_cv_mat(); // element-wise multiplication
    }
    sh = ComplexMat(sh).conj().to_cv_mat();

    // multiply with the transpose
    Vec1dMat hf_dst1(samplesf.size());
    ComplexMat sh_cmat(sh);
    for (int i = 0; i < samplesf.size(); ++i)
    {
        ComplexMat s_cmat(samplesf[i]);
        ComplexMat mul_res = sh_cmat.mul(s_cmat);
        hf_dst1[i] = mul_res.conj().to_cv_mat();
    }

    // compute the operation corresponding to the regularization term (convovle each feature dimension with the
    // DFT of w, and the transposed operation) add the regularization part
    int reg_pad = std::min(reg_filter[feat_ind].cols - 1, hf_filter[0].cols - 1);
    // add part needed for convolution
    Vec1dMat hf_conv(hf_filter.size());
    for (int i = 0; i < hf_filter.size(); ++i)
    {
        auto& hf_mat = hf_filter[i];
        auto hf_slice = hf_mat.colRange(hf_mat.cols - reg_pad - 1, hf_mat.cols - 1);
        cv::Mat hf_slice_rot;
        rot90(hf_slice, hf_slice_rot, ROTATE_180);
        hf_slice_rot = ComplexMat(hf_slice_rot).conj().to_cv_mat();

        // cat operation
        cv::hconcat(hf_mat, hf_slice_rot, hf_conv[i]);
    }

    // do first convolution
    for (int i = 0; i < hf_conv.size(); ++i)
    {
        conv2_complex(hf_conv[i], reg_filter[feat_ind], CONV_FULL, hf_conv[i]);
    }

    // do final convolution and put together result
    for (int i = 0; i < hf_dst1.size(); ++i)
    {
        cv::Mat conv_res;
        auto hf_conv_slice = hf_conv[i].colRange(0, hf_conv[i].cols - reg_pad);
        conv2_complex(hf_conv_slice, reg_filter[feat_ind], CONV_VALID, conv_res);
        hf_dst1[i] += conv_res;
    }

    // stuff related to the projection matrix
    cv::Mat feat_flat; // complex
    int height, width, channels;
    Vec1dMat feat_restore;
    ptr_feat_factory->flatten_feature(init_samplef, feat_flat, height, width, channels);
    // P is real
    feat_flat = (ComplexMat(feat_flat) * ComplexMat(P)).to_cv_mat();
    ptr_feat_factory->restore_flatten_feature(feat_flat, height, width, feat_flat.cols, feat_restore);

    cv::Mat BP_cell = cv::Mat::zeros(init_samplef[0].size(), CV_32FC2);
    for (int i = 0; i < feat_restore.size(); ++i)
    {
        ComplexMat feat_cmat(feat_restore[i]);
        ComplexMat init_hf_cmat(init_hf[i]);
        BP_cell += feat_cmat.mul(init_hf_cmat).to_cv_mat();
    }

    // multiply with the transpose: A^H * BP
    ComplexMat BP_cmat(BP_cell);
    for (int i = 0; i < hf_dst1.size(); ++i)
    {
        ComplexMat sample_cmat(samplesf[i]);
        ComplexMat mul_res = BP_cmat.mul(sample_cmat);
        hf_dst[0][i] = hf_dst1[i] + mul_res.conj().to_cv_mat();
    }

    // B^H * BP
    cv::Mat fBP(init_hf.size(), init_hf[0].rows * init_hf[0].cols, CV_32FC2);
    // Compute proj matrix part: B^H * A_m * f
    cv::Mat shBP(init_hf.size(), init_hf[0].rows * init_hf[0].cols, CV_32FC2);
    for (int i = 0; i < init_hf.size(); ++i)
    {
        ComplexMat conj_cmat = ComplexMat(init_hf[i]).conj();

        cv::Mat times1_t = conj_cmat.mul(BP_cmat).to_cv_mat().t();
        times1_t.reshape(0, 1).copyTo(fBP.row(i));

        cv::Mat times2_t = conj_cmat.mul(sh_cmat).to_cv_mat().t();
        times2_t.reshape(0, 1).copyTo(shBP.row(i));
    }
    // optimized cv::Mat col operation
    fBP = fBP.t();
    shBP = shBP.t();

    // index of the last frequency column starts
    int fi = hf_filter[0].rows * (hf_filter[0].cols - 1) + 1;
    // B^H *BP
    ComplexMat XH_cmat(init_sample_H);
    ComplexMat fBP_cmat(fBP);
    ComplexMat XH_slice_cmat(init_sample_H.colRange(fi - 1, init_sample_H.cols));
    ComplexMat fBP_slice_cmat(fBP.rowRange(fi - 1, fBP.rows));

    ComplexMat delta_res1 = XH_cmat * fBP_cmat - XH_slice_cmat * fBP_slice_cmat;
    cv::Mat hf_dst2 = delta_res1.real() * 2 + P * cg_paras.projection_reg;

    ComplexMat shBP_cmat(shBP);
    ComplexMat shBP_slice_cmat(shBP.rowRange(fi - 1, shBP.rows));

    ComplexMat delta_res2 = XH_cmat * shBP_cmat - XH_slice_cmat * shBP_slice_cmat;
    hf_dst[1][0] = hf_dst2 + delta_res2.real() * 2;

    hf_out = hf_dst;
}


void EcoTracker::lhs_operation_each(int feat_ind, const Vec1dMat& hf, const SampleSpaceModel& sample_mdl, Vec1dMat& hf_out)
{
    // Compute the operation corresponding to the data term in the optimization (blockwise matrix multiplication)
    // implements: A' diag(sample_weights) A f
    auto sample_capacity = sample_mdl.sample_capacity;
    // sample num x feat channels x Mat
    auto& sample_space = sample_mdl.sample_space[feat_ind];
    auto& sample_weights = sample_mdl.sample_weights;

    // sum over all features and feature blocks, assume the feature with the highest resolution is first
    Vec1dMat sh; // sample num x 1 x Mat
    init_vec1dmat(sample_capacity, hf[0].size(), CV_32FC2, sh);
    auto feat_channels = sample_space[0].size();
    for (int i = 0; i < sh.size(); ++i)
    {
        for (int j = 0; j < feat_channels; ++j)
        {
            ComplexMat s_cmat(sample_space[i][j]);
            ComplexMat hf_cmat(hf[j]);
            sh[i] += s_cmat.mul(hf_cmat).to_cv_mat();
        }
    }

    // weight all the samples and take conjugate
    for (int i = 0; i < sh.size(); ++i)
    {
        sh[i] *= sample_weights[i];
        sh[i] = ComplexMat(sh[i]).conj().to_cv_mat();
    }

    Vec1dMat hf_dst;
    init_vec1dmat(feat_channels, sample_space[0][0].size(), CV_32FC2, hf_dst);
    for (int i = 0; i < sh.size(); ++i)
    {
        ComplexMat sh_cmat(sh[i]);
        for (int j = 0; j < feat_channels; ++j)
        {
            ComplexMat sample_cmat(sample_space[i][j]);
            ComplexMat mul_res = sh_cmat.mul(sample_cmat);
            hf_dst[j] += mul_res.conj().to_cv_mat();
        }
    }

    // compute the operation corresponding to the regularization term (convovle each feature dimension with the
    // DFT of w, and the transposed operation) add the regularization part
    int reg_pad = std::min(reg_filter[feat_ind].cols - 1, hf[0].cols - 1);
    // add part needed for convolution
    Vec1dMat hf_conv(hf.size());
    for (int i = 0; i < hf.size(); ++i)
    {
        auto& hf_mat = hf[i];
        auto hf_slice = hf_mat.colRange(hf_mat.cols - reg_pad - 1, hf_mat.cols - 1);
        cv::Mat hf_slice_rot;
        rot90(hf_slice, hf_slice_rot, ROTATE_180);
        hf_slice_rot = ComplexMat(hf_slice_rot).conj().to_cv_mat();

        // cat operation
        cv::hconcat(hf_mat, hf_slice_rot, hf_conv[i]);
    }

    // do first convolution
    for (int i = 0; i < hf_conv.size(); ++i)
    {
        conv2_complex(hf_conv[i], reg_filter[feat_ind], CONV_FULL, hf_conv[i]);
    }

    // do final convolution and put together result
    for (int i = 0; i < hf_dst.size(); ++i)
    {
        cv::Mat conv_res;
        auto hf_conv_slice = hf_conv[i].colRange(0, hf_conv[i].cols - reg_pad);
        conv2_complex(hf_conv_slice, reg_filter[feat_ind], CONV_VALID, conv_res);
        hf_dst[i] += conv_res;
    }

    hf_out = std::move(hf_dst);
}


float EcoTracker::inner_product_joint(const Vec2dMat& xf, const Vec2dMat& yf)
{
    std::complex<float> ip(0, 0);
    // filter part, for each feat channel
    for (int i = 0; i < xf[0].size(); ++i)
    {
        auto& xf_mat = xf[0][i];
        auto& yf_mat = yf[0][i];

        ComplexMat xf_cmat(xf_mat);
        ComplexMat yf_cmat(yf_mat);
        ip += xf_cmat.conj().dot(yf_cmat) * 2.f;

        ComplexMat xcol_cmat(xf_mat.col(xf_mat.cols - 1));
        ComplexMat ycol_cmat(yf_mat.col(yf_mat.cols - 1));
        ip -= xcol_cmat.conj().dot(ycol_cmat);
    }

    // projection matrix part, only one channel
    ComplexMat xproj_cmat(xf[1][0]);
    ComplexMat yproj_cmat(yf[1][0]);
    ip += xproj_cmat.conj().dot(yproj_cmat);

    return ip.real();
}


float EcoTracker::inner_product_filter(const Vec1dMat &xf, const Vec1dMat &yf)
{
    std::complex<float> ip(0, 0);
    // for each channel
    for (int i = 0; i < xf.size(); ++i)
    {
        auto& xf_mat = xf[i];
        auto& yf_mat = yf[i];
        ComplexMat xf_cmat(xf_mat);
        ComplexMat yf_cmat(yf_mat);
        ip += xf_cmat.conj().dot(yf_cmat) * 2.f;

        ComplexMat xcol_cmat(xf_mat.col(xf_mat.cols - 1));
        ComplexMat ycol_cmat(yf_mat.col(yf_mat.cols - 1));
        ip -= xcol_cmat.conj().dot(ycol_cmat);
    }
    return ip.real();
}


// the preconditioner operation in Conjugate Gradient
void EcoTracker::diag_precond_joint(const Vec2dMat& hf, const Vec2dMat& M_diag, Vec2dMat& hf_out)
{
    Vec2dMat hf_dst(hf.size());
    for (int i = 0; i < hf.size(); ++i)
    {
        hf_dst[i].resize(hf[i].size());
        for (int j = 0; j < hf[i].size(); ++j)
        {
            // element-wise divide
            bool is_complex = (hf[i][j].channels() == 2) || (M_diag[i][j].channels() == 2);
            if(is_complex)
                hf_dst[i][j] = (ComplexMat(hf[i][j]) / ComplexMat(M_diag[i][j])).to_cv_mat();
            else
                hf_dst[i][j] = hf[i][j] / M_diag[i][j];
        }
    }
    hf_out = std::move(hf_dst);
}


void EcoTracker::diag_precond_filter(const Vec1dMat &hf, const Vec1dMat &M_diag, Vec1dMat &hf_out)
{
    Vec1dMat hf_dst(hf.size());
    for (int i = 0; i < hf.size(); ++i)
    {
        bool is_complex = (hf[i].channels() == 2) || (M_diag[i].channels() == 2);
        // element-wise divide
        if(is_complex)
            hf_dst[i] = (ComplexMat(hf[i]) / ComplexMat(M_diag[i])).to_cv_mat();
        else
            hf_dst[i] = hf[i] / M_diag[i];
    }
    hf_out = std::move(hf_dst);
}


float EcoTracker::pcg_ccot_joint_each(std::function<void(const Vec2dMat &, Vec2dMat &)> A, const Vec2dMat &b,
                                     int max_iter, std::function<void(const Vec2dMat &, Vec2dMat &)> M,
                                     std::function<float(const Vec2dMat &, const Vec2dMat &)> ip, const Vec2dMat &x0,
                                     Vec2dMat &x_out)
{
    Vec2dMat x;
    proc_batch_vec2dmat(lamda_copy_mat, x0, x);
    float rho = 1, rho1, beta, alpha;

    Vec2dMat Ax;
    A(x, Ax);
    Vec2dMat r;
    // r = b - Ax
    vec2dmat_linear_opt(r, Ax, b, -1.f);

    Vec2dMat y, p, q;
    float res_err = 0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        M(r, y);
        auto& z = y;
        rho1 = rho;
        rho = ip(r, z);

        if(iter == 0 && p.empty())
            p = z;
        else
        {
            // use Fletcher-Reeves
            beta = rho / rho1;
            assert(beta != 0.f && fabs(beta) != FLOAT_INF);
            beta = std::max(0.f, beta);
            // p = beta * p + z
            vec2dmat_linear_opt(p, p, z, beta);
        }

        A(p, q);
        float pq = ip(p, q);
        assert(pq > 0 && pq != FLOAT_INF);
        alpha = rho / pq;
        assert(fabs(alpha) != FLOAT_INF);

        // form new iterate, x = x + alpha * p
        vec2dmat_linear_opt(x, p, x, alpha);
        if(iter < max_iter - 1)
            vec2dmat_linear_opt(r, q, r, -alpha); // r = r - alpha * q

        res_err += sqrt(ip(r, r));
    }

    x_out = x;
    res_err /= max_iter;
    return res_err;
}


void EcoTracker::pcg_ccot_each(int feat_ind, std::function<void(const Vec1dMat &, Vec1dMat &)> A, const Vec1dMat &b,
                               int max_iter, std::function<void(const Vec1dMat &, Vec1dMat &)> M,
                               std::function<float(const Vec1dMat &, const Vec1dMat &)> ip,
                               const Vec1dMat &x0, Vec1dMat &x_out)
{
    Vec1dMat x;
    proc_batch_vec1dmat(lamda_copy_mat, x0, x);
    float rho = 1, rho1, rho2, beta, alpha;

    Vec1dMat p, r_prev;
    auto& cg_state = cg_states[feat_ind];
    bool is_load_state = !(cg_state.p.empty()) && cg_paras.init_forget_factor > 0;
    if(is_load_state)
    {
        p = cg_state.p;
        rho = cg_state.rho / cg_paras.init_forget_factor;
        r_prev = cg_state.r_prev;
    }

    Vec1dMat Ax;
    A(x, Ax);
    Vec1dMat r;
    // r = b - Ax
    vec1dmat_linear_opt(r, Ax, b, -1.f);

    Vec1dMat y, q;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        M(r, y);
        auto& z = y;
        rho1 = rho;
        rho = ip(r, z);

        if(iter == 0 && p.empty())
            p = z;
        else
        {
            // use Polak-Ribiere
            rho2 = ip(r_prev, z);
            beta = (rho - rho2) / rho1;
            assert(beta != 0.f && fabs(beta) != FLOAT_INF);
            beta = std::max(0.f, beta);
            // p = beta * p + z
            vec1dmat_linear_opt(p, p, z, beta);
        }

        A(p, q);
        float pq = ip(p, q);
        assert(pq > 0 && pq != FLOAT_INF);
        alpha = rho / pq;
        assert(fabs(alpha) != FLOAT_INF);

        r_prev = r;
        // form new iterate, x = x + alpha * p
        vec1dmat_linear_opt(x, p, x, alpha);

        if(iter < max_iter - 1)
            vec1dmat_linear_opt(r, q, r, -alpha); // r = r - alpha * q
    }

    cg_state.p = p;
    cg_state.rho = rho;
    cg_state.r_prev = r_prev;
    x_out = x;
}


void EcoTracker::train_joint_each(int feat_ind, Vec2dMat &hf, const Vec1dMat &init_samplef,
                                  const Vec1dMat &sample_energy, const cv::Mat &proj_energy, Vec1dMat &hf_out)
{
    // Get index for the start of the last column of frequencies
    int lf_ind = hf[0][0].rows * (hf[0][0].cols - 1) + 1;

    // Construct stuff for the proj matrix part
    auto init_channels = init_samplef.size();
    int numel = init_samplef[0].rows * init_samplef[0].cols;
    cv::Mat init_sample_H(init_channels, numel, init_samplef[0].type());
    for (int i = 0; i < init_channels; ++i)
    {
        // transpose before reshape, since cv::Mat is row-major, but matlab is column-major
        cv::Mat feat_i_t = init_samplef[i].t();
        cv::Mat feat_flat = feat_i_t.reshape(0, numel);
        feat_flat = ComplexMat(feat_flat).conj_t().to_cv_mat();
        feat_flat.copyTo(init_sample_H.row(i));
    }

    // Contruct preconditioner
    auto channels = sample_energy.size();
    Vec2dMat diag_M(hf.size());
    diag_M[0].resize(channels);
    diag_M[1].resize(1);

    cv::Mat mean_mat = cv::Mat::zeros(sample_energy[0].size(), sample_energy[0].type());
    mean_mat = std::accumulate(sample_energy.begin(), sample_energy.end(), mean_mat);
    mean_mat /= (float) channels;

    for (int i = 0; i < channels; ++i)
    {
        auto& s_mat = sample_energy[i];
        diag_M[0][i] = (s_mat * cg_paras.precond_data_param
                           + mean_mat * (1.f - cg_paras.precond_data_param)) * (1.f - cg_paras.precond_reg_param) \
                           + reg_energy[feat_ind] * cg_paras.precond_reg_param;
    }

    diag_M[1][0] = (proj_energy + cg_paras.projection_reg) * cg_paras.precond_proj_param;

    Vec1dMat init_samplef_proj;
    Vec2dMat rhs_samplef(hf.size());
    cv::Mat fyf(hf[0][0].rows * hf[0][0].cols, hf[0].size(), CV_32FC2);

    float prev_res_err = 0, delta_err;
    int init_cg_max_iter = (int)std::ceil(cg_paras.init_cg_iter / (float)cg_paras.init_gn_iter);
    for (int iter = 0; iter < cg_paras.init_gn_iter; ++iter)
    {
        // project sample with new matrix
        ptr_feat_factory->compress_each(init_samplef, feat_ind, init_samplef_proj);
        auto& init_hf = hf[0];

        // construct the right hand side vector for the filter part
        rhs_samplef[0].resize(init_samplef_proj.size());
        ComplexMat yf_cmat(guass_yf[feat_ind]);
        for (int j = 0; j < init_samplef_proj.size(); ++j)
        {
            ComplexMat init_sample_cmat(init_samplef_proj[j]);
            ComplexMat res = init_sample_cmat.conj().mul(yf_cmat);
            rhs_samplef[0][j] = res.to_cv_mat();
        }

        // construct the right hand side vector for the projection matrix part
        for (int j = 0; j < init_hf.size(); ++j)
        {
            ComplexMat hf_cmat(init_hf[j]);
            cv::Mat res_mat = hf_cmat.conj().mul(yf_cmat).to_cv_mat();
            cv::Mat res_mat_t = res_mat.t();
            res_mat_t.reshape(0, fyf.rows).copyTo(fyf.col(j));
        }

        const cv::Mat& proj_mat = ptr_feat_factory->project_mats[feat_ind];
        rhs_samplef[1].resize(1);
        ComplexMat xh_cmat(init_sample_H);
        ComplexMat fyf_cmat(fyf);
        ComplexMat xh_slice_cmat(init_sample_H.colRange(lf_ind - 1, init_sample_H.cols));
        ComplexMat fyf_slice_cmat(fyf.rowRange(lf_ind - 1, fyf.rows));

        ComplexMat res = xh_cmat * fyf_cmat - xh_slice_cmat * fyf_slice_cmat;
        rhs_samplef[1][0] = res.real() * 2 - proj_mat * cg_paras.projection_reg;

        // initialize the projection matrix increment to zero
        hf[1].resize(1);
        hf[1][0] = cv::Mat::zeros(proj_mat.size(), proj_mat.type());

        // do conjugate gradient
        auto A_fun = std::bind(&EcoTracker::lhs_operation_joint_each, this, feat_ind, std::placeholders::_1, 
                               init_samplef_proj, init_samplef, init_sample_H, init_hf, std::placeholders::_2);
        auto M_fun = std::bind(&EcoTracker::diag_precond_joint, this, std::placeholders::_1, diag_M, std::placeholders::_2);
        auto ip_fun = std::bind(&EcoTracker::inner_product_joint, this, std::placeholders::_1, std::placeholders::_2);
        float res_err = pcg_ccot_joint_each(A_fun, rhs_samplef, init_cg_max_iter, M_fun, ip_fun, hf, hf);

        // make the filter symmetric (avoid roundoff errors)
        proc_batch_vec1dmat(symmetrize_filter, hf[0], hf[0]);

        // add to the projection matrix
        ptr_feat_factory->modify_project_mat(hf[1][0], feat_ind);

        if(iter > 0)
        {
            delta_err = prev_res_err - res_err;
            if(delta_err >= 0.f && delta_err <= cg_paras.init_cg_eps)
               break;
        }
        prev_res_err = res_err;
    }

    hf_out = hf[0];
}


// Initial Gauss-Newton optimization of the filter and project matrix
void EcoTracker::train_joint(Vec3dMat &hf, const Vec2dMat &init_samplef, const Vec2dMat &sample_energy,
                             const Vec1dMat &proj_energy, Vec2dMat &hf_out)
{
    auto num_feats = hf.size();
    hf_out.resize(num_feats);
    for (int i = 0; i < num_feats; ++i)
    {
        hf_out[i].resize(hf[i][0].size());
    }

    std::vector<std::thread *> ptr_threads;
    for (int fid = 1; fid < num_feats; ++fid)
    {
        auto train_fun = std::bind(&EcoTracker::train_joint_each, this, fid, std::ref(hf[fid]), std::ref(init_samplef[fid]),
                                   std::ref(sample_energy[fid]), std::ref(proj_energy[fid]), std::ref(hf_out[fid]));
        ptr_threads.push_back(new std::thread(train_fun));
    }

    int rest_fid = 0;
    train_joint_each(rest_fid, hf[rest_fid], init_samplef[rest_fid], sample_energy[rest_fid],
                     proj_energy[rest_fid], hf_out[rest_fid]);

    for (int i = 0; i < num_feats - 1; ++i)
    {
        ptr_threads[i]->join();
        delete ptr_threads[i];
    }
}


void EcoTracker::train_filter_each(int feat_ind, const Vec1dMat& hf, const SampleSpaceModel& sample_mdl, Vec1dMat& hf_out)
{
    auto& sample_subspace = sample_mdl.sample_space[feat_ind];
    int num_samples = sample_mdl.sample_num;
    auto& sample_weights = sample_mdl.sample_weights;

    auto channels = sample_subspace[0].size();
    auto sample_sz = sample_subspace[0][0].size();
    // construct the right hand side vector
    Vec1dMat rhs_samplef(channels);
    for (int i = 0; i < channels; ++i)
    {
        rhs_samplef[i] = cv::Mat::zeros(sample_sz, CV_32FC2);
        for (int j = 0; j < num_samples; ++j)
        {
            float weights = sample_weights[j];
            rhs_samplef[i] += sample_subspace[j][i] * weights;
        }
    }

    ComplexMat yf_cmat(guass_yf[feat_ind]);
    for (int i = 0; i < rhs_samplef.size(); ++i)
    {
        ComplexMat rhs_cmat(rhs_samplef[i]);
        rhs_samplef[i] = rhs_cmat.conj().mul(yf_cmat).to_cv_mat();
    }

    // construct preconditioner
    auto s_energy = sample_energy[feat_ind];
    cv::Mat mean_mat = cv::Mat::zeros(s_energy[0].size(), s_energy[0].type());
    mean_mat = std::accumulate(s_energy.begin(), s_energy.end(), mean_mat);
    mean_mat /= (float) channels;

    Vec1dMat diag_M(channels);
    for (int i = 0; i < channels; ++i)
    {
        auto& s_mat = s_energy[i];
        diag_M[i] = (s_mat * cg_paras.precond_data_param
                     + mean_mat * (1.f - cg_paras.precond_data_param)) * (1.f - cg_paras.precond_reg_param)
                    + reg_energy[feat_ind] * cg_paras.precond_reg_param;
    }

    // do conjugate gradient
    auto A_fun = std::bind(&EcoTracker::lhs_operation_each, this, feat_ind, std::placeholders::_1,
                           sample_mdl, std::placeholders::_2);
    auto M_fun = std::bind(&EcoTracker::diag_precond_filter, this, std::placeholders::_1, diag_M, std::placeholders::_2);
    auto ip_fun = std::bind(&EcoTracker::inner_product_filter, this, std::placeholders::_1, std::placeholders::_2);
    pcg_ccot_each(feat_ind, A_fun, rhs_samplef, cg_paras.cg_iter, M_fun, ip_fun, hf, hf_out);
}


void EcoTracker::train_child_thread_entry(int ind)
{
    auto& is_quit = is_quit_train_childs[ind];
    auto& is_finished = finish_train_child_flags[ind];
    while(!is_quit.load())
    {
        while(is_finished.load() && !is_quit.load()) usleep(100);

        if (is_quit.load()) break;

        train_proc_funs[ind]();
        is_finished.store(true);
    }
}


void EcoTracker::train_filter(const Vec2dMat &hf, const Vec2dMat &sample_energy, const SampleSpaceModel &sample_mdl,
                              Vec2dMat &hf_out)
{
    auto num_feats = ptr_feat_factory->num_feat_blocks;
    Vec2dMat hf_dst(num_feats);
    for (int i = 0; i < num_feats; ++i)
    {
        hf_dst[i].resize(hf[i].size());
    }

    if(learn_paras.train_mode == SYNC || num_feats <= 1)
    {
        for (int i = 0; i < num_feats; ++i)
        {
            train_filter_each(i, hf[i], sample_mdl, hf_dst[i]);
        }
    }
    else
    {
        train_proc_funs.clear();
        for (int fid = 1; fid < num_feats; ++fid)
        {
            auto fun = std::bind(&EcoTracker::train_filter_each, this, fid,
                                  std::ref(hf[fid]), std::ref(sample_mdl), std::ref(hf_dst[fid]));
            train_proc_funs.push_back(std::move(fun));
        }

        if(train_child_threads.empty())
        {
            for (int i = 0; i < num_feats - 1; ++i)
            {
                is_quit_train_childs.emplace_back(false);
                finish_train_child_flags.emplace_back(false);

                auto child_proc = std::bind(&EcoTracker::train_child_thread_entry, this, i);
                train_child_threads.push_back(new std::thread(child_proc));
            }
        }
        else
        {
            for (int i = 0; i < finish_train_child_flags.size(); ++i)
            {
                finish_train_child_flags[i].store(false);
            }
        }

        int rest_fid = 0;
        train_filter_each(rest_fid, hf[rest_fid], sample_mdl, hf_dst[rest_fid]);

        while(1)
        {
            bool is_finish = true;
            for (int i = 0; i < num_feats - 1; ++i)
            {
                is_finish &= finish_train_child_flags[i].load();
            }
            if(is_finish)
                break;
            else
                usleep(100);
        }
    }
    hf_out = hf_dst;
}

void EcoTracker::fast_init_hf(const Sample& feats_proj, Vec2dMat& hf)
{
    for(int i = 0; i < hf.size(); i++)
    {
        ComplexMat yf_cmat(guass_yf[i] / feats_proj[i].size());
        for(int j = 0; j < feats_proj[i].size(); j++)
        {
            ComplexMat sample_cmat(feats_proj[i][j]);
            ComplexMat sample_cmat_conj = sample_cmat.conj();
            ComplexMat tp = sample_cmat.mul(sample_cmat_conj);
            cv::Mat lhs = tp.to_cv_mat();

            ComplexMat res = (sample_cmat_conj).mul(yf_cmat);
            cv::Mat rhs = res.to_cv_mat();
            HighClock c;
            c.Start();
            int size = lhs.rows * lhs.cols * lhs.channels();
            for(int n = 0; n < size; n++){
                if(*((float*)lhs.data + n) < pow(10, -9))
                    *((float*)lhs.data + n) = 0;
            }
            c.Stop();
            //LOGEO("for  time %lf", c.GetTime() / 1000);
            cv::Mat hf_t = rhs / lhs;
            hf[i].push_back(hf_t);
        }
    }
}


void EcoTracker::init_tracker_pos(const cv::Mat &img, const cv::Rect &box)
{
//    //LOGE("Debug fcvSetOperationMode start");
//    fcvSetOperationMode(FASTCV_OP_CPU_OFFLOAD);
//    //LOGE("Debug fcvSetOperationMode end");

    auto search_area = box.width * box.height * pow(search_area_scale, 2);
    if(search_area > max_image_sample_size)
        current_scale_factor = (float)sqrt(search_area / max_image_sample_size);
    else if(search_area < min_image_sample_size)
        current_scale_factor = (float)sqrt(search_area / min_image_sample_size);
    else
        current_scale_factor = 1.f;

    target_size = cv::Size2f(box.width, box.height);
    base_target_size = cv::Size2f(target_size.width / current_scale_factor, target_size.height / current_scale_factor);
    // square region
    image_sample_size.width = (float)sqrt(base_target_size.width * base_target_size.height * pow(search_area_scale, 2));
    image_sample_size.height = image_sample_size.width;

#ifdef PLATFORM_CAFFE
    image_support_size.width = ceil(image_sample_size.width / 4.) * 4;
    image_support_size.height = ceil(image_sample_size.height / 4.) * 4;
#else
    image_support_size.width = 160;
    image_support_size.height = 160;
#endif

    auto num_extractors = ptr_feat_factory->ptr_extractors.size();
    for(int eid = 0; eid<num_extractors; eid++)
    {
        ptr_feat_factory->ptr_extractors[eid]->input_size = image_support_size;
    }
    //image_support_size = ptr_feat_factory->get_image_support_sz(image_sample_size, 1);

    target_pos = cv::Point2f(box.x + 0.5f * (box.width - 1), box.y + 0.5f * (box.height - 1));
    cv::Point sample_pos((int)round(target_pos.x), (int)round(target_pos.y));
    float sample_scale = current_scale_factor;

    //init scale filter
    scale_filter.init(target_size);

    Sample feats;
    extract_features(img, sample_pos, sample_scale, feats);

    // init the projection matrix for feature compression
    ptr_feat_factory->compute_project_mats(feats);

    init_params(feats);

    // do windowing of features
    if(init_mode == NORMAL_INIT)
        apply_cosin_window(feats, feats);
    else
        apply_guass_window(feats, feats);


    // compute the fouries series
    proc_batch_vec2dmat(cfft2, feats, feats);

    // interpolate features to the coutinuous domain
    interpolate_dft(feats, interp1_fs, interp2_fs, feats);

    // new sample to be added
    proc_batch_vec2dmat(compact_fourier_coef, feats, feats);

    cv::Point2f shift_samp;
    shift_samp.x = 2 * CV_PI * (target_pos.x - sample_pos.x) / (sample_scale * image_support_size.width);
    shift_samp.y = 2 * CV_PI * (target_pos.y - sample_pos.y) / (sample_scale * image_support_size.height);

    // project samples
    Vec2dMat feats_proj;
    ptr_feat_factory->compress_all(feats, feats_proj);

    // init sample model
    sample_model = SampleSpaceModel(learn_paras.sample_capacity, filter_size,
                                    ptr_feat_factory->target_channels, learn_paras.learning_rate);
    sample_model_for_train = SampleSpaceModel(learn_paras.sample_capacity, filter_size,
                                              ptr_feat_factory->target_channels, learn_paras.learning_rate);
    sample_mdl_clone = SampleSpaceModel(learn_paras.sample_capacity, filter_size,
                                        ptr_feat_factory->target_channels, learn_paras.learning_rate);
    // update samples and their weights
    sample_model.update(feats_proj, changed_ids_main_thread);

    // for preconidtioning
    compute_sample_energy(feats_proj, sample_energy);

    // initialize stuff for filter learning
    Vec1dMat proj_energy;
    compute_proj_energy(proj_energy);

    auto feat_block_num = feats_proj.size();
    if(init_mode == NORMAL_INIT)
    {
        Vec3dMat hf(feat_block_num);
        for (int i = 0; i < feat_block_num; ++i)
        {
            hf[i].resize(2);
            int cmpr_channels = ptr_feat_factory->target_channels[i];
            cv::Size hf_sz((filter_size[i].width + 1) / 2, filter_size[i].height);
            init_vec1dmat(cmpr_channels, hf_sz, CV_32FC2, hf[i][0]);
        }

        train_joint(hf, feats, sample_energy, proj_energy, hf_half);
    }
    else
    {
        hf_half.clear();
        hf_half.resize(feat_block_num);
        fast_init_hf(feats_proj, hf_half);
    }
    proc_batch_vec2dmat(full_fourier_coef, hf_half, hf_full);

    // update scale filter
    if(scale_mode == SCALE_MODE_FILTER)
        scale_filter.update(img, target_pos, base_target_size, current_scale_factor);

    // re-project and insert training sample, for project-mat has been updated
    ptr_feat_factory->compress_all(feats, feats_proj);
    sample_model.insert_sample(feats_proj, 0);
    // update the gram matrix since the sample has changed
    float new_sample_norm = 0; // find the norm of the reprojected sample
    for (int i = 0; i < feats_proj.size(); ++i)
    {
        for (int j = 0; j < feats_proj[i].size(); ++j) // channels
        {
            ComplexMat feats_proj_cmat(feats_proj[i][j]);
            auto dot_pd = feats_proj_cmat.conj().dot(feats_proj_cmat);
            new_sample_norm += 2 * dot_pd.real();
        }
    }
    sample_model.gram_mat.at<float>(0, 0) = new_sample_norm;

    frames_since_last_train = 0;
    object_status = EXIST;
}


void EcoTracker::update_template()
{
    if(learn_paras.train_mode == SYNC)
    {
        train_filter(hf_half, sample_energy, sample_model, hf_half);
        proc_batch_vec2dmat(full_fourier_coef, hf_half, hf_full);
    }
    else
    {
        Vec2dMat sample_energy_clone;
        std::unique_lock<std::mutex> sample_lock(sample_mutex);
        proc_batch_vec2dmat(lamda_copy_mat, sample_energy, sample_energy_clone);
        sample_mdl_clone.apply_changes(sample_model_for_train, changed_ids_train_thread);
        changed_ids_train_thread.clear();
        sample_lock.unlock();

        train_filter(hf_half, sample_energy_clone, sample_mdl_clone, hf_half);

        Vec2dMat filters;
        proc_batch_vec2dmat(full_fourier_coef, hf_half, filters);

        std::unique_lock<std::mutex> hf_full_lock(hf_full_mutex);
        proc_batch_vec2dmat(lamda_copy_mat, filters, hf_full);
        hf_full_lock.unlock();
    }
}


void EcoTracker::train_thread_entry()
{
    while(!is_quit_train.load())
    {
        std::unique_lock<std::mutex> do_train_lock(do_train_mutex);
        do_train_cond_var.wait(do_train_lock);

        if(!is_quit_train.load())
            update_template();
    }
}


void EcoTracker::stop_train_thread()
{
    if(ptr_train_thread != NULL)
    {
        for (int i = 0; i < train_child_threads.size(); ++i)
        {
            if(train_child_threads[i] != NULL)
            {
                is_quit_train_childs[i].store(true);
                train_child_threads[i]->join();
                delete train_child_threads[i];
            }
        }

        is_quit_train.store(true);
        do_train_cond_var.notify_one();
        ptr_train_thread->join();
        delete ptr_train_thread;
    }
}


void EcoTracker::process_feats_each(int extractor_id, const cv::Mat& img, cv::Point sample_pos,
                                   float sample_scale, Sample& feats)
{
    extract_features_each(extractor_id, img, sample_pos, sample_scale, feats);
    ptr_feat_factory->compress_each(feats, extractor_id, feats);
    auto& feat_ids = ptr_feat_factory->id_convert_table[extractor_id];
    assert(feats.size() == feat_ids.size());
    for (int i = 0; i < feat_ids.size(); ++i)
    {
        int id = feat_ids[i];
        apply_cosin_window(feats[i], id, feats[i]);
        proc_batch_vec1dmat(cfft2, feats[i], feats[i]);
        interpolate_dft(feats[i], interp1_fs, interp2_fs, feats[i], id);
    }
}


void EcoTracker::feat_thread_entry(int ind)
{
    auto& is_quit = is_quit_feat_procs[ind];
    auto& is_finished = finish_feat_proc_flags[ind];
    while(!is_quit.load())
    {
        while(is_finished.load() && !is_quit.load()) usleep(100);

        if (is_quit.load()) break;

        feat_proc_funs[ind]();
        is_finished.store(true);
    }
}


void EcoTracker::stop_feat_threads()
{
    for (int i = 0; i < feat_threads.size(); ++i)
    {
        is_quit_feat_procs[i].store(true);
        feat_threads[i]->join();
        delete feat_threads[i];
    }
}


Sample EcoTracker::process_feats(const cv::Mat& img, cv::Point sample_pos, float sample_scale)
{
    auto& ptr_extractors = ptr_feat_factory->ptr_extractors;
    auto num_extractors = ptr_extractors.size();
    Sample out_sample;

    if(num_extractors == 1)
    {
        process_feats_each(0, img, sample_pos, sample_scale, out_sample);
    }
    else
    {
        std::vector<Sample> samples(num_extractors);
        feat_proc_funs.clear();
        for (int eid = 1; eid < num_extractors; ++eid)
        {
            auto fun = std::bind(&EcoTracker::process_feats_each, this, eid, std::ref(img), sample_pos,
                                 sample_scale, std::ref(samples[eid]));

            feat_proc_funs.push_back(std::move(fun));
        }

        if(feat_threads.empty())
        {
            for (int i = 0; i < num_extractors - 1; ++i)
            {
                is_quit_feat_procs.emplace_back(false);
                finish_feat_proc_flags.emplace_back(false);

                auto child_proc = std::bind(&EcoTracker::feat_thread_entry, this, i);
                feat_threads.push_back(new std::thread(child_proc));
            }
        }
        else
        {
            for (int i = 0; i < finish_feat_proc_flags.size(); ++i)
            {
                finish_feat_proc_flags[i].store(false);
            }
        }

        int rest_eid = 0;
        process_feats_each(rest_eid, img, sample_pos, sample_scale, samples[rest_eid]);

        while(1)
        {
            bool is_finish = true;
            for (int i = 0; i < num_extractors - 1; ++i)
            {
                is_finish &= finish_feat_proc_flags[i].load();
            }
            if(is_finish)
                break;
            else
                usleep(100);
        }

        out_sample.clear();
        for (int i = 0; i < num_extractors; ++i)
        {
            out_sample.insert(out_sample.end(), samples[i].begin(), samples[i].end());
        }
    }
    return out_sample;
}


void EcoTracker::track(const cv::Mat &img)
{
    cv::Point sample_pos((int)round(target_pos.x), (int)round(target_pos.y));
    float sample_scale = current_scale_factor;

    Sample feats = process_feats(img, sample_pos, sample_scale);

    // Compute convolution for each feature block in the Fourier domain and the sum over all blocks.
    if(learn_paras.train_mode == SYNC)
        compute_response(feats, hf_full);
    else
    {
        Vec2dMat filters;
        std::unique_lock<std::mutex> hf_full_lock(hf_full_mutex);
        // deep copy
        proc_batch_vec2dmat(lamda_copy_mat, hf_full, filters);
        hf_full_lock.unlock();

        compute_response(feats, filters);
    }

    // Optimize the continuous score function
    cv::Point2f trans_vec;
    trans_vec = optimize_scores(1);
    
    // predict tracking confidence by max_score and apce
    apce_conf.compute(sampled_scores);
    is_conf_high = apce_conf.judge();

    //Compute the translation vector in pixel-coordinates and round to the closest integer pixel
    cv::Point2f pre_target_pos = target_pos;
    trans_vec.x *= (image_support_size.width / (float)feat_output_size.width) * sample_scale;
    trans_vec.y *= (image_support_size.height / (float)feat_output_size.height) * sample_scale;
    target_pos += trans_vec;

    // new target pos is out of image boundary
    if(target_pos.x < 0 || target_pos.x >= img.cols ||
            target_pos.y < 0 || target_pos.y >= img.rows)
    {
        target_pos = pre_target_pos;
    }

    if(scale_mode == SCALE_MODE_FILTER)
    {
        float scale_change_factor = scale_filter.track(img, target_pos, base_target_size, current_scale_factor);
        current_scale_factor *= scale_change_factor;
    }
    target_size = base_target_size * current_scale_factor;

    // get the left hand side of new sample
    auto get_lhs = [](const cv::Mat& src, cv::Mat& dst){ dst = src.colRange(0, (src.cols + 1) / 2).clone(); };
    proc_batch_vec2dmat(get_lhs, feats, feats);

    // shift the sample so that the target is centered
    cv::Point2f shift_samp;
    shift_samp.x = (float)(2 * CV_PI * (target_pos.x - sample_pos.x) / (sample_scale * image_support_size.width));
    shift_samp.y = (float)(2 * CV_PI * (target_pos.y - sample_pos.y) / (sample_scale * image_support_size.height));
    do_shift_samples(feats, shift_samp, feats);

    sample_model.update(feats, changed_ids_main_thread);

    if (++frames_since_last_train >= learn_paras.frames_intv_do_train)
    {
        // for preconidtioning
        Vec2dMat new_sample_energy;
        compute_sample_energy(feats, new_sample_energy);
        if(learn_paras.train_mode == SYNC)
        {
            // update the approximate average sample energy using the learning rate.
            // This is only used to construct the preconditioner.
            vec2dmat_linear_opt(sample_energy, sample_energy, new_sample_energy,
                                1.f - learn_paras.learning_rate, learn_paras.learning_rate);

            update_template();
        }
        else
        {
            std::unique_lock<std::mutex> sample_lock(sample_mutex);
            vec2dmat_linear_opt(sample_energy, sample_energy, new_sample_energy,
                                1.f - learn_paras.learning_rate, learn_paras.learning_rate);

            changed_ids_train_thread = changed_ids_main_thread;
            sample_model_for_train.apply_changes(sample_model, changed_ids_main_thread);
            changed_ids_main_thread.clear();
            sample_lock.unlock();

            if(!is_quit_train.load() && ptr_train_thread == NULL)
            {
                auto train_entry = std::bind(&EcoTracker::train_thread_entry, this);
                ptr_train_thread = new std::thread(train_entry);
            }
            do_train_cond_var.notify_one();
        }

        // train scale filter
        if(scale_mode == SCALE_MODE_FILTER)
            scale_filter.update(img, target_pos, base_target_size, current_scale_factor);

        frames_since_last_train = 0;
    }

}

void EcoTracker::correct_with_box(const cv::Mat &img, const cv::Rect& box)
{
    cv::Point2f box_center;
    box_center.x = box.x + 0.5f * box.width;
    box_center.y = box.y + 0.5f * box.height;
    modify_pos(box_center);

    cv::Size2f box_sz(box.width, box.height);
    modify_scale(box_sz, img.size());

    if(scale_mode == SCALE_MODE_FILTER)
        scale_filter.update(img, target_pos, base_target_size, current_scale_factor);

    object_status = EXIST;
}


void EcoTracker::correct_with_box(const cv::Mat &img, const cv::Rect &box, float sample_weight)
{
    cv::Point2f box_center;
    box_center.x = box.x + 0.5f * box.width;
    box_center.y = box.y + 0.5f * box.height;
    modify_pos(box_center);

    cv::Size2f box_sz(box.width, box.height);
    modify_scale(box_sz, img.size());

    cv::Point sample_pos((int)round(target_pos.x), (int)round(target_pos.y));
    Sample feats = process_feats(img, sample_pos, current_scale_factor);
    auto get_lhs = [](const cv::Mat& src, cv::Mat& dst){ dst = src.colRange(0, (src.cols + 1) / 2).clone();};
    proc_batch_vec2dmat(get_lhs, feats, feats);
    sample_model.force_correct(feats, sample_weight, changed_ids_main_thread);

    if(scale_mode == SCALE_MODE_FILTER)
        scale_filter.update(img, target_pos, base_target_size, current_scale_factor);

    object_status = EXIST;
}


cv::Rect EcoTracker::get_target_bbox()
{
    cv::Rect rect;
    rect.x = (int)floor(std::max(0.f, target_pos.x - 0.5f * target_size.width));
    rect.y = (int)floor(std::max(0.f, target_pos.y - 0.5f * target_size.height));
    rect.width = (int)floor(target_size.width);
    rect.height = (int)floor(target_size.height);
    return rect;
}

void EcoTracker::modify_pos(cv::Point2f new_pos)
{
    target_pos = new_pos;
}

void EcoTracker::modify_scale(cv::Size2f new_box_sz, cv::Size img_sz)
{
    auto search_area = new_box_sz.width * new_box_sz.height * pow(search_area_scale, 2);
    if(search_area > max_image_sample_size)
        current_scale_factor = (float)sqrt(search_area / max_image_sample_size);
    else if(search_area < min_image_sample_size)
        current_scale_factor = (float)sqrt(search_area / min_image_sample_size);
    else
        current_scale_factor = 1.f;

    target_size = new_box_sz;
    base_target_size = cv::Size2f(new_box_sz.width / current_scale_factor, new_box_sz.height / current_scale_factor);
}

ObjStatus EcoTracker::set_no_object()
{
    object_status = DISAPPEAR;
    is_conf_high = false;
    return object_status;

}

ObjStatus EcoTracker::get_object_status()
{
    return object_status;
}

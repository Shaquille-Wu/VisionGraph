//
// Created by yuan on 17-9-22.
//

#ifndef ROBOT_TRACKING_ECO_TRACKER_H
#define ROBOT_TRACKING_ECO_TRACKER_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <numeric>
#include <condition_variable>
#include "feature_factory.h"
#include "sample_space_model.h"
#include "data_type.h"
#include "scale_filter.h"
#include "apce_conf.h"
#include "utils.h"

typedef enum ScaleMode_ {
    SCALE_MODE_FIX = 0,
    SCALE_MODE_FILTER,
//    MultiScale
} ScaleMode;


typedef enum TrainMode_ {
    SYNC = 0,
    ASYNC
} TrainMode;

typedef enum InitMode_ {
    NORMAL_INIT = 0,
    FAST_INIT
} InitMode;

class RegParas {
public:
    RegParas(float reg_win_edge = 0.01f, float reg_win_power = 2, float reg_win_min = 1e-4,
             float reg_spars_thresh = 0.05f) : reg_window_edge(reg_win_edge),
                                               reg_window_power(reg_win_power),
                                               reg_window_min(reg_win_min),
                                               reg_sparsity_thresh(reg_spars_thresh) {}

    float reg_window_edge; // 0.01
    float reg_window_power; // 2
    float reg_window_min; // 1e-4
    float reg_sparsity_thresh; // 0.05
};


class CGParas {
public:
    CGParas(float lrt = 0.009f, int train_cg_iter = 5, float train_init_cg_eps = 5e-4,
            int train_init_cg_iter = 150, int train_init_gn_iter = 10, int train_forget_rate = 75,
            float precond_data_coef = 0.3f, float precond_reg_coef = 0.015f,
            float precond_proj_coef = 35, float proj_reg = 5e-8) :
            cg_iter(train_cg_iter), init_cg_iter(train_init_cg_iter),
            init_cg_eps(train_init_cg_eps), init_gn_iter(train_init_gn_iter),
            forgetting_rate(train_forget_rate),
            precond_data_param(precond_data_coef), precond_reg_param(precond_reg_coef),
            precond_proj_param(precond_proj_coef), projection_reg(proj_reg) {
        compute_init_forget_factor(lrt);
    }

    void compute_init_forget_factor(float lrt) {
        init_forget_factor = pow(1.f - lrt, forgetting_rate);
    }

    int cg_iter; // 5
    int init_cg_iter; // 150
    float init_cg_eps; // 5e-4
    int init_gn_iter; // 10
    int forgetting_rate; // 75
    float init_forget_factor;
    float precond_data_param; // 0.3
    float precond_reg_param; // 0.015
    float precond_proj_param; // 35
    float projection_reg; // 5e-8
};


class LearningParas {
public:
    LearningParas(int nsamples = 50, TrainMode update_mode = ASYNC, int train_intv = 6,
                  float sigma = 1 / 12.f, float lrt = 0.009f) :
            sample_capacity(nsamples), train_mode(update_mode), frames_intv_do_train(train_intv),
            output_sigma_factor(sigma), learning_rate(lrt) {}

    int sample_capacity;
    TrainMode train_mode;
    int frames_intv_do_train;
    float output_sigma_factor;
    float learning_rate;
};


typedef struct CGState_ {
    Vec1dMat p;
    float rho;
    Vec1dMat r_prev;
} CGState;

enum ObjStatus {
    EXIST = 0, DISAPPEAR
};

class EcoTracker {
public:
    EcoTracker();

    EcoTracker(FeatureFactory *ptr_feat_handler, LearningParas lr_paras = LearningParas(),
               ScaleMode scale_pred_mode = SCALE_MODE_FILTER, InitMode init_pred_mode = NORMAL_INIT,
               bool use_conf = false, float area_scale = 4.5f, int min_img_sample_sz = 160 * 160,
               int max_img_sample_sz = 160 * 160, ApceConfidence apce = ApceConfidence(),
               float interp_coef = -0.75f,
               RegParas reg_filter_paras = RegParas(), CGParas train_cg_paras = CGParas());

    virtual ~EcoTracker();

    void init_tracker_pos(const cv::Mat &img, const cv::Rect &box);

    void track(const cv::Mat &img);

    void correct_with_box(const cv::Mat &img, const cv::Rect &box, float sample_weight);

    void correct_with_box(const cv::Mat &img, const cv::Rect &box);

    cv::Rect get_target_bbox();

    void modify_pos(cv::Point2f new_pos);

    void modify_scale(cv::Size2f new_box_sz, cv::Size img_sz);

    ObjStatus set_no_object();

    ObjStatus get_object_status();

private:

    void init_params(const Sample &sample);

    void compute_cosin_window(int feat_height, int width, cv::Mat &cos_win);

    void compute_guass_window(int feat_heigth, int width, cv::Mat &cos_win);

    void apply_cosin_window(const Vec2dMat &in_feat_mats, Vec2dMat &out_feat_mats);

    void apply_guass_window(const Vec2dMat &in_feat_mats, Vec2dMat &out_feat_mats);

    void apply_cosin_window(const Vec1dMat &in_feat_mat, int feat_ind, Vec1dMat &out_feat_mat);

    void get_interp_fourier(int len, cv::Mat &interp_mat, bool is_col_vec = true);

    void get_reg_filter(cv::Mat &sp_reg_filter);

    void
    sample_patch(const cv::Mat &img, cv::Point center_pos, cv::Size sample_sz, cv::Size output_size,
                 cv::Mat &img_patch);

    void
    extract_features(const cv::Mat &img, cv::Point center_pos, float sample_scale, Sample &feats);

    void extract_features_each(int extractor_id, const cv::Mat &img, cv::Point center_pos,
                               float sample_scale, Sample &feats);

    void compute_response(const Vec2dMat &feat_mats, const Vec2dMat &filters);

    cv::Point2f optimize_scores(int iter_cnt);

    void do_shift_samples(const Vec2dMat &in_mats, cv::Point2f shift_vec, Vec2dMat &out_mats);

    void compute_sample_energy(const Vec2dMat &sample, Vec2dMat &energy);

    void compute_proj_energy(Vec1dMat &proj_energy);

    void train_joint_each(int feat_ind, Vec2dMat &hf, const Vec1dMat &init_samplef,
                          const Vec1dMat &sample_energy,
                          const cv::Mat &proj_energy, Vec1dMat &hf_out);

    void train_joint(Vec3dMat &hf, const Vec2dMat &init_samplef, const Vec2dMat &sample_energy,
                     const Vec1dMat &proj_energy, Vec2dMat &hf_out);

    void train_filter_each(int feat_ind, const Vec1dMat &hf, const SampleSpaceModel &sample_mdl,
                           Vec1dMat &hf_out);

    void train_filter(const Vec2dMat &hf, const Vec2dMat &sample_energy,
                      const SampleSpaceModel &sample_mdl, Vec2dMat &hf_out);

    void lhs_operation_joint_each(int feat_ind, const Vec2dMat &hf, const Vec1dMat &samplesf,
                                  const Vec1dMat &init_samplef, const cv::Mat &init_sample_H,
                                  const Vec1dMat &init_hf,
                                  Vec2dMat &hf_out);

    void lhs_operation_each(int feat_ind, const Vec1dMat &hf, const SampleSpaceModel &sample_mdl,
                            Vec1dMat &hf_out);

    void fast_init_hf(const Sample &feats_proj, Vec2dMat &hf);

    float inner_product_joint(const Vec2dMat &xf, const Vec2dMat &yf);

    float inner_product_filter(const Vec1dMat &xf, const Vec1dMat &yf);

    void diag_precond_joint(const Vec2dMat &hf, const Vec2dMat &M_diag, Vec2dMat &hf_out);

    void diag_precond_filter(const Vec1dMat &hf, const Vec1dMat &M_diag, Vec1dMat &hf_out);

    float
    pcg_ccot_joint_each(std::function<void(const Vec2dMat &, Vec2dMat &)> A, const Vec2dMat &b,
                        int max_iter,
                        std::function<void(const Vec2dMat &, Vec2dMat &)> M,
                        std::function<float(const Vec2dMat &, const Vec2dMat &)> ip,
                        const Vec2dMat &x0, Vec2dMat &x_out);

    void pcg_ccot_each(int feat_ind, std::function<void(const Vec1dMat &, Vec1dMat &)> A,
                       const Vec1dMat &b, int max_iter,
                       std::function<void(const Vec1dMat &, Vec1dMat &)> M,
                       std::function<float(const Vec1dMat &, const Vec1dMat &)> ip,
                       const Vec1dMat &x0, Vec1dMat &x_out);

    void update_template();

    void train_thread_entry();

    void train_child_thread_entry(int ind);

    void stop_train_thread();

    void process_feats_each(int extractor_id, const cv::Mat &img, cv::Point sample_pos,
                            float sample_scale, Sample &feats);

    void feat_thread_entry(int ind);

    Sample process_feats(const cv::Mat &img, cv::Point sample_pos, float sample_scale);

    void stop_feat_threads();

public:

    FeatureFactory *ptr_feat_factory;

    Vec1dMat cos_window;
    Vec1dMat guass_window;
    std::vector<cv::Size> filter_size;
    int max_filter_key;
    std::vector<cv::Size> feat_pad_size;
    cv::Size feat_output_size;
    // fourier series indices and their transposes
    Vec1dMat fouri_ky;
    Vec1dMat fouri_kx;
    // construct the Gaussian label function using Poisson formula
    Vec1dMat guass_yf;

    cv::Point2f target_pos;
    cv::Size2f base_target_size;
    cv::Size2f target_size;
    cv::Size2f image_support_size;
    cv::Size2f image_sample_size;

    ScaleMode scale_mode;
    float current_scale_factor;
    float search_area_scale; // 4.5
    float min_image_sample_size; // 200^2
    float max_image_sample_size; // 250^2
    ScaleFilter scale_filter;
    ScaleFilter scale_filter_for_update;

    // Fourier series of interpolation function
    Vec1dMat interp1_fs;
    Vec1dMat interp2_fs;
    float interp_bicubic_a; // -0.75

    // spatial regularization filter
    Vec1dMat reg_filter;
    std::vector<float> reg_energy;
    RegParas reg_paras;

    // tracking template
    Vec2dMat hf_half;
    Vec2dMat hf_full;
    Vec2dMat sample_energy;
    cv::Mat scores_fs_sum;
    cv::Mat sampled_scores;
    Vec1dMat scores_fs_feat;
    float max_score;

    // learning parameters
    LearningParas learn_paras;
    int frames_since_last_train;
    // asynchronous template update in an independent thread
    std::thread *ptr_train_thread;
    std::mutex hf_full_mutex;
    std::mutex sample_mutex;
    std::mutex scale_mutex1;
    std::mutex scale_mutex2;
    std::atomic<bool> is_quit_train{false};
    std::mutex do_train_mutex;
    std::condition_variable do_train_cond_var;

    std::vector<std::thread *> train_child_threads;
    std::vector<std::function<void()>> train_proc_funs;
    std::deque<std::atomic<bool>> is_quit_train_childs;
    std::deque<std::atomic<bool>> finish_train_child_flags;

    std::vector<std::thread *> feat_threads;
    std::vector<std::function<void()>> feat_proc_funs;
    std::deque<std::atomic<bool>> is_quit_feat_procs;
    std::deque<std::atomic<bool>> finish_feat_proc_flags;

    // confidence
    bool is_conf_high; // true
    bool use_confidence;
    ApceConfidence apce_conf;

    // sample space model
    SampleSpaceModel sample_model;
    SampleSpaceModel sample_model_for_train;
    SampleSpaceModel sample_mdl_clone;
    std::vector<int> changed_ids_main_thread;
    std::vector<int> changed_ids_train_thread;

    // conjugate gradient parameters
    CGParas cg_paras;
    std::vector<CGState> cg_states;

    ObjStatus object_status;

    InitMode init_mode;
};

#endif //ROBOT_TRACKING_ECO_TRACKER_H

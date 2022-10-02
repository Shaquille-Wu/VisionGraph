//
//
//
#include "face_ecotrack_strategy.h"

#include <box.h>
#include <logging.h>

#include "face_alignment.h"
#include "reid_utils.h"


FaceEcoTrackStrategy::FaceEcoTrackStrategy() {
    ptr_feat_factory = NULL;
    ptr_fp_detector = NULL;
    ptr_face_verifier = NULL;
    ptr_face_keypoints = NULL;
    ptr_eco_tracker = NULL;

    face_thresh = 0.8f;
    face_reliable_thresh = 0.7f;

    tracker_input_scale = 1.f;
    reid_interval = 10;
    reid_retry_thresh = 5;
    reid_retry_count = 0;
    frame_count = 0;
    frames_since_reid = 0;
    face_pose_checked = false;

    face_status = TRACKING_FAILED;
    target_status = TRACKING_FAILED;
    target_pos = cv::Rect(-1, -1, -1, -1);
}

FaceEcoTrackStrategy::FaceEcoTrackStrategy(FeatureFactory *feat_factory, BaseDetector *fp_detector,
                                           BaseVerifier *face_verifier,
                                           BaseKeypoints *face_keypoints,
                                           float track_input_sz_scale, float face_reg_thresh,
                                           float face_conf_thresh,
                                           int reid_frame_intv, int retry_thresh) {
    ptr_feat_factory = feat_factory;
    ptr_fp_detector = fp_detector;
    ptr_face_verifier = face_verifier;
    ptr_face_keypoints = face_keypoints;

    create_tracker();

    face_thresh = face_reg_thresh;
    face_reliable_thresh = face_conf_thresh;

    tracker_input_scale = track_input_sz_scale;
    reid_interval = reid_frame_intv;
    reid_retry_thresh = retry_thresh;
    reid_retry_count = 0;
    frame_count = 0;
    frames_since_reid = 0;
    face_pose_checked = false;

    face_status = TRACKING_FAILED;
    target_status = TRACKING_FAILED;
    target_pos = cv::Rect(-1, -1, -1, -1);
}

FaceEcoTrackStrategy::~FaceEcoTrackStrategy() {
    destroy_tracker();
}

void FaceEcoTrackStrategy::create_tracker() {
    LearningParas lr_paras(20);
    ApceConfidence apce(100, 0.3f, 0.1f);
    ptr_eco_tracker = new EcoTracker(ptr_feat_factory, lr_paras, SCALE_MODE_FIX, FAST_INIT, false, 4.5f, 160 * 160, 160 * 160, apce);
}

void FaceEcoTrackStrategy::init_tracker(const cv::Mat &img, cv::Rect rect) {
    if (ptr_eco_tracker == NULL) {
        create_tracker();
    }

    cv::Mat tracker_input = get_tracker_input(img);
    cv::Rect scaled_rect = scale_rect(rect, tracker_input_scale);
    ptr_eco_tracker->init_tracker_pos(tracker_input, scaled_rect);
}

void FaceEcoTrackStrategy::destroy_tracker() {
    if (ptr_eco_tracker != NULL) {
        delete ptr_eco_tracker;
        ptr_eco_tracker = NULL;
    }
}

int FaceEcoTrackStrategy::init(const cv::Mat &img, cv::Rect *face_rect, cv::Rect *body_rect) {
    assert(face_rect != NULL);

    std::vector<cv::Point2f> kpts;
    RPYPose pose;
    cv::Mat face_img = img(*face_rect);
    ptr_face_keypoints->compute_keypoints_pose(face_img, kpts, pose);
    for (int i = 0; i < 106; ++i) {
        kpts[i].x = kpts[i].x + face_rect->x;
        kpts[i].y = kpts[i].y + face_rect->y;
    }
    float max_x = kpts[0].x, max_y = kpts[0].y, min_x = kpts[0].x, min_y = kpts[0].y;
    for (int i = 1; i < 106; i++) {
        if (min_x > kpts[i].x) {
            min_x = kpts[i].x;
        }
        if (max_x < kpts[i].x) {
            max_x = kpts[i].x;
        }
        if (min_y > kpts[i].y) {
            min_y = kpts[i].y;
        }
        if (max_y < kpts[i].y) {
            max_y = kpts[i].y;
        }
    }
    min_x = std::max(0.f, min_x);
    min_y = std::max(0.f, min_y);
    max_x = std::min((float)img.cols, max_x);
    max_y = std::min((float)img.rows, max_y);

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    face_templates.resize(1);
    cv::Mat compact_face = img(rect);

    // cv::Mat compact_face;
    // int suc = face_align_compact(img, kpts, *face_rect, compact_face);

    ptr_face_verifier->compute_feature(compact_face, face_templates[0]);

    init_tracker(img, *face_rect);

    target_status = TRACKING_SUCCESS;
    target_pos = *face_rect;
    return 1;
}

void FaceEcoTrackStrategy::reset() {
    destroy_tracker();

    face_templates.clear();
    det_face_bboxes.clear();
    kpts_vecs.clear();
    pose_vec.clear();

    face_status = TRACKING_FAILED;
    target_status = TRACKING_FAILED;
    target_pos = cv::Rect(-1, -1, -1, -1);
    reid_retry_count = 0;
    frame_count = 0;
    frames_since_reid = 0;
    face_pose_checked = false;
}

cv::Mat FaceEcoTrackStrategy::get_tracker_input(const cv::Mat &img) {
    cv::Mat tracker_input;
    if (tracker_input_scale == 1.f)
        tracker_input = img;
    else
        cv::resize(img, tracker_input, cv::Size(), tracker_input_scale, tracker_input_scale);
    return tracker_input;
}

cv::Rect FaceEcoTrackStrategy::get_tracker_rect() {
    cv::Rect eco_rect = ptr_eco_tracker->get_target_bbox();
    eco_rect = scale_rect(eco_rect, 1 / tracker_input_scale);
    return eco_rect;
}

FaceDirection FaceEcoTrackStrategy::check_face_direction(const std::vector<cv::Point2f> &kpts,
                                                         const RPYPose &pose) {
    FaceDirection face_type = FRONT_FACE;
    // strong rule
    if (fabs(pose.yaw) >= 25 || fabs(pose.pitch) >= 10 || fabs(pose.roll) >= 25)
        face_type = SIDE_FACE;

    return face_type;
}

static bool check_is_sideface(const std::vector<cv::Point2f> &kpt_vec) {
    int x_center_ind = 43;
    int y_center_ind = 49;
    int x1_ind = 52;
    int x2_ind = 61;
    int y1_ind = 43;
    int y2_ind = 16;

    float dis_x1 = std::fabs(kpt_vec[x_center_ind].x - kpt_vec[x1_ind].x) + 0.00000001;
    float dis_x2 = std::fabs(kpt_vec[x_center_ind].x - kpt_vec[x2_ind].x) + 0.00000001;
    float dis_y1 = std::fabs(kpt_vec[y_center_ind].y - kpt_vec[y1_ind].y) + 0.00000001;
    float dis_y2 = std::fabs(kpt_vec[y_center_ind].y - kpt_vec[y2_ind].y) + 0.00000001;

    float x_thresh = 1.7;
    float y_thresh = 2.0;

    float x_result = std::max(dis_x1, dis_x2) / std::min(dis_x1, dis_x2);
    float y_result = std::max(dis_y1, dis_y2) / std::min(dis_y1, dis_y2);

    if (x_result > x_thresh || y_result > y_thresh) {
        return true;
    }
    return false;
}

FaceDirection FaceEcoTrackStrategy::check_sideface_extra(const cv::Mat &face_img,
                                                         const std::vector<cv::Point2f> &kpts) {
    FaceDirection face_direction = SIDE_FACE;
    float sideface_score = 0.0f;
    Histogram hist;
    bool lightEnough = hist.isLightEnough(face_img);
    if (!lightEnough)
        face_direction = FRONT_FACE;
    else {
        do {
            bool is_sideface = check_is_sideface(kpts);
            if (is_sideface)
                break;
            cv::Mat face_img_gray;
            cv::cvtColor(face_img, face_img_gray, cv::COLOR_BGR2GRAY);
            sideface_score = extra_sideface_cls.compute_similarity(face_img_gray.data,
                                                                   face_img_gray.cols,
                                                                   face_img_gray.rows, kpts);
            if (sideface_score < 0.51f)
                break;
            face_direction = FRONT_FACE;
        } while (false);
    }

    return face_direction;
}
// ****************************keypoint********************************************
void FaceEcoTrackStrategy::compute_detface_keypoints(const cv::Mat &img) {
    size_t num = det_face_bboxes.size();
    kpts_vecs.clear();
    pose_vec.clear();
    kpts_vecs.resize(num);
    pose_vec.resize(num);

    for (int i = 0; i < num; ++i) {
        cv::Rect face_rect = bbox2rect(det_face_bboxes[i]);
        if (face_rect.y + face_rect.height > img.rows)
            //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, (int)(handBox.y2 - bgr.rows) + 1, 0, 0, cv::BORDER_CONSTANT);
            face_rect.height = img.rows - face_rect.y - 1;
        if (face_rect.x + face_rect.width > img.cols)
            //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, 0, 0, (int)(handBox.x2 - bgr.cols) + 1, cv::BORDER_CONSTANT);
            face_rect.width = img.cols - face_rect.x - 1;
        if (face_rect.x < 0) {
            face_rect.x = 0;
        }

        if (face_rect.y < 0) {
            face_rect.y = 0;
        }
        cv::Mat face_img = img(face_rect);
        ptr_face_keypoints->compute_keypoints_pose(face_img, kpts_vecs[i], pose_vec[i]);
    }
}
// ****************************keypoint********************************************


// ****************************detect********************************************
void FaceEcoTrackStrategy::face_det_reid(const cv::Mat &img) {
    std::vector<BBox> fp_bboxes = ptr_fp_detector->detect(img);
    // face_boxs.clear();
    // for (BBox &face : fp_bboxes) {
    //     vision::Box b;
    //     b.x1 = face.x1;
    //     b.x2 = face.x2;
    //     b.y1 = face.y1;
    //     b.y2 = face.y2;
    //     face_boxs.push_back(b);
    // }

    det_face_bboxes = ptr_fp_detector->combine_bboxes_p2f(fp_bboxes);

    compute_detface_keypoints(img);

    int reid_id = do_reid_face_compact(img, ptr_face_verifier, det_face_bboxes, kpts_vecs,
                                       face_templates, face_thresh,
                                       REID_BEST_SCORE, &face_bbox, &face_status);

    // LOGE("wyb debug reid_id %d det_face_bboxes.size() %d", reid_id, det_face_bboxes.size());
    // assert(reid_id < det_face_bboxes.size());
    if (reid_id >= 0) {
        cv::Rect possible_rect = bbox2rect(det_face_bboxes[reid_id]);

        if (possible_rect.y + possible_rect.height > img.rows)
            //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, (int)(handBox.y2 - bgr.rows) + 1, 0, 0, cv::BORDER_CONSTANT);
            possible_rect.height = img.rows - possible_rect.y - 1;
        if (possible_rect.x + possible_rect.width > img.cols)
            //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, 0, 0, (int)(handBox.x2 - bgr.cols) + 1, cv::BORDER_CONSTANT);
            possible_rect.width = img.cols - possible_rect.x - 1;
        if (possible_rect.x < 0) {
            possible_rect.x = 0;
        }

        if (possible_rect.y < 0) {
            possible_rect.y = 0;
        }
        cv::Mat face_img = img(possible_rect);
        const auto &kpts = kpts_vecs[reid_id];
        if (check_sideface_extra(face_img, kpts) == SIDE_FACE) {
            set_facepose(90, 90, 90);
        } else {
            set_facepose(pose_vec[reid_id]);
        }
    }

    frames_since_reid = 0;
}
// ****************************detect********************************************


void FaceEcoTrackStrategy::set_facepose(const RPYPose &pose) {
    face_pose = pose;
    face_pose_checked = true;
}

void FaceEcoTrackStrategy::set_facepose(float roll, float pitch, float yaw) {
    face_pose.roll = roll;
    face_pose.pitch = pitch;
    face_pose.yaw = yaw;
    face_pose_checked = true;
}

RPYPose FaceEcoTrackStrategy::get_facepose() {
    if (target_status == TRACKING_SUCCESS && face_pose_checked) {
        return face_pose;
    } else {
        RPYPose pose;
        pose.roll = 180;
        pose.pitch = 180;
        pose.yaw = 180;
        return pose;
    }
}

void FaceEcoTrackStrategy::combine_eco_with_rule(const cv::Mat &img, cv::Rect track_rect) {
    bool has_possible_bbox = false;
    cv::Rect possible_rect;
    int possible_ind = -1;
    FaceDirection face_direction = SIDE_FACE;
    bool extra_checked = false;

    BBox track_bbox = rect2bbox(track_rect);
    std::pair<int, float> overlap_bbox = find_max_iou(det_face_bboxes, track_bbox);
    if (overlap_bbox.second >= 0.5f) {
        possible_ind = overlap_bbox.first;
        BBox possible_bbox = det_face_bboxes[possible_ind];
        possible_rect = bbox2rect(possible_bbox);

        const auto &kpts = kpts_vecs[possible_ind];
        const auto &pose = pose_vec[possible_ind];

        if (possible_rect.y + possible_rect.height > img.rows)
            //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, (int)(handBox.y2 - bgr.rows) + 1, 0, 0, cv::BORDER_CONSTANT);
            possible_rect.height = img.rows - possible_rect.y - 1;
        if (possible_rect.x + possible_rect.width > img.cols)
            //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, 0, 0, (int)(handBox.x2 - bgr.cols) + 1, cv::BORDER_CONSTANT);
            possible_rect.width = img.cols - possible_rect.x - 1;
        if (possible_rect.x < 0) {
            possible_rect.x = 0;
        }

        if (possible_rect.y < 0) {
            possible_rect.y = 0;
        }
        cv::Mat face_img = img(possible_rect);
        if (check_sideface_extra(face_img, kpts) == SIDE_FACE) {
            face_direction = SIDE_FACE;
            extra_checked = true;
        } else {
            face_direction = check_face_direction(kpts, pose);
        }

        bool not_front_face = (face_direction == SIDE_FACE);

        BBox prev_target_bbox = rect2bbox(target_pos);
        float motion_dist = bbox_dist(possible_bbox, prev_target_bbox);
        float side_len = (float)sqrt(possible_rect.width * possible_rect.height);
        bool move_fast = motion_dist > 0.4f * side_len;

        // add limit for unexpected long-distance motion
        bool dist_normal = motion_dist <= 0.7 * side_len;

        // add face pose info to validate possible bbox:
        // discard it if face in bbox is front-face or motion-blurred
        // (which should be recognized by face verifier)
        if (not_front_face && move_fast && dist_normal) {
            has_possible_bbox = true;
        }
    }

    if (has_possible_bbox) {
        //if a side or blur face is detected at the location of eco tracking result
        //think eco track as reliable
        cv::Mat tracker_input = get_tracker_input(img);
        cv::Rect scaled_rect = scale_rect(possible_rect, tracker_input_scale);
        ptr_eco_tracker->correct_with_box(tracker_input, scaled_rect);

        //        target_pos = get_tracker_rect();
        target_pos = possible_rect;
        target_status = TRACKING_SUCCESS;
        tracking_action = ACTION_TRACK;

        if (face_direction == SIDE_FACE && extra_checked) {
            set_facepose(90, 90, 90);
        } else {
            set_facepose(pose_vec[possible_ind]);
        }
    } else {
        //else think eco track is wrong
        ptr_eco_tracker->set_no_object();
        target_status = TRACKING_FAILED;
        tracking_action = ACTION_NONE;
    }
}

void FaceEcoTrackStrategy::combine_reid_with(const cv::Mat &img, ObjStatus eco_status,
                                             bool eco_conf_high) {

    face_det_reid(img);

    bool reid_success =
        (face_status == TRACKING_SUCCESS) && (face_bbox.reid_conf < face_reliable_thresh);

    if (reid_success) {
        cv::Rect reid_face_rect = bbox2rect(face_bbox);
        cv::Mat tracker_input = get_tracker_input(img);
        cv::Rect scaled_rect = scale_rect(reid_face_rect, tracker_input_scale);
        ptr_eco_tracker->correct_with_box(tracker_input, scaled_rect, 0.2f);

        target_pos = reid_face_rect;
        target_status = TRACKING_SUCCESS;
        tracking_action = ACTION_REID;

    } else {
        if (eco_status != EXIST) {
            // eco is not tracking an object and reid failed,
            // set no object
            ptr_eco_tracker->set_no_object();
            target_status = TRACKING_FAILED;
            tracking_action = ACTION_NONE;
        } else if (eco_conf_high) {
            // reid failed but eco is tracking with high confidence, do track
            cv::Mat tracker_input = get_tracker_input(img);
            ptr_eco_tracker->track(tracker_input);
            BBox prev_target_bbox = rect2bbox(target_pos);
            cv::Rect tracker_rect = get_tracker_rect();
            BBox tracker_bbox = rect2bbox(tracker_rect);

            bool abnormal = false;
            std::pair<int, float> overlap_bbox = find_max_iou(det_face_bboxes, tracker_bbox);
            if (overlap_bbox.second > 0.5f) {
                float motion_dist = bbox_dist(tracker_bbox, prev_target_bbox);
                cv::Rect prev_target_rect = target_pos;
                float side_len = (float)sqrt(prev_target_rect.area());
                if (motion_dist > 0.4f * side_len) {
                    abnormal = true;
                }
            } else
                abnormal = true;

            if (abnormal) {
                target_status = TRACKING_FAILED;
                tracking_action = ACTION_NONE;
                ptr_eco_tracker->set_no_object();
            } else {
                int possible_ind = overlap_bbox.first;
                cv::Rect possible_rect = bbox2rect(det_face_bboxes[possible_ind]);
                if (possible_rect.y + possible_rect.height > img.rows)
                    //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, (int)(handBox.y2 - bgr.rows) + 1, 0, 0, cv::BORDER_CONSTANT);
                    possible_rect.height = img.rows - possible_rect.y - 1;
                if (possible_rect.x + possible_rect.width > img.cols)
                    //            cv::copyMakeBorder(bgr_copy, bgr_copy, 0, 0, 0, (int)(handBox.x2 - bgr.cols) + 1, cv::BORDER_CONSTANT);
                    possible_rect.width = img.cols - possible_rect.x - 1;
                if (possible_rect.x < 0) {
                    possible_rect.x = 0;
                }

                if (possible_rect.y < 0) {
                    possible_rect.y = 0;
                }
                cv::Mat face_img = img(possible_rect);
                const auto &kpts = kpts_vecs[possible_ind];
                if (check_sideface_extra(face_img, kpts) == SIDE_FACE) {
                    set_facepose(90, 90, 90);
                } else {
                    set_facepose(pose_vec[possible_ind]);
                }

                //target_pos = tracker_rect;
                target_pos = possible_rect;
                target_status = TRACKING_SUCCESS;
                tracking_action = ACTION_TRACK;
            }
        } else {
            // reid failed and eco is tracking with low confidence
            // why face reid failed:
            // case 1: disappear or sheltered
            // case 2: side-face (need to be tracked)
            // case 3: blurred, caused by fast motion or imagery (need to be tracked)
            // case 4: far-away, small face

            // if not failed at previous frame
            if (target_status != TRACKING_FAILED) {
                cv::Mat tracker_input = get_tracker_input(img);
                ptr_eco_tracker->track(tracker_input);
                cv::Rect track_rect = get_tracker_rect();

                combine_eco_with_rule(img, track_rect);
            } else {
                ptr_eco_tracker->set_no_object();
                target_status = TRACKING_FAILED;
                tracking_action = ACTION_NONE;
            }
        }
    }
}

void FaceEcoTrackStrategy::reid_check(const cv::Mat &img, cv::Rect track_rect) {
    face_det_reid(img);
    bool reid_success =
        (face_status == TRACKING_SUCCESS) && (face_bbox.reid_conf < face_reliable_thresh);

    if (reid_success) {
        // eco has done before this operation
        cv::Rect tracker_rect = get_tracker_rect();
        BBox tracker_bbox = rect2bbox(tracker_rect);

        cv::Rect reid_face_rect = bbox2rect(face_bbox);
        cv::Mat tracker_input = get_tracker_input(img);
        cv::Rect scaled_rect = scale_rect(reid_face_rect, tracker_input_scale);

        float iou = bbox_iou(tracker_bbox, face_bbox);
        if (iou > 0.5f) {
            // correct without repeatly update sample space for faster speed
            ptr_eco_tracker->correct_with_box(tracker_input, scaled_rect);
        } else {
            ptr_eco_tracker->correct_with_box(tracker_input, scaled_rect, 0.2f);
        }

        target_pos = reid_face_rect;
        target_status = TRACKING_SUCCESS;
        tracking_action = ACTION_REID;
    } else {
        combine_eco_with_rule(img, track_rect);
    }
}

void FaceEcoTrackStrategy::track(const cv::Mat &img) {
    frame_count++;
    face_pose_checked = false;


    ObjStatus prev_eco_status = ptr_eco_tracker->get_object_status();
    bool prev_eco_conf_high = ptr_eco_tracker->is_conf_high;

    if ((frames_since_reid + 1 >= reid_interval || prev_eco_status != EXIST)) {
        combine_reid_with(img, prev_eco_status, prev_eco_conf_high);

    } else {
        //do track

        cv::Mat tracker_input = get_tracker_input(img);
        ptr_eco_tracker->track(tracker_input);
        bool is_eco_conf_high = ptr_eco_tracker->is_conf_high;
        cv::Rect tracker_rect = get_tracker_rect();

        BBox tracker_bbox = rect2bbox(tracker_rect);
        cv::Rect prev_target_rect = target_pos;
        BBox prev_target_bbox = rect2bbox(prev_target_rect);

        bool abnormal = false;
        if (target_status == TRACKING_SUCCESS) {
            float motion_dist = bbox_dist(tracker_bbox, prev_target_bbox);
            float side_len = (float)sqrt(prev_target_rect.area());
            if (motion_dist > 0.4f * side_len) {
                abnormal = true;
            }
        }

        if (!abnormal) {
            target_pos = tracker_rect;
            target_status = TRACKING_SUCCESS;
            tracking_action = ACTION_TRACK;
        }

        det_face_bboxes.clear();
        kpts_vecs.clear();
        pose_vec.clear();

        frames_since_reid++;

        if (!is_eco_conf_high || abnormal) {
            //            LOGE("is_eco_conf_high == false");
            reid_check(img, tracker_rect);
        }

    }
}

//////////////////////////////////////////////////////
//created by anwenju 20180206
/////////////////////////////////////////////////////
#include "lead_strategy.h"
#include "reid_utils.h"
#include "face_alignment.h"

LeadStrategy::LeadStrategy() {
    ptr_feat_factory = NULL;
    ptr_fp_detector = NULL;
    ptr_face_verifier = NULL;
    ptr_body_verifier = NULL;
    ptr_face_keypointer = NULL;
    ptr_eco_tracker = NULL;

    face_status = TRACKING_FAILED;
    body_status = TRACKING_FAILED;

    face_thresh = 0.6f;
    face_reliable_thresh = 0.6f;
    body_thresh = 0.9f;
    body_reliable_thresh = 0.9f;

    tracker_input_scale = 1.f;

    frame_count = 0;
}

LeadStrategy::LeadStrategy(FeatureFactory *feat_factory, BaseDetector *fp_detector,
                           BaseVerifier *face_verifier,
                           BaseVerifier *body_verifier, BaseKeypoints *face_keypointer,
                           float tracker_input_sz_scale,
                           float face_reg_thresh, float face_conf_thresh, float body_reg_thresh,
                           float body_conf_thresh,
                           int reid_frame_intv) {
    ptr_feat_factory = feat_factory;
    ptr_eco_tracker = new EcoTracker(ptr_feat_factory);

    ptr_fp_detector = fp_detector;
    ptr_face_verifier = face_verifier;
    ptr_body_verifier = body_verifier;
    ptr_face_keypointer = face_keypointer;

    face_status = TRACKING_FAILED;
    body_status = TRACKING_FAILED;

    face_thresh = face_reg_thresh;
    face_reliable_thresh = face_conf_thresh;
    body_thresh = body_reg_thresh;
    body_reliable_thresh = body_conf_thresh;

    tracker_input_scale = tracker_input_sz_scale;

    frame_count = 0;
    reid_interval = reid_frame_intv;
}

LeadStrategy::~LeadStrategy() {
    destroy_tracker();
}

void LeadStrategy::init_tracker(const cv::Mat &img, cv::Rect rect) {
    if (ptr_eco_tracker == NULL) {
        ptr_eco_tracker = new EcoTracker(ptr_feat_factory);
    }
    cv::Mat tracker_input = get_tracker_input(img);
    cv::Rect scaled_rect = scale_rect(rect, tracker_input_scale);
    ptr_eco_tracker->init_tracker_pos(tracker_input, scaled_rect);
}


void LeadStrategy::destroy_tracker() {
    if (ptr_eco_tracker != NULL) {
        delete ptr_eco_tracker;
        ptr_eco_tracker = NULL;
    }
}

int LeadStrategy::init(const cv::Mat &img, cv::Rect *face_rect, cv::Rect *body_rect) {
    if (face_rect != NULL) {
        cv::Mat face_img = img(*face_rect);
        std::vector<cv::Point2f> kpts;
        ptr_face_keypointer->compute_keypoints(face_img, kpts);
        cv::Mat compact_face;
        face_align_compact(img, kpts, *face_rect, compact_face);

        std::vector<float> face_feat;
        ptr_face_verifier->compute_feature(compact_face, face_feat);
        face_templates.push_back(face_feat);
    }

    if (body_rect != NULL) {
        init_tracker(img, *body_rect);

        cv::Mat body_img = img(*body_rect);

        std::vector<float> body_feat;
        ptr_body_verifier->compute_feature((cv::Mat &) body_img, body_feat);
        body_templates.push_back(body_feat);
    }
    return 0;
}


void LeadStrategy::reset() {
    destroy_tracker();

    face_templates.clear();
    body_templates.clear();

    face_status = TRACKING_FAILED;
    body_status = TRACKING_FAILED;
    target_status = TRACKING_FAILED;
    target_pos = cv::Rect(-1, -1, -1, -1);

    frame_count = 0;

    det_face_bboxes.clear();
    det_body_bboxes.clear();
}


cv::Mat LeadStrategy::get_tracker_input(const cv::Mat &img) {
    cv::Mat tracker_input;
    if (tracker_input_scale == 1.f)
        tracker_input = img;
    else
        cv::resize(img, tracker_input, cv::Size(), tracker_input_scale, tracker_input_scale);
    return tracker_input;
}

cv::Rect LeadStrategy::get_tracker_rect() {
    cv::Rect eco_rect = ptr_eco_tracker->get_target_bbox();
    eco_rect = scale_rect(eco_rect, 1 / tracker_input_scale);
    return eco_rect;
}

void LeadStrategy::track(const cv::Mat &img) {
    frame_count++;
    ObjStatus prev_eco_status = ptr_eco_tracker->get_object_status();
    bool prev_eco_conf_high = ptr_eco_tracker->is_conf_high;
    bool is_eco_abnormal = ((!prev_eco_conf_high) || (prev_eco_status != EXIST));
    float max_iou = 1.0;
//    LOG(INFO) << "is_eco_abnormal: " << is_eco_abnormal;

    if (frame_count % reid_interval == 0 || is_eco_abnormal) {
//        LOG(INFO) << "reid status: "  << "done";
        get_face_people_bboxes(img, ptr_fp_detector, det_face_bboxes, det_body_bboxes);
        if (!is_eco_abnormal) {
            cv::Rect eco_rect = get_tracker_rect();
            BBox eco_bbox = rect2bbox(eco_rect);
            std::pair<int, float> matched_info = find_max_iou(det_body_bboxes, eco_bbox);
            max_iou = matched_info.second;
        }
//        LOG(INFO) << "max_iou: " << max_iou;

        do_reid_compactface_body(img, det_face_bboxes, det_body_bboxes, ptr_face_verifier,
                                 ptr_body_verifier,
                                 ptr_face_keypointer, face_templates, body_templates, face_thresh,
                                 body_thresh,
                                 REID_BEST_SCORE, REID_BEST_SCORE, &face_bbox, &body_bbox,
                                 &face_status, &body_status);

        bool face_reid_success = false;
        bool body_reid_success = false;
        cv::Rect reid_body_rect;
        bool reid_high_conf = false;

        if (face_status == TRACKING_SUCCESS) {
            if (face_bbox.is_attached) {
//                LOG(INFO) << "reid type: " << "face";
//                LOG(INFO) << "reid score: " << face_bbox.reid_conf;
                reid_body_rect = face_bbox.attached_rect;
                face_reid_success = true;
                if (face_bbox.reid_conf < face_reliable_thresh)
                    reid_high_conf = true;
            }
        } else if (body_status == TRACKING_SUCCESS) {
//            LOG(INFO) << "reid type: " << "body";
//            LOG(INFO) << "reid score: " << body_bbox.reid_conf;
            reid_body_rect = bbox2rect(body_bbox);
            body_reid_success = true;
            if (body_bbox.reid_conf < body_reliable_thresh)
                reid_high_conf = true;
        }

        if (face_reid_success || body_reid_success) {
            bool reid_correct = true;
            if (is_eco_abnormal
                || max_iou < 0.5
                || reid_high_conf) {
//                LOG(INFO) << "reid result: " << "correct eco sample space";
                cv::Mat tracker_input = get_tracker_input(img);
                cv::Rect scaled_rect = scale_rect(reid_body_rect, tracker_input_scale);
                ptr_eco_tracker->correct_with_box(tracker_input, scaled_rect, 0.3f);
                ptr_eco_tracker->is_conf_high = true;
            } else {
                BBox eco_bbox = rect2bbox(get_tracker_rect());
                BBox reid_bbox = rect2bbox(reid_body_rect);
                float iou = bbox_iou(eco_bbox, reid_bbox);
                if (iou >= 0.6f) {
//                    LOG(INFO) << "reid result: " << "correct eco location";
                    cv::Mat tracker_input = get_tracker_input(img);
                    cv::Rect scaled_rect = scale_rect(reid_body_rect, tracker_input_scale);
                    ptr_eco_tracker->correct_with_box(tracker_input, scaled_rect);
                } else {
//                    LOG(INFO) << "reid result: " << "reid not used";
                    reid_correct = false;
                    // do track
                    cv::Mat tracker_input = get_tracker_input(img);
                    ptr_eco_tracker->track(tracker_input);
                }
            }
            if (reid_correct) {
                target_pos = reid_body_rect;
                target_status = TRACKING_SUCCESS;
                tracking_action = ACTION_REID;
//                DLOG(INFO) << "reid status: " << "[SUCCESS]" ;
            } else {
                target_pos = get_tracker_rect();
                target_status = TRACKING_SUCCESS;
                tracking_action = ACTION_TRACK;
            }

        } else {
//            LOG(INFO) << "reid result: " << "reid failed";
            if (is_eco_abnormal) {
//                LOG(INFO) << "set no object";
                ptr_eco_tracker->set_no_object();
                target_status = TRACKING_FAILED;
                tracking_action = ACTION_NONE;
            } else {
//                LOG(INFO) << "eco track";
                // do track
                cv::Mat tracker_input = get_tracker_input(img);
                ptr_eco_tracker->track(tracker_input);

                target_pos = get_tracker_rect();
                target_status = TRACKING_SUCCESS;
                tracking_action = ACTION_TRACK;
            }
//            DLOG(INFO) << "reid status: " << "[FAILED]" ;
        }
    } else {
//        LOG(INFO) << "reid status: " << "undo";
        // do track
        cv::Mat tracker_input = get_tracker_input(img);
        ptr_eco_tracker->track(tracker_input);

        // bool is_eco_conf_high = ptr_eco_tracker->is_conf_high;
        // ObjStatus eco_status = ptr_eco_tracker->get_object_status();
        cv::Rect eco_rect = get_tracker_rect();

        target_pos = eco_rect;
        target_status = TRACKING_SUCCESS;
        tracking_action = ACTION_TRACK;

        det_face_bboxes.clear();
        det_body_bboxes.clear();
    }
}

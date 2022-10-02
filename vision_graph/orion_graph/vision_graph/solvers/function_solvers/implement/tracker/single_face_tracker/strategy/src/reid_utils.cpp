//
// Created by yuan on 18-1-31.
//

#include "reid_utils.h"

#include <set>

#include "face_alignment.h"
#include "logging.h"
bool compair_pair_by_second_asscend(const std::pair<int, float> &p1, const std::pair<int, float> &p2) {
    return p1.second < p2.second;
}

bool compair_pair_by_second_descend(const std::pair<int, float> &p1, const std::pair<int, float> &p2) {
    return p1.second > p2.second;
}

#if 0
std::vector<std::pair<int, float>> find_possible_bboxes(const cv::Mat & image,
                                                        const std::vector<BBox> & bboxes,
                                                        BaseVerifier * verifier,
                                                        const std::vector<std::vector<float>> & init_feats,
                                                        float thresh)
{
    std::vector<std::pair<int, float>> good_bboxes;
    for(int i = 0; i < bboxes.size(); i++)
    {
        cv::Mat sub_img = image(bbox2rect(bboxes[i]));
        std::vector<float> sub_feat;
        verifier->compute_feature(sub_img, sub_feat);
        float diff = verifier->compute_feature_diff(sub_feat, (std::vector<std::vector<float>>&)(init_feats));
//        DLOG(INFO) << "diff: " << diff << ", thresh: " << thresh ;
        if (diff < thresh)
        {
            std::pair<int, float> sub_pair(i, diff);
            good_bboxes.push_back(sub_pair);
        }
    }
    std::sort(good_bboxes.begin(), good_bboxes.end(), compair_pair_by_second_asscend);
    return good_bboxes;
}
#else

std::vector<std::pair<int, float>> find_possible_bboxes(const cv::Mat &image,
                                                        const std::vector<BBox> &bboxes,
                                                        BaseVerifier *verifier,
                                                        const std::vector<std::vector<float>> &init_feats,
                                                        float thresh) {
    std::vector<cv::Mat> sub_imgs;
    for (int i = 0; i < bboxes.size(); i++) {
        cv::Mat sub_img = image(bbox2rect(bboxes[i]));
        sub_imgs.push_back(sub_img);
    }
    std::vector<std::vector<float>> sub_feats;
    verifier->compute_feature(sub_imgs, sub_feats);

    std::vector<std::pair<int, float>> good_bboxes = find_possible_bboxes(verifier, sub_feats,
                                                                          init_feats, thresh);
    return good_bboxes;
}

#endif

std::vector<std::pair<int, float>> find_possible_bboxes(BaseVerifier *verifier,
                                                        const std::vector<std::vector<float>> &feats,
                                                        const std::vector<std::vector<float>> &init_feats,
                                                        float thresh) {
    std::vector<std::pair<int, float>> good_bboxes;
    for (int i = 0; i < feats.size(); i++) {
        std::vector<float> &sub_feat = (std::vector<float> &)feats[i];
        float diff = verifier->compute_feature_diff(sub_feat,
                                                    (std::vector<std::vector<float>> &)init_feats);
        //    DLOG(INFO) << "diff: " << diff << ", thresh: " << thresh ;
        // LOGE("wyb debug diff %f  thresh %f", diff, thresh);
        if (diff < thresh) {
            std::pair<int, float> sub_pair(i, diff);
            good_bboxes.push_back(sub_pair);
        }
    }
    std::sort(good_bboxes.begin(), good_bboxes.end(), compair_pair_by_second_asscend);
    return good_bboxes;
};

std::vector<std::pair<int, float>>
find_overlap_bboxes(BBox &prev_bbox, std::vector<BBox> &bboxes, float iou_thresh) {
    std::vector<std::pair<int, float>> good_bboxes;
    for (int i = 0; i < bboxes.size(); i++) {
        float iou = bbox_iou(bboxes[i], prev_bbox);
        if (iou > iou_thresh) {
            std::pair<int, float> sub_pair(i, iou);
            good_bboxes.push_back(sub_pair);
        }
    }
    std::sort(good_bboxes.begin(), good_bboxes.end(), compair_pair_by_second_descend);
    return good_bboxes;
}

int choose_nearest_bboxes(BBox &prev_bbox, std::vector<std::pair<int, float>> &possible_bboxes,
                          std::vector<BBox> &bboxes) {
    float max_iou = 0;
    int max_i = -1;
    for (int i = 0; i < possible_bboxes.size(); i++) {
        int id = possible_bboxes[i].first;
        BBox bbox = bboxes[id];
        float iou = bbox_iou(bbox, prev_bbox);
        if (iou > max_iou) {
            max_iou = iou;
            max_i = i;
        }
    }
    return max_i;
}

std::pair<int, float> find_min_dis_bbox(const std::vector<BBox> &bboxes,
                                        const BBox &query_bbox) {
    float min_dist = std::numeric_limits<float>::max();
    int min_ind = -1;
    for (int i = 0; i < bboxes.size(); ++i) {
        float dist = bbox_dist((BBox &)query_bbox, (BBox &)bboxes[i]);
        if (dist == 0.f)
            continue;

        if (dist < min_dist) {
            min_dist = dist;
            min_ind = i;
        }
    }
    assert(min_dist != 0);
    return std::pair<int, float>(min_ind, min_dist);
}

std::pair<int, float> find_max_iou(std::vector<BBox> &bbox_vec, BBox &bbox) {
    float max_iou = 0;
    int matched_ind = -1;
    for (int i = 0; i < bbox_vec.size(); ++i) {
        float iou = bbox_iou(bbox, bbox_vec[i]);
        if (iou > max_iou) {
            max_iou = iou;
            matched_ind = i;
        }
    }
    return std::pair<int, float>(matched_ind, max_iou);
}

int do_reid_base(const std::vector<BBox> &bboxes,
                 const std::vector<std::pair<int, float>> &possible_bboxes,
                 REID_SEL_MODE mode,
                 BBox *bbox,
                 TrackingStatus *status) {
    auto possible_num = possible_bboxes.size();
    int id = -1;
    if (possible_num == 1) {
        // tracking success
        id = possible_bboxes[0].first;
        float reid_conf = possible_bboxes[0].second;
        *bbox = bboxes[id];
        bbox->reid_conf = reid_conf;
        *status = TRACKING_SUCCESS;
    } else if (possible_num > 1) {
        if (*status == TRACKING_SUCCESS && mode == REID_MAX_IOU) {
            // chooses nearest
            int i = choose_nearest_bboxes(*bbox,
                                          (std::vector<std::pair<int, float>> &)possible_bboxes,
                                          (std::vector<BBox> &)(bboxes));

            if (i == -1)
                i = 0;
            id = possible_bboxes[i].first;
            float reid_conf = possible_bboxes[i].second;
            *bbox = bboxes[id];
            bbox->reid_conf = reid_conf;
            *status = TRACKING_SUCCESS;
        } else {
            // choose best score
            id = possible_bboxes[0].first;
            float reid_conf = possible_bboxes[0].second;
            *bbox = bboxes[id];
            bbox->reid_conf = reid_conf;
            *status = TRACKING_SUCCESS;
        }
    } else {
        // reid failed
        *status = TRACKING_FAILED;
        // id = 0;
    }
    return id;
}

int do_reid_base(const cv::Mat &img,
                 const std::vector<BBox> &bboxes,
                 BaseVerifier *verifier,
                 const std::vector<std::vector<float>> &init_feats,
                 float thresh,
                 REID_SEL_MODE mode,
                 BBox *bbox,
                 TrackingStatus *status) {
    std::vector<std::pair<int, float>> possible_bboxes = find_possible_bboxes(img, bboxes, verifier,
                                                                              init_feats, thresh);
    return do_reid_base(bboxes, possible_bboxes, mode, bbox, status);
}

void do_reid_face(const cv::Mat &img,
                  BaseDetector *fp_detector,
                  BaseVerifier *face_verifier,
                  const std::vector<std::vector<float>> &face_init_feats,
                  float face_thresh,
                  REID_SEL_MODE mode,
                  BBox *face_bbox,
                  TrackingStatus *face_status) {
    std::vector<BBox> fp_bboxes = fp_detector->detect(img);
    std::vector<BBox> face_bboxes = fp_detector->combine_bboxes_p2f(fp_bboxes);
    do_reid_base(img, face_bboxes, face_verifier, face_init_feats, face_thresh, mode, face_bbox,
                 face_status);
}

void do_reid_face(const cv::Mat &img,
                  BaseDetector *fp_detector,
                  BaseVerifier *face_verifier,
                  const std::vector<std::vector<float>> &face_init_feats,
                  float face_thresh,
                  REID_SEL_MODE mode,
                  std::vector<BBox> &det_face_bboxes,
                  BBox *face_bbox,
                  TrackingStatus *face_status) {
    std::vector<BBox> fp_bboxes = fp_detector->detect(img);
    det_face_bboxes = fp_detector->combine_bboxes_p2f(fp_bboxes);
    do_reid_base(img, det_face_bboxes, face_verifier, face_init_feats, face_thresh, mode, face_bbox,
                 face_status);
}

int do_reid_face_compact(const cv::Mat &img,
                         BaseVerifier *face_verifier,
                         const std::vector<BBox> &face_bboxes,
                         const std::vector<std::vector<cv::Point2f>> &kpts_vecs,
                         const std::vector<std::vector<float>> &face_init_feats,
                         float face_thresh,
                         REID_SEL_MODE mode,
                         BBox *face_bbox,
                         TrackingStatus *face_status) {
    size_t face_num = face_bboxes.size();
    std::vector<std::vector<float>> feats(face_num);
    for (int i = 0; i < face_num; ++i) {
        cv::Rect face_rect = bbox2rect(face_bboxes[i]);
        const auto &kpts = kpts_vecs[i];
        cv::Mat compact_face = img(face_rect);
        // face_align_compact(img, kpts, face_rect, compact_face);
        face_verifier->compute_feature(compact_face, feats[i]);
    }

    std::vector<std::pair<int, float>> possible_bboxes = find_possible_bboxes(face_verifier, feats,
                                                                              face_init_feats,
                                                                              face_thresh);
    return do_reid_base(face_bboxes, possible_bboxes, mode, face_bbox, face_status);
}

void do_reid_face_compact(const cv::Mat &img,
                          BaseVerifier *face_verifier,
                          BaseKeypoints *face_keypointer,
                          const std::vector<BBox> &face_bboxes,
                          const std::vector<std::vector<float>> &face_init_feats,
                          float face_thresh,
                          REID_SEL_MODE mode,
                          BBox *face_bbox,
                          TrackingStatus *face_status) {
    size_t face_num = face_bboxes.size();
    std::vector<std::vector<cv::Point2f>> kpts_vecs(face_num);
    for (int i = 0; i < face_num; ++i) {
        cv::Rect face_rect = bbox2rect(face_bboxes[i]);
        cv::Mat face_img = img(face_rect);
        face_keypointer->compute_keypoints(face_img, kpts_vecs[i]);
    }

    do_reid_face_compact(img, face_verifier, face_bboxes, kpts_vecs, face_init_feats,
                         face_thresh, mode, face_bbox, face_status);
}

void do_reid_body(const cv::Mat &img,
                  BaseDetector *fp_detector,
                  BaseVerifier *body_verifier,
                  const std::vector<std::vector<float>> &body_init_feats,
                  float body_thresh,
                  REID_SEL_MODE mode,
                  BBox *body_bbox,
                  TrackingStatus *body_status) {
    std::vector<BBox> fp_bboxes = fp_detector->detect(img);
    std::vector<BBox> body_bboxes = fp_detector->combine_bboxes_f2p(fp_bboxes);
    do_reid_base(img, body_bboxes, body_verifier, body_init_feats, body_thresh, mode, body_bbox,
                 body_status);
}

void do_reid_face_body(const cv::Mat &img,
                       BaseDetector *fp_detector,
                       BaseVerifier *face_verifier,
                       BaseVerifier *body_verifier,
                       const std::vector<std::vector<float>> &face_init_feats,
                       const std::vector<std::vector<float>> &body_init_feats,
                       float face_thresh,
                       float body_thresh,
                       REID_SEL_MODE face_mode,
                       REID_SEL_MODE body_mode,
                       BBox *face_bbox,
                       BBox *body_bbox,
                       TrackingStatus *face_status,
                       TrackingStatus *body_status) {
    std::vector<BBox> face_bboxes, body_bboxes;
    get_face_people_bboxes(img, fp_detector, face_bboxes, body_bboxes);

    if (face_init_feats.empty()) {
        *face_status = TRACKING_UNCERTAIN;
    } else {
        do_reid_base(img, face_bboxes, face_verifier, face_init_feats, face_thresh, face_mode,
                     face_bbox, face_status);
    }

    if (*face_status == TRACKING_SUCCESS || body_init_feats.empty()) {
        *body_status = TRACKING_UNCERTAIN;
    } else {
        do_reid_base(img, body_bboxes, body_verifier, body_init_feats, body_thresh, body_mode,
                     body_bbox, body_status);
    }
}

void get_face_people_bboxes(const cv::Mat &img,
                            BaseDetector *fp_detector,
                            std::vector<BBox> &face_bboxes,
                            std::vector<BBox> &body_bboxes) {
    std::vector<BBox> fp_bboxes = fp_detector->detect(img);
    face_bboxes = fp_detector->combine_bboxes_p2f(fp_bboxes);
    //    for(int i=0; i<face_bboxes.size(); i++)
    //    {
    //        LOG(INFO) << "faceloc " << i << ": " << face_bboxes[i].x1 << " "
    //                << face_bboxes[i].y1 << " " << face_bboxes[i].x2 << " " << face_bboxes[i].y2;
    //    }
    body_bboxes = fp_detector->combine_bboxes_f2p(fp_bboxes);
    //    for(int i=0; i<body_bboxes.size(); i++)
    //    {
    //        LOG(INFO) << "peopleloc " << i << ": " << body_bboxes[i].x1 << " "
    //                << body_bboxes[i].y1 << " " << body_bboxes[i].x2 << " " << body_bboxes[i].y2;
    //    }
}

void do_reid_face_body_bboxes_input(const cv::Mat &img,
                                    const std::vector<BBox> &face_bboxes,
                                    const std::vector<BBox> &body_bboxes,
                                    BaseVerifier *face_verifier,
                                    BaseVerifier *body_verifier,
                                    const std::vector<std::vector<float>> &face_init_feats,
                                    const std::vector<std::vector<float>> &body_init_feats,
                                    float face_thresh,
                                    float body_thresh,
                                    REID_SEL_MODE face_mode,
                                    REID_SEL_MODE body_mode,
                                    BBox *face_bbox,
                                    BBox *body_bbox,
                                    TrackingStatus *face_status,
                                    TrackingStatus *body_status) {
    do_reid_base(img, face_bboxes, face_verifier, face_init_feats, face_thresh, face_mode,
                 face_bbox, face_status);
    if (*face_status == TRACKING_SUCCESS) {
        *body_status = TRACKING_UNCERTAIN;
    } else {
        do_reid_base(img, body_bboxes, body_verifier, body_init_feats, body_thresh, body_mode,
                     body_bbox, body_status);
    }
}

static int search_b2f(std::vector<BBox> &body_bboxes, BBox &face_bbox) {
    cv::Rect face_rect = bbox2rect(face_bbox);
    float face_area = face_rect.area();
    int max_ind = -1;
    float max_iou = 0;
    for (int i = 0; i < body_bboxes.size(); ++i) {
        cv::Rect body_rect = bbox2rect(body_bboxes[i]);
        cv::Rect cross_rect = face_rect & body_rect;
        float cross_area = cross_rect.area();
        if (cross_area >= 0.7f * face_area) {
            if (max_iou < cross_area) {
                max_iou = cross_area;
                max_ind = i;
            }
        }
    }
    return max_ind;
}

void do_reid_compactface_body(const cv::Mat &img,
                              const std::vector<BBox> &face_bboxes,
                              const std::vector<BBox> &body_bboxes,
                              BaseVerifier *face_verifier,
                              BaseVerifier *body_verifier,
                              BaseKeypoints *face_keypointer,
                              const std::vector<std::vector<float>> &face_init_feats,
                              const std::vector<std::vector<float>> &body_init_feats,
                              float face_thresh,
                              float body_thresh,
                              REID_SEL_MODE face_mode,
                              REID_SEL_MODE body_mode,
                              BBox *face_bbox,
                              BBox *body_bbox,
                              TrackingStatus *face_status,
                              TrackingStatus *body_status) {
    size_t face_num = face_bboxes.size();
    std::vector<std::vector<cv::Point2f>> kpts_vecs(face_num);
    std::vector<RPYPose> pose_vec(face_num);
    std::vector<std::vector<float>> face_feats(face_num);
    for (int i = 0; i < face_num; ++i) {
        cv::Rect face_rect = bbox2rect(face_bboxes[i]);
        cv::Mat face_img = img(face_rect);
        face_keypointer->compute_keypoints_pose(face_img, kpts_vecs[i], pose_vec[i]);

        cv::Mat compact_face;
        face_align_compact(img, kpts_vecs[i], face_rect, compact_face);
        face_verifier->compute_feature(compact_face, face_feats[i]);
    }

    auto possible_face_bboxes = find_possible_bboxes(face_verifier, face_feats, face_init_feats,
                                                     face_thresh);
    do_reid_base(face_bboxes, possible_face_bboxes, face_mode, face_bbox, face_status);

    if (*face_status == TRACKING_SUCCESS) {
        *body_status = TRACKING_UNCERTAIN;
    } else {
        // remove bodies attached to rejected faces by face reid
        std::set<int> rm_body_inds;
        for (int i = 0; i < face_num; ++i) {
            cv::Rect face_rect = bbox2rect(face_bboxes[i]);
            RPYPose &pose = pose_vec[i];
            float face_diff = std::numeric_limits<float>::max();
            for (int j = 0; j < possible_face_bboxes.size(); ++j) {
                if (possible_face_bboxes[j].first == i)
                    face_diff = possible_face_bboxes[j].second;
            }

            // rejected by face verifier with high confidence, not the object
            if (face_rect.area() >= 80 * 80 && face_diff > 1.f &&
                fabs(pose.yaw) < 20 && fabs(pose.pitch) < 20 && fabs(pose.roll) < 30) {
                int matched_ind = search_b2f((std::vector<BBox> &)body_bboxes,
                                             (BBox &)face_bboxes[i]);
                if (matched_ind >= 0) {
                    rm_body_inds.insert(matched_ind);
                }
            }
        }

        std::vector<BBox> filtered_body_bboxes;
        for (int i = 0; i < body_bboxes.size(); ++i) {
            if (rm_body_inds.find(i) == rm_body_inds.end()) {
                filtered_body_bboxes.push_back(body_bboxes[i]);
            }
        }

        do_reid_base(img, filtered_body_bboxes, body_verifier, body_init_feats, body_thresh,
                     body_mode,
                     body_bbox, body_status);
    }
}

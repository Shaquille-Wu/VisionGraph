//
// Created by yuan on 18-1-31.
//

#ifndef GRAPH_ROBOT_TRACKING_REID_UTILS_H
#define GRAPH_ROBOT_TRACKING_REID_UTILS_H

#include "base_detector.h"
#include "base_keypoints.h"
#include "base_track_def.h"
#include "base_verifier.h"

typedef enum {
    REID_BEST_SCORE = 0,
    REID_MAX_IOU
} REID_SEL_MODE;

std::vector<std::pair<int, float>> find_possible_bboxes(const cv::Mat& image,
                                                        const std::vector<BBox>& bboxes,
                                                        BaseVerifier* verifier,
                                                        const std::vector<std::vector<float>>& init_feats,
                                                        float thresh);

std::vector<std::pair<int, float>> find_possible_bboxes(BaseVerifier* verifier,
                                                        const std::vector<std::vector<float>>& feats,
                                                        const std::vector<std::vector<float>>& init_feats,
                                                        float thresh);

std::vector<std::pair<int, float>> find_overlap_bboxes(BBox& prev_bbox,
                                                       std::vector<BBox>& bboxes,
                                                       float iou_thresh);

int choose_nearest_bboxes(BBox& prev_bbox,
                          std::vector<std::pair<int, float>>& possible_bboxes,
                          std::vector<BBox>& bboxes);

std::pair<int, float> find_min_dis_bbox(const std::vector<BBox>& bboxes,
                                        const BBox& query_bbox);

std::pair<int, float> find_max_iou(std::vector<BBox>& bbox_vec, BBox& bbox);

int do_reid_base(const std::vector<BBox>& bboxes,
                 const std::vector<std::pair<int, float>>& possible_bboxes,
                 REID_SEL_MODE mode,
                 BBox* bbox,
                 TrackingStatus* status);

int do_reid_base(const cv::Mat& img,
                 const std::vector<BBox>& bboxes,
                 BaseVerifier* verifier,
                 const std::vector<std::vector<float>>& init_feats,
                 float thresh,
                 REID_SEL_MODE mode,
                 BBox* bbox,
                 TrackingStatus* status);

void do_reid_face(const cv::Mat& img,
                  BaseDetector* fp_detector,
                  BaseVerifier* face_verifier,
                  const std::vector<std::vector<float>>& face_init_feats,
                  float face_thresh,
                  REID_SEL_MODE mode,
                  BBox* face_bbox,
                  TrackingStatus* face_status);

void do_reid_face(const cv::Mat& img,
                  BaseDetector* fp_detector,
                  BaseVerifier* face_verifier,
                  const std::vector<std::vector<float>>& face_init_feats,
                  float face_thresh,
                  REID_SEL_MODE mode,
                  std::vector<BBox>& det_face_bboxes,
                  BBox* face_bbox,
                  TrackingStatus* face_status);

int do_reid_face_compact(const cv::Mat& img,
                         BaseVerifier* face_verifier,
                         const std::vector<BBox>& face_bboxes,
                         const std::vector<std::vector<cv::Point2f>>& kpts_vecs,
                         const std::vector<std::vector<float>>& face_init_feats,
                         float face_thresh,
                         REID_SEL_MODE mode,
                         BBox* face_bbox,
                         TrackingStatus* face_status);

void do_reid_face_compact(const cv::Mat& img,
                          BaseVerifier* face_verifier,
                          BaseKeypoints* face_keypointer,
                          const std::vector<BBox>& face_bboxes,
                          const std::vector<std::vector<float>>& face_init_feats,
                          float face_thresh,
                          REID_SEL_MODE mode,
                          BBox* face_bbox,
                          TrackingStatus* face_status);

void do_reid_body(const cv::Mat& img,
                  BaseDetector* fp_detector,
                  BaseVerifier* body_verifier,
                  const std::vector<std::vector<float>>& body_init_feats,
                  float body_thresh,
                  REID_SEL_MODE mode,
                  BBox* body_bbox,
                  TrackingStatus* body_status);

void do_reid_face_body(const cv::Mat& img,
                       BaseDetector* fp_detector,
                       BaseVerifier* face_verifier,
                       BaseVerifier* body_verifier,
                       const std::vector<std::vector<float>>& face_init_feats,
                       const std::vector<std::vector<float>>& body_init_feats,
                       float face_thresh,
                       float body_thresh,
                       REID_SEL_MODE face_mode,
                       REID_SEL_MODE body_mode,
                       BBox* face_bbox,
                       BBox* body_bbox,
                       TrackingStatus* face_status,
                       TrackingStatus* body_status);

void get_face_people_bboxes(const cv::Mat& img,
                            BaseDetector* fp_detector,
                            std::vector<BBox>& face_bboxes,
                            std::vector<BBox>& body_bboxes);

void do_reid_face_body_bboxes_input(const cv::Mat& img,
                                    const std::vector<BBox>& face_bboxes,
                                    const std::vector<BBox>& body_bboxes,
                                    BaseVerifier* face_verifier,
                                    BaseVerifier* body_verifier,
                                    const std::vector<std::vector<float>>& face_init_feats,
                                    const std::vector<std::vector<float>>& body_init_feats,
                                    float face_thresh,
                                    float body_thresh,
                                    REID_SEL_MODE face_mode,
                                    REID_SEL_MODE body_mode,
                                    BBox* face_bbox,
                                    BBox* body_bbox,
                                    TrackingStatus* face_status,
                                    TrackingStatus* body_status);

void do_reid_compactface_body(const cv::Mat& img,
                              const std::vector<BBox>& face_bboxes,
                              const std::vector<BBox>& body_bboxes,
                              BaseVerifier* face_verifier,
                              BaseVerifier* body_verifier,
                              BaseKeypoints* face_keypointer,
                              const std::vector<std::vector<float>>& face_init_feats,
                              const std::vector<std::vector<float>>& body_init_feats,
                              float face_thresh,
                              float body_thresh,
                              REID_SEL_MODE face_mode,
                              REID_SEL_MODE body_mode,
                              BBox* face_bbox,
                              BBox* body_bbox,
                              TrackingStatus* face_status,
                              TrackingStatus* body_status);

#endif  //ROBOT_TRACKING_REID_UTILS_H

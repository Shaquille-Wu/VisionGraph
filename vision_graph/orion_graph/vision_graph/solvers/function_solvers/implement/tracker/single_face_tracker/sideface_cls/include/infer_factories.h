//
// Created by yuan on 18-5-19.
//

#ifndef GRAPH_ROBOT_TRACKING_INFER_FACTORIES_H
#define GRAPH_ROBOT_TRACKING_INFER_FACTORIES_H

#include "base_detector.h"
#include "base_verifier.h"
#include "base_keypoints.h"
#include "graph_solver.h"

#ifdef PLATFORM_CAFFE

// detector factories
BaseDetector* get_fp_detector(const std::string& prototxt,
                              const std::string& caffemodel,
                              int net_wid,
                              int net_hei);

BaseDetector* get_rcnn_detector(const std::string& prototxt,
                                const std::string& caffemodel);

// verifier factories
BaseVerifier* get_face_verifier(const std::string& prototxt,
                                const std::string& caffemodel);

BaseVerifier* get_newest_people_verifier(const std::string& prototxt,
                                         const std::string& caffemodel);

BaseVerifier* get_aligned_people_verifier(const std::string& prototxt,
                                          const std::string& caffemodel,
                                          const std::vector<std::string>& output_layers);

// keypointer factories
BaseKeypoints* get_face_keypoints_regressor(const std::string& prototxt,
                                            const std::string& caffemodel);

// // classifier factories
// BaseClassfier* get_body_pose_classfier(const std::string & prototxt,
//                                        const std::string & caffemodel);

// BaseClassfier* get_person_classfier(const std::string & prototxt,
//                                     const std::string & caffemodel);

// BaseClassfier* get_face_quality_classfier(const std::string & prototxt,
//                                           const std::string & caffemodel,
//                                           const std::string & meanfile);

#else
// iml or tensorrt

BaseDetector* get_fp_detector(vision_graph::Solver* solver, int w = 320, int h = 320);

BaseDetector* get_fp_detector();

BaseVerifier* get_face_verifier(const std::string& model);

BaseVerifier* get_face_verifier();

BaseVerifier* get_newest_people_verifier(const std::string& model);

BaseVerifier* get_newest_people_verifier();

BaseKeypoints* get_face_keypoints_regressor(vision_graph::Solver* solver);

BaseKeypoints* get_face_keypoints_regressor();


// BaseClassfier* get_body_pose_classifier(const std::string& model);

// BaseClassfier* get_body_pose_classifier();

#endif


#ifdef USE_RFCN_PFD
// no out dependencies
BaseDetector* get_pfd_detector(int net_wid, int net_hei);
#endif


#endif //GRAPH_ROBOT_TRACKING_INFER_FACTORIES_H

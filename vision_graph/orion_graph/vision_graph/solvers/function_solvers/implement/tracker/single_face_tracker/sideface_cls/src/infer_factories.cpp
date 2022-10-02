//
// Created by yuan on 18-5-19.
//

#include "infer_factories.h"

#ifdef USE_RFCN_PFD
#include "pfd_detection.h"
#endif

#ifdef PLATFORM_CAFFE

#include "caffe_align_reid.h"
#include "caffe_body_pose.h"
#include "caffe_face_keypoints.h"
#include "caffe_person_cls.h"
#include "caffe_reid.h"
#include "caffe_yolov2_detection.h"
#include "rcnn_detection.h"
BaseDetector* get_fp_detector(const std::string& prototxt, const std::string& caffemodel, int net_wid, int net_hei) {
    float people_anchors[] = {
        0.625, .625, 0.625, 1.250, 0.625, 2.500, 1, 0.5,
        1, 1, 1, 2, 1, 4, 1.6, 0.8,
        1.6, 1.6, 1.6, 3.2, 1.6, 6.4, 2.56, 1.28,
        2.56, 2.56, 2.56, 5.12, 2.56, 10.24, 4.1, 2.05,
        4.1, 4.1, 4.1, 8.2, 4.1, 16.4, 6.56, 3.28,
        6.56, 6.56, 6.56, 13.12, 10.5, 5.25, 10.5, 10.5};
    float face_anchors[] = {
        0.625, 0.750, 0.625, 0.750, 0.625, 0.750, 0.625, 0.750,
        0.625, 0.750, 1.000, 1.200, 1.000, 1.200, 1.000, 1.200,
        1.000, 1.200, 1.600, 1.920, 2.560, 3.072, 4.096, 4.915,
        6.554, 7.864, 10.486, 12.583};

    CaffeYolov2DetectionFP* detector = new CaffeYolov2DetectionFP(prototxt.c_str(), caffemodel.c_str());
    detector->setFaceParams(14, face_anchors, 0.6, 0.4);
    detector->setPeopleParams(24, people_anchors, 0.6, 0.4);
    detector->set_net_size(net_hei, net_wid);
    return detector;
}

BaseDetector* get_rcnn_detector(const std::string& prototxt, const std::string& caffemodel) {
    RCNNDetection* detector = new RCNNDetection(prototxt.c_str(), caffemodel.c_str());
std:
    vector<int> keep_classes;
    keep_classes.push_back(1);
    float nms_t = 0.3;
    float conf_t = 0.8;
    detector->set_param(nms_t, keep_classes, conf_t);
    return detector;
}

BaseVerifier* get_face_verifier(const std::string& prototxt, const std::string& caffemodel) {
    CaffeReID* verifier = new CaffeReID();  //(64, 128);
    verifier->loadModel(prototxt.c_str(), caffemodel.c_str());
    //resnet_50_d4
    float bn_beta[3] = {3.12500215, -0.97326922, -0.30886102};
    float bn_gama[3] = {0.16505176, 0.60845631, 0.59119201};
    float bn_mean[3] = {103.65445709, 121.083992, 158.1137085};
    float bn_var[3] = {3049.10253906, 3112.63256836, 3859.73632812};

    verifier->set_swap_channels(false);
    verifier->set_input_bn(bn_beta, bn_gama, bn_mean, bn_var);
    verifier->set_input_normalize(false);
    verifier->set_output_normalize(true);
    verifier->set_input_blob_name("data");
    verifier->set_output_blob_name("conv_last");
    return verifier;
}

BaseVerifier* get_aligned_people_verifier(const std::string& prototxt,
                                          const std::string& caffemodel,
                                          const std::vector<std::string>& output_layers) {
    std::string input_blob("data");
    CaffeAlignReID* verifier = new CaffeAlignReID(prototxt,
                                                  caffemodel,
                                                  input_blob,
                                                  output_layers,
                                                  true,    // input norm
                                                  true,    // output norm
                                                  false);  // is_bgr

    return verifier;
}

#ifdef USE_ALIGN_REID

BaseVerifier* get_newest_people_verifier(const std::string& prototxt, const std::string& caffemodel) {
    std::vector<std::string> output_blobs(2);
    output_blobs[0] = "layer12-fc";
    output_blobs[1] = "layer14-conv";
    return get_aligned_people_verifier(prototxt, caffemodel, output_blobs);
}

#else

BaseVerifier* get_newest_people_verifier(const std::string& prototxt, const std::string& caffemodel) {
    CaffeReID* verifier = new CaffeReID();  //(64, 128);
    verifier->loadModel(prototxt.c_str(), caffemodel.c_str());
    verifier->set_input_normalize(true);
    verifier->set_output_normalize(true);
    verifier->set_input_blob_name("data");
    verifier->set_output_blob_name("layer12-fc");
    return verifier;
}

#endif

BaseKeypoints* get_face_keypoints_regressor(const std::string& prototxt, const std::string& caffemodel) {
    CaffeFaceKeypoints* ptr = new CaffeFaceKeypoints(prototxt, caffemodel, "data", "fc_reg", 106);

    float bn_mean = 95.29972076f;
    float bn_var = 4672.56933594f;
    float bn_gamma = 0.2678552f;
    float bn_beta = -0.02005605f;

    ptr->set_input_bn(bn_beta, bn_mean, bn_gamma, bn_var);
    ptr->set_preprocess_param(false, false, false);

    return ptr;
}

BaseClassfier* get_body_pose_classfier(const std::string& prototxt, const std::string& caffemodel) {
    cv::Scalar_<float> pixel_mean(102.9801, 115.9465, 122.7717);
    BaseClassfier* ptr = new BodyPoseClassifier(prototxt, caffemodel, pixel_mean);
    return ptr;
}

BaseClassfier* get_person_classfier(const std::string& prototxt, const std::string& caffemodel) {
    cv::Scalar_<float> pixel_mean(122.7717, 122.7717, 122.7717);
    BaseClassfier* ptr = new PersonClassifier(prototxt, caffemodel, pixel_mean);
    return ptr;
}

#else

#include "inference_detection.h"
#include "inference_face_keypoints.h"
#include "inference_face_verify.h"

BaseDetector* get_fp_detector(vision_graph::Solver* solver, int w, int h) {
    InferenceDetection* detector = new InferenceDetection(solver, w, h);
    return detector;
}

BaseDetector* get_fp_detector() {
    InferenceDetection* detector = new InferenceDetection();
    return detector;
}

BaseVerifier* get_face_verifier(const std::string& model) {
    InferenceVerifier* verifier = new InferenceVerifier((char*)model.c_str());
    return verifier;
}

BaseVerifier* get_face_verifier() {
    InferenceVerifier* verifier = new InferenceVerifier();
    return verifier;
}


BaseKeypoints* get_face_keypoints_regressor(vision_graph::Solver* solver) {
    InferenceKeypoint* ptr = new InferenceKeypoint(solver);
    return ptr;
}

BaseKeypoints* get_face_keypoints_regressor() {
    InferenceKeypoint* ptr = new InferenceKeypoint();
    return ptr;
}

#endif

#ifdef USE_RFCN_PFD
BaseDetector* get_pfd_detector(int net_width, int net_height) {
    PFDDetection* detector = new PFDDetection(net_width, net_height);
    return detector;
}
#endif

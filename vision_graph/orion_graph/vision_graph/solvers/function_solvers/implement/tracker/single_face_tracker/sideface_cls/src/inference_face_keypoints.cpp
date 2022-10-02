#include "inference_face_keypoints.h"

#include "timer.h"

void changepoint(std::vector<cv::Point2f>& kpts) {
    std::vector<int> eye = {52, 53, 72, 54, 56, 57, 73, 74, 55, 104, 58, 59, 60, 61, 62, 63, 75, 76,
                            105, 77};
    std::vector<int> eye_brow = {33, 34, 35, 36, 37, 38, 39, 40, 41, 64, 65, 66, 67, 68, 69, 70, 71,
                                 42};
    std::vector<int> nose = {43, 44, 45, 46, 47, 48, 49, 50, 51, 78, 79, 80, 81, 82, 83};
    std::vector<int> mouth = {84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                              101, 102, 103};
    float outx[106] = {0};
    float outy[106] = {0};
    //    float* outx;
    //    float* outy;
    for (int i = 33; i < 53; i++) {
        outx[eye[i - 33]] = kpts[i].x;
        outy[eye[i - 33]] = kpts[i].y;
    }

    for (int i = 53; i < 71; i++) {
        outx[eye_brow[i - 53]] = kpts[i].x;
        outy[eye_brow[i - 53]] = kpts[i].y;
    }

    for (int i = 71; i < 86; i++) {
        outx[nose[i - 71]] = kpts[i].x;
        outy[nose[i - 71]] = kpts[i].y;
    }

    for (int i = 86; i < 106; i++) {
        outx[mouth[i - 86]] = kpts[i].x;
        outy[mouth[i - 86]] = kpts[i].y;
    }

    for (int i = 33; i < 106; i++) {
        kpts[i].x = outx[i];
        kpts[i].y = outy[i];
    }
}

InferenceKeypoint::InferenceKeypoint() {

}

InferenceKeypoint::InferenceKeypoint(vision_graph::Solver* solver) {
    keypoint_ptr = solver;
    if(!tensor_fkp)
        tensor_fkp = new vision_graph::TensorKeypointsVector();
    if(!tensor_att)
        tensor_att = new vision_graph::TensorAttributes();
    if(!tensor_image)
        tensor_image = new vision_graph::TensorImageVector();
}

InferenceKeypoint::~InferenceKeypoint() {
    if (tensor_fkp) {
        delete tensor_fkp;
        tensor_fkp = NULL;
    }
    if (tensor_image) {
        delete tensor_image;
        tensor_image = NULL;
    }
    if (tensor_att) {
        delete tensor_att;
        tensor_att = NULL;
    }
}

void InferenceKeypoint::compute_keypoints(const cv::Mat& img, std::vector<cv::Point2f>& kpts) {

    kpts.clear();
    kpts.resize(106);
    
    tensor_image->clear();

    tensor_image->push_back(img);

    std::vector<vision_graph::Tensor*>  in;
    in.push_back(tensor_image);
    std::vector<vision_graph::Tensor*> out;
    out.push_back(tensor_fkp);
    out.push_back(tensor_att);

    keypoint_ptr->Solve(in, out);
    kpts = tensor_fkp->front();

    changepoint(kpts);
}

void InferenceKeypoint::compute_keypoints_pose(const cv::Mat& img, std::vector<cv::Point2f>& kpts, RPYPose& pose) {

    kpts.clear();
    kpts.resize(106);
    tensor_image->clear();

    tensor_image->push_back(img);

    std::vector<vision_graph::Tensor*>  in;
    in.push_back(tensor_image);
    std::vector<vision_graph::Tensor*> out;
    out.push_back(tensor_fkp);
    out.push_back(tensor_att);

    keypoint_ptr->Solve(in, out);


    // tensor_base_fkp = out.front();
    // tensor_fkp               = dynamic_cast<vision_graph::TensorKeypointsVector*>(tensor_base_fkp);
    kpts = tensor_fkp->front();
    changepoint(kpts);

    // tensor_base_att          = out[1];
    // tensor_att               = dynamic_cast<vision_graph::TensorAttributes*>(tensor_base_att);


    std::vector<std::vector<float>>angle = *tensor_att;
    pose.roll = angle[0][0] * 90;
    pose.yaw = angle[0][1] * 90;
    pose.pitch = angle[0][2] * 90;

    cv::Point2f left_eye = kpts[104];
    cv::Point2f right_eye = kpts[105];
    pose.roll = atan2f(right_eye.y - left_eye.y, right_eye.x - left_eye.x) * 180 / (float)CV_PI;

}
void InferenceKeypoint::compute_keypoints_pose_hog(const cv::Mat& img, std::vector<cv::Point2f>& kpts, RPYPose& pose, float& hog_score) {
    // std::vector<std::vector<float> > attributes;
    // std::vector<cv::Point> points;

    // keypoint_ptr->run((cv::Mat&)img, points, attributes);

    // kpts.clear();
    // kpts.resize(106);
    // for (int ix = 0; ix < 106; ix++) {
    //     kpts[ix].x = points[ix].x;
    //     kpts[ix].y = points[ix].y;
    // }
    // changepoint(kpts);
    // CFacialLandmarks extra_sideface_cls;

    // cv::Mat face_img_gray;
    // cv::cvtColor(img, face_img_gray, cv::COLOR_BGR2GRAY);
    // hog_score = extra_sideface_cls.compute_similarity(face_img_gray.data,
    //                                                   face_img_gray.cols,
    //                                                   face_img_gray.rows, kpts);
    // pose.roll = attributes[0][0] * 90;
    // pose.yaw = attributes[0][1] * 90;
    // pose.pitch = attributes[0][2] * 90;
}

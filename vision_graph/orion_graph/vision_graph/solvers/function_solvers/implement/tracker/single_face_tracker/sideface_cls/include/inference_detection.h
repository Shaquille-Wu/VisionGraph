#ifndef GRAPH_INFERENCE_DETECTION_H
#define GRAPH_INFERENCE_DETECTION_H

#include <vector>

#include "base_detector.h"
#include "base_utils.h"
#include "detector.h"
#include "opencv2/opencv.hpp"
#include "vision_graph.h"
#include "logging.h"
#include "graph_solver.h"

using namespace vision;

class InferenceDetection : public BaseDetector {
public:
    InferenceDetection();
    InferenceDetection(vision_graph::Solver* solver, int w = 320, int h = 320);
    ~InferenceDetection();
    std::vector<BBox> detect(const cv::Mat& img);
    std::vector<BBox> get_face_bboxes(std::vector<BBox>& bboxes);
    std::vector<BBox> get_people_bboxes(std::vector<BBox>& bboxes);
    std::vector<BBox> combine_bboxes_f2p(std::vector<BBox>& bboxes);
    std::vector<BBox> combine_bboxes_p2f(std::vector<BBox>& bboxes);

private:
    // vision::Detector* detect_ptr;
    vision_graph::Solver* detect_ptr;
    vision_graph::TensorBoxesMap*     tensor_box = NULL;
    vision_graph::TensorImage*        tensor_image = NULL;

};

#endif

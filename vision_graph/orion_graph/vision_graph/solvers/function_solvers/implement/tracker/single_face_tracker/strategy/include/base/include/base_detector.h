#ifndef BASE_DETECTOR_H
#define BASE_DETECTOR_H

#include "base_utils.h"
class BaseDetector {
public:
    BaseDetector(){}
    virtual ~BaseDetector() {}
    virtual std::vector<BBox> detect(const cv::Mat & img) = 0;
    virtual std::vector<BBox> get_face_bboxes(std::vector<BBox> & bboxes) {return bboxes;};
    virtual std::vector<BBox> get_people_bboxes(std::vector<BBox> & bboxes) {return bboxes;};
    virtual std::vector<BBox> combine_bboxes_f2p(std::vector<BBox> & bboxes) {return bboxes;};
    virtual std::vector<BBox> combine_bboxes_p2f(std::vector<BBox> & bboxes) {return bboxes;};
};

#endif

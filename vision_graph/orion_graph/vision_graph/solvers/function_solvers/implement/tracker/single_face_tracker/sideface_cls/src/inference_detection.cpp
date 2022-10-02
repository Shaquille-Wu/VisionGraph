#include "inference_detection.h"

#include "timer.h"

static void convert2BBox(const vision::Box& src, BBox& des) {
    des.x1 = src.x1;
    des.x2 = src.x2;
    des.y1 = src.y1;
    des.y2 = src.y2;
    if (src.cls == 0)
        des.class_id = FACE;
    else {
        des.class_id = BODY;
    }
}

InferenceDetection::InferenceDetection() {
    // detect_ptr = NULL;
}

InferenceDetection::InferenceDetection(vision_graph::Solver* solver, int w, int h) {

    detect_ptr = solver;
    tensor_box = new vision_graph::TensorBoxesMap();
    tensor_image = new vision_graph::TensorImage();
}

InferenceDetection::~InferenceDetection() {
    if (tensor_box) {
        delete tensor_box;
        tensor_box = nullptr;
    }
    if (tensor_image) {
        delete tensor_image;
        tensor_image = nullptr;
    }
}

std::vector<BBox> InferenceDetection::detect(const cv::Mat& img) {
    std::vector<vision::Box> fp_vec;
    std::map<std::string, std::vector<vision::Box>> fp_vec_dlcv;

    *tensor_image = img;

    std::vector<vision_graph::Tensor*>  in;
    in.push_back(tensor_image);
    std::vector<vision_graph::Tensor*> out;
    out.push_back(tensor_box);

    detect_ptr->Solve(in, out);

    fp_vec_dlcv = *tensor_box;
    std::map<std::string, std::vector<Box>>::iterator iter;
    for (iter = fp_vec_dlcv.begin(); iter != fp_vec_dlcv.end(); iter++) {
        std::vector<Box> box = iter->second;
        fp_vec.insert(fp_vec.end(), box.begin(), box.end());
    }
    std::vector<BBox> ret(fp_vec.size());
    for (int i = 0; i < fp_vec.size(); ++i) {
        convert2BBox(fp_vec[i], ret[i]);
    }
#ifdef PRINT_TIME_COST
    //    DLOG(INFO)<<"detect time cost is "<<clock.GetTime()/1000<<" ms";
#endif
    return ret;
}

std::vector<BBox> InferenceDetection::combine_bboxes_f2p(std::vector<BBox>& bboxes) {
    std::vector<BBox> face_bboxes;
    std::vector<BBox> people_bboxes;
    for (size_t i = 0; i < bboxes.size(); i++) {
        BBox box = bboxes[i];
        if (box.class_id == BODY)
            people_bboxes.push_back(box);
        if (box.class_id == FACE)
            face_bboxes.push_back(box);
    }

    for (size_t i = 0; i < people_bboxes.size(); i++) {
        BBox& people_box = people_bboxes[i];
        cv::Rect people_rect = bbox2rect(people_box);
        people_box.is_attached = false;
        float max_iou = 0;
        int max_id = -1;
        for (size_t j = 0; j < face_bboxes.size(); j++) {
            BBox face_box = face_bboxes[j];
            cv::Rect face_rect = bbox2rect(face_box);
            cv::Rect cross_rect = (face_rect & people_rect);
            if (face_rect.area() * 0.8 <= cross_rect.area() && face_rect.width * 4 >= people_rect.width) {
                float iou = cross_rect.area();
                if (iou > max_iou) {
                    max_iou = iou;
                    max_id = j;
                }
            }
        }

        if (max_id != -1) {
            people_box.attached_rect = bbox2rect(face_bboxes[max_id]);
            people_box.is_attached = true;
        }
    }
    return people_bboxes;
}

std::vector<BBox> InferenceDetection::combine_bboxes_p2f(std::vector<BBox>& bboxes) {
    std::vector<BBox> face_bboxes;
    std::vector<BBox> people_bboxes;
    for (size_t i = 0; i < bboxes.size(); i++) {
        BBox box = bboxes[i];
        if (box.class_id == BODY)
            people_bboxes.push_back(box);
        if (box.class_id == FACE)
            face_bboxes.push_back(box);
    }
    for (size_t i = 0; i < face_bboxes.size(); i++) {
        BBox& face_box = face_bboxes[i];
        cv::Rect face_rect = bbox2rect(face_box);
        face_box.is_attached = false;
        float max_iou = 0;
        int max_id = -1;
        for (size_t j = 0; j < people_bboxes.size(); j++) {
            BBox people_box = people_bboxes[j];
            cv::Rect people_rect = bbox2rect(people_box);
            cv::Rect cross_rect = (face_rect & people_rect);
            if (face_rect.area() * 0.8 <= cross_rect.area() && face_rect.width * 4 >= people_rect.width) {
                float iou = cross_rect.area();
                if (iou > max_iou) {
                    max_iou = iou;
                    max_id = j;
                }
            }
        }
        if (max_id != -1) {
            face_box.attached_rect = bbox2rect(people_bboxes[max_id]);
            face_box.is_attached = true;
        }
    }
    return face_bboxes;
}

std::vector<BBox> InferenceDetection::get_face_bboxes(std::vector<BBox>& bboxes) {
    std::vector<BBox> face_bboxes;
    for (size_t i = 0; i < bboxes.size(); i++) {
        BBox box = bboxes[i];
        if (box.class_id == FACE)
            face_bboxes.push_back(box);
    }
    return face_bboxes;
}

std::vector<BBox> InferenceDetection::get_people_bboxes(std::vector<BBox>& bboxes) {
    std::vector<BBox> people_bboxes;
    for (size_t i = 0; i < bboxes.size(); i++) {
        BBox box = bboxes[i];
        if (box.class_id == BODY)
            people_bboxes.push_back(box);
    }
    return people_bboxes;
}

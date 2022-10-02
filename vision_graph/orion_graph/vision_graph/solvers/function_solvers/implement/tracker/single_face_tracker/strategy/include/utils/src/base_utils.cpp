#include "base_utils.h"

#include <math.h>
#include <stdio.h>  // sprintf

bool file_type(const char *file, const char *type) {
    int len1 = strlen(file);
    int len2 = strlen(type);
    if (len2 >= len1)
        return 0;
    int i;
    for (i = 0; i < len2; i++) {
        if (file[len1 - 1 - i] != type[len2 - 1 - i])
            return 0;
    }
    return 1;
}

vision::Box getfacebox(std::vector<cv::Point2f> points) {
    float x1 = points[0].x, y1 = points[0].y;
    float x2 = points[0].x, y2 = points[0].y;
    for (int ix = 1; ix < 106; ++ix) {
        if (x1 > points[ix].x) {
            x1 = points[ix].x;
        }
        if (x2 < points[ix].x) {
            x2 = points[ix].x;
        }
        if (y1 > points[ix].y) {
            y1 = points[ix].y;
        }
        if (y2 < points[ix].y) {
            y2 = points[ix].y;
        }
    }
    vision::Box b;
    b.x1 = x1;
    b.y1 = y1;
    b.x2 = x2;
    b.y2 = y2;
    return b;
}

float calDistance_480(float width) {
    static float distance_cof[5] = {-4872.420140229189f, 1119.1057581403209f,
                                    307.1303608134672f, -98.30410997841248f,
                                    7.727243645589179f};
    // Normalize width
    width /= 480;
    float dist;
    if (width < 0.15f)
        dist = ((((distance_cof[0] * width + distance_cof[1]) * width + distance_cof[2]) *
                 width) +
                distance_cof[3]) *
                   width +
               distance_cof[4];
    else {
        dist = 1 / (width * 5.544561130411958f);
    }
    //        //LOGE("calDistance 480: width %f, view_width %d, dis %f", width, view_width, dist);
    return dist;
}

float calDistance_720(float width) {
    static float distance_cof[5] = {-3666.3427198961995f, 1100.056546470614f,
                                    127.96660184705027f, -69.12997108857867f,
                                    6.903147785271374f};
    // Normalize width
    width /= 720;
    float dist;
    if (width < 0.15f)
        dist = ((((distance_cof[0] * width + distance_cof[1]) * width + distance_cof[2]) *
                 width) +
                distance_cof[3]) *
                   width +
               distance_cof[4];
    else {
        dist = 1 / (width * 5.251388541053607f);
    }
    //        //LOGE("calDistance 720: width %f, view_width %d, dis %f", width, view_width, dist);
    return dist;
}

float sigmoid(float x) {
    return 1.0 / (exp(-x) + 1);
}
float box_area(vision::Box b) {
    return b.height() * b.width();
}
void bBoxToBox(std::vector<BBox> &bboxs, std::vector<vision::Box> &boxs) {
    auto sort_by_area = [](const vision::Box &b1, const vision::Box &b2) {
        return box_area(b1) > box_area(b2);
    };

    for (BBox bBox : bboxs) {
        vision::Box box;
        box.x1 = bBox.x1;
        box.y1 = bBox.y1;
        box.x2 = bBox.x2;
        box.y2 = bBox.y2;
        box.cls = bBox.class_id;
        boxs.push_back(box);
    }
    std::sort(boxs.begin(), boxs.end(), sort_by_area);
    int tid = 1;
    for (vision::Box &b : boxs) {
        b.tid = tid;
        tid++;
    }
}

static bool sort_cmp_ascending_func(const std::pair<float, int> &di, const std::pair<float, int> &dj) {
    if (di.first < dj.first)
        return true;
    else if (di.first > dj.first)
        return false;
    else if (di.second < dj.second)
        return true;
    else if (di.second > dj.second)
        return false;
    return true;
}

static bool sort_cmp_descending_func(const std::pair<float, int> &di, const std::pair<float, int> &dj) {
    if (di.first < dj.first)
        return false;
    else if (di.first > dj.first)
        return true;
    else if (di.second < dj.second)
        return false;
    else if (di.second > dj.second)
        return true;
    return true;
}

static std::vector<int> get_sort_ids(std::vector<float> vals, bool is_ascending) {
    std::vector<std::pair<float, int> > valids(vals.size());
    for (int i = 0; i < vals.size(); i++) {
        valids[i].first = vals[i];
        valids[i].second = i;
    }
    if (is_ascending)
        std::sort(valids.begin(), valids.end(), sort_cmp_ascending_func);
    else
        std::sort(valids.begin(), valids.end(), sort_cmp_descending_func);

    std::vector<int> ids(vals.size());
    int i = 0;
    for (std::vector<std::pair<float, int> >::iterator it = valids.begin(); it != valids.end(); it++) {
        ids[i++] = it->second;
    }
    return ids;
}

float bbox_dist(BBox &bbox1, BBox &bbox2) {
    float cx1 = (bbox1.x1 + bbox1.x2) / 2.0;
    float cy1 = (bbox1.y1 + bbox1.y2) / 2.0;
    float cx2 = (bbox2.x1 + bbox2.x2) / 2.0;
    float cy2 = (bbox2.y1 + bbox2.y2) / 2.0;
    float dist = sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
    return dist;
}

float bbox_iou(BBox &bbox1, BBox &bbox2) {
    float mx = fmin(bbox1.x1, bbox2.x1);
    float Mx = fmax(bbox1.x2, bbox2.x2);
    float my = fmin(bbox1.y1, bbox2.y1);
    float My = fmax(bbox1.y2, bbox2.y2);
    float w1 = bbox1.x2 - bbox1.x1;
    float h1 = bbox1.y2 - bbox1.y1;
    float w2 = bbox2.x2 - bbox2.x1;
    float h2 = bbox2.y2 - bbox2.y1;
    float uw = Mx - mx;
    float uh = My - my;
    float cw = w1 + w2 - uw;
    float ch = h1 + h2 - uh;
    if (cw <= 0 || ch <= 0)
        return 0;

    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float carea = cw * ch;
    float uarea = area1 + area2 - carea;
    return carea / uarea;
}

cv::Point2f bbox_center(const BBox &bbox) {
    return cv::Point2f(0.5f * (bbox.x1 + bbox.x2), 0.5f * (bbox.y1 + bbox.y2));
}

std::vector<BBox> nms(std::vector<BBox> boxes, float thresh) {
    if (boxes.empty())
        return boxes;
    int num_boxes = boxes.size();
    std::vector<float> det_confs(num_boxes);
    for (int i = 0; i < num_boxes; i++) det_confs[i] = boxes[i].det_conf;
    std::vector<int> sortIds = get_sort_ids(det_confs, false);

    std::vector<BBox> out_boxes;
    for (int i = 0; i < num_boxes; i++) {
        BBox box_i = boxes[sortIds[i]];
        if (box_i.det_conf > 0) {
            out_boxes.push_back(box_i);
            for (int j = i + 1; j < num_boxes; j++) {
                BBox &box_j = boxes[sortIds[j]];
                if (j != i && bbox_iou(box_i, box_j) > thresh)
                    box_j.det_conf = 0;
            }
        }
    }
    return out_boxes;
}

BBox rect2bbox(cv::Rect rect) {
    BBox box;
    box.x1 = rect.x;
    box.y1 = rect.y;
    box.x2 = rect.x + rect.width;
    box.y2 = rect.y + rect.height;
    return box;
}

cv::Rect bbox2rect(BBox box) {
    //    cv::Rect2d rect;
    cv::Rect rect;
    rect.x = (int)round(box.x1);
    rect.y = (int)round(box.y1);
    rect.width = (int)round(box.x2 - box.x1);
    rect.height = (int)round(box.y2 - box.y1);
    return rect;
}

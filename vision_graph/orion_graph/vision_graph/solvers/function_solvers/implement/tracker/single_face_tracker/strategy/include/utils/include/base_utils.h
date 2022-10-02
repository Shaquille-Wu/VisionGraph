#ifndef BASE_UTILS_H
#define BASE_UTILS_H

#include <box.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

enum Box_ty {
    FACE = 1,
    BODY = 2,
    HEAD = 3,
    HAND = 4,
    MAX
};
struct Color {
    Color() {
        r = 0;
        g = 0;
        b = 0;
    }
    Color(int rr, int gg, int bb) {
        r = rr;
        g = gg;
        b = bb;
    }
    int r, g, b;
};
#define RED_COLOR Color(255, 0, 0)
#define GREEN_COLOR Color(0, 255, 0)
#define BLUE_COLOR Color(0, 0, 255)
#define YELLO_COLOR Color(255, 255, 0)
#define BLACK_COLOR Color(0, 0, 0)

struct BBox {
    float x1, y1, x2, y2;
    float det_conf;
    float reid_conf;
    float reid_conf_hlf;
    int class_id;
    Color color;

    cv::Rect_<double> attached_rect;
    bool is_attached;

    bool track_lost;  // tracking status

    inline float width() {
        return x2 - x1;
    }

    inline float height() {
        return y2 - y1;
    }
};

bool file_type(const char *file, const char *type);
/*
class VideoCaptureExt {
public:
    VideoCaptureExt(const char *filename) {
        fp_ = NULL;
        if (file_type(filename, ".txt") || file_type(filename, ".list")) {
            fp_ = fopen(filename, "r");
        } else if (isdigit(filename[0])) {
            cap_ = cv::VideoCapture(atoi(filename));
        } else {
            cap_ = cv::VideoCapture(filename);
        }
    }

    VideoCaptureExt &operator>>(cv::Mat &image) {
        if (fp_) {
            char line[1000];
            if (fgets(line, 1000, fp_) != NULL) {
                line[strlen(line) - 1] = '\0';
                image = cv::imread(line, cv::IMREAD_UNCHANGED);
                cur_filename = line;
            }
        } else {
            cap_ >> image;
        }
        return *this;
    }

    bool isOpened() {
        return (cap_.isOpened() || fp_);
    }

public:
    cv::VideoCapture cap_;
    FILE *fp_;
    std::string cur_filename;
};
*/
float bbox_iou(BBox &bbox1, BBox &bbox2);
float bbox_dist(BBox &bbox1, BBox &bbox2);
cv::Point2f bbox_center(const BBox &bbox);
void bBoxToBox(std::vector<BBox> &bboxs, std::vector<vision::Box> &boxs);
float calDistance_720(float width);
float calDistance_480(float width);
vision::Box getfacebox(std::vector<cv::Point2f> points);
std::vector<BBox> nms(std::vector<BBox> boxes, float thresh);

BBox rect2bbox(cv::Rect rect);
cv::Rect bbox2rect(BBox box);

#endif

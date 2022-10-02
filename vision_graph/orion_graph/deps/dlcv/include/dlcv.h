#ifndef _DLCV_H_
#define _DLCV_H_

#include <map>
#include <memory>
#include <vector>

#include "box.h"

using namespace std;

namespace vision {

#define DLCV_VERSION_MAJOR 1
#define DLCV_VERSION_MINOR 1
#define DLCV_VERSION_TINY  0
#define DLCV_VERSION (DLCV_VERSION_MAJOR*10000 + DLCV_VERSION_MINOR*100 +DLCV_VERSION_TINY)

std::string get_dlcv_version_string();

/**
 * @brief element type of managed buffer
 *
 * format: SFWWWWWW
 * S: 1: signend 0: unsigned
 * F: 1: float point 0: integer
 * W: width in bits
 */
enum ElementType {
  //! uint8_t
  U8 = 0b00001000,
  //! int8_t
  I8 = 0b10001000,
  //! uint16_t
  U16 = 0b00010000,
  //! int16_t
  I16 = 0b10010000,
  //! uint32_t
  U32 = 0b00100000,
  //! int32_t
  I32 = 0b10100000,
  //! float
  F32 = 0b11100000
};


/**
 * @brief output structure of featuremap models, such as segmentation eco featuremap.
 */
class FeatureMap {
 public:
  //! shape of data
  std::vector<int> shape;
  //! element type of data
  ElementType type;
  //! pointer to data.
  std::weak_ptr<unsigned char> data;
};


class DLCVOut {
public:
    bool has_boxes = false;
    bool has_keypoints = false;
    bool has_feature = false;
    bool has_featuremaps = false;
    bool has_multiclass = false;

    std::vector<cv::Point> keypoints;
    std::vector<float> feature;
    std::vector<FeatureMap> featuremaps;
    std::map<std::string, std::vector<Box>> boxes_map;
    std::vector<std::vector<float> > multiclass;

    void reset()
    {
        has_boxes = false;
        has_keypoints = false;
        has_feature = false;
        has_featuremaps = false;
        has_multiclass = false;
    };
};

} // namespace vision

#endif

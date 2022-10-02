#ifndef _MULTICLASS_H_
#define _MULTICLASS_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "dlcv.h"

namespace vision {

/**
 * @brief End-to-end runner for Keypoints with Attributes models
 */
class KeypointsAttributes{
public:
  //! Keypoints use std::vector<cv::Point> as output type
  using output_type_keypoints = std::vector<cv::Point>;
  //! Attributes use std::vector<std::vector<float> > as output type
  using output_type_attributes = std::vector<std::vector<float> >;

  /**
   * @brief Construct a new KeypointsAttributes object via JSON config
   * 
   * @param config_file path to config file.
   */
  explicit KeypointsAttributes(std::string config_file);

  ~KeypointsAttributes();

  /**
   * @brief Run KeypointsAttributes model
   * 
   * @param image Input image.
   * @param keypoints reference to output keypoints
   * @param attributes reference to output attributes
   * @return int 
   */
  int run(cv::Mat &image, std::vector<cv::Point>& keypoints, std::vector<std::vector<float> > &attributes);

private:
  class Impl;
  Impl* pimpl;
};

} // namespace vision
#endif

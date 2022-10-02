#ifndef _REID_FEATURE_H_
#define _REID_FEATURE_H_

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "dlcv.h"

namespace vision{

class Feature {
public:
  //! Feature use std::vector<float> as output type
  using output_type = std::vector<float>;

  /**
   * @brief Construct a new Feature object via JSON config
   *
   * @param config_file path to config file.
   */
  explicit Feature(std::string config_file);

  ~Feature();

  /**
   * @brief Run Feature model
   * 
   * @param image Input image.
   * @param feature reference to output
   * @return int 
   */
  int run(cv::Mat &image, std::vector<float> &feature);

private:
  class Impl;
  Impl* pimpl;
};

}
#endif

#ifndef _FeaturemapRunner_H_
#define _FeaturemapRunner_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "dlcv.h"

namespace vision {

/**
 * @brief End-to-end runner for featuremap models, such as segmentataion 
 */
class FeaturemapRunner {
public:
  //! FeaturemapRunner use std::vector<FeatureMap> as output type
  using output_type = std::vector<FeatureMap>;

  /**
   * @brief Construct a new FeaturemapRunner object via JSON config
   * 
   * @param config_file path to config file.
   */
  explicit FeaturemapRunner(std::string config_file);

  ~FeaturemapRunner();

  /**
   * @brief Run featuremap model
   * 
   * @param image Input image.
   * @param featuremaps reference to output
   * @return int 
   */
  int run(cv::Mat &image, std::vector<FeatureMap>& featuremaps);

private:
  class Impl;
  Impl* pimpl;
};

} // namespace vision
#endif

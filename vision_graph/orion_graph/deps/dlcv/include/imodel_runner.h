#ifndef _IModelRunner_H_
#define _IModelRunner_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "dlcv.h"

namespace vision {

/**
 * @brief End-to-end general runner for models
 */
class IModelRunner {
public:
  //! IModelRunner use std::vector<FeatureMap> as output type
  using output_type = vision::DLCVOut;

  /**
   * @brief Construct a new IModelRunner object via JSON config
   * 
   * @param config_file path to config file.
   * @param jpatch json string to patch config file. see demo/test_imodel_runner.cpp
   */
  explicit IModelRunner(std::string config_file, std::string jpatch="");

  ~IModelRunner();

  /**
   * @brief Run model
   * 
   * @param image Input image.
   * @param DLCVOut reference to output
   * @return int 
   */
  int run(cv::Mat &image, vision::DLCVOut& out);

private:
  class Impl;
  Impl* pimpl;
};

} // namespace vision
#endif

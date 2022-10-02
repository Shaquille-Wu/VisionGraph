#ifndef _MULTICLASS_H_
#define _MULTICLASS_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "dlcv.h"

namespace vision {

/**
 * @brief End-to-end runner for multiclass models
 */
class Multiclass {
public:
  //! Multiclass use std::vector<std::vector<float> > as output type
  using output_type = std::vector<std::vector<float> >;

  /**
   * @brief Construct a new Multiclass object via JSON config
   * 
   * @param config_file path to config file.
   */
  explicit Multiclass(std::string config_file);

  ~Multiclass();

  /**
   * @brief Run multiclass model
   * 
   * @param image Input image.
   * @param multiclass reference to output
   * @return int 
   */
  int run(cv::Mat &image, std::vector<std::vector<float> > &multiclass);

private:
  class Impl;
  Impl* pimpl;
};

} // namespace vision
#endif

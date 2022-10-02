#ifndef _DETECTOR_H_ 
#define _DETECTOR_H_

#include "box.h"

namespace vision{

class Detector{
public:
    Detector(const std::string cfgfile);
    ~Detector();
    int run(cv::Mat& image, std::map<std::string, std::vector<Box> >& boxes_map, int frameid=-1);

private:
    class Impl;
    Impl* _pimpl; //Hide inner implements from interface
};


}
#endif

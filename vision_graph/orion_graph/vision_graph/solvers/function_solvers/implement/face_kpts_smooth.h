#ifndef FACE_KPTS_SMOOTH_H__
#define FACE_KPTS_SMOOTH_H__

#include "../../../include/graph_tensor.h"

namespace vision_graph{

class FaceKptsSmooth
{
public:
    FaceKptsSmooth() noexcept: kpts_(0) {;};
    virtual ~FaceKptsSmooth() noexcept { kpts_.clear() ; } ;

    static const int                   kKptsCount = 106;

    void                               Update(std::vector<cv::Point2f> const& in) noexcept ;
    std::vector<cv::Point2f> const&    GetProcKpts() const noexcept { return kpts_; };

protected:
    std::vector<cv::Point2f>           kpts_;
};//class FaceKptsSmooth

}//namespace vision_graph

#endif
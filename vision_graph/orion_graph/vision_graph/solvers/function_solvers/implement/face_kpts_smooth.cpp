#include "face_kpts_smooth.h"
#include <math.h>

namespace vision_graph{

void FaceKptsSmooth::Update(std::vector<cv::Point2f> const& in) noexcept
{
    int   i = 0 ;
    std::vector<float> ex(kKptsCount, 0.0f), ey(kKptsCount, 0.0f);
    for (i = 0; i < kKptsCount; ++i) 
    {
        ex[i] = in[i].x;
        ey[i] = in[i].y;
    }

    if(kpts_.size() <= 0)
        kpts_ = in;
    else 
    {
        double x = 1.1;
        float  k = 25.0;
        for (i = 0; i < kKptsCount; ++i) 
        {
            float kcx = 1.0f / (float)(pow(x, (k - fabsf(ex[i] - kpts_[i].x))) + 1.0f);
            float kcy = 1.0f / (float)(pow(x, (k - fabsf(ey[i] - kpts_[i].y))) + 1.0f);
            ex[i]       = kpts_[i].x + (ex[i] - kpts_[i].x) * kcx;
            ey[i]       = kpts_[i].y + (ey[i] - kpts_[i].y) * kcy;

            kpts_[i].x  = ex[i];
            kpts_[i].y  = ey[i];
        }
    }
}

}//namespace vision_graph
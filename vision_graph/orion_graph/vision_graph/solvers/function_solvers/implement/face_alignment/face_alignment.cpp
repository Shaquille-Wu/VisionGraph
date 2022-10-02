#include "face_alignment.h"

template<typename T>
int twoPointsAlignMatrix( const cv::Point_<T>& srcPointA ,
                          const cv::Point_<T>& srcPointB ,
                          const cv::Point_<T>& dstPointA ,
                          const cv::Point_<T>& dstPointB ,
                          cv::Mat & outmatrix )
{

    cv::Point_<T> srcAB = srcPointB - srcPointA ;
    cv::Point_<T> dstAB = dstPointB - dstPointA ;

    if (cv::norm(srcAB) == 0 || cv::norm(dstAB) == 0)
    {
        outmatrix = cv::Mat() ;
        return -1 ;
    }

    double lenSrcAB =  cv::norm(srcAB);
    double lenDstAB = cv::norm(dstAB);
    double scale = lenDstAB / lenSrcAB;

    double cosA = ( srcAB.x * dstAB.x + srcAB.y * dstAB.y ) / ( lenSrcAB * lenDstAB ) ;
    double sinA = ( srcAB.x * dstAB.y - srcAB.y * dstAB.x ) / ( lenSrcAB * lenDstAB ) ;

    outmatrix = cv::Mat( 2, 3 , CV_64F , cv::Scalar(0)) ;
    outmatrix.at<double>( 0 , 0 ) = scale * cosA ;
    outmatrix.at<double>( 0 , 1 ) = - scale * sinA ;
    outmatrix.at<double>( 0 , 2 ) = - scale * cosA * srcPointA.x + scale * sinA * srcPointA.y + dstPointA.x ;
    outmatrix.at<double>( 1 , 0 ) = scale * sinA ;
    outmatrix.at<double>( 1 , 1 ) = scale * cosA ;
    outmatrix.at<double>( 1 , 2 ) = - scale * sinA * srcPointA.x - scale * cosA * srcPointA.y + dstPointA.y;

    return 0 ;
}

cv::Mat face_align(const cv::Mat& src,
                   const std::vector<cv::Point2f>& kpts_vec,
                   cv::Rect face_rect,
                   cv::Size dst_sz,
                   int eye_h)
{
    assert(kpts_vec.size() == 106);

    cv::Point2f leyeCtr = kpts_vec[104] ;
    cv::Point2f reyeCtr = kpts_vec[105] ;
    cv::Point2f eyeCtr = (leyeCtr + reyeCtr) * 0.5f ;
    cv::Point2f mouthUp = (kpts_vec[84]+ kpts_vec[90] + kpts_vec[97] + kpts_vec[99]) * 0.25f ;
    eyeCtr.x += face_rect.x;
    eyeCtr.y += face_rect.y;
    mouthUp.x += face_rect.x;
    mouthUp.y += face_rect.y;

    float dst_x = dst_sz.width * 0.5f;
    cv::Point2f desPoint1(dst_x, eye_h);
    cv::Point2f desPoint2(dst_x, eye_h + 64) ;
    cv::Mat affineMat;
    twoPointsAlignMatrix(eyeCtr, mouthUp, desPoint1, desPoint2, affineMat);
    cv::Mat dst;
    cv::warpAffine(src, dst, affineMat, dst_sz, 1);
    return dst;
}


template<typename T>
int twoPointsAlignMatrix( const cv::Point_<T>& srcPointA ,
                         const cv::Point_<T>& srcPointB ,
                         const cv::Point_<T>& dstPointA ,
                         const cv::Point_<T>& dstPointB ,
                         std::vector<cv::Point_<T>> inPointx ,
                         cv::Mat & outmatrix ,
                         std::vector<cv::Point_<T>>& outPointx)
{       
        
    cv::Point_<T> srcAB = srcPointB - srcPointA ;
    cv::Point_<T> dstAB = dstPointB - dstPointA ;
        
    if (cv::norm(srcAB) == 0 || cv::norm(dstAB) == 0)
    {   
        outmatrix = cv::Mat() ;
        return -1 ;
    }   
        
    double lenSrcAB =  cv::norm(srcAB) ;
    double lenDstAB = cv::norm( dstAB ) ; 
    double scale = lenDstAB / lenSrcAB ;
        
    double cosA = ( srcAB.x * dstAB.x + srcAB.y * dstAB.y ) / ( lenSrcAB * lenDstAB ) ; 
    double sinA = ( srcAB.x * dstAB.y - srcAB.y * dstAB.x ) / ( lenSrcAB * lenDstAB ) ; 
        
    for(int i=0;i<(int)(inPointx.size());i++)
    {   
        cv::Point_<T> dst;
        dst.x = (inPointx[i].x-srcPointB.x)*cosA - (inPointx[i].y-srcPointB.y)*sinA+srcPointB.x;
        dst.y = (inPointx[i].x-srcPointB.x)*sinA + (inPointx[i].y-srcPointB.y)*cosA+srcPointB.y;
        outPointx.push_back(dst);
    }   
        
    outmatrix = cv::Mat( 2, 3 , CV_64F , cv::Scalar(0)) ;
    outmatrix.at<double>( 0 , 0 ) = scale * cosA ;
    outmatrix.at<double>( 0 , 1 ) = - scale * sinA ;
    outmatrix.at<double>( 0 , 2 ) = - scale * cosA * srcPointA.x + scale * sinA * srcPointA.y + dstPointA.x ;
    outmatrix.at<double>( 1 , 0 ) = scale * sinA ;
    outmatrix.at<double>( 1 , 1 ) = scale * cosA ;
    outmatrix.at<double>( 1 , 2 ) = - scale * sinA * srcPointA.x - scale * cosA * srcPointA.y + dstPointA.y;
        
    return 0 ; 
}


int face_align_compact(const cv::Mat& src,
                           const std::vector<cv::Point2f>& kpts_vec,
                           cv::Rect face_rect, cv::Mat &out_img)
{
    assert(kpts_vec.size() == 106);

    std::vector<cv::Point2f> pts(kpts_vec.size());
    for(int i=0 ;i < (int)(pts.size()); ++i)
    {
        pts[i].x = kpts_vec[i].x + face_rect.x;
        pts[i].y = kpts_vec[i].y + face_rect.y;
    }

	cv::Point2f leyeCtr = (pts[52] + pts[55]) * 0.5f;
    cv::Point2f reyeCtr = (pts[61] + pts[58]) * 0.5f;

    cv::Point2f eyeCtr = (leyeCtr + reyeCtr) * 0.5f;
    cv::Point2f mouthUp = (pts[86] + pts[87] + pts[88]) * 0.333333333f;
    float distance = sqrt(pow(mouthUp.x-eyeCtr.x,2) + pow(mouthUp.y-eyeCtr.y, 2));
    cv::Point2f desPoint1(mouthUp.x, mouthUp.y-distance);
    cv::Mat affineMat;
    std::vector<cv::Point2f> output;
    int suc = twoPointsAlignMatrix(eyeCtr, mouthUp, desPoint1, mouthUp, pts, affineMat, output);
    if(suc == -1)
        return -1;
    cv::Mat outimg;
    warpAffine(src, outimg, affineMat, src.size(), 1);
    //compact
    float x_min=outimg.cols, x_max=0, y_min=outimg.rows, y_max=0;
    for(int i=0;i<106;i++)
    {
        //DLOG(INFO) << output[i].x << " " << output[i].y;
        if(output[i].x < x_min)
            x_min=output[i].x;
        if(output[i].x > x_max)
            x_max=output[i].x;
        if(output[i].y < y_min)
            y_min=output[i].y;
        if(output[i].y > y_max)
            y_max=output[i].y;
    }

    float top_padding=(y_max-y_min) * 0.25f;
    if(y_min > top_padding)
        y_min -= top_padding;
    else
        y_min=0;

    if(y_max>outimg.rows)
        y_max=outimg.rows;
    if(x_min>5)
        x_min -= 5;
    else
        x_min = 0;

    if(outimg.cols-x_max>5)
        x_max += 5;
    else
        x_max = outimg.cols;
    
    cv::Rect result_rect(x_min,y_min,x_max-x_min,y_max-y_min);
    out_img = outimg(result_rect);
	return 0;
}
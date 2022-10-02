#include "FacialLandmarks.h"
#include "CComputSim.h"

CFacialLandmarks::CFacialLandmarks(void)
{
    m_ccsim = new CComputSim();
    CComputSim *ptr = (CComputSim*)m_ccsim;
    ptr->InitModel();
}

CFacialLandmarks::~CFacialLandmarks(void)
{
    if(m_ccsim)
    {
        CComputSim *ptr = (CComputSim*)m_ccsim;
        delete ptr;
    }
}

float CFacialLandmarks::compute_similarity(unsigned char* pdata , int w, int h, const std::vector<cv::Point2f>& keypts)
{
	CM13PT_KEY_POINT_2D locate_key_pt_last[CM13PT_face_3D_key_point_num];
	for(int i = 0; i < CM13PT_face_3D_key_point_num; i++)
    {
		locate_key_pt_last[i].x = keypts[i].x;
		locate_key_pt_last[i].y = keypts[i].y;
	}

    CComputSim *ptr = (CComputSim*)m_ccsim;
	ptr->setprevlandmark_prev_2D_pt(locate_key_pt_last);
	ptr->GetRegressionTrackLocateResult(pdata, w, h);
	float score = ptr->computeprehogfeature();
	return score;
}

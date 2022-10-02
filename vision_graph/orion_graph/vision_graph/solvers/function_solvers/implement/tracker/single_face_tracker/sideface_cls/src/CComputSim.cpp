#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CComputSim.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

// extern const float vision_graph::mean_face_kpts_61[];
// extern const float vision_graph::mean_face_hog_feat[];


CComputSim::CComputSim()
{
	prev_hog_feature = NULL;
	m_hog_feature = NULL;
	m_norm_image_reg = NULL;
	m_new_pt_xy = NULL;
}

CComputSim::~CComputSim(void)
{
	ReleaseModel();
}

bool CComputSim::InitModel()
{
	memset(new_key_norm_pt, 0, sizeof(CM13PT_KEY_POINT_2D)*point_num_seleted);
	int i,j;
//	FILE *fp;

	face_2D_key_point_num = CM13PT_face_3D_key_point_num;
	total_hog_feat_dim = HOG_FEATURE_ONE_POINT*point_num_seleted;
	m_hog_feature = (float *)malloc(sizeof(float)*total_hog_feat_dim);
	m_new_pt_xy = (float *)malloc(sizeof(float)*point_num_seleted*2);
	m_norm_image_reg = (unsigned char *)malloc(sizeof(unsigned char)*norm_regression_wd_ht*norm_regression_wd_ht);

//	fp = fopen(MeanFileFullname.c_str(), "r");
//	if(fp == NULL)
//		return false;
//	fscanf(fp, "%d\n", &i);
//	if(i != point_num_seleted)
//		return false;
//	for(i = 0;i < point_num_seleted; ++i) {
//		fscanf(fp, "%f	%f	", &mean_2D_pt[i].x, &mean_2D_pt[i].y);
//
//	}
//	fclose(fp);

    for(i = 0;i < point_num_seleted; ++i)
    {
        mean_2D_pt[i].x = vision_graph::mean_face_kpts_61[2 * i];
        mean_2D_pt[i].y = vision_graph::mean_face_kpts_61[2 * i + 1];
	}


	for(j = 0;j < point_num_seleted; ++j) {
		mean_2D_regression_pt[j].x = mean_2D_pt[j].x + norm_regression_wd_ht/2 - norm_texture_wd_ht/2;   // mean_2D_regression_pt 的坐标是基于norm_texture_wd_ht/2,所以要先减掉归零,在加norm_regression_wd_ht/2变成基于norm_regression_wd_ht/2的坐标
		mean_2D_regression_pt[j].y = mean_2D_pt[j].y + norm_regression_wd_ht/2 - norm_texture_wd_ht/2;
	}

	for(j = 0;j < point_num_seleted; ++j) {
		m_pt1_x[j] = mean_2D_regression_pt[j].x;
		m_pt1_y[j] = mean_2D_regression_pt[j].y;
	}

	prev_hog_feature = (float *)malloc(sizeof(float)*total_hog_feat_dim);
//	avg_hog_feature = (float *)malloc(sizeof(float)*total_hog_feat_dim);
	memset(prev_hog_feature, 0, sizeof(float)*total_hog_feat_dim);
//	memset(avg_hog_feature, 0, sizeof(float)*total_hog_feat_dim);

//	fp = fopen(AvgFeatureFileFullname.c_str(), "rb");
//	if(fp == NULL)
//	{
//		free(avg_hog_feature);
//		free(prev_hog_feature);
//		return false;
//	}
//
//	fread(avg_hog_feature, sizeof(float), total_hog_feat_dim, fp);
//	fclose(fp);

    avg_hog_feature = (float *)vision_graph::mean_face_hog_feat;
    return true;
}

void CComputSim::ReleaseModel()
{

	if(prev_hog_feature)
		free(prev_hog_feature);
	prev_hog_feature = NULL;
//	if(avg_hog_feature)
//		free(avg_hog_feature);
	avg_hog_feature = NULL;

	if(m_hog_feature){
		free(m_hog_feature);
		m_hog_feature = NULL;
	}
	if(m_norm_image_reg){
		free(m_norm_image_reg);
		m_norm_image_reg = NULL;
	}
	if(m_new_pt_xy)
	{
		free(m_new_pt_xy);
		m_new_pt_xy = NULL;
	}
}


void CComputSim::setprevlandmark_prev_2D_pt(CM13PT_KEY_POINT_2D* locate_key_pt_last)
{
	for( int i = 0; i < point_num_seleted; i++){
		prev_2D_pt[i].x = locate_key_pt_last[m_select_pt_num[i]].x;
		prev_2D_pt[i].y = locate_key_pt_last[m_select_pt_num[i]].y;

	}
}

void CComputSim::GetRegressionTrackLocateResult(unsigned char *image, int wd, int ht )
{
	float pt2_x[CM13PT_face_3D_key_point_num], pt2_y[CM13PT_face_3D_key_point_num];
	float rot_s_x = 1.f;
    float rot_s_y = 1.f;
    float move_x = 0.f;
    float move_y = 0.f;
	float rot_s_x_inv = 1.f;
    float rot_s_y_inv = 1.f;
    float move_x_inv = 0.f;
    float move_y_inv = 0.f;

	for(int j = 0;j < point_num_seleted; ++j) {
		//pt1_x[j] = mean_2D_regression_pt[j].x;
		//pt1_y[j] = mean_2D_regression_pt[j].y;
		pt2_x[j] = prev_2D_pt[j].x ;
		pt2_y[j] = prev_2D_pt[j].y ;

	}
	CM13PT_CalAffineTransformData_float(pt2_x, pt2_y,m_pt1_x, m_pt1_y, point_num_seleted,rot_s_x, rot_s_y, move_x, move_y);
	CM13PT_AffineTransformImage_Sam_Bilinear(rot_s_x, rot_s_y, move_x, move_y, m_norm_image_reg, norm_regression_wd_ht, norm_regression_wd_ht, image, ht, wd);
	CM13PT_CalAffineTransInv(rot_s_x, rot_s_y, move_x, move_y, rot_s_x_inv, rot_s_y_inv, move_x_inv, move_y_inv);
    //for debug
//    for (int i = 0; i < norm_regression_wd_ht*norm_regression_wd_ht; i++){
//        m_norm_image_reg[i] = i%200;
//    }

	m_HogFeatureCls.SetSourceImage(m_norm_image_reg, norm_regression_wd_ht, norm_regression_wd_ht);
    //cv::Mat matshow(norm_regression_wd_ht, norm_regression_wd_ht, CV_8U, m_norm_image_reg);


	for(int k = 0;k < point_num_seleted; ++k)
	{

		new_key_norm_pt[k].x = prev_2D_pt[k].x*rot_s_x_inv - prev_2D_pt[k].y*rot_s_y_inv + move_x_inv;
		new_key_norm_pt[k].y = prev_2D_pt[k].y*rot_s_x_inv + prev_2D_pt[k].x*rot_s_y_inv + move_y_inv;
        //cv::circle(matshow,  cv::Point(new_key_norm_pt[k].x,new_key_norm_pt[k].y), 2, cv::Scalar(0, 255, 0));
	}

    //cv::imshow("histNormlize", matshow);
    //cv::waitKey(-1);
}
//////////
float CComputSim::computeprehogfeature()
{
	float hog_similarity = 1.0;
	float weight_1 = 0, weight_2 = 0;
	float sum_multi = 0;

	int i,k;
	for(k = 0;k < point_num_seleted; ++k)
	{
		m_new_pt_xy[k*2] = new_key_norm_pt[k].x;
		m_new_pt_xy[k*2 + 1] = new_key_norm_pt[k].y;
        //DLOG(INFO)<<m_select_pt_num[k] ;
        //DLOG(INFO)<<m_new_pt_xy[k*2]<<" "<<m_new_pt_xy[k*2 + 1] ;
	}
	//
	m_HogFeatureCls.GetHogFeature_Interpolation(m_new_pt_xy, point_num_seleted, prev_hog_feature);

	for(i = 0;i < total_hog_feat_dim; ++i)
	{
		sum_multi += prev_hog_feature[i]*avg_hog_feature[i];
		weight_1 += prev_hog_feature[i]*prev_hog_feature[i];
		weight_2 += avg_hog_feature[i]*avg_hog_feature[i];
        //DLOG(INFO)<<i<<":"<<avg_hog_feature[i] ;
        //DLOG(INFO)<<i<<":"<<prev_hog_feature[i] ;

	}
	weight_1 = sqrt(weight_1);
	weight_2 = sqrt(weight_2);
	hog_similarity = sum_multi/(weight_1*weight_2+ 0.000000001f);
	//double start3 = util::GetCurrentMSTime();
	return hog_similarity;
}


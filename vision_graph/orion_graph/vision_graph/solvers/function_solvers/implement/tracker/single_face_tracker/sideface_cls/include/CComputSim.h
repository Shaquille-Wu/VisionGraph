#ifndef GRAPH_CCOMPUT_SIM_H
#define GRAPH_CCOMPUT_SIM_H

#include "string"
#include "face_hog_const.h"

#define  CM13PT_face_3D_key_point_num 106
#define D_NUM_POINTS 212
#define CM13PT_max(a,b)    (((a) > (b)) ? (a) : (b))
#define CM13PT_min(a,b)    (((a) < (b)) ? (a) : (b))
struct CM13PT_KEY_POINT_2D
{
    float x, y;
};
const int norm_texture_wd_ht = 64;
const int norm_regression_wd_ht = 84;
//const int norm_regression_wd_ht = 96;
const int point_num_seleted = 61;
const int HOG_FEATURE_ONE_POINT = 128;

#define MM_PI					3.1415926535897932384626433832795
#define DIV_M_PI				0.3183098861837906715377675267450
#define FEATURE_WINDOW_SIZE		16
#define DESC_NUM_BINS			8
#define FVSIZE					128
#define	FV_THRESHOLD			0.2
#define TOTALDIM 7808

class CHogFeatureCls
{
public:
	CHogFeatureCls();
	~CHogFeatureCls();
	
	void SetSourceImage(unsigned char *image, int wd, int ht);
	void ReleaseImageData();
	void GetHogFeature_Interpolation(float *key_points, int nkeyPt, float *hog_feature);
private:
	void BuildScaleSpace(unsigned char *image, int wd, int ht);	
	void ExtractKeypointDescriptors(float *key_points, int nkeyPt, float *hog_feature);
	void BuildInterpolatedGaussianTable(unsigned short *G, unsigned int size, float sigma);
	float gaussian2D(float x, float y, float sigma);

private:
	unsigned short* imgInterpolatedMagnitude;
	int* imgInterpolatedOrientation;

	int img_width, img_height;
	unsigned short *Gauss_weight;

	float **img_hog_feature;
	char *feat_cal_flag;

	float m_pkey_points_left_up[D_NUM_POINTS];
	float m_pkey_points_right_up[D_NUM_POINTS];
	float m_pkey_points_left_down[D_NUM_POINTS];
	float m_pkey_points_right_down[D_NUM_POINTS];

	float m_phog_feature_left_up[TOTALDIM];
	float m_phog_feature_right_up[TOTALDIM];
	float m_phog_feature_left_down[TOTALDIM];
	float m_phog_feature_right_down[TOTALDIM];

	unsigned int* m_weight;
};

//main class for SDM based pose tracking
//Each SDM model will befine a class, total three classes are used
class CComputSim
{
public:
	CComputSim(void);
	~CComputSim(void);
	
	//Mean 2D points
	CM13PT_KEY_POINT_2D mean_2D_pt[point_num_seleted];

private:

	const int m_select_pt_num[point_num_seleted] = {
			0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,
			34,35,36,37,38,39,40,41,
			44,45,46,
			49,
			52,53,54,55,56,57,58,59,
			60,61,62,63,
			72,75,
			80,81,82,83,84,85,87,89,
			90,91,93,95,97,98,99,
			101,102,103
	};
	//
	float* m_hog_feature;
	unsigned char* m_norm_image_reg;
	float* m_new_pt_xy;
	float m_pt1_x[point_num_seleted];
	float m_pt1_y[point_num_seleted];
	float *avg_hog_feature;
	CHogFeatureCls m_HogFeatureCls;
	int total_hog_feat_dim;
	CM13PT_KEY_POINT_2D mean_2D_regression_pt[point_num_seleted];
	int face_2D_key_point_num;
	float *prev_hog_feature;
	CM13PT_KEY_POINT_2D prev_2D_pt[point_num_seleted];
	CM13PT_KEY_POINT_2D new_key_norm_pt[point_num_seleted];
public:
	bool InitModel();
	void ReleaseModel();
	float computeprehogfeature( );
	void GetRegressionTrackLocateResult(unsigned char *image, int wd, int ht);
	void setprevlandmark_prev_2D_pt(CM13PT_KEY_POINT_2D* locate_key_pt_last);

};
void CM13PT_CalAffineTransInv(float rot_x, float rot_y, float move_x, float move_y, float &rot_x_inv, float &rot_y_inv, float &move_x_inv, float &move_y_inv);
void CM13PT_CalAffineTransformData_float(float *pt1_x, float *pt1_y, float *pt2_x, float *pt2_y, int npt, float &rot_s_x, float &rot_s_y, float &move_x, float &move_y);
void CM13PT_AffineTransformImage_Sam_Bilinear(float rot_s_x, float rot_s_y, float move_x, float move_y, unsigned char *image, int ht, int wd, unsigned char *ori_image, int oriht, int oriwd);
bool CM13PT_MatrixInverse(float *m1, int row1, int col1);
bool CM13PT_MatrixMulti(float *m1, int row1, int col1, float *m2, int row2, int col2, float *m3);
bool CM13PT_MatrixTranspose(float *m1, int row1, int col1, float *m2);
void CM13PT_CalSobelImage_X(unsigned char *image, short *sobel_x, int wd, int ht);
void CM13PT_CalSobelImage_Y(unsigned char *image, short *sobel_y, int wd, int ht);

#endif

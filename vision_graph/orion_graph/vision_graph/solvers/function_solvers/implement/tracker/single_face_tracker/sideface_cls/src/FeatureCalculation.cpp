
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "CComputSim.h"
#include <iostream>

using namespace std;

CHogFeatureCls::CHogFeatureCls() {
    Gauss_weight = (unsigned short *) malloc(
            sizeof(unsigned short) * FEATURE_WINDOW_SIZE * FEATURE_WINDOW_SIZE);
    BuildInterpolatedGaussianTable(Gauss_weight, FEATURE_WINDOW_SIZE, 0.5 * FEATURE_WINDOW_SIZE);
    imgInterpolatedMagnitude = NULL;
    imgInterpolatedOrientation = NULL;
    img_hog_feature = (float **) malloc(
            sizeof(float *) * norm_regression_wd_ht * norm_regression_wd_ht);
    for (int i = 0; i < norm_regression_wd_ht * norm_regression_wd_ht; ++i)
        img_hog_feature[i] = (float *) malloc(sizeof(float) * FVSIZE);
    feat_cal_flag = (char *) malloc(sizeof(char) * norm_regression_wd_ht * norm_regression_wd_ht);
    imgInterpolatedMagnitude = (unsigned short *) malloc(
            sizeof(unsigned short) * norm_regression_wd_ht * norm_regression_wd_ht);
    imgInterpolatedOrientation = (int *) malloc(
            sizeof(int) * norm_regression_wd_ht * norm_regression_wd_ht);
    m_weight = (unsigned int *) malloc(sizeof(int) * FEATURE_WINDOW_SIZE * FEATURE_WINDOW_SIZE);
}

CHogFeatureCls::~CHogFeatureCls() {

    ReleaseImageData();
    if (Gauss_weight) {
        free(Gauss_weight);
        Gauss_weight = NULL;
    }


    for (int i = 0; i < norm_regression_wd_ht * norm_regression_wd_ht; ++i) {
        if (img_hog_feature[i]) {
            free(img_hog_feature[i]);
            img_hog_feature[i] = NULL;
        }

    }
    if (img_hog_feature) {
        free(img_hog_feature);
        img_hog_feature = NULL;
    }

    if (feat_cal_flag) {
        free(feat_cal_flag);
        feat_cal_flag = NULL;
    }

    if (imgInterpolatedOrientation) {
        free(imgInterpolatedOrientation);
        imgInterpolatedOrientation = NULL;
    }
    if (imgInterpolatedMagnitude) {
        free(imgInterpolatedMagnitude);
        imgInterpolatedMagnitude = NULL;
    }
    if (m_weight) {
        free(m_weight);
        m_weight = NULL;
    }
}

void CHogFeatureCls::SetSourceImage(unsigned char *image, int wd, int ht) {
    img_width = wd;
    img_height = ht;
    memset(imgInterpolatedMagnitude, 0, sizeof(unsigned short) * wd * ht);
    memset(imgInterpolatedOrientation, 0, sizeof(int) * wd * ht);
    memset(feat_cal_flag, 0, sizeof(char) * norm_regression_wd_ht * norm_regression_wd_ht);
    BuildScaleSpace(image, wd, ht);
}

void CHogFeatureCls::ReleaseImageData() {
    if (imgInterpolatedMagnitude)
        free(imgInterpolatedMagnitude);
    imgInterpolatedMagnitude = NULL;
    if (imgInterpolatedOrientation)
        free(imgInterpolatedOrientation);
    imgInterpolatedOrientation = NULL;
}


void
CHogFeatureCls::GetHogFeature_Interpolation(float *key_points, int nkeyPt, float *hog_feature) {
    float Error = 0.00001;
    float temp, b1, b2, b3, b4;
    for (int i = 0; i < nkeyPt; i++) {
        m_pkey_points_left_up[2 * i] = (int) key_points[2 * i];
        m_pkey_points_left_down[2 * i] = (int) key_points[2 * i];

        temp = key_points[2 * i] - (int) (key_points[2 * i]);
        if (temp < Error) {
            m_pkey_points_right_up[2 * i] = (int) (key_points[2 * i]);
            m_pkey_points_right_down[2 * i] = (int) (key_points[2 * i]);
        } else {
            m_pkey_points_right_up[2 * i] = (int) (key_points[2 * i] + 1);
            m_pkey_points_right_down[2 * i] = (int) (key_points[2 * i] + 1);
        }

        m_pkey_points_left_up[2 * i + 1] = (int) key_points[2 * i + 1];
        m_pkey_points_right_up[2 * i + 1] = (int) (key_points[2 * i + 1]);

        temp = key_points[2 * i + 1] - (int) (key_points[2 * i + 1]);
        if (temp < Error) {
            m_pkey_points_left_down[2 * i + 1] = (int) key_points[2 * i + 1];
            m_pkey_points_right_down[2 * i + 1] = (int) key_points[2 * i + 1];
        } else {
            m_pkey_points_left_down[2 * i + 1] = (int) (key_points[2 * i + 1] + 1);
            m_pkey_points_right_down[2 * i + 1] = (int) (key_points[2 * i + 1] + 1);
        }

    }
    ExtractKeypointDescriptors(m_pkey_points_left_up, nkeyPt, m_phog_feature_left_up);
    ExtractKeypointDescriptors(m_pkey_points_right_up, nkeyPt, m_phog_feature_right_up);
    ExtractKeypointDescriptors(m_pkey_points_left_down, nkeyPt, m_phog_feature_left_down);
    ExtractKeypointDescriptors(m_pkey_points_right_down, nkeyPt, m_phog_feature_right_down);
    //todo
    for (int i = 0; i < nkeyPt; i++) {
        float x = key_points[2 * i] - (int) (key_points[2 * i]);
        float y = key_points[2 * i + 1] - (int) (key_points[2 * i + 1]);
        for (int j = 0; j < FVSIZE; j++) {
            b1 = m_phog_feature_left_up[i * FVSIZE + j];
            b2 = m_phog_feature_right_up[i * FVSIZE + j] - b1;
            b3 = m_phog_feature_left_down[i * FVSIZE + j] - b1;
            b4 = m_phog_feature_right_down[i * FVSIZE + j] - b2 - b3 - b1;
            hog_feature[i * FVSIZE + j] = b1 + b2 * x + b3 * y + b4 * x * y;
        }
    }

}

//void CHogFeatureCls::BuildScaleSpace(unsigned char *image, int wd, int ht)
//{
//	Sobel_Neon_XY(image, wd, ht, imgInterpolatedMagnitude, imgInterpolatedOrientation);
//}

void CHogFeatureCls::BuildScaleSpace(unsigned char *image, int wd, int ht) {
    int i;
    short *sobel_image_x, *sobel_image_y;

    sobel_image_x = (short *) malloc(sizeof(short) * wd * ht);
    sobel_image_y = (short *) malloc(sizeof(short) * wd * ht);

    CM13PT_CalSobelImage_X(image, sobel_image_x, wd, ht);
    CM13PT_CalSobelImage_Y(image, sobel_image_y, wd, ht);

    for (i = 0; i < wd * ht; i++) {
        short dx = sobel_image_x[i];
        short dy = sobel_image_y[i];

        imgInterpolatedMagnitude[i] = int(sqrt(float(dx * dx + dy * dy)) + 0.5);
        float theta = atan2(float(dy), float(dx));
        //cout<<theta<<endl;
        int sample_orien;
        if (theta < 0)
            theta += 2 * MM_PI;

        sample_orien = int(theta * 4 * DIV_M_PI);                    // The bin
        //if(sample_orien >= DESC_NUM_BINS)
        //	sample_orien = 0;
        imgInterpolatedOrientation[i] = sample_orien;
    }
    if (sobel_image_x)
        free(sobel_image_x);
    if (sobel_image_y)
        free(sobel_image_y);
}


void CHogFeatureCls::ExtractKeypointDescriptors(float *key_points, int nkeyPt, float *hog_feature) {

    int width = img_width;
    // int height = img_height;
    int i, j, k, x, y, ipt, n;
    int dis_x, dis_y;
    int pt_x, pt_y;


    int search_block_num = 4;
    int search_block_size = 4;
    int half_window_size = 8;
    float fv[FVSIZE];
    unsigned int hist[DESC_NUM_BINS];
    int feat_index = 0;
    for (ipt = 0; ipt < nkeyPt; ipt++) {
        //double start = util::GetCurrentMSTime();
        pt_x = key_points[ipt * 2];
        pt_y = key_points[ipt * 2 + 1];
        if (pt_x < half_window_size - 1) pt_x = half_window_size - 1;
        if (pt_x >= width - half_window_size) pt_x = width - half_window_size - 1;
        if (pt_y < half_window_size - 1) pt_y = half_window_size - 1;
        if (pt_y >= width - half_window_size) pt_y = width - half_window_size - 1;


        if (feat_cal_flag[pt_y * norm_regression_wd_ht + pt_x]) {
            memcpy(hog_feature + ipt * FVSIZE, img_hog_feature[pt_y * norm_regression_wd_ht + pt_x],
                   sizeof(float) * FVSIZE);

            continue;
        }
        n = 0;
        dis_y = pt_y + 1 - half_window_size;
        dis_x = pt_x + 1 - half_window_size;

        for (y = 0; y < FEATURE_WINDOW_SIZE; y++) {
            int temp_pos = (dis_y + y) * width + dis_x;
            for (x = 0; x < FEATURE_WINDOW_SIZE; x++) {
                m_weight[n] = Gauss_weight[n] * imgInterpolatedMagnitude[temp_pos + x];
                n++;
            }
        }

        int n = 0;
        for (i = 0; i < search_block_num; i++)            // 4x4 thingy
        {
            for (j = 0; j < search_block_num; j++) {
                // Clear the histograms
                memset(hist, 0, sizeof(unsigned int) * DESC_NUM_BINS);

                // Calculate the coordinates of the 4x4 block
                int startx = pt_x - half_window_size + 1 + search_block_size * i;
                int limitx = startx + search_block_size;
                int starty = pt_y - half_window_size + 1 + search_block_size * j;
                int limity = starty + search_block_size;

                // Go though this 4x4 block and do the thingy :D
                for (y = starty; y < limity; y++) {
                    int temp_pos = (y - dis_y) * FEATURE_WINDOW_SIZE - dis_x;

                    for (x = startx; x < limitx; x++) {

                        hist[imgInterpolatedOrientation[y * width + x]] += m_weight[temp_pos + x];

                    }
                    //
                    //
                }

                for (k = 0; k < DESC_NUM_BINS; ++k) {
                    fv[n] = hist[k];
                    n++;
                }
            }
        }

        float norm = 0;
        float sum_norm2 = 0;
        float norm_thres = 0, norm_thres_2;

        for (i = 0; i < FVSIZE; i++)
            sum_norm2 += fv[i] * fv[i];
        if (sum_norm2 < 0.00001) {
            sum_norm2 = 0.00001;
        }

        norm = sqrt(sum_norm2);


        norm_thres = FV_THRESHOLD * norm;
        norm_thres_2 = norm_thres * norm_thres;
        // Now, threshold the vector
        for (i = 0; i < FVSIZE; i++) {
            if (fv[i] > norm_thres) {
                sum_norm2 += norm_thres_2 - fv[i] * fv[i];
                fv[i] = norm_thres;
            }
        }

        if (sum_norm2 < 0.00001) {
            sum_norm2 = 0.00001;
        }
        norm = 1.0 / sqrt(sum_norm2);


        feat_index = ipt * FVSIZE;
        for (i = 0; i < FVSIZE; i++) {
            hog_feature[feat_index] = fv[i] * norm;
            feat_index++;
        }
        feat_cal_flag[pt_y * norm_regression_wd_ht + pt_x] = 1;
        memcpy(img_hog_feature[pt_y * norm_regression_wd_ht + pt_x], hog_feature + ipt * FVSIZE,
               sizeof(float) * FVSIZE);

    }

}

void
CHogFeatureCls::BuildInterpolatedGaussianTable(unsigned short *G, unsigned int size, float sigma) {
    // 构造函数调一次。
    unsigned int i, j;
    float half_kernel_size = size / 2 - 0.5;
    float *float_G_value;
    //float sog=0;
    float temp = 0;

    float_G_value = (float *) malloc(sizeof(float) * size * size);

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            temp = gaussian2D(j - half_kernel_size, i - half_kernel_size, sigma);
            float_G_value[i * size + j] = temp;
            //sog+=temp;
        }
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            G[i * size + j] = int(
                    float_G_value[i * size + j] * 1024 * 1024 + 0.5);//1.0/sog * G[i*size + j];
        }
    }
    if (float_G_value)
        free(float_G_value);
}

float CHogFeatureCls::gaussian2D(float x, float y, float sigma) {
    float ret = 1.0 / (2 * MM_PI * sigma * sigma) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));
    return ret;
}


bool CM13PT_MatrixInverse(float *m1, int row1, int col1) {
    int i, j, k;
    float div, temp;
    float *out;
    int *is, *js;

    if (row1 != col1)
        return false;

    out = (float *) malloc(sizeof(float) * row1 * col1);
    is = (int *) malloc(sizeof(int) * row1);
    js = (int *) malloc(sizeof(int) * row1);
    for (i = 0; i < row1; ++i) {
        is[i] = i;
        js[i] = i;
    }

    // start from first column to the next
    for (k = 0; k < row1; ++k) {
        div = 0;
        for (i = k; i < row1; ++i)
            for (j = k; j < row1; ++j) {
                if (fabs(m1[i * col1 + j]) > div) {
                    div = fabs(m1[i * col1 + j]);
                    is[k] = i;
                    js[k] = j;
                }
            }
        if (fabs(div) < 1e-40) {
            if (out)
                free(out);
            if (is)
                free(is);
            if (js)
                free(js);
            return false;
        }
        if (is[k] != k) {
            for (j = 0; j < row1; ++j) {
                temp = m1[k * col1 + j];
                m1[k * col1 + j] = m1[is[k] * col1 + j];
                m1[is[k] * col1 + j] = temp;
            }
        }
        if (js[k] != k) {
            for (i = 0; i < row1; ++i) {
                temp = m1[i * col1 + k];
                m1[i * col1 + k] = m1[i * col1 + js[k]];
                m1[i * col1 + js[k]] = temp;
            }
        }
        m1[k * col1 + k] = 1 / m1[k * col1 + k];
        for (j = 0; j < row1; ++j) {
            if (j != k)
                m1[k * col1 + j] = m1[k * col1 + j] * m1[k * col1 + k];
        }
        for (i = 0; i < row1; ++i) {
            if (i != k) {
                for (j = 0; j < row1; ++j) {
                    if (j != k)
                        m1[i * col1 + j] -= m1[i * col1 + k] * m1[k * col1 + j];
                }
            }
        }
        for (i = 0; i < row1; ++i) {
            if (i != k)
                m1[i * col1 + k] = -m1[i * col1 + k] * m1[k * col1 + k];
        }
    }

    for (k = row1 - 1; k >= 0; --k) {
        for (j = 0; j < row1; ++j)
            if (js[k] != k) {
                temp = m1[k * col1 + j];
                m1[k * col1 + j] = m1[js[k] * col1 + j];
                m1[js[k] * col1 + j] = temp;
            }
        for (i = 0; i < row1; ++i)
            if (is[k] != k) {
                temp = m1[i * col1 + k];
                m1[i * col1 + k] = m1[i * col1 + is[k]];
                m1[i * col1 + is[k]] = temp;
            }
    }
    if (out)
        free(out);
    if (is)
        free(is);
    if (js)
        free(js);
    return true;
}

bool CM13PT_MatrixMulti(float *m1, int row1, int col1, float *m2, int row2, int col2,
                        float *m3) {
    int i, j, k;

    for (i = 0; i < row1; ++i) {
        int tetem = i * col1;
        for (j = 0; j < col2; ++j) {
            float sum = 0;
            for (k = 0; k < col1; ++k)
                sum += m1[tetem + k] * m2[k * col2 + j];
            m3[i * col2 + j] = sum;
        }
    }

    return true;
}


bool CM13PT_MatrixTranspose(float *m1, int row1, int col1, float *m2) {
    int i, j;
    if (m2 == NULL) {
        float *m3;

        m3 = (float *) malloc(sizeof(float) * row1 * col1);
        for (i = 0; i < col1; ++i)
            for (j = 0; j < row1; ++j) {
                m3[i * row1 + j] = m1[j * col1 + i];
            }
        for (i = 0; i < row1; ++i)
            for (j = 0; j < col1; ++j)
                m1[i * col1 + j] = m3[j * col1 + i];
        if (m3)
            free(m3);
    } else {
        for (i = 0; i < col1; ++i)
            for (j = 0; j < row1; ++j) {
                m2[i * row1 + j] = m1[j * col1 + i];
            }
    }
    return true;
}


void
CM13PT_CalAffineTransformData_float(float *pt1_x, float *pt1_y, float *pt2_x, float *pt2_y, int npt,
                                    float &rot_s_x, float &rot_s_y,
                                    float &move_x, float &move_y) {
    float *X, *A, *B;
    float *temp, *TA;
    int nDim = 4, nrow = npt * 2;
    int i, ii;
    int n1, n2;

    X = (float *) malloc(sizeof(float) * nDim);
    A = (float *) malloc(sizeof(float) * npt * nDim * 2);
    TA = (float *) malloc(sizeof(float) * npt * nDim * 2);
    B = (float *) malloc(sizeof(float) * npt * 2);
    temp = (float *) malloc(sizeof(float) * nDim * nDim);

    for (i = 0; i < npt; ++i) {
        ii = (i << 1);
        n1 = ii * nDim;
        n2 = (ii + 1) * nDim;
        B[ii] = pt1_x[i];
        B[ii + 1] = pt1_y[i];
        A[n1] = pt2_x[i];
        A[n1 + 1] = -pt2_y[i];
        A[n1 + 2] = 1;
        A[n1 + 3] = 0;
        A[n2] = pt2_y[i];
        A[n2 + 1] = pt2_x[i];
        A[n2 + 2] = 0;
        A[n2 + 3] = 1;
    }


    CM13PT_MatrixTranspose(A, nrow, nDim, TA);
    CM13PT_MatrixMulti(TA, nDim, nrow, A, nrow, nDim, temp);
    CM13PT_MatrixInverse(temp, nDim, nDim);
    CM13PT_MatrixMulti(TA, nDim, nrow, B, nrow, 1, A);
    CM13PT_MatrixMulti(temp, nDim, nDim, A, nDim, 1, X);

    rot_s_x = X[0];
    rot_s_y = X[1];
    move_x = X[2];
    move_y = X[3];
    if (TA)
        free(TA);
    if (X)
        free(X);
    if (A)
        free(A);
    if (B)
        free(B);
    if (temp)
        free(temp);
}

void
CM13PT_AffineTransformImage_Sam_Bilinear(float rot_s_x, float rot_s_y, float move_x, float move_y,
                                         unsigned char *image, int ht, int wd,
                                         unsigned char *ori_image, int oriht, int oriwd) {
    int i, j;
    float x1, y1;
    float *rx, *ry;
    int max_ht_wd = CM13PT_max(ht, wd) + 1;
    float tx1, ty1;

    rx = (float *) malloc(sizeof(float) * max_ht_wd);
    ry = (float *) malloc(sizeof(float) * max_ht_wd);
    for (i = 0; i < max_ht_wd; ++i)
        rx[i] = rot_s_x * i;
    if (rot_s_y == 0)
        memset(ry, 0, sizeof(int) * max_ht_wd);
    else {
        for (i = 0; i < max_ht_wd; ++i)
            ry[i] = rot_s_y * i;
    }


    //todo can be optimaized.

    for (i = 0; i < ht; ++i) {
        tx1 = -ry[i] + move_x;
        ty1 = rx[i] + move_y;
        for (j = 0; j < wd; ++j) {
            x1 = rx[j] + tx1;
            y1 = ry[j] + ty1;
            image[i * wd + j] = 0;
            if (x1 < 0 || y1 < 0 || x1 >= oriwd - 1 || y1 >= oriht - 1) {
                continue;
            }
            int x_int = int(x1), y_int = int(y1);
            float x_tail = x1 - x_int, y_tail = y1 - y_int;
            int x_round = x_int + 1, y_round = y_int + 1;


            float pixel1 = ori_image[y_int * oriwd + x_int] * (1 - x_tail) +
                           ori_image[y_int * oriwd + x_round] * x_tail;
            float pixel2 = ori_image[y_round * oriwd + x_int] * (1 - x_tail) +
                           ori_image[y_round * oriwd + x_round] * x_tail;

            image[i * wd + j] = int(pixel1 * (1 - y_tail) + pixel2 * y_tail + 0.5);
        }
    }
    if (rx)
        free(rx);
    if (ry)
        free(ry);
}


void
CM13PT_CalAffineTransInv(float rot_x, float rot_y, float move_x, float move_y, float &rot_x_inv,
                         float &rot_y_inv, float &move_x_inv, float &move_y_inv) {
    float temp = 1.0f / (rot_x * rot_x + rot_y * rot_y);
    rot_x_inv = temp * rot_x;
    rot_y_inv = -temp * rot_y;
    move_x_inv = 1 - (rot_x_inv * (rot_x + move_x) - rot_y_inv * (rot_y + move_y));
    move_y_inv = -(rot_y_inv * (rot_x + move_x) + rot_x_inv * (rot_y + move_y));

}


void CM13PT_CalSobelImage_X(unsigned char *image, short *sobel_x, int wd, int ht) {
    static int matrix_x[25] = {2, 1, 0, -1, -2,
                               3, 2, 0, -2, -3,
                               4, 3, 0, -3, -4,
                               3, 2, 0, -2, -3,
                               2, 1, 0, -1, -2};
    // int sobel_wd = 5, half_sobel_wd = 2;
    int half_sobel_wd = 2;
    int i, j, k, l, n;
    int x, y;
    int sobel_expand_wd = 2;

    memset(sobel_x, 0, sizeof(short) * wd * ht);

    for (i = sobel_expand_wd; i < ht - sobel_expand_wd; ++i)
        for (j = sobel_expand_wd; j < wd - sobel_expand_wd; ++j) {
            n = 0;
            for (k = -half_sobel_wd; k <= half_sobel_wd; ++k)
                for (l = -half_sobel_wd; l <= half_sobel_wd; ++l) {
                    x = j + l;
                    y = i + k;
                    sobel_x[i * wd + j] += -image[y * wd + x] * matrix_x[n];
                    n++;
                }
        }
}

void CM13PT_CalSobelImage_Y(unsigned char *image, short *sobel_y, int wd, int ht) {
    static int matrix_y[25] = {2, 3, 4, 3, 2,
                               1, 2, 3, 2, 1,
                               0, 0, 0, 0, 0,
                               -1, -2, -3, -2, -1,
                               -2, -3, -4, -3, -2};
    // int sobel_wd = 5, half_sobel_wd = 2;
    int half_sobel_wd = 2;
    int i, j, k, l, n;
    int x, y;
    int sobel_expand_wd = 2;

    memset(sobel_y, 0, sizeof(short) * wd * ht);

    for (i = sobel_expand_wd; i < ht - sobel_expand_wd; ++i)
        for (j = sobel_expand_wd; j < wd - sobel_expand_wd; ++j) {
            n = 0;
            for (k = -half_sobel_wd; k <= half_sobel_wd; ++k)
                for (l = -half_sobel_wd; l <= half_sobel_wd; ++l) {
                    x = j + l;
                    y = i + k;
                    sobel_y[i * wd + j] += -image[y * wd + x] *
                                           matrix_y[n];//(k + half_sobel_wd)*sobel_wd + l + half_sobel_wd];
                    n++;
                }

        }
}

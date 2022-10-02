//
// Created by An Wenju and yuan on 17-10-16.
//

#include "fourier_tools.h"
#include "utils.h"

//x is a single channel mat
cv::Mat circshift(cv::Mat x, int x_rot, int y_rot)
{
    cv::Mat circx = x.clone();
    cv::Mat tempx = x.clone();

    if(x_rot < 0)
    {
        cv::Range orig_range(-x_rot, x.cols);
        cv::Range rot_range(0, x.cols-(-x_rot));
        x(cv::Range::all(), orig_range).copyTo(tempx(cv::Range::all(), rot_range));

        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(x.cols-(-x_rot), x.cols);
        x(cv::Range::all(), orig_range).copyTo(tempx(cv::Range::all(), rot_range));
    }
    else if(x_rot > 0)
    {
        cv::Range orig_range(0, x.cols-x_rot);
        cv::Range rot_range(x_rot, x.cols);
        x(cv::Range::all(), orig_range).copyTo(tempx(cv::Range::all(), rot_range));

        orig_range = cv::Range(x.cols - x_rot, x.cols);
        rot_range = cv::Range(0, x_rot);
        x(cv::Range::all(), orig_range).copyTo(tempx(cv::Range::all(), rot_range));
    }
    else
    {
        cv::Range orig_range(0, x.cols);
        cv::Range rot_range(0, x.cols);
        x(cv::Range::all(), orig_range).copyTo(tempx(cv::Range::all(), rot_range));
    }

    if(y_rot < 0)
    {
        cv::Range orig_range(-y_rot, x.rows);
        cv::Range rot_range(0, x.rows-(-y_rot));
        tempx(orig_range, cv::Range::all()).copyTo(circx(rot_range, cv::Range::all()));

        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(x.rows-(-y_rot), x.rows);
        tempx(orig_range, cv::Range::all()).copyTo(circx(rot_range, cv::Range::all()));
    }
    else if(y_rot > 0)
    {
        cv::Range orig_range(0, x.rows-y_rot);
        cv::Range rot_range(y_rot, x.rows);
        tempx(orig_range, cv::Range::all()).copyTo(circx(rot_range, cv::Range::all()));

        orig_range = cv::Range(x.rows-y_rot, x.rows);
        rot_range = cv::Range(0, y_rot);
        tempx(orig_range, cv::Range::all()).copyTo(circx(rot_range, cv::Range::all()));
    }
    else
    {
        cv::Range orig_range(0, x.rows);
        cv::Range rot_range(0, x.rows);
        tempx(orig_range, cv::Range::all()).copyTo(circx(rot_range, cv::Range::all()));
    }

    return circx;
}


cv::Mat fftshift(const cv::Mat& x, int dim)
{
    int xwid = x.cols;
    int xhei = x.rows;
    // int xcha = x.channels();
    std::vector<cv::Mat> xshift_vec;

    std::vector<cv::Mat> x_vec;
    cv::split(x, x_vec);

    if(dim==1)
    {
        for(int i=0; i<x_vec.size(); i++)
        {
            cv::Mat xtmp;
            int xrot =floor((float)xwid/2.);
            xtmp = circshift(x_vec[i], xrot, 0 );
            xshift_vec.push_back(xtmp);
        }
    }
    else if(dim==0)
    {
        for(int i=0; i<x_vec.size();i++)
        {
            cv::Mat xtmp;
            int yrot = floor((float)xhei/2.);
            xtmp = circshift(x_vec[i], 0, yrot);
            xshift_vec.push_back(xtmp);
        }
    }

    cv::Mat xshift;
    if(xshift_vec.size() == 1)
        xshift = xshift_vec[0].clone();
    else
        cv::merge(xshift_vec, xshift);

    return xshift;

}


cv::Mat fftshift(const cv::Mat& src)
{
    return fftshift(fftshift(src, 0), 1);
}

// input: real, single channel; output: complex, double channel
void cfft2(const cv::Mat& x, cv::Mat& xf)
{
    int xwid = x.cols;
    int xhei = x.rows;

    cv::Mat complex_result_ori;
    cv::dft(x, complex_result_ori, cv::DFT_COMPLEX_OUTPUT);
    
    cv::Mat shift_result_v = fftshift(complex_result_ori, 0);
    cv::Mat shift_result = fftshift(shift_result_v, 1);

    cv::Mat complex_result(shift_result);
    if(xhei%2 == 0)
        cv::copyMakeBorder(shift_result, complex_result, 0, 1, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0));
    if(xwid%2 == 0)
        cv::copyMakeBorder(complex_result, complex_result, 0, 0, 0, 1,cv::BORDER_CONSTANT, cv::Scalar(0,0));

    if(xhei%2 == 0)
    {
        float* pfrom = complex_result.ptr<float>(0)+(complex_result.cols)*2-1;
        float* pto = complex_result.ptr<float>(complex_result.rows-1);
        for(int j=0; j<complex_result.cols; j++)
        {
            *pto = *(pfrom-1);
            pto++;
            *pto = -*pfrom;
            pto++;
            pfrom-=2;
        }
    }
    if(xwid%2 == 0)
    {
        for(int j=0; j<complex_result.rows; j++)
        {
            float* pfrom = complex_result.ptr<float>(complex_result.rows-j-1);
            float* pto = complex_result.ptr<float>(j)+(complex_result.cols)*2-2;
            *pto = *pfrom;
            pto++;
            pfrom++;
            *pto = -*pfrom;
        }
    }

    xf = std::move(complex_result);
//    xf = complex_result;
}

cv::Mat ifftshift(const cv::Mat& x, int dim)
{

    int xwid = x.cols;
    int xhei = x.rows;
    std::vector<cv::Mat> xshift_vec;
    std::vector<cv::Mat> x_vec;
    cv::split(x, x_vec);

    if(dim==1)
    {
        for(int i=0; i<x_vec.size(); i++)
        {
            cv::Mat xtmp;
            int xrot =ceil((float)xwid/2.);
            xtmp = circshift(x_vec[i], xrot, 0 );
            xshift_vec.push_back(xtmp);
        }
    }
    else if(dim==0)
    {
        for(int i=0; i<x_vec.size();i++)
        {
            cv::Mat xtmp;
            int yrot = ceil((float)xhei/2.);
            xtmp = circshift(x_vec[i], 0, yrot);
            xshift_vec.push_back(xtmp);
        }
    }

    cv::Mat xshift;
    cv::merge(xshift_vec, xshift);

    return xshift;
}


cv::Mat ifftshift(const cv::Mat& x)
{
    return ifftshift(ifftshift(x, 0), 1);
}


// input: complex, double channel; output: real, single channel
void cifft2(const cv::Mat& xf, cv::Mat& x)
{
//    cv::Mat xf_shift_v = ifftshift(xf, 0);
//    cv::Mat xf_shift = ifftshift(xf_shift_v, 1);
    cv::Mat xf_shift = ifftshift(xf);
    cv::dft(xf_shift, x, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
}

void full_fourier_coef(const cv::Mat& xf, cv::Mat& xf_full)
{
    cv::Mat xf_slice = xf.colRange(0, xf.cols - 1);
    cv::Mat xf_slice_rot;
    rot90(xf_slice, xf_slice_rot, ROTATE_180);
//    complexf_mat_conj(xf_slice_rot, xf_slice_rot);
    xf_slice_rot = ComplexMat(xf_slice_rot).conj().to_cv_mat();
    cv::hconcat(xf, xf_slice_rot, xf_full);
}

void compact_fourier_coef(const cv::Mat& xf, cv::Mat& xf_compact)
{
    cv::Mat xf_com;
    xf(cv::Rect(0, 0, (xf.cols+1)/2, xf.rows)).copyTo(xf_com);
    xf_compact = std::move(xf_com);
//    xf_compact = xf_com;
}

cv::Mat cos_mat(cv::Mat f)
{
    cv::Mat cosm = f.clone();
    for(int i=0; i<f.rows; i++)
    {
        float* pf = f.ptr<float>(i);
        float* pr = cosm.ptr<float>(i);
        for(int j=0; j<f.step/f.elemSize(); j++)
        {
            *pr++ = cos(*pf);
            pf++;
        }
    }
    return cosm;
}

cv::Mat sin_mat(cv::Mat f)
{
    cv::Mat sinm = f.clone();

    for(int i=0; i<f.rows; i++)
    {
        float* pf = f.ptr<float>(i);
        float* pi = sinm.ptr<float>(i);
        for(int j=0; j<f.step/f.elemSize(); j++)
        {
            *pi++ = sin(*pf);
            pf++;
        }
    }
    return sinm;
}

void cubic_spline_fourier(const cv::Mat& f, float a, cv::Mat& df)
{
    cv::Mat tmp;
    cv::Mat tmp1;

    float x1 = -12*a;

    tmp = CV_PI*f*2;
    cv::Mat x2 = cos_mat(tmp)*12*2;

    //for exp^(xi)= cosx+i*sinx exp^(-xi)=cosx-isinx
    //we can simplify exp^(xi)+exp^(-xi) = 2*cosx
    cv::Mat tmp3;
    tmp3 = tmp*2;//pi*f*4
    float t3 = 6*a;
    cv::Mat x3 = cos_mat(tmp3)*t3*2;

    //simplify exp^(-xi)*yi-exp^(xi)*yi = y*2*sin(x)
    cv::Mat tmp4 = tmp;
    float t4 = CV_PI*12*2;
    cv::Mat x41 = sin_mat(tmp4)*t4;
    cv::Mat x4 = f.mul(x41);

    cv::Mat tmp5 = tmp;
    float t5 = CV_PI*16*2;
    cv::Mat x51 = sin_mat(tmp5)*t5;
    cv::Mat x5 = f.mul(x51)*a;

    cv::Mat tmp6 = tmp*2;
    float t6 = CV_PI*4*2;
    cv::Mat x61 = sin_mat(tmp6)*t6;
    cv::Mat x6 = f.mul(x61)*a;

    float x7 = -24;

    cv::Mat x = -(x1+x2+x3+x4+x5+x6+x7);

    float t = pow(CV_PI,4);
    cv::Mat y1;
    pow(f, 4, y1);
    y1 = y1*t;
    cv::Mat y = 16*y1;

    df = x/y;

    cv::Mat ad1 = (df==0);
    cv::Mat add;
    ad1.convertTo(add, CV_32F, 1.0/255);

    df = df+add;
}


cv::Mat bsxfun_times(cv::Mat xf, cv::Mat p1)
{
    cv::Mat xf_inter;
    int xfwid = xf.cols;
    int xfhei = xf.rows;
    int xfcha = xf.channels();

    int p1wid = p1.cols;
    int p1hei = p1.rows;
    int p1cha = p1.channels();

    // bool con1 = (xfwid == p1wid && xfhei == p1hei && xfcha == p1cha);
    // bool con2 = ( (p1hei == xfhei && p1wid == 1 && p1cha == 1) ||  (p1hei == 1 && p1wid == xfwid && p1cha ==1) );
    // bool con3 = ( ((p1hei == xfhei && p1wid == 1 ) ||  (p1hei == 1 && p1wid == xfwid )) && xfcha == p1cha );
    // bool con = con1||con2||con3;
    assert((xfwid == p1wid && xfhei == p1hei && xfcha == p1cha)||( (p1hei == xfhei && p1wid == 1 && p1cha == 1) ||  (p1hei == 1 && p1wid == xfwid && p1cha ==1) )||( ((p1hei == xfhei && p1wid == 1 ) ||  (p1hei == 1 && p1wid == xfwid )) && xfcha == p1cha ));

    if(xfwid == p1wid && xfhei == p1hei && xfcha == p1cha)
    {
        xf_inter = xf.mul(p1);
    }
    else if( ((p1hei == xfhei && p1wid ==1) || (p1hei ==1 && p1wid == xfwid)) && p1cha == xfcha)//complex mat mul
    {
        cv::Mat p1_e;
        if(p1hei == xfhei && p1wid == 1)
        {
            int padwid = xfwid - p1wid;
            cv::copyMakeBorder(p1, p1_e, 0, 0, 0, padwid, cv::BORDER_REPLICATE);
        }
        else if(p1hei == 1 && p1wid == xfwid)
        {
            int padhei = xfhei - p1hei;
            cv::copyMakeBorder(p1, p1_e, 0, padhei, 0, 0, cv::BORDER_REPLICATE);
        }

        ComplexMat xfc(xf);
        ComplexMat p1c(p1_e);
        ComplexMat xf_inter_c = xfc.mul(p1c);

        xf_inter = xf_inter_c.to_cv_mat();
    }
    else if( (p1hei == xfhei && p1wid == 1 && p1cha == 1) ||  (p1hei == 1 && p1wid == xfwid && p1cha ==1))
    {
        cv::Mat p1_e(xfhei, xfwid, CV_32FC1);
        if(p1hei == xfhei && p1wid == 1)
        {
            for(int j=0; j<p1_e.cols; j++)
            {
                cv::Range to(j,j+1);
                cv::Range from(0,1);

                p1(cv::Range::all(), from).copyTo(p1_e(cv::Range::all(), to));
            }
        }
        else if(p1hei == 1 && p1wid == xfwid)
        {
            for(int j=0; j<p1_e.rows; j++)
            {
                cv::Range to(j, j+1);
                cv::Range from(0,1);

                p1(from, cv::Range::all()).copyTo(p1_e(to, cv::Range::all()));
            }

        }

        cv::Mat p1_fs;
        if(xfcha == 1)
            p1_fs = p1_e.clone();
        else
        {
            std::vector<cv::Mat> p1_e_vec;
            for(int j=0; j<xfcha; j++)
            {
                p1_e_vec.push_back(p1_e);
            }

            cv::merge(p1_e_vec, p1_fs);
        }

        xf_inter = xf.mul(p1_fs);
    }
    return xf_inter;
}


void interpolate_dft(const Vec2dMat& xf, Vec1dMat& interp1_fs, Vec1dMat& interp2_fs, Vec2dMat& xf_out)
{
    Vec2dMat xf_inter_vec;
    xf_inter_vec.resize(xf.size());
    for(int i=0; i<xf.size(); i++)
    {
        xf_inter_vec[i].resize(xf[i].size());
        for (int j = 0; j < xf[i].size(); ++j)
        {
            auto& xfi = xf[i][j];
            auto& p1 = interp1_fs[i];
            auto& p2 = interp2_fs[i];
            auto xfi_p1 = bsxfun_times(xfi, p1);
            xf_inter_vec[i][j] = bsxfun_times(xfi_p1, p2);
        }
    }
    xf_out = std::move(xf_inter_vec);
//    xf_out = xf_inter_vec;
}

void interpolate_dft(const Vec1dMat& xf, Vec1dMat& interp1_fs, Vec1dMat& interp2_fs, Vec1dMat& xf_out, int i)
{
    Vec1dMat xf_inter_vec;
    xf_inter_vec.resize(xf.size());
    for (int j = 0; j < xf.size(); ++j)
    {
        auto& xfi = xf[j];
        auto& p1 = interp1_fs[i];
        auto& p2 = interp2_fs[i];
        auto xfi_p1 = bsxfun_times(xfi, p1);
        xf_inter_vec[j] = bsxfun_times(xfi_p1, p2);
    }
    xf_out = std::move(xf_inter_vec);
}


void resize_dft(const cv::Mat& input_dft, int desiredLen, cv::Mat& output_dft)
{
    int dftwid = input_dft.cols;
    int dfthei = input_dft.rows;

    // bool con = (dftwid == 1 && dfthei > 1) || (dftwid > 1 && dfthei == 1 );
    assert((dftwid == 1 && dfthei > 1) || (dftwid > 1 && dfthei == 1 ));

    cv::Mat resizedmat;
    if(dftwid == 1 && dfthei > 1)
    {
        int dftLen = dfthei;
        int minsz =std::min(dftLen, desiredLen);
        float scaling = (float)desiredLen/(float)dftLen;

        resizedmat = cv::Mat(desiredLen, 1, CV_32FC2);
        resizedmat.setTo(0);

        int mids = ceil((float) minsz/2.);
        int mide = floor((float)(minsz-1)/2.) - 1;

        input_dft(cv::Rect(0, 0, input_dft.cols, mids)).copyTo(resizedmat(cv::Rect(0, 0, resizedmat.cols, mids)));
        input_dft(cv::Rect(0, input_dft.rows-mide-1, input_dft.cols, mide+1 )).copyTo(resizedmat(cv::Rect(0, resizedmat.rows-mide-1, resizedmat.cols, mide+1)));
        resizedmat = resizedmat*scaling;
    }
    else if(dfthei == 1 && dftwid >1)
    {
        int dftLen = dftwid;
        int minsz = std::min(dftLen, desiredLen);
        float scaling = (float)desiredLen/(float)dftLen;

        resizedmat = cv::Mat(1, desiredLen, CV_32FC2);
        resizedmat.setTo(0);

        int mids = ceil((float) minsz/2.);
        int mide = floor((float)(minsz-1)/2.) - 1;

        input_dft(cv::Rect(0, 0, mids, input_dft.rows)).copyTo(resizedmat(cv::Rect(0, 0, mids, resizedmat.rows)));
        input_dft(cv::Rect(input_dft.cols-mide-1, 0, mide+1, input_dft.rows)).copyTo(resizedmat(cv::Rect(resizedmat.cols-mide-1, 0, mide+1, resizedmat.rows)));
        resizedmat = resizedmat*scaling;
    }

    output_dft = std::move(resizedmat);
//    output_dft = resizedmat;
}

//grid_sz (width, height)
void sample_fs(const cv::Mat& fs, cv::Mat& fs_dst, int* grid_sz)
{
    int fswid = fs.cols;
    int fshei = fs.rows;
    cv::Mat x;
    if(grid_sz == NULL || (fswid == grid_sz[0] && fshei == grid_sz[1]) )
    {
        int prod = fswid*fshei;
        cv::Mat fs_cifft;
        cifft2(fs, fs_cifft);
        x = prod * fs_cifft;
    }
    else
    {
        // bool con = (grid_sz[0] > fswid && grid_sz[1] > fshei);
        assert((grid_sz[0] > fswid && grid_sz[1] > fshei));

        int gridwid = grid_sz[0];
        int gridhei = grid_sz[1];

        int tot_pad_wid = gridwid - fswid;
        int tot_pad_hei = gridhei - fshei;

        int pad_sz_wid = ceil((float)tot_pad_wid/2.);
        int pad_sz_hei = ceil((float)tot_pad_hei/2.);

        int pad_sz_left = pad_sz_wid;
        int pad_sz_right;
        if(tot_pad_wid%2 == 1)
            pad_sz_right = pad_sz_wid-1;
        else
            pad_sz_right = pad_sz_wid;

        int pad_sz_top = pad_sz_hei;
        int pad_sz_bottom;
        if(tot_pad_hei%2 == 1)
            pad_sz_bottom = pad_sz_hei-1;
        else
            pad_sz_bottom = pad_sz_hei;

        cv::Mat fs_mat = fs.clone();
        cv::copyMakeBorder(fs_mat, fs_mat, pad_sz_top, pad_sz_bottom, pad_sz_left, pad_sz_right, cv::BORDER_CONSTANT, cv::Scalar(0,0));

        int prod = gridwid*gridhei;
        cv::Mat fs_cifft;
        cifft2(fs_mat, fs_cifft);
        x = prod * fs_cifft;
    }
    fs_dst = std::move(x);
//    fs_dst = x;
}

cv::Mat exp_mat(cv::Mat x)
{
    cv::Mat real = cos_mat(x);
    cv::Mat imag = sin_mat(x);

    std::vector<cv::Mat> exp_mat_vec;
    exp_mat_vec.push_back(real);
    exp_mat_vec.push_back(imag);

    cv::Mat exp;
    cv::merge(exp_mat_vec, exp);
    return exp;
}

void shift_sample(const Vec1dMat& xf, cv::Point2f shift, const cv::Mat& kx, const cv::Mat& ky, Vec1dMat& xf_sample)
{
    int matnum = xf.size();
    float shift1 = shift.y;
    float shift2 = shift.x;

    Vec1dMat shift_xf_vec;
    shift_xf_vec.resize(matnum);
    for(int i=0; i<matnum; i++)
    {
        cv::Mat tmpy = shift1*ky;
        cv::Mat shift_exp_y = exp_mat(tmpy);

        cv::Mat tmpx = shift2*kx;
        cv::Mat shift_exp_x = exp_mat(tmpx);

        cv::Mat shift_xf_y =  bsxfun_times(xf[i], shift_exp_y);
        shift_xf_vec[i] = bsxfun_times(shift_xf_y, shift_exp_x);
    }
    xf_sample = std::move(shift_xf_vec);
//    xf_sample = shift_xf_vec;
}

void symmetrize_filter(const cv::Mat& hf, cv::Mat& hf_symm)
{
    float dc_ind = ((float)hf.rows+1)/2.;

    std::vector<cv::Mat> hf_vec;
    cv::split(hf, hf_vec);

    cv::Mat hf_real = hf_vec[0];
    cv::Mat hf_imag = hf_vec[1];

    cv::Mat real_end_col = hf_real(cv::Rect(hf_real.cols-1, 0, 1, ceil(dc_ind-1)));
    cv::Mat real_end_col_flip ;
    cv::flip(real_end_col, real_end_col_flip, 0);
    cv::Mat hfi_symm_real = hf_real.clone();
    real_end_col_flip.copyTo(hfi_symm_real(cv::Rect(hf_real.cols-1, floor(dc_ind), 1, ceil(dc_ind-1))));

    cv::Mat imag_end_col = hf_imag(cv::Rect(hf_imag.cols-1, 0, 1, ceil(dc_ind-1)));
    cv::Mat imag_end_col_flip ;
    cv::flip(imag_end_col, imag_end_col_flip, 0);
    imag_end_col_flip = -imag_end_col_flip;
    cv::Mat hf_symm_imag = hf_imag.clone();
    imag_end_col_flip.copyTo(hf_symm_imag(cv::Rect(hf_imag.cols-1, floor(dc_ind), 1, ceil(dc_ind-1))));

    std::vector<cv::Mat> hf_symm_vec;
    hf_symm_vec.push_back(hfi_symm_real);
    hf_symm_vec.push_back(hf_symm_imag);
    cv::merge(hf_symm_vec, hf_symm);
}

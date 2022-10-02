//
// Created by yuan on 17-9-20.
//

#include "hog_feature_extractor.h"
#include "utils.h"


HogFeatureExtractor::HogFeatureExtractor(int bin, int orients_num, float clip_value, bool crop)
{
    this->input_size = cv::Size(244, 244);
    this->bin_size = bin;
    this->orients = orients_num;
    this->clip = clip_value;
    this->is_crop =  crop;
    this->output_blocks = 1;
}

HogFeatureExtractor::~HogFeatureExtractor() {}

void HogFeatureExtractor::extract_feature(const cv::Mat &img, Sample &out_feature)
{
    cv::Mat img_proc;
    if(img.size() != input_size)
        cv::resize(img, img_proc, input_size);
    else
        img_proc = img;

    // keep the same channel sequence with the data format in matlab
    assert(img_proc.channels() == 3);
//    cv::cvtColor(img_proc, img_proc, cv::COLOR_BGR2RGB);

    int height = img_proc.rows;
    int width = img_proc.cols;
    int channels = img_proc.channels();

    //  float *img_data = split_bgrmat_channels(img_proc);
    float *img_data = new float[height * width * channels];
    int count = 0;
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uchar* ptr = img_proc.ptr<uchar>(i);
            for (size_t k = 0; k < channels; k++)
            {
                img_data[count] = ptr[j * channels + k];
                count++;
            }
        }
    }

    float *img_data_t = new float[height * width * channels];
    // re-arrange data to split channels and height-major format
    change_format(img_data_t, img_data, height, width, channels);
    int out_h, out_w, out_d;
    float* hog_data = fhog(img_data_t, height, width, channels, &out_h, &out_w, &out_d,
                           bin_size, orients, clip, is_crop);

    Feature feat(out_d - 1);
    for (int k = 0; k < out_d-1; ++k)
    {
        cv::Mat feat_ch(out_h, out_w, CV_32FC1);
        for (int i = 0; i < out_h; ++i)
        {
            float *ptr = feat_ch.ptr<float>(i);
            for (int j = 0; j < out_w; ++j)
            {
                ptr[j] = hog_data[i + j * out_h + k * out_h * out_w];
            }
        }
        feat[k] = feat_ch;
    }
    out_feature.push_back(feat);

    delete[] img_data;
    delete[] img_data_t;
    delete[] hog_data;
}

cv::Size HogFeatureExtractor::get_image_support_sz(cv::Size2f new_sample_sz, float scale)
{
    //the max cell to extract feature
    //there is only one hog feature for now,
    //that's why max_cell_size = orients
    int max_cell_size = bin_size;

    std::vector<float> new_image_sample_sz(2);
    new_image_sample_sz[0] = (1 + 2*round(new_sample_sz.height/(2*max_cell_size)))*max_cell_size;
    new_image_sample_sz[1] = (1 + 2*round(new_sample_sz.width/(2*max_cell_size)))*max_cell_size;
    std::vector<std::vector<float>> feature_sz_choices(max_cell_size, std::vector<float>(2, 0));
    for(int i=0; i<max_cell_size; i++)
    {
        feature_sz_choices[i][0] = floor((new_image_sample_sz[0] + i)/max_cell_size);
        feature_sz_choices[i][1] = floor((new_image_sample_sz[1] + i)/max_cell_size);
    }

    int best_choice = 0;
    float max_n = 0.0;
    std::vector<float> num_odd_dimensions(max_cell_size);
    for(int i=0; i<max_cell_size; i++)
    {
        num_odd_dimensions[i] = (int)(feature_sz_choices[i][0])%2 + (int)(feature_sz_choices[i][1])%2;
        if(max_n < num_odd_dimensions[i])
        {
            max_n = num_odd_dimensions[i];
            best_choice = i;
        }
    }

    float pixels_added = best_choice;
    input_size.width = (int)round(new_image_sample_sz[1] + pixels_added);
    input_size.height = (int)round(new_image_sample_sz[0] + pixels_added);

    return input_size;
}

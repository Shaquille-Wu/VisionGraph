#include "../vision_graph/include/vision_graph.h"
#include <opencv2/opencv.hpp>
#include <chrono> 

int main(int argc, char* argv[])
{
    std::string           graph_json_file  = argv[1];
    std::string           test_image       = argv[2];
    std::string           result_image     = argv[3];
    int                   loops            = atoi((argv[4]));
    vision_graph::Graph   face_graph;

    face_graph.Build(graph_json_file);

    std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map = face_graph.GetTensors(0);
    vision_graph::Tensor*             tensor_base_image        = ((tensor_map.find("image"))->second).first;
    vision_graph::Tensor*             tensor_base_pos          = ((tensor_map.find("box_selector"))->second).first;
    vision_graph::Tensor*             tensor_base_keypoint     = ((tensor_map.find("face_keypoint"))->second).first;
    vision_graph::Tensor*             tensor_base_hog          = ((tensor_map.find("face_hog_similarity"))->second).first;
    vision_graph::Tensor*             tensor_base_kpts_offset  = ((tensor_map.find("face_kpts_offset"))->second).first;
    vision_graph::Tensor*             tensor_base_box          = ((tensor_map.find("face_box_smooth"))->second).first;
    vision_graph::TensorImage*        tensor_image             = dynamic_cast<vision_graph::TensorImage*>(tensor_base_image);
    vision_graph::TensorReference*    tensor_reference         = dynamic_cast<vision_graph::TensorReference*>(tensor_base_pos);
    vision_graph::TensorKeypoints*    tensor_keypoints         = dynamic_cast<vision_graph::TensorKeypoints*>(tensor_base_keypoint);
    vision_graph::TensorFloat32*      tensor_float32           = dynamic_cast<vision_graph::TensorFloat32*>(tensor_base_hog);
    vision_graph::TensorKeypoints*    tensor_kpts_offset       = dynamic_cast<vision_graph::TensorKeypoints*>(tensor_base_kpts_offset);
    vision_graph::TensorBox*          tensor_smooth_box        = dynamic_cast<vision_graph::TensorBox*>(tensor_base_box);


    *tensor_image = cv::imread(test_image);

    if(loops < 30)
        loops = 30;

    auto start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < loops ; i ++)
    {
        face_graph.Start(0); 
        face_graph.Wait(0);
    }
    auto end      = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("%d loops, cost %lld\n", loops, (long long int)(duration.count()));

    vision_graph::TensorBox*         tensor_select_box     = dynamic_cast<vision_graph::TensorBox*>(tensor_reference->reference_);
    //if(tensor_box > 0)
    {
        //if(tensor_select_box.size() > 0)
        {
            cv::Rect rect((*tensor_select_box).x1, (*tensor_select_box).y1, (*tensor_select_box).x2-(*tensor_select_box).x1, (*tensor_select_box).y2-(*tensor_select_box).y1);
            cv::rectangle(*(tensor_image), rect, cv::Scalar(255, 0, 0), 2, 8);

            cv::Rect rect_smooth(FLT2INT32((*tensor_smooth_box).x1), 
                                 FLT2INT32((*tensor_smooth_box).y1), 
                                 FLT2INT32((*tensor_smooth_box).x2 - (*tensor_smooth_box).x1), 
                                 FLT2INT32((*tensor_smooth_box).y2 - (*tensor_smooth_box).y1));
            cv::rectangle(*(tensor_image), rect_smooth, cv::Scalar(0, 0, 255), 2, 8);

            int  kpt_size = (int)(tensor_kpts_offset->size());
            for(int i = 0 ; i < kpt_size ; i ++)
            {
                cv::Point2f   cur_pt((*tensor_kpts_offset)[i].x, (*tensor_kpts_offset)[i].y);
                cv::circle(*tensor_image, cur_pt, 2, cv::Scalar(0, 255, 0), -1);
            }
        }
    }

    cv::imwrite(result_image, *(tensor_image));

    std::cout << "hog similarity: " << tensor_float32->value_ << std::endl;
    std::cout << "face graph test exit" << std::endl;
    face_graph.Destroy();
}
#include "../vision_graph/include/vision_graph.h"
#include <opencv2/opencv.hpp>
#include <chrono> 

int main(int argc, char* argv[])
{
    std::string           graph_json_file  = argv[1];
    std::string           test_image       = argv[2];
    std::string           result_image     = argv[3];
    int                   loops            = atoi((argv[4]));
    vision_graph::Graph   mini_graph;

    mini_graph.Build(graph_json_file);

    std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map0 = mini_graph.GetTensors(0);
    vision_graph::Tensor*             tensor_base_image                    = ((tensor_map0.find("image"))->second).first;
    vision_graph::TensorImage*        tensor_image                         = dynamic_cast<vision_graph::TensorImage*>(tensor_base_image);
    std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map1 = mini_graph.GetTensors(1);
    vision_graph::Tensor*             tensor_base_single_value             = ((tensor_map1.find("single_value"))->second).first;
    vision_graph::TensorInt32*        tensor_single_value                  = dynamic_cast<vision_graph::TensorInt32*>(tensor_base_single_value);
    vision_graph::Tensor*             tensor_base_single_image             = ((tensor_map1.find("single_image"))->second).first;
    vision_graph::TensorImage*        tensor_single_image                  = dynamic_cast<vision_graph::TensorImage*>(tensor_base_single_image);
    vision_graph::Tensor*             tensor_base_single_box               = ((tensor_map1.find("single_Box"))->second).first;
    vision_graph::TensorBox*          tensor_single_box                    = dynamic_cast<vision_graph::TensorBox*>(tensor_base_single_box);
    std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map2 = mini_graph.GetTensors(2);
    vision_graph::Tensor*             tensor_base_quality_images           = ((tensor_map2.find("quality_image"))->second).first;
    vision_graph::TensorImageVector*  tensor_quality_images                = dynamic_cast<vision_graph::TensorImageVector*>(tensor_base_quality_images);
    std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map3 = mini_graph.GetTensors(3);
    vision_graph::Tensor*             tensor_base_people_attributes_image  = ((tensor_map3.find("people_attributes_image"))->second).first;
    vision_graph::TensorImage*        tensor_people_attributes_image       = dynamic_cast<vision_graph::TensorImage*>(tensor_base_people_attributes_image);
    std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map4 = mini_graph.GetTensors(4);
    vision_graph::Tensor*             tensor_base_arm_face_feature_image   = ((tensor_map4.find("arm_face_feature_image"))->second).first;
    vision_graph::TensorImage*        tensor_arm_face_feature_image        = dynamic_cast<vision_graph::TensorImage*>(tensor_base_arm_face_feature_image);

    *tensor_image                   = cv::imread(test_image);
    *tensor_single_image            = cv::imread(test_image);
    *tensor_single_value            = 0;
    tensor_single_box->x1 = 40;
    tensor_single_box->y1 = 40;
    tensor_single_box->x2 = 140;
    tensor_single_box->y2 = 140;
    tensor_quality_images->resize(4);
    (*tensor_quality_images)[0]     = cv::imread(test_image);
    (*tensor_quality_images)[1]     = cv::imread(test_image);
    (*tensor_quality_images)[2]     = cv::imread(test_image);
    (*tensor_quality_images)[3]     = cv::imread(test_image);
    *tensor_people_attributes_image = cv::imread(test_image);
    *tensor_arm_face_feature_image  = cv::imread(test_image);

    if(loops < 100)
        loops = 100;

    char input = getchar();

    auto start    = std::chrono::system_clock::now();
    //for(int i = 0 ; i < loops ; i ++)
    {
        mini_graph.Start(0); 
        mini_graph.Wait(0);
    }
    auto end      = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("0, %d loops, cost %lld\n", loops, (long long int)(duration.count()));

    start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < loops ; i ++)
    {
        *tensor_single_value = (0 == i ? 0 : 1);
        mini_graph.Start(1); 
        mini_graph.Wait(1);
    }
    end      = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("1, %d loops, cost %lld\n", loops, (long long int)(duration.count()));

    start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < loops ; i ++)
    {
        mini_graph.Start(2); 
        mini_graph.Wait(2);
    }
    end      = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("2, %d loops, cost %lld\n", loops, (long long int)(duration.count()));

    start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < loops ; i ++)
    {
        mini_graph.Start(3); 
        mini_graph.Wait(3);
    }
    end      = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("3, %d loops, cost %lld\n", loops, (long long int)(duration.count()));

    mini_graph.Destroy();
}
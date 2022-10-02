#include <opencv2/opencv.hpp>
#include <detector.h>
#include <keypoints_attributes.h>
#include <reidfeature.h>

int main(int argc, char* argv[])
{
    std::string   det_cfgfile  = "/data/local/tmp/shaquille/test_dlcv/model_zoo/detection/face_ssdlite1_qf_0.35_r2.0/snpe-1.36/det_face_ssdlite1_qf_0.35_r2.0_nbn_gpu.snpe.json";
    std::string   kpts_cfgfile = "/data/local/tmp/shaquille/test_dlcv/model_zoo/keypoints/face_keypoint_classifier_0428_avgpool/snpe-1.36/keypoint_classifier_0428_avgpool_gpu.snpe.json";
    std::string   reid_cfgfile = "/data/local/tmp/shaquille/test_dlcv/model_zoo/reid_feature/face_mobilenet_v2_relu_finetune/snpe-1.36/face_mobilenet_v2_relu_finetune_gpu.snpe.json";
    printf("create Detector\n");
    vision::Detector*            det     = new vision::Detector(det_cfgfile);
    printf("create KeypointsAttributes\n");
    vision::KeypointsAttributes* kpts    = new vision::KeypointsAttributes(kpts_cfgfile);
    printf("create Feature\n");
    vision::Feature*             feature = new vision::Feature(reid_cfgfile);
    printf("create Feature end\n");

    delete det;
    delete kpts;
    delete feature;

    return 0;
}
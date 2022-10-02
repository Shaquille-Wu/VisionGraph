#include <string.h>
#include "../../include/graph_error.h"
#include "solver_face_hog_verify.h"
#include <logging.h>

namespace vision_graph{

static const std::vector<TENSOR_TYPE>  kInTensorTypeSupported        = {
    kTensorImageVector,
    kTensorKeypointsVector
};
static const TENSOR_TYPE               kOutTensorTypeSupported       = kTensorFloat32Vector;

static const std::string               kJSONNodeTarget               = std::string("target_feature");
static const std::string               kJSONNodeTargetSource         = std::string("source");
static const std::string               kJSONNodeTargetKptsSource     = std::string("kpts");
static const std::string               kJSONNodeTargethogSource      = std::string("hog");

SolverFaceHOGVerify::SolverFaceHOGVerify(nlohmann::json const& param) noexcept : Solver(param), 
                                                                                 target_src_type_(FROM_DEFAULT)
{
    //LOG(ERROR) << "SolverFaceHOGVerify start";

    int           i         = 0;
    std::string   kpts_file = std::string("");
    std::string   hog_file  = std::string("");

    hog_verify_ = new FaceHOGVerify;

    if(true == param.contains(kJSONNodeTarget))
    {
        nlohmann::json   template_json  = param.at(kJSONNodeTarget);
        std::string      target_source  = template_json.at(kJSONNodeTargetSource);
        if(std::string("default") == target_source)
            target_src_type_ = FROM_DEFAULT;
        else if(std::string("file") == target_source)
        {
            target_src_type_ = FROM_FILE;
            kpts_file        = template_json.at(kJSONNodeTargetKptsSource);
            hog_file         = template_json.at(kJSONNodeTargethogSource);
        }
        else if(std::string("var") == target_source)
            target_src_type_ = FROM_VARIABLE;
    }
    if(FROM_DEFAULT == target_src_type_)
        hog_verify_->SetTargetDefault();
    else if(FROM_FILE == target_src_type_)
        hog_verify_->SetTargetFromFile(kpts_file, hog_file);
    hog_verify_->Init();
}

SolverFaceHOGVerify::~SolverFaceHOGVerify() noexcept
{
    if(nullptr != hog_verify_)
        delete hog_verify_;
    hog_verify_ = nullptr;
}

Tensor* SolverFaceHOGVerify::CreateOutTensor(int out_tensor_idx) noexcept
{ 
    Tensor*  out_tensor = new TensorFloat32Vector;
    return out_tensor;
}

bool SolverFaceHOGVerify::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(FROM_VARIABLE != target_src_type_)
    {
        if(in.size() != 2)
            return false;
    }
    if(out.size() != 1)
        return false;

    if(in[0]->GetType() != kInTensorTypeSupported[0] ||
       in[1]->GetType() != kInTensorTypeSupported[1])
        return false;

    if(FROM_VARIABLE == target_src_type_)
    {
        if(in.size() != 4)
            return false;
        if(in[2]->GetType() != kTensorKeyPoints ||
           in[3]->GetType() != kTensorFeature)
            return false;
    }

    if(out[0]->GetType() != kOutTensorTypeSupported)
        return false;

    return true;
}

int SolverFaceHOGVerify::Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept
{
    if(false == CheckTensor(in, out))
    {
        LOG(ERROR) << "CheckTensor failed";
        ABORT();
        return vision_graph::kErrCodeParamInvalid;
    }
        

    int                           i                      = 0;
    Tensor const*                 tensor_img             = in[0];
    Tensor const*                 tensor_kpts            = in[1];
    Tensor const*                 tensor_target_kpts     = nullptr;
    Tensor const*                 tensor_target_hog      = nullptr;
    TensorImageVector const*      tensor_src_image       = dynamic_cast<TensorImageVector const*>(tensor_img);
    TensorKeypointsVector const*  tensor_src_kpts        = dynamic_cast<TensorKeypointsVector const*>(tensor_kpts);
    TensorKeypointsVector const*  tensor_src_target_kpts = nullptr;
    TensorFeature const*    tensor_src_target_hog  = nullptr;
    Tensor*                 tensor_out             = out[0];
    TensorFloat32Vector*   tensor_similarity      = dynamic_cast<TensorFloat32Vector*>(tensor_out);
    tensor_similarity->clear();
    // if(FROM_VARIABLE == target_src_type_)
    // {
    //     tensor_target_kpts     = in[2];
    //     tensor_target_hog      = in[3];
    //     tensor_src_target_kpts = dynamic_cast<TensorKeypointsVector const*>(tensor_target_kpts);
    //     tensor_src_target_hog  = dynamic_cast<TensorFeature const*>(tensor_target_hog);

    //     hog_verify_->SetTargetData(*tensor_src_target_kpts, *tensor_src_target_hog);
    // }
    // *tensor_similarity = 0.0f;

    // if(tensor_src_image->cols <= 0 ||
    //    tensor_src_image->rows <= 0 ||
    //    tensor_src_kpts->size() <= 0)
    //    return 0;
    for (int i = 0; i < tensor_src_image->size(); i++) {
        float hog = hog_verify_->Solve(*(tensor_src_image->begin() + i), *(tensor_src_kpts->begin() + i));
        tensor_similarity->push_back(hog);
    }
        // *tensor_similarity = hog_verify_->Solve(*tensor_src_image, *tensor_src_kpts);

    return 0;
}

};//namespace vision_graph
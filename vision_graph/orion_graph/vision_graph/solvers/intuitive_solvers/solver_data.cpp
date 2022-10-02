#include "../../include/graph_error.h"
#include "solver_data.h"
#include "implement/scalar_compare/scalar_compare.h"
#include <logging.h>
namespace vision_graph{

static const std::map<std::string, TENSOR_TYPE>  kDataTypes = {
    { "UInt8",               kTensorUInt8             },
    { "Int8",                kTensorInt8              },
    { "UInt16",              kTensorUInt16            },
    { "Int16",               kTensorInt16             },
    { "UInt32",              kTensorUInt32            },
    { "Int32",               kTensorInt32             },
    { "UInt64",              kTensorUInt64            },
    { "Int64",               kTensorInt64             },
    { "Float32",             kTensorFloat32           },
    { "Float64",             kTensorFloat64           },
    { "String",              kTensorString            },
    { "Point",               kTensorPoint             },
    { "Box",                 kTensorBox               },
    { "Image",               kTensorImage             },
    { "BoxesMap",            kTensorBoxesMap          },
    { "KeyPoints",           kTensorKeyPoints         },
    { "Attributes",          kTensorAttributes        },
    { "Feature",             kTensorFeature           },
    { "FeatureMaps",         kTensorFeatureMaps       },
    //{ "DLCVOut",           kTensorDLCVOut           },
    { "Reference",           kTensorReference         },
    { "UInt8Vector",         kTensorUInt8Vector       },
    { "Int8Vector",          kTensorInt8Vector        },
    { "UInt16Vector",        kTensorUInt16Vector      },
    { "Int16Vector",         kTensorInt16Vector       },
    { "UInt32Vector",        kTensorUInt32Vector      },
    { "Int32Vector",         kTensorInt32Vector       },
    { "UInt64Vector",        kTensorUInt64Vector      },
    { "Int64Vector",         kTensorInt64Vector       },
    { "Float32Vector",       kTensorFloat32Vector     },
    { "Float64Vector",       kTensorFloat64Vector     },
    { "BoxVector",           kTensorBoxVector         },
    { "ImageVector",         kTensorImageVector       },
    { "KeypointsVector",     kTensorKeypointsVector   },
    { "AttributesVector",    kTensorAttributesVector  },
    //{ "DLCVOutVector",     kTensorDLCVOutVector     },
    { "PtrVector",           kTensorPtrVector         }
};

static const std::string    kJSONNodeDataType   = "data_type" ;
static const std::string    kJSONNodeInitValue  = "init_value" ;

SolverData::SolverData(nlohmann::json const& param) noexcept : Solver(param), init_value_(nullptr)
{
    if(false == param.contains(kJSONNodeDataType))
    {
        //LOG(ERROR) << "SolverData, cannot find type json node";
        ABORT();
    }
    std::string  data_type = param.at(kJSONNodeDataType).get<std::string>();
    std::map<std::string, TENSOR_TYPE>::const_iterator  iter = kDataTypes.find(data_type);
    TENSOR_TYPE                                         tensor_type = 0;                                                 
    if(kDataTypes.end() == iter)
    {
        //LOG(ERROR) << "SolverData, cannot support data_type: " << data_type << std::endl;
        LOG(ERROR) << "supported types: " << std::endl;
        for(auto support_type : kDataTypes)
            LOG(ERROR) << support_type.first << std::endl;
        ABORT();
    }
    tensor_type = iter->second;
    if(true == param.contains(kJSONNodeInitValue))
    {
        if(tensor_type >= kTensorUInt8 && tensor_type <= kTensorString)
            ReadScalarStringValue(param, kJSONNodeInitValue, init_value_);
        else if(tensor_type == kTensorPoint)
        {
            std::vector<float>   point_value = param.at(kJSONNodeInitValue).get<std::vector<float> >();
            TensorPoint* point_tensor = new TensorPoint(0.0f, 0.0f);
            if(point_value.size() > 0)
                point_tensor->x = point_value[0];
            if(point_value.size() > 1)
                point_tensor->y = point_value[1];
            init_value_ = point_tensor;
        }
        else if(tensor_type == kTensorBox)
        {
            std::vector<float>   box_value = param.at(kJSONNodeInitValue).get<std::vector<float> >();
            TensorBox* box_tensor = new TensorBox;
            if(box_value.size() > 0)
                box_tensor->x1 = box_value[0];
            if(box_value.size() > 1)
                box_tensor->y1 = box_value[1];
            if(box_value.size() > 2)
                box_tensor->x2 = box_value[2];
            if(box_value.size() > 3)
                box_tensor->y2 = box_value[3];
            init_value_ = box_tensor;
        }
    }

    data_type_ = iter->second;
};

SolverData::~SolverData() noexcept
{
    if(nullptr != init_value_)
        delete init_value_;
    init_value_ = nullptr;
};

bool SolverData::CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept
{
    if(out.size() != 1)
        return false;

    return true;
}

Tensor* SolverData::CreateOutTensor(int out_tensor_idx) noexcept 
{ 
    Tensor*   tensor = nullptr;
    (void)out_tensor_idx;
    switch(data_type_)
    {
        case kTensorUInt8:
            if(nullptr == init_value_)
                tensor = new TensorUInt8(0);
            else
                tensor = new TensorUInt8(GetTensorNumericValue<unsigned char>(*init_value_));
            break;
        case kTensorInt8:
            if(nullptr == init_value_)
                tensor = new TensorInt8(0);
            else
                tensor = new TensorInt8(GetTensorNumericValue<char>(*init_value_));
            break;
        case kTensorUInt16:
            if(nullptr == init_value_)
                tensor = new TensorUInt16(0);
            else
                tensor = new TensorUInt16(GetTensorNumericValue<unsigned short int>(*init_value_));
            break;
        case kTensorInt16:
            if(nullptr == init_value_)
                tensor = new TensorInt16(0);
            else
                tensor = new TensorInt16(GetTensorNumericValue<short int>(*init_value_));
            break;
        case kTensorUInt32:
            if(nullptr == init_value_)
                tensor = new TensorUInt32(0);
            else
                tensor = new TensorUInt32(GetTensorNumericValue<unsigned int>(*init_value_));
            break;
        case kTensorInt32:
            if(nullptr == init_value_)
                tensor = new TensorInt32(0);
            else
                tensor = new TensorInt32(GetTensorNumericValue<int>(*init_value_));
            break;
        case kTensorUInt64:
            if(nullptr == init_value_)
                tensor = new TensorUInt64(0);
            else
                tensor = new TensorUInt64(GetTensorNumericValue<unsigned long long int>(*init_value_));
            break;
        case kTensorInt64:
            if(nullptr == init_value_)
                tensor = new TensorInt64(0);
            else
                tensor = new TensorInt64(GetTensorNumericValue<long long int>(*init_value_));
            break;
        case kTensorFloat32:
            if(nullptr == init_value_)
                tensor = new TensorFloat32(0.0f);
            else
                tensor = new TensorFloat32(GetTensorNumericValue<float>(*init_value_));
            break;
        case kTensorFloat64:
            if(nullptr == init_value_)
                tensor = new TensorFloat64(0.0);
            else
                tensor = new TensorFloat64(GetTensorNumericValue<double>(*init_value_));
            break;
        case kTensorString:
            if(nullptr == init_value_)
                tensor = new TensorString;
            else
                tensor = new TensorString(*(dynamic_cast<TensorString*>(init_value_)));
            break;
        case kTensorPoint:
            if(nullptr == init_value_)
                tensor = new TensorPoint(0.0f, 0.0f);
            else
                tensor = new TensorPoint(*(dynamic_cast<TensorPoint*>(init_value_)));
            break;
        case kTensorBox:
            if(nullptr == init_value_)
                tensor = new TensorBox;
            else
                tensor = new TensorBox(*(dynamic_cast<TensorBox*>(init_value_)));
            break;
        case kTensorImage:
            tensor = new TensorImage;
            break;   
        case kTensorBoxesMap:
            tensor = new TensorBoxesMap;
            break; 
        case kTensorKeyPoints:
            tensor = new TensorKeypoints;
            break; 
        case kTensorAttributes:
            tensor = new TensorAttributes;
            break; 
        case kTensorFeature:
            tensor = new TensorFeature;
            break; 
        case kTensorFeatureMaps:
            tensor = new TensorFeatureMaps;
            break; 
/*
        case kTensorDLCVOut:
            tensor = new TensorDLCVOut;
            break; 
*/
        case kTensorReference:
            tensor = new TensorReference;
            break; 
        case kTensorUInt8Vector:
            tensor = new TensorUInt8Vector;
            break; 
        case kTensorInt8Vector:
            tensor = new TensorInt8Vector;
            break; 
        case kTensorUInt16Vector:
            tensor = new TensorUInt16Vector;
            break; 
        case kTensorInt16Vector:
            tensor = new TensorInt16Vector;
            break; 
        case kTensorUInt32Vector:
            tensor = new TensorUInt32Vector;
            break; 
        case kTensorInt32Vector:
            tensor = new TensorUInt32Vector;
            break; 
        case kTensorUInt64Vector:
            tensor = new TensorUInt64Vector;
            break; 
        case kTensorInt64Vector:
            tensor = new TensorInt64Vector;
            break; 
        case kTensorFloat32Vector:
            tensor = new TensorFloat32Vector;
            break; 
        case kTensorFloat64Vector:
            tensor = new TensorFloat64Vector;
            break; 
        case kTensorBoxVector:
            tensor = new TensorBoxVector;
            break; 
        case kTensorImageVector:
            tensor = new TensorImageVector;
            break; 
        case kTensorKeypointsVector:
            tensor = new TensorKeypointsVector;
            break; 
        case kTensorAttributesVector:
            tensor = new TensorAttributesVector;
            break; 
/*
        case kTensorDLCVOutVector:
            tensor = new TensorDLCVOutVector;
            break;
*/
        case kTensorPtrVector:
            tensor = new TensorPtrVector;
            break; 
        default:
            break;
    }

    return tensor;
};

}//namespace vision_graph
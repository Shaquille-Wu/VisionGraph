/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file graph_tensor.h
 * @brief This header file defines basic Tensor.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef GRAPH_TENSOR_H_
#define GRAPH_TENSOR_H_

#include <string>
#include <vector>
#include <map>
#include <box.h>
#include <featuremap_runner.h>

#define FLT2INT32(x)      ((int)((x) > 0.0f ? ((x) + 0.5f) : ((x) - 0.5f)))
#define DBL2INT32(x)      ((int)((x) > 0.0  ? ((x) + 0.5)  : ((x) - 0.5)))

#define FLT2INT64(x)      ((long long int)((x) > 0.0f ? ((x) + 0.5f) : ((x) - 0.5f)))
#define DBL2INT64(x)      ((long long int)((x) > 0.0  ? ((x) + 0.5)  : ((x) - 0.5)))
namespace vision_graph
{
/**
 * @brief The Tensor object is the base class for all input&ouput in graph.
 *
 */

typedef unsigned int      TENSOR_TYPE;

static const TENSOR_TYPE  kTensorBase             = 0;
static const TENSOR_TYPE  kTensorUInt8            = 1;
static const TENSOR_TYPE  kTensorInt8             = 2;
static const TENSOR_TYPE  kTensorUInt16           = 3;
static const TENSOR_TYPE  kTensorInt16            = 4;
static const TENSOR_TYPE  kTensorUInt32           = 5;
static const TENSOR_TYPE  kTensorInt32            = 6;
static const TENSOR_TYPE  kTensorUInt64           = 7;
static const TENSOR_TYPE  kTensorInt64            = 8;
static const TENSOR_TYPE  kTensorFloat32          = 9;
static const TENSOR_TYPE  kTensorFloat64          = 10;
static const TENSOR_TYPE  kTensorString           = 11;
static const TENSOR_TYPE  kTensorPoint            = 12;
static const TENSOR_TYPE  kTensorBox              = 13;
static const TENSOR_TYPE  kTensorImage            = 100;
static const TENSOR_TYPE  kTensorBoxesMap         = 101;
static const TENSOR_TYPE  kTensorKeyPoints        = 102;
static const TENSOR_TYPE  kTensorAttributes       = 103;
static const TENSOR_TYPE  kTensorFeature          = 104;
static const TENSOR_TYPE  kTensorFeatureMaps      = 105;
//static const TENSOR_TYPE  kTensorDLCVOut         = 106;
static const TENSOR_TYPE  kTensorReference        = 1000;
static const TENSOR_TYPE  kTensorUInt8Vector      = 2000;
static const TENSOR_TYPE  kTensorInt8Vector       = 2001;
static const TENSOR_TYPE  kTensorUInt16Vector     = 2002;
static const TENSOR_TYPE  kTensorInt16Vector      = 2003;
static const TENSOR_TYPE  kTensorUInt32Vector     = 2004;
static const TENSOR_TYPE  kTensorInt32Vector      = 2005;
static const TENSOR_TYPE  kTensorUInt64Vector     = 2006;
static const TENSOR_TYPE  kTensorInt64Vector      = 2007;
static const TENSOR_TYPE  kTensorFloat32Vector    = 2008;
static const TENSOR_TYPE  kTensorFloat64Vector    = 2009;
static const TENSOR_TYPE  kTensorBoxVector        = 2010;
static const TENSOR_TYPE  kTensorImageVector      = 2011;
static const TENSOR_TYPE  kTensorKeypointsVector  = 2012;
static const TENSOR_TYPE  kTensorAttributesVector = 2013;
//static const TENSOR_TYPE  kTensorDLCVOutVector   = 2014;
static const TENSOR_TYPE  kTensorPtrVector        = 2014;

class Tensor
{
public:
    /**
     * @brief Constructor
     *
     */
    Tensor() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    Tensor(Tensor const& other) noexcept {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    Tensor(Tensor&& other) noexcept {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    Tensor& operator=(Tensor const& other) noexcept
    {
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    Tensor& operator=(Tensor&& other) noexcept
    {
        return *this;
    };

    /**
     * @brief desonstructor
     *
     */
    virtual ~Tensor() noexcept {;};

    virtual TENSOR_TYPE  GetType() const noexcept   { return kTensorBase; };

    static  TENSOR_TYPE  GetClassType() noexcept    { return kTensorBase; };

}; //Tensor

template <typename NumericType>
class TensorNumeric : public Tensor
{
public:
    using numeric_type = NumericType;
    TensorNumeric() noexcept : value_(0) { ; } ;

    template<typename OtherType, typename std::enable_if<std::is_arithmetic<OtherType>::value, void>::type* = nullptr>
    TensorNumeric(OtherType value) noexcept : value_(cast_value<NumericType, OtherType>(value)) {;};

    template<typename OtherType, typename std::enable_if<std::is_arithmetic<OtherType>::value, void>::type* = nullptr>
    TensorNumeric(TensorNumeric<OtherType> const& other) noexcept : value_(cast_value<NumericType, OtherType>(other.value_)) {;};


    template<typename OtherType, typename std::enable_if<std::is_arithmetic<OtherType>::value, void>::type* = nullptr>
    TensorNumeric& operator=(OtherType other) noexcept
    {
        value_ = cast_value<NumericType, OtherType>(other);

        return *this;
    };

    template<typename OtherType, typename std::enable_if<std::is_arithmetic<OtherType>::value, void>::type* = nullptr>
    TensorNumeric& operator=(TensorNumeric<OtherType> const& other) noexcept
    {
        if(&other == this)
            return *this;
        
        value_ = cast_value<NumericType, OtherType>(other.value_);

        return *this;
    }

    TENSOR_TYPE  GetType() const noexcept
    { 
        return get_numeric_type<NumericType>();
    };

    static TENSOR_TYPE  GeClassType() noexcept
    {
        return get_numeric_type<NumericType>();
    }

    template<typename T>
    T CastValue() const noexcept
    {
        return cast_value<T, NumericType>(value_);
    }
private:
    template<typename T, typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && sizeof(T) == 1, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorUInt8;
    };
    
    template<typename T, typename std::enable_if<std::is_integral<T>::value && (!std::is_unsigned<T>::value) && sizeof(T) == 1, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorInt8; 
    };

    template<typename T, typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && sizeof(T) == 2, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorUInt16; 
    };
    
    template<typename T, typename std::enable_if<std::is_integral<T>::value && (!std::is_unsigned<T>::value) && sizeof(T) == 2, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorInt16; 
    };

    template<typename T, typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && sizeof(T) == 4, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorUInt32; 
    };
    
    template<typename T, typename std::enable_if<std::is_integral<T>::value && (!std::is_unsigned<T>::value) && sizeof(T) == 4, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorInt32; 
    };

    template<typename T, typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && sizeof(T) == 8, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorUInt64; 
    };

    template<typename T, typename std::enable_if<std::is_integral<T>::value && (!std::is_unsigned<T>::value) && sizeof(T) == 8, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorInt64; 
    };

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value && sizeof(T) == 4, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorFloat32; 
    };

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value && sizeof(T) == 8, void>::type* = nullptr>
    static TENSOR_TYPE  get_numeric_type() noexcept
    {
        return kTensorFloat64;
    };

    template<typename T0, typename T1, typename std::enable_if<std::is_same<T0, T1>::value, void>::type* = nullptr>
    static T0 cast_value(T1 src) noexcept
    {
        return src;
    }

    template<typename T0, typename T1, typename std::enable_if<!std::is_same<T0, T1>::value && std::is_integral<T0>::value && std::is_floating_point<T1>::value  && (4 == sizeof(T1)), void>::type* = nullptr>
    static T0 cast_value(T1 src) noexcept
    {
        return (T0)(src < 0.0f ? src - 0.5f : src + 0.5f);
    }

    template<typename T0, typename T1, typename std::enable_if<!std::is_same<T0, T1>::value && std::is_integral<T0>::value && std::is_floating_point<T1>::value  && (8 == sizeof(T1)), void>::type* = nullptr>
    static T0 cast_value(T1 src) noexcept
    {
        return (T0)(src < 0.0 ? src - 0.5 : src + 0.5);
    }

    template<typename T0, typename T1, typename std::enable_if<!std::is_same<T0, T1>::value && !(std::is_integral<T0>::value && std::is_floating_point<T1>::value), void>::type* = nullptr>
    static T0 cast_value(T1 src) noexcept
    {
        return (T0)(src);
    }


public:
    NumericType    value_;
};

typedef TensorNumeric<unsigned char>                TensorUInt8;
typedef TensorNumeric<char>                         TensorInt8;
typedef TensorNumeric<unsigned short int>           TensorUInt16;
typedef TensorNumeric<short int>                    TensorInt16;
typedef TensorNumeric<unsigned int>                 TensorUInt32;
typedef TensorNumeric<int>                          TensorInt32;
typedef TensorNumeric<unsigned long long int>       TensorUInt64;
typedef TensorNumeric<long long int>                TensorInt64;
typedef TensorNumeric<float>                        TensorFloat32;
typedef TensorNumeric<double>                       TensorFloat64;

template<typename T, typename std::enable_if<std::is_arithmetic<T>::value, void>::type* = nullptr>
inline T GetTensorNumericValue(Tensor const& numeric_tensor) noexcept
{
    T result = 0;
    switch(numeric_tensor.GetType())
    {
        case kTensorUInt8:
            result = (static_cast<TensorUInt8 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorInt8:
            result = (static_cast<TensorInt8 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorUInt16:
            result = (static_cast<TensorUInt16 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorInt16:
            result = (static_cast<TensorInt16 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorUInt32:
            result = (static_cast<TensorUInt32 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorInt32:
            result = (static_cast<TensorInt32 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorUInt64:
            result = (static_cast<TensorUInt64 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorInt64:
            result = (static_cast<TensorInt64 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorFloat32:
            result = (static_cast<TensorFloat32 const&>(numeric_tensor)).CastValue<T>();
            break;
        case kTensorFloat64:
            result = (static_cast<TensorFloat64 const&>(numeric_tensor)).CastValue<T>();
            break;
        default:
            return (T)(0);
            break;
    }
    return result;
}

class TensorString : public Tensor, public std::string
{
public:
    TensorString() noexcept {;};
    virtual ~TensorString() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorString(TensorString const& other) noexcept : std::string(other) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param other other std::string object.
     *
     */
    TensorString(std::string const& other) noexcept : std::string(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorString(TensorString&& other) noexcept : std::string(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other std::string object
     */
    TensorString(std::string&& other) noexcept : std::string(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorString& operator=(TensorString const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::string::operator=(other);
        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other std::string object
     */
    TensorString& operator=(std::string const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::string::operator=(other);
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorString& operator=(TensorString&& other) noexcept
    {
        if(&other == this)
            return *this;
        
        this->std::string::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorString& operator=(std::string&& other) noexcept
    {
        if(&other == this)
            return *this;
        
        this->std::string::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorString; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorString; };
};

class TensorPoint : public Tensor, public cv::Point2f
{
public:
    TensorPoint() noexcept {;};
    virtual ~TensorPoint() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorPoint(TensorPoint const& other) noexcept : cv::Point2f(other) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param other other point object.
     *
     */
    TensorPoint(cv::Point2f const& other) noexcept : cv::Point2f(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorPoint(TensorPoint&& other) noexcept : cv::Point2f(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorPoint(cv::Point2f&& other) noexcept : cv::Point2f(std::move(other)) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param x
     * 
     * @param y
     *
     */
    TensorPoint(float x, float y) noexcept : cv::Point2f(x, y) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorPoint& operator=(TensorPoint const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Point2f::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other point object
     */
    TensorPoint& operator=(cv::Point2f const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Point2f::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorPoint& operator=(TensorPoint&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Point2f::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other point object
     */
    TensorPoint& operator=(cv::Point2f&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Point2f::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorPoint; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorPoint; };
};

class TensorBox : public Tensor, public vision::Box
{
public:
    TensorBox() noexcept {;};
    virtual ~TensorBox() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorBox(TensorBox const& other) noexcept : vision::Box(other) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param other other box object.
     *
     */
    TensorBox(vision::Box const& other) noexcept : vision::Box(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorBox(TensorBox&& other) noexcept : vision::Box(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorBox(vision::Box&& other) noexcept : vision::Box(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorBox& operator=(TensorBox const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::Box::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other box object
     */
    TensorBox& operator=(vision::Box const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::Box::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorBox& operator=(TensorBox&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::Box::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other point object
     */
    TensorBox& operator=(vision::Box&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::Box::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorBox; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorBox; };

    static void  NormalizeBox(vision::Box& other)
    {
        float tmp = 0.0f;
        if(other.x2 < other.x1)
        {
            tmp      = other.x1;
            other.x1 = other.x2;
            other.x2 = tmp;
        }

        if(other.y2 < other.y1)
        {
            tmp      = other.y1;
            other.y1 = other.y2;
            other.y2 = tmp;
        }
    }
};

class TensorImage : public Tensor, public cv::Mat
{
public:
    TensorImage() noexcept {;};
    virtual ~TensorImage() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorImage(TensorImage const& other) noexcept : cv::Mat(other) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param other other image object.
     *
     */
    TensorImage(cv::Mat const& other) noexcept : cv::Mat(other) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param other other image object.
     *
     */
    TensorImage(int rows, int cols, int type) noexcept : cv::Mat(rows, cols, type) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorImage(TensorImage&& other) noexcept : cv::Mat(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorImage(cv::Mat&& other) noexcept : cv::Mat(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorImage& operator=(TensorImage const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Mat::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other image object
     */
    TensorImage& operator=(cv::Mat const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Mat::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorImage& operator=(TensorImage&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Mat::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other image object
     */
    TensorImage& operator=(cv::Mat&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->cv::Mat::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorImage; };  

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorImage; };
};

class TensorBoxesMap : public Tensor, public std::map<std::string, std::vector<vision::Box> >
{
public:
    TensorBoxesMap() noexcept {;};
    virtual ~TensorBoxesMap() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorBoxesMap(TensorBoxesMap const& other) noexcept : std::map<std::string, std::vector<vision::Box> >(other) {;};

    /**
     * @brief parameter Constructor
     * 
     * @param other other detect object.
     *
     */
    TensorBoxesMap(std::map<std::string, std::vector<vision::Box> > const& other) noexcept : std::map<std::string, std::vector<vision::Box> >(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorBoxesMap(TensorBoxesMap&& other) noexcept : std::map<std::string, std::vector<vision::Box> >(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorBoxesMap(std::map<std::string, std::vector<vision::Box> >&& other) noexcept : std::map<std::string, std::vector<vision::Box> >(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorBoxesMap& operator=(TensorBoxesMap const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::map<std::string, std::vector<vision::Box> >::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other map object
     */
    TensorBoxesMap& operator=(std::map<std::string, std::vector<vision::Box> > const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::map<std::string, std::vector<vision::Box> >::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorBoxesMap& operator=(TensorBoxesMap&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::map<std::string, std::vector<vision::Box> >::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other map object
     */
    TensorBoxesMap& operator=(std::map<std::string, std::vector<vision::Box> >&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::map<std::string, std::vector<vision::Box> >::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorBoxesMap; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorBoxesMap; };

    static void  NormalizeBox(std::vector<vision::Box>& other)
    {
        for (vision::Box& b : other) {
            float tmp = 0.0f;
            if(b.x2 < b.x1)
            {
                tmp  = b.x1;
                b.x1 = b.x2;
                b.x2 = tmp;
            }

            if(b.y2 < b.y1)
            {
                tmp  = b.y1;
                b.y1 = b.y2;
                b.y2 = tmp;
            }
        }
        
    }
};

class TensorKeypoints : public Tensor, public std::vector<cv::Point2f>
{
public:
    TensorKeypoints() noexcept {;};
    virtual ~TensorKeypoints() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorKeypoints(TensorKeypoints const& other) noexcept : std::vector<cv::Point2f>(other) {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other keypoints object.
     *
     */
    TensorKeypoints(std::vector<cv::Point2f> const& other) noexcept : std::vector<cv::Point2f>(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorKeypoints(TensorKeypoints&& other) noexcept : std::vector<cv::Point2f>(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other keypoints object
     */
    TensorKeypoints(std::vector<cv::Point2f>&& other) noexcept : std::vector<cv::Point2f>(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorKeypoints& operator=(TensorKeypoints const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<cv::Point2f>::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other keypoints object
     */
    TensorKeypoints& operator=(std::vector<cv::Point2f> const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<cv::Point2f>::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorKeypoints& operator=(TensorKeypoints&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<cv::Point2f>::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other keypoints object
     */
    TensorKeypoints& operator=(std::vector<cv::Point2f>&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<cv::Point2f>::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorKeyPoints; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorKeyPoints; };
};

class TensorAttributes : public Tensor, public std::vector<std::vector<float> >
{
public:
    TensorAttributes() noexcept {;};
    virtual ~TensorAttributes() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorAttributes(TensorAttributes const& other) noexcept : std::vector<std::vector<float> >(other) {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other attributes object.
     *
     */
    TensorAttributes(std::vector<std::vector<float> > const& other) noexcept : std::vector<std::vector<float> >(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorAttributes(TensorAttributes&& other) noexcept : std::vector<std::vector<float> >(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other attributes object
     */
    TensorAttributes(std::vector<std::vector<float> >&& other) noexcept : std::vector<std::vector<float> >(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorAttributes& operator=(TensorAttributes const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<std::vector<float> >::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other attributes object
     */
    TensorAttributes& operator=(std::vector<std::vector<float> > const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<std::vector<float> >::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorAttributes& operator=(TensorAttributes&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<std::vector<float> >::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other attributes object
     */
    TensorAttributes& operator=(std::vector<std::vector<float> >&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<std::vector<float> >::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorAttributes; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorAttributes; };
};

class TensorFeature : public Tensor, public std::vector<float>
{
public:
    TensorFeature() noexcept {;};
    virtual ~TensorFeature() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorFeature(TensorFeature const& other) noexcept : std::vector<float>(other) {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other reid object.
     *
     */
    TensorFeature(std::vector<float> const& other) noexcept : std::vector<float>(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorFeature(TensorFeature&& other) noexcept : std::vector<float>(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other reid object
     */
    TensorFeature(std::vector<float>&& other) noexcept : std::vector<float>(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorFeature& operator=(TensorFeature const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<float>::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other reid object
     */
    TensorFeature& operator=(std::vector<float> const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<float>::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorFeature& operator=(TensorFeature&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<float>::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other reid object
     */
    TensorFeature& operator=(std::vector<float>&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<float>::operator=(std::move(other));
        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorFeature; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorFeature; };
};

class TensorFeatureMaps : public Tensor, public std::vector<vision::FeatureMap>
{
public:
    TensorFeatureMaps() noexcept {;};
    virtual ~TensorFeatureMaps() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorFeatureMaps(TensorFeatureMaps const& other) noexcept : std::vector<vision::FeatureMap>(other) {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other feature_maps object.
     *
     */
    TensorFeatureMaps(std::vector<vision::FeatureMap> const& other) noexcept : std::vector<vision::FeatureMap>(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorFeatureMaps(TensorFeatureMaps&& other) noexcept : std::vector<vision::FeatureMap>(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other feature_maps object
     */
    TensorFeatureMaps(std::vector<vision::FeatureMap>&& other) noexcept : std::vector<vision::FeatureMap>(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorFeatureMaps& operator=(TensorFeatureMaps const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<vision::FeatureMap>::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other feature_maps object
     */
    TensorFeatureMaps& operator=(std::vector<vision::FeatureMap> const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<vision::FeatureMap>::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorFeatureMaps& operator=(TensorFeatureMaps&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<vision::FeatureMap>::operator=(other);
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other feature_maps object
     */
    TensorFeatureMaps& operator=(std::vector<vision::FeatureMap>&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<vision::FeatureMap>::operator=(std::move(other));

        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorFeatureMaps; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorFeatureMaps; };
};

#if 0
class TensorDLCVOut : public Tensor, public vision::DLCVOut
{
public:
    TensorDLCVOut() noexcept {;};
    virtual ~TensorDLCVOut() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor object.
     *
     */
    TensorDLCVOut(TensorDLCVOut const& other) noexcept : vision::DLCVOut(other) {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other dlcv_out object.
     *
     */
    TensorDLCVOut(vision::DLCVOut const& other) noexcept : vision::DLCVOut(other) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor object
     */
    TensorDLCVOut(TensorDLCVOut&& other) noexcept : vision::DLCVOut(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other dlcv_out object
     */
    TensorDLCVOut(vision::DLCVOut&& other) noexcept : vision::DLCVOut(std::move(other)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor object
     */
    TensorDLCVOut& operator=(TensorDLCVOut const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::DLCVOut::operator=(other);

        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param other other dlcv_out object
     */
    TensorDLCVOut& operator=(vision::DLCVOut const& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::DLCVOut::operator=(other);

        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor object
     */
    TensorDLCVOut& operator=(TensorDLCVOut&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::DLCVOut::operator=(other);
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other dlcv_out object
     */
    TensorDLCVOut& operator=(vision::DLCVOut&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->vision::DLCVOut::operator=(std::move(other));

        return *this;
    };

    TENSOR_TYPE         GetType() const noexcept   { return kTensorDLCVOut; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorDLCVOut; };
};
#endif

class TensorReference : public Tensor
{
public:
    TensorReference() noexcept : reference_(nullptr) {;};
    TensorReference(TensorReference const& other) noexcept:reference_(other.reference_){;};
    TensorReference(Tensor* tensor) noexcept:reference_(tensor){;};
    virtual ~TensorReference() noexcept { reference_ = nullptr;};

    /**
     * @brief Copy assignment
     *
     * @param other other reference object
     */
    TensorReference& operator=(TensorReference const& other) noexcept
    {
        if(&other == this)
            return *this;

        reference_= other.reference_;

        return *this;
    };

    /**
     * @brief Move constructor
     *
     * @param other other tensor reference.
     *
     */
    TensorReference(TensorReference&& other) noexcept : reference_(other.reference_) {;};

    /**
     * @brief Move assignment
     *
     * @param other other tensor reference
     */
    TensorReference& operator=(TensorReference&& other) noexcept
    {
        if(&other == this)
            return *this;

        reference_ = other.reference_;
        return *this;
    };

    TensorReference(Tensor const& other) = delete;
    TensorReference& operator=(Tensor const& other) = delete;
    TensorReference(Tensor&& other) = delete;

    TENSOR_TYPE         GetType() const noexcept   { return kTensorReference; };

    static TENSOR_TYPE  GeClassType() noexcept     { return kTensorReference; };
public:
    Tensor*      reference_;
};

template <typename ELEMENT_TYPE>
class TensorVector : public Tensor, public std::vector<ELEMENT_TYPE>
{
public:
    TensorVector() noexcept : std::vector<ELEMENT_TYPE>(0) { ; } ;
    virtual ~TensorVector() noexcept {;};

    /**
     * @brief copy Constructor
     * 
     * @param other other tensor vector.
     *
     */
    TensorVector(TensorVector const& other) noexcept : std::vector<ELEMENT_TYPE>(other) { ; } ;

    /**
     * @brief copy Constructor
     * 
     * @param vec_tensor other tensor vector.
     *
     */
    TensorVector(std::vector<ELEMENT_TYPE> const& vec_tensor) noexcept : std::vector<ELEMENT_TYPE>(vec_tensor) { ; } ;

    /**
     * @brief Move constructor
     *
     * @param other other tensor vector.
     *
     */
    TensorVector(TensorVector&& other) noexcept : std::vector<ELEMENT_TYPE>(std::move(other)) {;};

    /**
     * @brief Move constructor
     *
     * @param other other tensor vector
     * 
     */
    TensorVector(std::vector<ELEMENT_TYPE>&& vec_tensor) noexcept : std::vector<ELEMENT_TYPE>(std::move(vec_tensor)) {;};

    /**
     * @brief Copy assignment
     *
     * @param other other tensor vector
     */
    TensorVector& operator=(TensorVector const& other) noexcept
    {
        if(&other == this)
            return *this;
        this->std::vector<ELEMENT_TYPE>::operator=(other);
        return *this;
    };

    /**
     * @brief Copy assignment
     *
     * @param vec_tensor other tensor vector
     */
    TensorVector& operator=(std::vector<ELEMENT_TYPE> const& vec_tensor) noexcept
    {
        if(&vec_tensor == this)
            return *this;
        this->std::vector<ELEMENT_TYPE>::operator=(vec_tensor);
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param other other tensor vector
     */
    TensorVector& operator=(TensorVector&& other) noexcept
    {
        if(&other == this)
            return *this;

        this->std::vector<ELEMENT_TYPE>::operator=(std::move(other));
        return *this;
    };

    /**
     * @brief Move assignment
     *
     * @param vec_tensor other tensor vector
     */
    TensorVector& operator=(std::vector<ELEMENT_TYPE>&& vec_tensor) noexcept
    {
        if(&vec_tensor == this)
            return *this;

        this->std::vector<ELEMENT_TYPE>::operator=(std::move(vec_tensor));

        return *this;
    };


    TENSOR_TYPE  GetType() const noexcept   
    { 
        return get_vector_type<ELEMENT_TYPE>();
    };

    static TENSOR_TYPE  GeClassType() noexcept
    {
        return get_vector_type<ELEMENT_TYPE>();
    };

private:
    template<typename T, typename std::enable_if<std::is_same<T, unsigned char>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorUInt8Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, char>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorInt8Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, unsigned short int>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorUInt16Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, short int>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorInt16Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, unsigned int>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorUInt32Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, int>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorInt32Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, unsigned long long int>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorUInt64Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, long long int>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorInt64Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, float>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorFloat32Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, double>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorFloat64Vector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, vision::Box>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorBoxVector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, cv::Mat>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorImageVector; 
    };
    
    template<typename T, typename std::enable_if<std::is_same<T, std::vector<cv::Point2f> >::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorKeypointsVector; 
    };

    template<typename T, typename std::enable_if<std::is_same<T, std::vector<std::vector<float> > >::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorAttributesVector;
    };
/*
    template<typename T, typename std::enable_if<std::is_same<T, std::vector<vision::DLCVOut> >::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorDLCVOutVector; 
    };
*/
    template<typename T, typename std::enable_if<std::is_same<T, Tensor*>::value, void>::type* = nullptr>
    static TENSOR_TYPE  get_vector_type() noexcept
    {
        return kTensorPtrVector;
    };
};

typedef TensorVector<unsigned char>                      TensorUInt8Vector;
typedef TensorVector<char>                               TensorInt8Vector;
typedef TensorVector<unsigned short int>                 TensorUInt16Vector;
typedef TensorVector<short int>                          TensorInt16Vector;
typedef TensorVector<unsigned int>                       TensorUInt32Vector;
typedef TensorVector<int>                                TensorInt32Vector;
typedef TensorVector<unsigned long long>                 TensorUInt64Vector;
typedef TensorVector<long long>                          TensorInt64Vector;
typedef TensorVector<float>                              TensorFloat32Vector;
typedef TensorVector<double>                             TensorFloat64Vector;
typedef TensorVector<vision::Box>                        TensorBoxVector;
typedef TensorVector<cv::Mat>                            TensorImageVector;
typedef TensorVector<std::vector<cv::Point2f> >          TensorKeypointsVector;
typedef TensorVector<std::vector<std::vector<float> > >  TensorAttributesVector;
//typedef TensorVector<vision::DLCVOut>            TensorDLCVOutVector;
typedef TensorVector<Tensor*>                            TensorPtrVector;

}  //namespace vision_graph

#endif
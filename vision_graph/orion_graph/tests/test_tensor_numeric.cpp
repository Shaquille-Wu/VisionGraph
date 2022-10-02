#include <algorithm> 
#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "../vision_graph/include/graph_tensor.h"

TEST(TensorNumericTest, tensor_uint8) 
{
    vision_graph::TensorUInt8  t0 = 1;
    vision_graph::TensorUInt8  t1 = 2.0f;

    EXPECT_EQ(t0.value_, 1);
    EXPECT_EQ(t1.value_, 2);

    t0 = 3.0f;
    t1 = 4.0;

    EXPECT_EQ(t0.value_, 3);
    EXPECT_EQ(t1.value_, 4);
}

TEST(TensorNumericTest, tensor_int8) 
{
    vision_graph::TensorInt8  t0 = 1;
    vision_graph::TensorInt8  t1 = -2.0f;

    EXPECT_EQ(t0.value_,  1);
    EXPECT_EQ(t1.value_, -2);

    t0 = 3.0f;
    t1 = -4.0;

    EXPECT_EQ(t0.value_,  3);
    EXPECT_EQ(t1.value_, -4);
}

TEST(TensorNumericTest, tensor_uint16) 
{
    vision_graph::TensorUInt16  t0 = 1;
    vision_graph::TensorUInt16  t1 = 2.0f;

    EXPECT_EQ(t0.value_, 1);
    EXPECT_EQ(t1.value_, 2);

    t0 = 3.0f;
    t1 = 4.0;

    EXPECT_EQ(t0.value_, 3);
    EXPECT_EQ(t1.value_, 4);
}

TEST(TensorNumericTest, tensor_int16) 
{
    vision_graph::TensorInt16  t0 = 1;
    vision_graph::TensorInt16  t1 = -2.0f;

    EXPECT_EQ(t0.value_,  1);
    EXPECT_EQ(t1.value_, -2);

    t0 = 3.0f;
    t1 = -4.0;

    EXPECT_EQ(t0.value_,  3);
    EXPECT_EQ(t1.value_, -4);
}

TEST(TensorNumericTest, tensor_uint32) 
{
    vision_graph::TensorUInt32  t0 = 1;
    vision_graph::TensorUInt32  t1 = 2.0f;

    EXPECT_EQ(t0.value_, 1U);
    EXPECT_EQ(t1.value_, 2U);

    t0 = 3.0f;
    t1 = 4.0;

    EXPECT_EQ(t0.value_, 3U);
    EXPECT_EQ(t1.value_, 4U);
}

TEST(TensorNumericTest, tensor_int32) 
{
    vision_graph::TensorInt32  t0 = 1;
    vision_graph::TensorInt32  t1 = -2.0f;

    EXPECT_EQ(t0.value_,  1);
    EXPECT_EQ(t1.value_, -2);

    t0 = 3.0f;
    t1 = -4.0;

    EXPECT_EQ(t0.value_,  3);
    EXPECT_EQ(t1.value_, -4);
}

TEST(TensorNumericTest, tensor_uint64) 
{
    vision_graph::TensorUInt64  t0 = 1;
    vision_graph::TensorUInt64  t1 = 2.0f;

    EXPECT_EQ(t0.value_, 1ULL);
    EXPECT_EQ(t1.value_, 2ULL);

    t0 = 3.0f;
    t1 = 4.0;

    EXPECT_EQ(t0.value_, 3ULL);
    EXPECT_EQ(t1.value_, 4ULL);
}

TEST(TensorNumericTest, tensor_int64) 
{
    vision_graph::TensorInt64  t0 = 1;
    vision_graph::TensorInt64  t1 = -2.0f;

    EXPECT_EQ(t0.value_,  1LL);
    EXPECT_EQ(t1.value_, -2LL);

    t0 = 3.0f;
    t1 = -4.0;

    EXPECT_EQ(t0.value_,  3LL);
    EXPECT_EQ(t1.value_, -4LL);
}

TEST(TensorNumericTest, tensor_float32) 
{
    vision_graph::TensorFloat32  t0 = 1;
    vision_graph::TensorFloat32  t1 = 2.0f;

    EXPECT_NEAR(t0.value_, 1.0f, 1e-6);
    EXPECT_EQ(t1.value_,   2.0f);

    t0 = 3.0f;
    t1 = 4.0;

    EXPECT_EQ(t0.value_,   3.0f);
    EXPECT_NEAR(t1.value_, 4.0f, 1e-6f);
}

TEST(TensorNumericTest, tensor_float64) 
{
    vision_graph::TensorFloat64  t0 = 1.0;
    vision_graph::TensorFloat64  t1 = -2.0f;

    EXPECT_EQ(t0.value_,    1.0);
    EXPECT_NEAR(t1.value_, -2.0, 1e-15);

    t0 = 3.0f;
    t1 = -4.0;

    EXPECT_NEAR(t0.value_,  3.0, 1e-15);
    EXPECT_EQ(t1.value_,   -4.0);
}

TEST(TensorNumericTest, tensor_operator_assign) 
{
    vision_graph::TensorUInt8    t00 = 1;
    vision_graph::TensorInt8     t01 = 2;
    vision_graph::TensorUInt16   t02 = 3;
    vision_graph::TensorInt16    t03 = 4;
    vision_graph::TensorUInt32   t04 = 5;
    vision_graph::TensorInt32    t05 = 6;
    vision_graph::TensorUInt64   t06 = 7;
    vision_graph::TensorInt64    t07 = 8;
    vision_graph::TensorFloat32  t08 = 1.0f;
    vision_graph::TensorFloat64  t09 = 2.0;

    vision_graph::TensorUInt8    t10 = t09;
    vision_graph::TensorInt8     t11 = t08;
    vision_graph::TensorUInt16   t12 = t07;
    vision_graph::TensorInt16    t13 = t06;
    vision_graph::TensorUInt32   t14 = t05;
    vision_graph::TensorInt32    t15 = t04;
    vision_graph::TensorUInt64   t16 = t03;
    vision_graph::TensorInt64    t17 = t02;
    vision_graph::TensorFloat32  t18 = t01;
    vision_graph::TensorFloat64  t19 = t00;

    EXPECT_EQ(t10.value_,     2U);
    EXPECT_EQ(t11.value_,     1);
    EXPECT_EQ(t12.value_,     8U);
    EXPECT_EQ(t13.value_,     7);
    EXPECT_EQ(t14.value_,     6U);
    EXPECT_EQ(t15.value_,     5);
    EXPECT_EQ(t16.value_,     4ULL);
    EXPECT_EQ(t17.value_,     3LL);
    EXPECT_NEAR(t18.value_,   2.0f, 1e-6f);
    EXPECT_NEAR(t19.value_,   1.0,  1e-15);

    t10 = t00;
    t11 = t01;
    t12 = t02;
    t13 = t03;
    t14 = t04;
    t15 = t05;
    t16 = t06;
    t17 = t07;
    t18 = t08;
    t19 = t09;

    EXPECT_EQ(t10.value_,     1);
    EXPECT_EQ(t11.value_,     2);
    EXPECT_EQ(t12.value_,     3);
    EXPECT_EQ(t13.value_,     4);
    EXPECT_EQ(t14.value_,     5U);
    EXPECT_EQ(t15.value_,     6);
    EXPECT_EQ(t16.value_,     7ULL);
    EXPECT_EQ(t17.value_,     8);
    EXPECT_NEAR(t18.value_,   1.0f, 1e-6f);
    EXPECT_NEAR(t19.value_,   2.0,  1e-15);
}

TEST(TensorNumericTest, tensor_getvalue) 
{
    vision_graph::TensorUInt8    t00 = 1;
    vision_graph::TensorInt8     t01 = 2;
    vision_graph::TensorUInt16   t02 = 3;
    vision_graph::TensorInt16    t03 = 4;
    vision_graph::TensorUInt32   t04 = 5;
    vision_graph::TensorInt32    t05 = 6;
    vision_graph::TensorUInt64   t06 = 7;
    vision_graph::TensorInt64    t07 = 8;
    vision_graph::TensorFloat32  t08 = 1.0f;
    vision_graph::TensorFloat64  t09 = 2.0;

    double                   t10 = t00.CastValue<double>();
    float                    t11 = t01.CastValue<float>();
    long long int            t12 = t02.CastValue<long long int>();
    unsigned long long int   t13 = t03.CastValue<unsigned long long int>();
    int                      t14 = t04.CastValue<int>();
    unsigned int             t15 = t05.CastValue<unsigned int>();
    short int                t16 = t06.CastValue<short int>();
    unsigned short int       t17 = t07.CastValue<unsigned short int>();
    char                     t18 = t08.CastValue<char>();
    unsigned char            t19 = t09.CastValue<unsigned char>();

    EXPECT_NEAR(t10,     1.0,  1e-15);
    EXPECT_NEAR(t11,     2.0f, 1e-6);
    EXPECT_EQ(t12,       3);
    EXPECT_EQ(t13,       4ULL);
    EXPECT_EQ(t14,       5);
    EXPECT_EQ(t15,       6U);
    EXPECT_EQ(t16,       7);
    EXPECT_EQ(t17,       8U);
    EXPECT_EQ(t18,       1);
    EXPECT_EQ(t19,       2U);
}
// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include "libvis/logging.h"
#include <gtest/gtest.h>

#include "libvis/image.h"

using namespace vis;

// Tests the provided ways to iterate over images.
TEST(Image, Iteration) {
  Image<u32> image(16, 16);
  
  // Set the image to these two values alternatingly, starting with kTestValue1.
  constexpr u32 kTestValue1 = 42;
  constexpr u32 kTestValue2 = 43;
  image.SetTo(kTestValue1);
  
  // Test FOREACH_PIXEL_XY.
  FOREACH_PIXEL_XY(x, y, image) {
    EXPECT_EQ(kTestValue1, image(x, y));
    image(x, y) = kTestValue2;
  }
  
  // Test pixels().
  for (u32& value : image.pixels()) {
    EXPECT_EQ(kTestValue2, value);
    value = kTestValue1;
  }
  
  // Iterate manually.
  for (u32 y = 0; y < image.height(); ++ y) {
    const u32* row = image.row(y);
    for (u32 x = 0; x < image.width(); ++ x) {
      EXPECT_EQ(kTestValue1, row[x]);
    }
  }
}

// Tests copying one image to another.
TEST(Image, Copy) {
  u32 image_data[] = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9};
  Image<u32> image(3, 3, image_data);
  
  // Test copy constructor.
  {
    Image<u32> other_image(image);
    EXPECT_TRUE(image == other_image);
  }
  
  // Test SetTo().
  {
    Image<u32> other_image;
    other_image.SetSizeToMatch(image);
    other_image.SetTo(image);
    EXPECT_TRUE(image == other_image);
    
    Image<u32> other_image_different_settings;
    other_image_different_settings.SetSize(3, 3, 16, 1 * sizeof(void*));
    other_image_different_settings.SetTo(image);
    EXPECT_TRUE(image == other_image_different_settings);
  }
  
  // Test SetRectTo().
  {
    Image<u32> other_image;
    other_image.SetSizeToMatch(image);
    other_image.SetRectTo(
        /*dest_offset*/ Vector2i(0, 0),
        /*size*/ image.size(),
        /*src_offset*/ Vector2i(0, 0),
        image);
    EXPECT_TRUE(image == other_image);
  }
}

// Tests bilinear interpolation.
TEST(Image, InterpolateBilinear) {
  u32 image_data[] = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9};
  Image<u32> image(3, 3, image_data);
  
  EXPECT_EQ(1.f, image.InterpolateBilinear(Vector2f(0, 0)));
  EXPECT_EQ(2.f, image.InterpolateBilinear(Vector2f(1, 0)));
  EXPECT_EQ(4.f, image.InterpolateBilinear(Vector2f(0, 1)));
  EXPECT_EQ(1.5f, image.InterpolateBilinear(Vector2f(0.5f, 0)));
}

// Tests bicubic interpolation.
TEST(Image, InterpolateBicubic) {
  Vec2f image_data[] = {
      Vec2f(0, 0), Vec2f(0, 0), Vec2f(0, 0), Vec2f(0, 0),
      Vec2f(0, 0), Vec2f(1, 1), Vec2f(2, 2), Vec2f(3, 3),
      Vec2f(0, 0), Vec2f(4, 4), Vec2f(5, 5), Vec2f(6, 6),
      Vec2f(0, 0), Vec2f(7, 7), Vec2f(8, 8), Vec2f(9, 9)};
  Image<Vec2f> image(4, 4, image_data);
  
  EXPECT_EQ(Vec2f(1, 1), image.InterpolateBicubicVector(Vector2f(1, 1)));
  EXPECT_EQ(Vec2f(2, 2), image.InterpolateBicubicVector(Vector2f(2, 1)));
  EXPECT_EQ(Vec2f(4, 4), image.InterpolateBicubicVector(Vector2f(1, 2)));
}

// Tests downscaling to half size.
TEST(Image, DownscaleToHalfSize) {
  float image_data[] = {
      1, 1, 2, 2,
      2, 2, 3, 3,
      4, 4, 5, 5,
      6, 6, 5, 5};
  float image_data_result[] = {
      1.5, 2.5,
      5, 5};
  Image<float> image(4, 4, image_data);
  Image<float> image_result;
  image.DownscaleToHalfSize(&image_result);
  Image<float> image_result_expected(2, 2, image_data_result);
  EXPECT_TRUE(image_result_expected == image_result);
}

// Tests average calculation.
TEST(Image, CalcAverage) {
  u32 image_data[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9};
  Image<u32> image(3, 3, image_data);
  
  EXPECT_FLOAT_EQ(5.f, image.CalcAverage<float>());
  EXPECT_FLOAT_EQ(5.f, image.CalcAverageWhileExcluding<float>(5.f));
}

// Tests median calculation.
TEST(Image, CalcMedian) {
  u32 image_data[] = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9};
  Image<u32> image(3, 3, image_data);
  
  EXPECT_EQ(5u, image.CalcMedian());
  EXPECT_EQ(6u, image.CalcMedianWhileExcluding(5));
}

// Tests minimum calculation.
TEST(Image, CalcMin) {
  u32 image_data[] = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9};
  Image<u32> image(3, 3, image_data);
  
  EXPECT_EQ(1u, image.CalcMin());
  EXPECT_EQ(2u, image.CalcMinWhileExcluding(1));
}

// Tests maximum calculation.
TEST(Image, CalcMax) {
  u32 image_data[] = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9};
  Image<u32> image(3, 3, image_data);
  
  EXPECT_EQ(9u, image.CalcMax());
  EXPECT_EQ(8u, image.CalcMaxWhileExcluding(9));
}

// Tests the FlipX() function on uneven- and even-sized images.
TEST(Image, FlipX) {
  // Test uneven size.
  {
    u32 image_data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    u32 image_data_flipped[] = {
        3, 2, 1,
        6, 5, 4,
        9, 8, 7};
    
    Image<u32> image(3, 3, image_data);
    Image<u32> image_flipped(3, 3, image_data_flipped);
    
    EXPECT_FALSE(image == image_flipped);
    image.FlipX();
    EXPECT_TRUE(image == image_flipped);
  }
  
  // Test even size.
  {
    u32 image_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};
    u32 image_data_flipped[] = {
        4, 3, 2, 1,
        8, 7, 6, 5,
        12, 11, 10, 9,
        16, 15, 14, 13};
    
    Image<u32> image(4, 4, image_data);
    Image<u32> image_flipped(4, 4, image_data_flipped);
    
    EXPECT_FALSE(image == image_flipped);
    image.FlipX();
    EXPECT_TRUE(image == image_flipped);
  }
}

// Tests the FlipY() function on uneven- and even-sized images.
TEST(Image, FlipY) {
  // Test uneven size.
  {
    u32 image_data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    u32 image_data_flipped[] = {
        7, 8, 9,
        4, 5, 6,
        1, 2, 3};
    
    Image<u32> image(3, 3, image_data);
    Image<u32> image_flipped(3, 3, image_data_flipped);
    
    EXPECT_FALSE(image == image_flipped);
    image.FlipY();
    EXPECT_TRUE(image == image_flipped);
  }
  
  // Test even size.
  {
    u32 image_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};
    u32 image_data_flipped[] = {
        13, 14, 15, 16,
        9, 10, 11, 12,
        5, 6, 7, 8,
        1, 2, 3, 4};
    
    Image<u32> image(4, 4, image_data);
    Image<u32> image_flipped(4, 4, image_data_flipped);
    
    EXPECT_FALSE(image == image_flipped);
    image.FlipY();
    EXPECT_TRUE(image == image_flipped);
  }
}

// Tests that ApplyAffineFunction() gives the correct result for an example.
TEST(Image, ApplyAffineFunction) {
  Image<u32> image(3, 3);
  image.SetTo(2);
  image.ApplyAffineFunction(2, 1);
  for (const u32& value : image.pixels()) {
    EXPECT_EQ(5u, value);
  }
}

// Tests that an image retains the same content after writing it to disk and
// reading it again.
TEST(Image, ReadWrite) {
  const string kFilepath = "/tmp/libvis_Image_Test_temp_file.png";
  
  // TODO: Can this test iterate over all image I/O handlers which are available
  //       and test them all?
  
  // Test u8.
  {
    u8 image_data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    Image<u8> image(3, 3, image_data);
    
    // Write image to temporary file.
    ASSERT_TRUE(image.Write(kFilepath));
    
    // Read image again.
    Image<u8> image_read;
    ASSERT_TRUE(image_read.Read(kFilepath));
    
    EXPECT_TRUE(image == image_read);
  }
  
  // Test Vec3u8.
  {
    Vec3u8 image_data[] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
        {10, 11, 12}, {13, 14, 15}, {16, 17, 18},
        {19, 20, 21}, {22, 23, 24}, {25, 26, 27}};
    Image<Vec3u8> image(3, 3, image_data);
    
    // Write image to temporary file.
    ASSERT_TRUE(image.Write(kFilepath));
    
    // Read image again.
    Image<Vec3u8> image_read;
    ASSERT_TRUE(image_read.Read(kFilepath));
    
    EXPECT_TRUE(image == image_read);
  }
  
  // Test Vec3u8 conversion to u8.
  {
    Vec3u8 image_data[] = {
        {1, 2, 3}};
    Image<Vec3u8> image(1, 1, image_data);
    
    // Write image to temporary file.
    ASSERT_TRUE(image.Write(kFilepath));
    
    // Read image again.
    Image<u8> image_read;
    ASSERT_TRUE(image_read.Read(kFilepath));
    
    // Conversion takes the first component.
    // TODO: Shall this be changed to do averaging?
    EXPECT_EQ(1, image_read(0, 0));
  }
  
  // Test u8 conversion to Vec3u8.
  {
    u8 image_data[] = {
        2};
    Image<u8> image(1, 1, image_data);
    
    // Write image to temporary file.
    ASSERT_TRUE(image.Write(kFilepath));
    
    // Read image again.
    Image<Vec3u8> image_read;
    ASSERT_TRUE(image_read.Read(kFilepath));
    
    EXPECT_EQ(2, image_read(0, 0)(0));
    EXPECT_EQ(2, image_read(0, 0)(1));
    EXPECT_EQ(2, image_read(0, 0)(2));
  }
}

// Verifies that the channel_count() function returns correct results for some
// examples, and that the bytes per pixels are as expected (tight packing).
TEST(Image, ChannelCountAndBytesPerPixel) {
  Image<u8> image_u8;
  EXPECT_EQ(1u, image_u8.channel_count());
  EXPECT_EQ(1u, image_u8.bytes_per_pixel());
  
  Image<u16> image_u16;
  EXPECT_EQ(1u, image_u16.channel_count());
  EXPECT_EQ(2u, image_u16.bytes_per_pixel());
  
  Image<float> image_float;
  EXPECT_EQ(1u, image_float.channel_count());
  EXPECT_EQ(4u, image_float.bytes_per_pixel());
  
  Image<Vec2f> image_Vec2f;
  EXPECT_EQ(2u, image_Vec2f.channel_count());
  EXPECT_EQ(2u * 4u, image_Vec2f.bytes_per_pixel());
  
  Image<Vec3f> image_Vec3f;
  EXPECT_EQ(3u, image_Vec3f.channel_count());
  EXPECT_EQ(3u * 4u, image_Vec3f.bytes_per_pixel());
  
  Image<Vec4u8> image_Vec4u8;
  EXPECT_EQ(4u, image_Vec4u8.channel_count());
  EXPECT_EQ(4u * 1u, image_Vec4u8.bytes_per_pixel());
  
  Image<Matrix<float, 2, 3>> image_MatrixFloat2x3;
  EXPECT_EQ(2u * 3u, image_MatrixFloat2x3.channel_count());
  EXPECT_EQ(6u * 4u, image_MatrixFloat2x3.bytes_per_pixel());
}

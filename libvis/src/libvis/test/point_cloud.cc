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


#include <gtest/gtest.h>

#include "libvis/point_cloud.h"

using namespace vis;

// Tests that the bounds computation is correct.
TEST(PointCloud, ComputeMinMax) {
  Point3fCloud cloud(4);
  cloud[0] = Point3f(Vec3f(1, 2, 3));
  cloud[1] = Point3f(Vec3f(2, 3, 4));
  cloud[2] = Point3f(Vec3f(3, 4, 5));
  cloud[3] = Point3f(Vec3f(4, 5, 6));
  
  Vec3f min, max;
  cloud.ComputeMinMax(&min, &max);
  
  EXPECT_FLOAT_EQ(1, min.x());
  EXPECT_FLOAT_EQ(2, min.y());
  EXPECT_FLOAT_EQ(3, min.z());
  EXPECT_FLOAT_EQ(4, max.x());
  EXPECT_FLOAT_EQ(5, max.y());
  EXPECT_FLOAT_EQ(6, max.z());
}

TEST(PointCloud, TransformSE3) {
  Point3fCloud cloud(4);
  cloud[0] = Point3f(Vec3f(1, 2, 3));
  cloud[1] = Point3f(Vec3f(2, 3, 4));
  cloud[2] = Point3f(Vec3f(3, 4, 5));
  cloud[3] = Point3f(Vec3f(4, 5, 6));
  
  SE3f transformation(
      AngleAxisf(0.2f, Vec3f(1, 2, 3).normalized()).toRotationMatrix(),
      Vec3f(5, 6, 7));
  
  Point3fCloud expected_result(cloud.size());
  for (usize i = 0; i < expected_result.size(); ++ i) {
    expected_result[i] = Point3f(transformation * cloud[i].position());
  }
  
  Point3fCloud actual_result(cloud);
  actual_result.Transform(transformation);
  
  ASSERT_EQ(expected_result.size(), actual_result.size());
  for (usize i = 0; i < expected_result.size(); ++ i) {
    EXPECT_FLOAT_EQ(expected_result[i].position().x(),
                    actual_result[i].position().x());
    EXPECT_FLOAT_EQ(expected_result[i].position().y(),
                    actual_result[i].position().y());
    EXPECT_FLOAT_EQ(expected_result[i].position().z(),
                    actual_result[i].position().z());
  }
}

TEST(PointCloud, Save) {
  // Position-only cloud
  Point3fCloud position_cloud(2);
  position_cloud[0] = Point3f(Vec3f(1, 2, 3));
  position_cloud[1] = Point3f(Vec3f(2, 3, 4));
  
  std::ofstream obj_file1("/tmp/__position_cloud_test.obj", std::ios::out);  // TODO: use random temporary filename
  position_cloud.WriteAsOBJ(&obj_file1);
  obj_file1.close();
  
  position_cloud.WriteAsPLY("/tmp/__position_cloud_test.ply");
  
  
  // Position-and-color cloud
  Point3fC3u8Cloud position_color_cloud(2);
  position_color_cloud[0] = Point3fC3u8(Vec3f(1, 2, 3), Vec3u8(255, 0, 0));
  position_color_cloud[1] = Point3fC3u8(Vec3f(2, 3, 4), Vec3u8(0, 255, 0));
  
  std::ofstream obj_file2("/tmp/__position_color_cloud_test.obj", std::ios::out);  // TODO: use random temporary filename
  position_color_cloud.WriteAsOBJ(&obj_file2);
  obj_file2.close();
  
  position_color_cloud.WriteAsPLY("/tmp/__position_color_cloud_test.ply");
  
  
  // Position-color-normal cloud
  Point3fC3u8NfCloud position_color_normal_cloud(2);
  position_color_normal_cloud[0] = Point3fC3u8Nf(Vec3f(1, 2, 3), Vec3u8(255, 0, 0), Vec3f(2, 5, 7).normalized());
  position_color_normal_cloud[1] = Point3fC3u8Nf(Vec3f(2, 3, 4), Vec3u8(0, 255, 0), Vec3f(8, 9, 10).normalized());
  
  std::ofstream obj_file3("/tmp/__position_color_normal_cloud_test.obj", std::ios::out);  // TODO: use random temporary filename
  position_color_normal_cloud.WriteAsOBJ(&obj_file3);
  obj_file3.close();
  
  position_color_normal_cloud.WriteAsPLY("/tmp/__position_color_normal_cloud_test.ply");
  
  
  // TODO: Extend this test to load the saved files again and ensure that the results are equal to the original clouds
}

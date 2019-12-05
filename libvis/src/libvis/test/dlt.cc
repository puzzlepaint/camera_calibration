// Copyright 2019 ETH Zürich, Thomas Schöps
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

#include "libvis/dlt.h"

using namespace vis;

// Tests a known result of a homography estimation.
TEST(DLT, KnownTransforms1) {
  vector<Vec2f> input{
      {0, 0},
      {0, 1},
      {1, 1},
      {1, 0}};
  vector<Vec2f> output{
      {0, 0},
      {0, 2},
      {2, 2},
      {2, 0}};
  
  Mat3f GT = Mat3f::Identity();
  GT(0, 0) = 2;
  GT(1, 1) = 2;
  
  constexpr float kEpsilon = 1e-5f;
  
  // DLT
  Mat3f H = DLT(input.data(), output.data(), input.size());
  H /= H(2, 2);  // normalize H(2, 2) = 1
  for (int row = 0; row < 3; ++ row) {
    for (int col = 0; col < 3; ++ col) {
      EXPECT_NEAR(GT(row, col), H(row, col), kEpsilon) << " at (" << row << ", " << col << ")";
    }
  }
  
  // Normalized DLT
  H = NormalizedDLT(input.data(), output.data(), input.size());
  H /= H(2, 2);  // normalize H(2, 2) = 1
  for (int row = 0; row < 3; ++ row) {
    for (int col = 0; col < 3; ++ col) {
      EXPECT_NEAR(GT(row, col), H(row, col), kEpsilon) << " at (" << row << ", " << col << ")";
    }
  }
}

// Tests a known result of a homography estimation.
TEST(DLT, KnownTransforms2) {
  vector<Vec2f> input{
      {0, 0},
      {0, 1},
      {1, 1},
      {1, 0}};
  vector<Vec2f> output{
      {0, 0},
      {-1, 0},
      {-1, 1},
      {0, 1}};
  
  Mat3f GT;
  GT << 0, -1, 0,
        1,  0, 0,
        0,  0, 1;
  
  constexpr float kEpsilon = 1e-5f;
  
  // DLT
  Mat3f H = DLT(input.data(), output.data(), input.size());
  H /= H(2, 2);  // normalize H(2, 2) = 1
  for (int row = 0; row < 3; ++ row) {
    for (int col = 0; col < 3; ++ col) {
      EXPECT_NEAR(GT(row, col), H(row, col), kEpsilon) << " at (" << row << ", " << col << ")";
    }
  }
  
  // Normalized DLT
  H = NormalizedDLT(input.data(), output.data(), input.size());
  H /= H(2, 2);  // normalize H(2, 2) = 1
  for (int row = 0; row < 3; ++ row) {
    for (int col = 0; col < 3; ++ col) {
      EXPECT_NEAR(GT(row, col), H(row, col), kEpsilon) << " at (" << row << ", " << col << ")";
    }
  }
}

// Tests a known result of a homography estimation by applying it.
TEST(DLT, KnownTransformsResult) {
  vector<Vec2f> input{
      {0, 0},
      {0, 1},
      {1, 1},
      {1, 0}};
  vector<Vec2f> output{
      {0, 0},
      {0, 1},
      {2, 2},
      {2, 0}};
  
  Vec2f input_point(1, 0.5);
  Vec2f GT(2, 1);
  
  constexpr float kEpsilon = 1e-5f;
  
  // DLT
  Mat3f H = DLT(input.data(), output.data(), input.size());
  Vec2f result = Vec3f(H * input_point.homogeneous()).hnormalized();
  for (int row = 0; row < GT.rows(); ++ row) {
    EXPECT_NEAR(GT(row), result(row), kEpsilon) << " at (" << row << ")";
  }
  
  // Normalized DLT
  H = NormalizedDLT(input.data(), output.data(), input.size());
  result = Vec3f(H * input_point.homogeneous()).hnormalized();
  for (int row = 0; row < GT.rows(); ++ row) {
    EXPECT_NEAR(GT(row), result(row), kEpsilon) << " at (" << row << ")";
  }
}

// Tests that identical sets of points result in the identity transform.
TEST(DLT, IdentityTransform) {
  srand(0);
  
  vector<Vec2f> points(4);
  for (int i = 0; i < 4; ++ i) {
    points[i] = Vec2f::Random();
  }
  
  Mat3f GT = Mat3f::Identity();
  
  constexpr float kEpsilon = 1e-5f;
  
  // DLT
  Mat3f H = DLT(points.data(), points.data(), points.size());
  H /= H(2, 2);  // normalize H(2, 2) = 1
  for (int row = 0; row < 3; ++ row) {
    for (int col = 0; col < 3; ++ col) {
      EXPECT_NEAR(GT(row, col), H(row, col), kEpsilon) << " at (" << row << ", " << col << ")";
    }
  }
  
  // Normalized DLT
  H = NormalizedDLT(points.data(), points.data(), points.size());
  H /= H(2, 2);  // normalize H(2, 2) = 1
  for (int row = 0; row < 3; ++ row) {
    for (int col = 0; col < 3; ++ col) {
      EXPECT_NEAR(GT(row, col), H(row, col), kEpsilon) << " at (" << row << ", " << col << ")";
    }
  }
}

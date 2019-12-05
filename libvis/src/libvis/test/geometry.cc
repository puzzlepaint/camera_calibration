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


#include <gtest/gtest.h>

#include "libvis/geometry.h"

using namespace vis;

// Tests several cases with known results.
TEST(Geometry, LineLineIntersection) {
  Vec2f intersection;
  
  EXPECT_TRUE(LineLineIntersection(Vec2f(0, 0), Vec2f(2, 0), Vec2f(1, 1), Vec2f(1, -1), &intersection));
  EXPECT_FLOAT_EQ(1, intersection.x());
  EXPECT_FLOAT_EQ(0, intersection.y());
  
  EXPECT_TRUE(LineLineIntersection(Vec2f(0, 0), Vec2f(2, 0), Vec2f(1, -1), Vec2f(1, 1), &intersection));
  EXPECT_FLOAT_EQ(1, intersection.x());
  EXPECT_FLOAT_EQ(0, intersection.y());
  
  EXPECT_TRUE(LineLineIntersection(Vec2f(2, 0), Vec2f(0, 0), Vec2f(1, 1), Vec2f(1, -1), &intersection));
  EXPECT_FLOAT_EQ(1, intersection.x());
  EXPECT_FLOAT_EQ(0, intersection.y());
  
  EXPECT_FALSE(LineLineIntersection(Vec2f(2, 0), Vec2f(0, 0), Vec2f(2, 10), Vec2f(0, 10), &intersection));
}

// Tests several cases with known results.
TEST(Geometry, ConvexPolygonOrientation) {
  vector<Vec2f> polygon = {Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0)};
  EXPECT_EQ(-1, ConvexPolygonOrientation(polygon));
  
  polygon = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1)};
  EXPECT_EQ(1, ConvexPolygonOrientation(polygon));
  
  polygon = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(0, 10)};
  EXPECT_EQ(1, ConvexPolygonOrientation(polygon));
  
  polygon = {Vec2f(0, 0), Vec2f(0, 10), Vec2f(1, 0)};
  EXPECT_EQ(-1, ConvexPolygonOrientation(polygon));
}

// Tests several cases with known results.
TEST(Geometry, PointInsidePolygon) {
  // Square
  vector<Vec2f> polygon = {Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0)};
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.5f, 0.5f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 0.1f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 0.9f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.9f, 0.9f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.9f, 0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(-0.9f, 0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(0.9f, -0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(1.9f, 0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(0.9f, 1.1f), polygon));
  
  // Square, opposite point ordering
  polygon = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1)};
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.5f, 0.5f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 0.1f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 0.9f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.9f, 0.9f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.9f, 0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(-0.9f, 0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(0.9f, -0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(1.9f, 0.1f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(0.9f, 1.1f), polygon));
  
  // Triangle (non-symmetric)
  polygon = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(0, 10)};
  EXPECT_FALSE(PointInsidePolygon(Vec2f(-1.0f, 5.0f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 5.0f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 0.1f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.9f, 0.01f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(0.9f, 5.0f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(2.0f, 5.0f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(5.0f, 0), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(-5.0f, 0), polygon));
  
  // Triangle (non-symmetric), opposite point ordering
  polygon = {Vec2f(0, 0), Vec2f(0, 10), Vec2f(1, 0)};
  EXPECT_FALSE(PointInsidePolygon(Vec2f(-1.0f, 5.0f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 5.0f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.1f, 0.1f), polygon));
  EXPECT_TRUE(PointInsidePolygon(Vec2f(0.9f, 0.01f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(0.9f, 5.0f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(2.0f, 5.0f), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(5.0f, 0), polygon));
  EXPECT_FALSE(PointInsidePolygon(Vec2f(-5.0f, 0), polygon));
}

// Tests several cases with known results.
TEST(Geometry, PolygonArea) {
  vector<Vec2f> polygon = {Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0)};
  EXPECT_FLOAT_EQ(1, PolygonArea(polygon));
  
  polygon = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1)};
  EXPECT_FLOAT_EQ(1, PolygonArea(polygon));
  
  polygon = {Vec2f(0, 0), Vec2f(1, 0), Vec2f(0, 10)};
  EXPECT_FLOAT_EQ(5, PolygonArea(polygon));
  
  polygon = {Vec2f(0, 0), Vec2f(0, 10), Vec2f(1, 0)};
  EXPECT_FLOAT_EQ(5, PolygonArea(polygon));
}

// Tests several cases with known results.
TEST(Geometry, ConvexClipPolygon) {
  // Test the clip path being completely within the polygon. This means that the
  // result must be equal to the clip path (except for possible point order
  // changes).
  vector<Vec2f> polygon = {Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0)};
  vector<Vec2f> convex_clip = {Vec2f(0.1f, 0.1f), Vec2f(0.1f, 0.9f), Vec2f(0.9f, 0.9f), Vec2f(0.9f, 0.1f)};
  vector<Vec2f> result;
  ConvexClipPolygon(polygon, convex_clip, &result);
  
  EXPECT_EQ(4, result.size());
  for (const Vec2f point : result) {
    bool found = false;
    for (const Vec2f clip_point : convex_clip) {
      if ((point - clip_point).squaredNorm() < 1e-10f) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

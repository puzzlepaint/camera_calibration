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


#pragma once

#include <vector>

#include "libvis/eigen.h"

namespace vis {

/// Determinant of 2x2 matrix.
template <typename T>
inline T Det(T a, T b, T c, T d) {
  return a*d - b*c;
}

/// Computes the intersection point between the two given 2D lines a and b. Each
/// line is specified with two points (0 and 1) on the line. Returns false if no
/// intersection was found.
/// Adapted from: https://gist.github.com/TimSC/47203a0f5f15293d2099507ba5da44e6
template <typename T>
bool LineLineIntersection(
    const Eigen::Matrix<T, 2, 1>& a0,
    const Eigen::Matrix<T, 2, 1>& a1,
    const Eigen::Matrix<T, 2, 1>& b0,
    const Eigen::Matrix<T, 2, 1>& b1,
    Eigen::Matrix<T, 2, 1>* result) {
  // http://mathworld.wolfram.com/Line-LineIntersection.html
  T detL1 = Det(a0.x(), a0.y(), a1.x(), a1.y());
  T detL2 = Det(b0.x(), b0.y(), b1.x(), b1.y());
  T x1mx2 = a0.x() - a1.x();
  T x3mx4 = b0.x() - b1.x();
  T y1my2 = a0.y() - a1.y();
  T y3my4 = b0.y() - b1.y();
  
  T xnom = Det(detL1, x1mx2, detL2, x3mx4);
  T ynom = Det(detL1, y1my2, detL2, y3my4);
  T denom = Det(x1mx2, y1my2, x3mx4, y3my4);
  if (denom == 0) {
    // Lines don't seem to cross
    return false;
  }
  
  result->x() = xnom / denom;
  result->y() = ynom / denom;
  if (!isfinite(result->x()) || !isfinite(result->y())) {
    // Probably a numerical issue
    return false;
  }
  
  return true;
}

/// Determines the winding order of the given convex polygon. Given a
/// right-handed coordinate system, returns 1 for counter-clockwise and -1 for clockwise.
/// NOTE: This function requires all edges in the polygon to have a significant
///       length (that is not close to zero), and no successive parallel line
///       segments are allowed.
/// Implementation from: https://en.wikipedia.org/wiki/Curve_orientation#Practical_considerations
template <typename T>
int ConvexPolygonOrientation(const vector<Eigen::Matrix<T, 2, 1>>& polygon) {
  typedef Eigen::Matrix<T, 2, 1> PointT;
  const PointT& a = polygon[0];
  const PointT& b = polygon[1];
  const PointT& c = polygon[2];
  
  float det = (b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y());
  return (det > 0) ? 1 : -1;
}

/// Computes whether the given 2D point lies within the given 2D polygon (given
/// as vector of corner points). The first point of the polygon does not need to
/// be repeated as its last point.
/// For self-intersecting polygons, this uses the 'alternating' fill rule.
/// NOTE: Several values could be pre-computed to speed up this computation if
///       performing it many times for the same polygon.
template <typename T>
bool PointInsidePolygon(
    const Eigen::Matrix<T, 2, 1>& point,
    const vector<Eigen::Matrix<T, 2, 1>>& polygon) {
  typedef Eigen::Matrix<T, 2, 1> PointT;
  
  int right_of_edge_count = 0;
  
  usize size = polygon.size();
  int prev_i = size - 1;
  for (int i = 0; i < size; ++ i) {
    const PointT& edge_start = polygon[prev_i];
    const PointT& edge_end = polygon[i];
    
    T min_y = min(edge_start.y(), edge_end.y());
    T max_y = max(edge_start.y(), edge_end.y());
    if (point.y() >= min_y && point.y() < max_y) {
      T min_x = min(edge_start.x(), edge_end.x());
      T max_x = max(edge_start.x(), edge_end.x());
      if (point.x() >= min_x) {
        if (point.x() < max_x) {
          float relative_x = (point.x() - edge_start.x()) / (edge_end.x() - edge_start.x());
          float relative_y = (point.y() - edge_start.y()) / (edge_end.y() - edge_start.y());
          if ((edge_end.x() > edge_start.x() && relative_x > relative_y) ||
              (edge_end.x() <= edge_start.x() && relative_x < relative_y)) {
            ++ right_of_edge_count;
          }
        } else {
          ++ right_of_edge_count;
        }
      }
    }
    
    prev_i = i;
  }
  
  return right_of_edge_count & 1;
}

/// Computes the given polygon's area. For self-intersecting polygons, parts
/// with differing orientation count against each other, reducing the overall
/// reported area. Parts with equal orientation sum up.
template <typename T>
T PolygonArea(const vector<Eigen::Matrix<T, 2, 1>>& polygon) {
  T result = 0;
  usize size = polygon.size();
  int prev_i = size - 1;
  for (int i = 0; i < size; ++ i) {
    result += (polygon[i].x() - polygon[prev_i].x()) * (polygon[i].y() + polygon[prev_i].y());
    prev_i = i;
  }
  return fabs(0.5f * result);
}

// Clips the given polygon to the area of the given clip polygon, which must be
// convex. Implements the Sutherland–Hodgman algorithm:
// https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
template <typename T>
void ConvexClipPolygon(
    const vector<Eigen::Matrix<T, 2, 1>>& polygon,
    const vector<Eigen::Matrix<T, 2, 1>>& convex_clip,
    vector<Eigen::Matrix<T, 2, 1>>* output) {
  typedef Eigen::Matrix<T, 2, 1> PointT;
  
  usize clip_size = convex_clip.size();
  
  // Determine the winding order of convex_clip (clockwise or counter-clockwise)
  int clip_orientation = ConvexPolygonOrientation(convex_clip);
  
  // Loop over all edges in the clip path
  for (usize clip_edge_start = 0; clip_edge_start < clip_size; ++ clip_edge_start) {
    usize clip_edge_end = (clip_edge_start + 1) % clip_size;
    
    PointT edge = convex_clip[clip_edge_end] - convex_clip[clip_edge_start];
    PointT right_vector = PointT(edge.y(), -edge.x());
    float edge_right = right_vector.dot(convex_clip[clip_edge_end]);
    
    // Clip the polygon with this edge, reading from input and writing to clipped
    const vector<PointT>* input = (clip_edge_start == 0) ? &polygon : output;
    vector<PointT> clipped;
    
    // Loop over input points
    usize input_size = input->size();
    for (usize i = 0; i < input_size; ++ i) {
      const PointT& current_point = input->at(i);
      const PointT& prev_point = input->at((i + input_size - 1) % input_size);
      
      if (clip_orientation * ((right_vector.dot(current_point) > edge_right) ? 1 : -1) < 0) {  // current_point in convex_clip?
        if (clip_orientation * ((right_vector.dot(prev_point) > edge_right) ? 1 : -1) > 0) {  // prev_point not in convex_clip?
          PointT intersection = prev_point;
          LineLineIntersection(prev_point, current_point, convex_clip[clip_edge_end], convex_clip[clip_edge_start], &intersection);
          clipped.push_back(intersection);
        }
        clipped.push_back(current_point);
      } else if (clip_orientation * ((right_vector.dot(prev_point) > edge_right) ? 1 : -1) < 0) {  // prev_point in convex_clip?
        PointT intersection = prev_point;
        LineLineIntersection(prev_point, current_point, convex_clip[clip_edge_end], convex_clip[clip_edge_start], &intersection);
        clipped.push_back(intersection);
      }
    }
    
    *output = clipped;  // TODO: avoid copies
  }
}

}

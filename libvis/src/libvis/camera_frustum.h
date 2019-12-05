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


#pragma once

#include <vector>

#include <Eigen/Geometry>

#include "libvis/camera.h"
#include "libvis/libvis.h"
#include "libvis/sophus.h"

namespace vis {

// Represents a camera frustum for intersection tests.
class CameraFrustum {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // No-op constructor.
  inline CameraFrustum()
      : axes_and_planes_computed_(false) {}
  
  // Creates a frustum for the given camera.
  inline CameraFrustum(const Camera& camera, float min_depth, float max_depth, const SE3f& global_T_camera)
      : axes_and_planes_computed_(false) {
    Create(camera, min_depth, max_depth, global_T_camera);
  }
  
  // Creates a frustum for the given camera.
  inline void Create(const Camera& camera, float min_depth, float max_depth, const SE3f& global_T_camera) {
    IDENTIFY_CAMERA(camera, _Create(_camera, min_depth, max_depth, global_T_camera));
  }
  
  // Tests whether the bounding box of the frustum intersects the bounding box
  // of another frustum.
  inline bool BBoxIntersects(const CameraFrustum& other) {
    return !bbox_.intersection(other.bbox_).isEmpty();
  }
  
  // Tests for intersection with another frustum. Starts with a bounding box
  // test, and if this does not rule out an intersection, uses the separating
  // axis test. If the frustums are touching but not intersecting, returns
  // false.
  bool Intersects(CameraFrustum* other) {
    if (!BBoxIntersects(*other)) {
      return false;
    }
    
    if (!axes_and_planes_computed_) {
      ComputeAxesAndPlanes();
    }
    
    // Test this frustum's planes vs. the other vertices.
    for (int plane = 0; plane < 6; ++ plane) {
      int vertex = 0;
      for (; vertex < 8; ++ vertex) {
        if (planes_[plane].signedDistance(other->points_[vertex]) < 0) {
          break;
        }
      }
      if (vertex == 8) {
        return false;
      }
    }
    
    if (!other->axes_and_planes_computed_) {
      other->ComputeAxesAndPlanes();
    }
    
    // Test the other frustum's planes vs. these vertices.
    for (int plane = 0; plane < 6; ++ plane) {
      int vertex = 0;
      for (; vertex < 8; ++ vertex) {
        if (other->planes_[plane].signedDistance(points_[vertex]) < 0) {
          break;
        }
      }
      if (vertex == 8) {
        return false;
      }
    }
    
    // Test whether an axis computed from edge cross products separates the
    // frustums.
    for (int this_edge = 0; this_edge < 6; ++ this_edge) {
      for (int other_edge = 0; other_edge < 6; ++ other_edge) {
        Vec3f direction = axes_[this_edge].cross(axes_[other_edge]);
        constexpr float kEpsilon = 1e-5f;  // TODO: Which threshold would be appropriate?
        if (direction.squaredNorm() < kEpsilon) {
          // The edges are (almost) parallel, which gives numerical problems for
          // the separating direction. Thus, cannot use this direction for
          // testing.
          continue;
        }
        
        float this_min = numeric_limits<float>::infinity();
        float this_max = -numeric_limits<float>::infinity();
        float other_min = numeric_limits<float>::infinity();
        float other_max = -numeric_limits<float>::infinity();
        
        for (int point = 0; point < 8; ++ point) {
          float this_value = direction.dot(points_[point]);
          this_min = std::min(this_min, this_value);
          this_max = std::max(this_max, this_value);
          
          float other_value = direction.dot(other->points_[point]);
          other_min = std::min(other_min, other_value);
          other_max = std::max(other_max, other_value);
        }
        
        if (this_max <= other_min || this_min >= other_max) {
          return false;
        }
      }
    }
    
    return true;
  }
  
 private:
  template <class CameraT>
  void _Create(const CameraT& camera, float min_depth, float max_depth, const SE3f& global_T_camera) {
    // TODO: For different camera types than pinhole, the frustum could model
    //       their viewing cone more accurately than by using the image corners only.
    
    Matrix<float, 3, 4> matrix = global_T_camera.matrix3x4();
    
    const Vec3f top_left = camera.UnprojectFromPixelCornerConv(Vec2i(0, 0)).template cast<float>();
    points_[0] = matrix * (min_depth * top_left).homogeneous();
    bbox_.extend(points_[0]);
    points_[1] = matrix * (max_depth * top_left).homogeneous();
    bbox_.extend(points_[1]);
    
    const Vec3f top_right = camera.UnprojectFromPixelCornerConv(Vec2i(camera.width(), 0)).template cast<float>();
    points_[2] = matrix * (min_depth * top_right).homogeneous();
    bbox_.extend(points_[2]);
    points_[3] = matrix * (max_depth * top_right).homogeneous();
    bbox_.extend(points_[3]);
    
    const Vec3f bottom_left = camera.UnprojectFromPixelCornerConv(Vec2i(0, camera.height())).template cast<float>();
    points_[4] = matrix * (min_depth * bottom_left).homogeneous();
    bbox_.extend(points_[4]);
    points_[5] = matrix * (max_depth * bottom_left).homogeneous();
    bbox_.extend(points_[5]);
    
    const Vec3f bottom_right = camera.UnprojectFromPixelCornerConv(Vec2i(camera.width(), camera.height())).template cast<float>();
    points_[6] = matrix * (min_depth * bottom_right).homogeneous();
    bbox_.extend(points_[6]);
    points_[7] = matrix * (max_depth * bottom_right).homogeneous();
    bbox_.extend(points_[7]);
  }
  
  void ComputeAxesAndPlanes() {
    axes_[0] = points_[7] - points_[6];  // bottom right edge
    axes_[1] = points_[3] - points_[2];  // top right edge
    axes_[2] = points_[5] - points_[4];  // bottom left edge
    axes_[3] = points_[1] - points_[0];  // top left edge
    axes_[4] = points_[2] - points_[6];  // top <-> down edge
    axes_[5] = points_[0] - points_[2];  // left <-> right edge
    
    // Plane normals point away from the frustum.
    // They are in general not normalized.
    
    // Far plane:
    //planes_[0] = Hyperplane<float, 3>(matrix.block<3, 1>(0, 2), matrix.block<3, 1>(0, 2).dot(points_[1]));
    Vec3f forward_normal_dir = axes_[5].cross(axes_[4]);
    planes_[0] = Hyperplane<float, 3>(forward_normal_dir, -forward_normal_dir.dot(points_[1]));
    // Near plane:
    //planes_[1] = Hyperplane<float, 3>(-matrix.block<3, 1>(0, 2), -matrix.block<3, 1>(0, 2).dot(points_[0]));
    planes_[1] = Hyperplane<float, 3>(-forward_normal_dir, forward_normal_dir.dot(points_[0]));
    
    // Right plane:
    Vec3f right_normal_dir = axes_[0].cross(axes_[4]);
    planes_[2] = Hyperplane<float, 3>(right_normal_dir, -right_normal_dir.dot(points_[6]));
    // Top plane:
    Vec3f top_normal_dir = axes_[1].cross(axes_[5]);
    planes_[3] = Hyperplane<float, 3>(top_normal_dir, -top_normal_dir.dot(points_[2]));
    // Left plane:
    Vec3f left_normal_dir = axes_[4].cross(axes_[2]);
    planes_[4] = Hyperplane<float, 3>(left_normal_dir, -left_normal_dir.dot(points_[4]));
    // Bottom plane:
    Vec3f bottom_normal_dir = axes_[5].cross(axes_[0]);
    planes_[5] = Hyperplane<float, 3>(bottom_normal_dir, -bottom_normal_dir.dot(points_[6]));
    
    axes_and_planes_computed_ = true;
  }
  
  Vec3f points_[8];
  
  Vec3f axes_[6];
  Hyperplane<float, 3> planes_[6];
  bool axes_and_planes_computed_;
  
  AlignedBox<float, 3> bbox_;
};

}

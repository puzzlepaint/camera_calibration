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

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>

#include "camera_calibration/b_spline.h"
#include "camera_calibration/models/camera_model.h"
#include "camera_calibration/local_parametrizations/direction_parametrization.h"

namespace vis {

/// Base class for central cameras with a direction grid.
template <class Derived>
class CentralGridModel : public CameraModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  CentralGridModel(
      CameraModel::Type type,
      int grid_resolution_x, int grid_resolution_y,
      int calibration_min_x, int calibration_min_y,
      int calibration_max_x, int calibration_max_y,
      int width, int height)
      : CameraModel(width, height,
                    calibration_min_x, calibration_min_y,
                    calibration_max_x, calibration_max_y,
                    type) {
    m_grid.SetSize(grid_resolution_x, grid_resolution_y);
  }
  
  /// Copy constructor.
  CentralGridModel(const CentralGridModel& other)
      : CameraModel(other.width(), other.height(),
                    other.calibration_min_x(), other.calibration_min_y(),
                    other.calibration_max_x(), other.calibration_max_y(),
                    other.type()) {
    m_grid = other.m_grid;
  }
  
  /// Applies the given rotation matrix to all directions in the model.
  inline void Rotate(const Mat3d& rotation) {
    for (u32 y = 0; y < m_grid.height(); ++ y) {
      for (u32 x = 0; x < m_grid.width(); ++ x) {
        m_grid(x, y) = rotation * m_grid(x, y);
      }
    }
  }
  
  
  inline virtual bool Project(
      const Vec3d& local_point,
      Vec2d* result) const override {
    const Vec3d point_direction = local_point.normalized();
    return ProjectDirection(point_direction, result);
  }
  
  inline virtual bool ProjectWithInitialEstimate(const Vec3d& local_point, Vec2d* result) const override {
    return static_cast<const Derived*>(this)->ProjectDirectionWithInitialEstimate(local_point.normalized(), result);
  }
  
  /// Version of Project() which assumes that the direction passed in is already
  /// normalized.
  inline bool ProjectDirection(
      const Vec3d& point_direction,
      Vec2d* result) const {
    *result = CenterOfCalibratedArea();
    return static_cast<const Derived*>(this)->ProjectDirectionWithInitialEstimate(point_direction, result);
  }
  
  inline virtual bool Unproject(double x, double y, Line3d* result) const override {
    if (!static_cast<const Derived*>(this)->Unproject(x, y, &result->direction())) {
      return false;
    }
    result->origin() = Vec3d::Zero();
    return true;
  }
  
  /// Unproject the given point in grid coordinates to a unit-length direction
  /// in local camera coordinates.
  inline Vec3d UnprojectFromGrid(double x, double y) const {
    return EvalUniformCubicBSplineSurface(m_grid, x, y).normalized();
  }
  
  
  virtual inline bool GetGridResolution(int* resolution_x, int* resolution_y) const override {
    *resolution_x = m_grid.width();
    *resolution_y = m_grid.height();
    return true;
  }
  
  inline virtual int update_parameter_count() const override {
    return 2 * grid().width() * grid().height();
  }
  
  
  /// For x and y in [0, n_grid.width/height()[, returns the location of that
  /// grid point in pixel-corner coordinate origin convention.
  inline Vec2d GridPointToPixelCornerConv(int x, int y) const {
    return Vec2d(
        m_calibration_min_x + ((x - 1.f) / (m_grid.width() - 3.f)) * (m_calibration_max_x + 1 - m_calibration_min_x),
        m_calibration_min_y + ((y - 1.f) / (m_grid.height() - 3.f)) * (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  static inline Vec2d GridPointToPixelCornerConv(
      int x, int y,
      int calibration_min_x, int calibration_min_y,
      int calibration_max_x, int calibration_max_y,
      int grid_width, int grid_height) {
    return Vec2d(
        calibration_min_x + ((x - 1.f) / (grid_width - 3.f)) * (calibration_max_x + 1 - calibration_min_x),
        calibration_min_y + ((y - 1.f) / (grid_height - 3.f)) * (calibration_max_y + 1 - calibration_min_y));
  }
  
  inline double GridScaleToPixelScaleX(double length) const {
    return length * ((m_calibration_max_x + 1 - m_calibration_min_x) / (m_grid.width() - 3.f));
  }
  inline double GridScaleToPixelScaleY(double length) const {
    return length * ((m_calibration_max_y + 1 - m_calibration_min_y) / (m_grid.height() - 3.f));
  }
  
  /// Inverse of GridPointToPixelCornerConv().
  inline Vec2d PixelCornerConvToGridPoint(double x, double y) const {
    return Vec2d(
        1.f + (m_grid.width() - 3.f) * (x - m_calibration_min_x) / (m_calibration_max_x + 1 - m_calibration_min_x),
        1.f + (m_grid.height() - 3.f) * (y - m_calibration_min_y) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  inline double PixelScaleToGridScaleX(double length) const {
    return length * ((m_grid.width() - 3.f) / (m_calibration_max_x + 1 - m_calibration_min_x));
  }
  inline double PixelScaleToGridScaleY(double length) const {
    return length * ((m_grid.height() - 3.f) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  inline void SetGrid(const Image<Vec3d>& grid) { m_grid = grid; }
  inline const Image<Vec3d>& grid() const { return m_grid; }
  inline Image<Vec3d>& grid() { return m_grid; }
  
  /// For optimizing the model manually.
  template <typename MatrixDerived>
  inline void SubtractDelta(const MatrixBase<MatrixDerived>& delta) {
    int i = 0;
    for (u32 y = 0; y < m_grid.height(); ++ y) {
      for (u32 x = 0; x < m_grid.width(); ++ x) {
        // TODO: Cache tangents earlier, do not recompute here
        DirectionTangents tangents;
        ComputeTangentsForDirectionOrLine(m_grid.at(x, y), &tangents);
        ApplyLocalUpdateToDirection(
            &m_grid.at(x, y), tangents,
            -delta(i + 0), -delta(i + 1));
        i += 2;
      }
    }
    CHECK_EQ(i, delta.size());
    DebugVerifyGridValues();
  }
  
  /// NOTE: This function is not re-entrant!
  template <typename DerivedA, typename DerivedB>
  bool ProjectionJacobianWrtIntrinsics(
      const Vec3d& local_point,
      const Vec2d& projected_pixel,
      const Image<DirectionTangents>& tangents_image,
      double numerical_diff_delta,
      MatrixBase<DerivedA>* grid_update_indices,
      MatrixBase<DerivedB>* pixel_wrt_grid_updates) const { 
    Vec3d point_direction = local_point.normalized();
    
    Vec2d grid_point = PixelCornerConvToGridPoint(projected_pixel.x(), projected_pixel.y());
    
    int ix = std::floor(grid_point.x());
    int iy = std::floor(grid_point.y());
    
    int local_index = 0;
    for (int y = 0; y < 4; ++ y) {
      int gy = iy + y - 1;
      CHECK_GE(gy, 0);
      CHECK_LT(gy, m_grid.height());
      
      for (int x = 0; x < 4; ++ x) {
        int gx = ix + x - 1;
        CHECK_GE(gx, 0);
        CHECK_LT(gx, m_grid.width());
        
        int sequential_index = gx + gy * m_grid.width();
        (*grid_update_indices)(local_index + 0) = 2 * sequential_index + 0;
        (*grid_update_indices)(local_index + 1) = 2 * sequential_index + 1;
        
        const DirectionTangents& tangents = tangents_image.data()[sequential_index];  // (gx, gy);
        
        Vec3d* test_direction = const_cast<Vec3d*>(grid().data() + sequential_index);  // this is reset to the original value at the end
        Vec3d original_direction = *test_direction;
        
        for (int d = 0; d < 2; ++ d) {
          ApplyLocalUpdateToDirection(
              test_direction, tangents,
              (d == 0) ? numerical_diff_delta : 0,
              (d == 1) ? numerical_diff_delta : 0);
          
          Vec2d test_projected_pixel = projected_pixel;
          bool success = static_cast<const Derived*>(this)->ProjectDirectionWithInitialEstimate(point_direction, &test_projected_pixel);
          *test_direction = original_direction;
          
          if (!success) {
            return false;
          }
          
          (*pixel_wrt_grid_updates)(0, local_index + d) = (test_projected_pixel.x() - projected_pixel.x()) / numerical_diff_delta;
          (*pixel_wrt_grid_updates)(1, local_index + d) = (test_projected_pixel.y() - projected_pixel.y()) / numerical_diff_delta;
        }
        
        local_index += 2;
      }
    }
//     CHECK_EQ(local_index, grid_update_indices->size());
    return true;
  }
  
  /// Verifies that the directions stored in the grid are all normalized.
  void DebugVerifyGridValues() {
    for (u32 y = 0; y < m_grid.height(); ++ y) {
      for (u32 x = 0; x < m_grid.width(); ++ x) {
        if (fabs(1 - m_grid(x, y).squaredNorm()) > 0.001f) {
          LOG(ERROR) << "Grid value at " << x << ", " << y << " is not normalized (length: " << m_grid(x, y).norm() << ")";
          LOG(ERROR) << "Not reporting possible additional errors to avoid log spam.";
          return;
        }
      }
    }
  }
  
 protected:
  Image<Vec3d> m_grid;
};

}

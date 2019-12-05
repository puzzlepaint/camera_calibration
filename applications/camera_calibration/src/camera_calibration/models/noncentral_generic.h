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

#include <Eigen/Geometry>
#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>

#include "camera_calibration/b_spline.h"
#include "camera_calibration/local_parametrizations/line_parametrization.h"
#include "camera_calibration/models/camera_model.h"

namespace vis {

class CentralGenericModel;
struct LineTangents;

/// Generic camera model for noncentral cameras. Performs unprojection by
/// B-Spline based interpolation in a grid of 3D lines.
/// 
/// Each grid point stores one 3D point on the line and the line direction.
/// B-Spline based interpolation is applied to both the point and the direction.
class NoncentralGenericModel : public CameraModel {
 public:
  /// Creates an invalid model.
  NoncentralGenericModel();
  
  /// Constructor. Leaves the model uninitialized.
  NoncentralGenericModel(
      int grid_resolution_x, int grid_resolution_y,
      int calibration_min_x, int calibration_min_y,
      int calibration_max_x, int calibration_max_y,
      int width, int height);
  
  /// Copy constructor.
  NoncentralGenericModel(const NoncentralGenericModel& other);
  
  NoncentralGenericModel& operator= (const NoncentralGenericModel& other);
  
  virtual CameraModel* duplicate() override;
  
  
  /// Sets this noncentral model to the existing central model. This can be used
  /// as initialization for near-central cameras, followed by bundle adjustment.
  void InitializeFromCentralGenericModel(const CentralGenericModel& other);
  
  // TODO: Since there is currently no support for initializing with a non-central
  //       model, the two functions commented out here are not implemented.
//   /// Initializes the model to match as good as possible with the dense model provided.
//   /// The dense model is given by an image of the same size as the camera images,
//   /// where each pixel's value specifies the observation line of that pixel.
//   /// Pixels with NaN values in the dense model are treated as invalid.
//   /// Returns true if successful, false if an error ocurred.
//   bool FitToDenseModel(const Image<Line3d>& dense_model);
//   
//   /// Pixels must be given in pixel-corner coordinate origin convention.
//   /// This function optimizes for TODO error (TODO: explain cost function).
//   void FitToPixelLines(const vector<Vec2d>& pixels, const vector<Line3d>& lines, int max_iteration_count);
  
  
  inline virtual bool Project(
      const Vec3d& local_point,
      Vec2d* result) const override {
    *result = CenterOfCalibratedArea();
    return ProjectWithInitialEstimate(local_point, result);
  }
  
  virtual bool ProjectWithInitialEstimate(
      const Vec3d& local_point,
      Vec2d* result) const override;
  
  /// Unproject the given point in grid coordinates to a line in local camera coordinates.
  inline Line3d UnprojectFromGrid(double x, double y) const {
    Line3d line;
    EvalTwoUniformCubicBSplineSurfaces(m_direction_grid, m_point_grid, x, y, &line.direction(), &line.origin());
    line.direction().normalize();
    return line;
  }
  
  /// Unprojects the given pixel to a line in local camera coordinates.
  inline virtual bool Unproject(double x, double y, Line3d* result) const override {
    if (IsInCalibratedArea(x, y)) {
      Vec2d grid_point = PixelCornerConvToGridPoint(x, y);
      *result = UnprojectFromGrid(grid_point.x(), grid_point.y());
      return true;
    } else {
      return false;
    }
  }
  
  /// Unprojects the given pixel to a line in local camera coordinates.
  /// At the same time, determines the Jacobian of this line wrt. the pixel
  /// coordinates in the image. The Jacobian contains the direction first and
  /// then the origin.
  bool UnprojectWithJacobian(
      double x, double y,
      Line3d* result,
      Matrix<double, 6, 2>* dresult_dxy) const;
  
  
  virtual inline Mat3d ChooseNiceCameraOrientation() override {
    // TODO: Implement this function.
    LOG(WARNING) << "ChooseNiceCameraOrientation() is not implemented for NoncentralGenericModel.";
    return Mat3d::Identity();
  }
  
  virtual void Scale(double factor) override;
  
  virtual inline bool GetGridResolution(int* resolution_x, int* resolution_y) const override {
    *resolution_x = m_point_grid.width();
    *resolution_y = m_point_grid.height();
    return true;
  }
  
  inline virtual int update_parameter_count() const override {
    return 5 * direction_grid().width() * direction_grid().height();
  }
  
  static inline int exterior_cells_per_side() {
    return 1;
  }
  
  
  /// For x and y in [0, n_grid.width/height()[, returns the location of that
  /// grid point in pixel-corner coordinate origin convention.
  inline Vec2d GridPointToPixelCornerConv(double x, double y) const {
    return Vec2d(
        m_calibration_min_x + ((x - 1.f) / (m_direction_grid.width() - 3.f)) * (m_calibration_max_x + 1 - m_calibration_min_x),
        m_calibration_min_y + ((y - 1.f) / (m_direction_grid.height() - 3.f)) * (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  inline double GridScaleToPixelScaleX(double length) const {
    return length * ((m_calibration_max_x + 1 - m_calibration_min_x) / (m_direction_grid.width() - 3.f));
  }
  inline double GridScaleToPixelScaleY(double length) const {
    return length * ((m_calibration_max_y + 1 - m_calibration_min_y) / (m_direction_grid.height() - 3.f));
  }
  
  /// Inverse of GridPointToPixelCornerConv().
  inline Vec2d PixelCornerConvToGridPoint(double x, double y) const {
    return Vec2d(
        1.f + (m_direction_grid.width() - 3.f) * (x - m_calibration_min_x) / (m_calibration_max_x + 1 - m_calibration_min_x),
        1.f + (m_direction_grid.height() - 3.f) * (y - m_calibration_min_y) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  inline double PixelScaleToGridScaleX(double length) const {
    return length * ((m_direction_grid.width() - 3.f) / (m_calibration_max_x + 1 - m_calibration_min_x));
  }
  inline double PixelScaleToGridScaleY(double length) const {
    return length * ((m_direction_grid.height() - 3.f) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  
  inline void SetPointGrid(const Image<Vec3d>& point_grid) {
    m_point_grid = point_grid;
  }
  inline const Image<Vec3d>& point_grid() const { return m_point_grid; }
  inline Image<Vec3d>& point_grid() { return m_point_grid; }
  
  inline void SetDirectionGrid(const Image<Vec3d>& direction_grid) {
    m_direction_grid = direction_grid;
  }
  inline const Image<Vec3d>& direction_grid() const { return m_direction_grid; }
  inline Image<Vec3d>& direction_grid() { return m_direction_grid; }
  
  
  /// For optimizing the model manually.
  template <typename Derived>
  inline void SubtractDelta(const MatrixBase<Derived>& delta) {
    int i = 0;
    for (u32 y = 0; y < m_point_grid.height(); ++ y) {
      for (u32 x = 0; x < m_point_grid.width(); ++ x) {
        // TODO: Cache tangents earlier, do not recompute here
        LineTangents tangents;
        ComputeTangentsForDirectionOrLine(m_direction_grid.at(x, y), &tangents);
        
        ParametrizedLine<double, 3> line(m_point_grid(x, y), m_direction_grid(x, y));
        
        ApplyLocalUpdateToLine(
            &line, tangents, -delta(i + 0), -delta(i + 1), -delta(i + 2),
            -delta(i + 3), -delta(i + 4));
        i += 5;
        
        m_point_grid(x, y) = line.origin();
        m_direction_grid(x, y) = line.direction();
        
        if (i == delta.size()) {
          return;
        }
      }
    }
  }
  
  static constexpr const int IntrinsicsJacobianSize = 5 * 16;
  
  /// NOTE: This function is not re-entrant!
  template <typename DerivedA, typename DerivedB>
  bool ProjectionJacobianWrtIntrinsics(
      const Vec3d& local_point,
      const Vec2d& projected_pixel,
      const Image<LineTangents>& tangents_image,
      double numerical_diff_delta,
      MatrixBase<DerivedA>* grid_update_indices,
      MatrixBase<DerivedB>* pixel_wrt_grid_updates) const {
    Vec2d grid_point = PixelCornerConvToGridPoint(projected_pixel.x(), projected_pixel.y());
    
    int ix = std::floor(grid_point.x());
    int iy = std::floor(grid_point.y());
    
    int local_index = 0;
    for (int y = 0; y < 4; ++ y) {
      int gy = iy + y - 1;
      for (int x = 0; x < 4; ++ x) {
        int gx = ix + x - 1;
        int sequential_index = gx + gy * m_point_grid.width();
        for (int i = 0; i < 5; ++ i) {
          (*grid_update_indices)(local_index + i) = 5 * sequential_index + i;
        }
        
        const LineTangents& tangents = tangents_image.data()[sequential_index];
        
        // These are reset to their original values at the end
        Vec3d* test_origin = const_cast<Vec3d*>(&m_point_grid(gx, gy));
        Vec3d original_origin = *test_origin;
        
        Vec3d* test_direction = const_cast<Vec3d*>(&m_direction_grid(gx, gy));
        Vec3d original_direction = *test_direction;
        
        // Test offsets in all dimensions:
        for (int d = 0; d < 5; ++ d) {
          Line3d test_line(original_origin, original_direction);
          
          double delta[5] = {0, 0, 0, 0, 0};
          delta[d] = numerical_diff_delta;
          ApplyLocalUpdateToLine(
              &test_line, tangents,
              delta[0], delta[1], delta[2], delta[3], delta[4]);
          *test_origin = test_line.origin();
          *test_direction = test_line.direction();
          Vec2d test_projected_pixel = projected_pixel;
          if (!ProjectWithInitialEstimate(local_point, &test_projected_pixel)) {
            *test_origin = original_origin;
            *test_direction = original_direction;
            return false;
          }
          (*pixel_wrt_grid_updates)(0, local_index + d) = (test_projected_pixel.x() - projected_pixel.x()) / numerical_diff_delta;
          (*pixel_wrt_grid_updates)(1, local_index + d) = (test_projected_pixel.y() - projected_pixel.y()) / numerical_diff_delta;
        }
        
        *test_origin = original_origin;
        *test_direction = original_direction;
        local_index += 5;
      }
    }
    return true;
  }
  
  void DebugVerifyGridValues();
  
 private:
  Image<Vec3d> m_point_grid;
  Image<Vec3d> m_direction_grid;
};

}

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
#include "camera_calibration/models/central_grid.h"
#include "camera_calibration/local_parametrizations/direction_parametrization.h"

struct float3;

namespace vis {

template <typename T>
class CUDABuffer;

/// Generic camera model for central cameras. Performs unprojection by
/// B-Spline interpolation in a grid of direction vectors.
class CentralGenericModel : public CentralGridModel<CentralGenericModel> {
 public:
  /// Constructor, creating an invalid model.
  CentralGenericModel();
  
  /// Constructor. Leaves the model uninitialized.
  CentralGenericModel(
      int grid_resolution_x, int grid_resolution_y,
      int calibration_min_x, int calibration_min_y,
      int calibration_max_x, int calibration_max_y,
      int width, int height);
  
  /// Copy constructor.
  CentralGenericModel(const CentralGenericModel& other);
  
  CentralGenericModel& operator= (const CentralGenericModel& other);
  
  virtual CameraModel* duplicate() override;
  
  ~CentralGenericModel();
  
  
  /// Initializes the model to match as good as possible with the dense model
  /// provided. The dense model is given by an image of the same size as the
  /// camera images, where each pixel's value specifies the observation
  /// direction of that pixel. Pixels with NaN values in the dense model (more
  /// specifically, the x coordinate) are treated as invalid. Returns true if
  /// successful, false if an error ocurred.
  bool FitToDenseModel(
      const Image<Vec3d>& dense_model,
      int subsample_step,
      int max_iteration_count = 10);
  
  /// Pixels must be given in pixel-corner coordinate origin convention.
  /// This function optimizes for direction error (difference of un-projected
  /// direction and given direction).
  void FitToPixelDirections(
      const vector<Vec2d>& pixels,
      const vector<Vec3d>& directions,
      int max_iteration_count);
  
  
  /// Version of ProjectWithInitialEstimate() which
  /// assumes that the direction passed in is already normalized.
  bool ProjectDirectionWithInitialEstimate(
      const Vec3d& local_direction,
      Vec2d* result) const;
  
  
  inline virtual bool Unproject(double x, double y, Vec3d* result) const override {
    if (IsInCalibratedArea(x, y)) {
      Vec2d grid_point = PixelCornerConvToGridPoint(x, y);
      *result = EvalUniformCubicBSplineSurface(m_grid, grid_point.x(), grid_point.y()).normalized();
      return true;
    } else {
      return false;
    }
  }
  
  using CentralGridModel<CentralGenericModel>::Unproject;
  
  /// Unprojects the given pixel to a direction in local camera coordinates.
  /// At the same time, determines the Jacobian of this direction wrt. the pixel
  /// coordinates in the image.
  bool UnprojectWithJacobian(
      double x, double y,
      Vec3d* result,
      Matrix<double, 3, 2>* dresult_dxy) const;
  
  
  virtual Mat3d ChooseNiceCameraOrientation() override;
  
  
  virtual CUDACameraModel* CreateCUDACameraModel() override;
  
  static inline int exterior_cells_per_side() {
    return 1;
  }
  
  
  static constexpr const int IntrinsicsJacobianSize = 2 * 16;
  
  static void ComputeUnprojectedDirectionResidualAndJacobianWrtGridUpdates(
      const Vec3d& measurement,
      const Vec2d& grid_point,
      const Image<Vec3d>& grid,
      const Image<DirectionTangents>& tangents_image,
      Matrix<double, 3, 1>* residuals,
      Matrix<int, 2*16, 1>* local_indices,
      Matrix<double, 3, 2*16, Eigen::RowMajor>* local_jacobian);
  
 private:
  void FitToPixelDirectionsImpl(const vector<Vec2d>& grid_points, const vector<Vec3d>& directions, int max_iteration_count);
  
  shared_ptr<CUDABuffer<float3>> m_cuda_grid;
};

}

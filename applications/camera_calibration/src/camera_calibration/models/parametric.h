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

#include "camera_calibration/local_parametrizations/line_parametrization.h"
#include "camera_calibration/models/camera_model.h"

namespace vis {

/// Linearly fits a simple parametric model to the given dense model. Optionally,
/// an 'equidistant' radial distortion step is used.
/// Parameters:
/// fx fy cx cy k1 k2
bool FitSimpleParametricToDenseModelLinearly(
    const Image<Vec3d>& dense_model,
    int width, int height,
    bool use_equidistant_projection,
    double* fx, double* fy,
    double* cx, double* cy,
    double* k1, double* k2);

bool FitPinholeToDenseModelLinearly(
    const Image<Vec3d>& dense_model,
    int width, int height,
    bool use_equidistant_projection,
    double* fx, double* fy,
    double* cx, double* cy);


template <class CameraT>
bool UnprojectWithGaussNewton(double x, double y, const CameraT& camera, Vec3d* direction) {
  const double& fx = camera.parameters()(0);
  const double& fy = camera.parameters()(1);
  const double& cx = camera.parameters()(2);
  const double& cy = camera.parameters()(3);
  
  Vec2d distorted_point = Vec2d((x - cx) / fx, (y - cy) / fy);
  
  double cur_x = distorted_point.coeff(0);
  double cur_y = distorted_point.coeff(1);
  
  // Levenberg-Marquardt optimization algorithm.
  const double kUndistortionEpsilon = 1e-10f;
  const usize kMaxIterations = 100;
  
  double lambda = -1;
  
  bool converged = false;
  for (usize i = 0; i < kMaxIterations; ++i) {
    Matrix<double, 2, 2> ddxy_dxy;
    Matrix<double, 2, 1> distorted;
    camera.ProjectInnerPartWithJacobian(Vec2d(cur_x, cur_y), &distorted, &ddxy_dxy);
    
    // (Non-squared) residuals.
    double dx = distorted.x() - distorted_point.x();
    double dy = distorted.y() - distorted_point.y();
    double cost = dx * dx + dy * dy;
    
    // Accumulate H and b.
    double H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0);
    double H_1_0_and_0_1 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1);
    double H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1);
    double b_0 = dx * ddxy_dxy(0, 0) + dy * ddxy_dxy(1, 0);
    double b_1 = dx * ddxy_dxy(0, 1) + dy * ddxy_dxy(1, 1);
    
    if (lambda < 0) {
      lambda = 1.0 * (0.5 * (H_0_0 + H_1_1));
    }
    
    bool update_found = false;
    for (int lm_iteration = 0; lm_iteration < 5; ++ lm_iteration) {
      double H_0_0_LM = H_0_0 + lambda;
      double H_1_1_LM = H_1_1 + lambda;
      
      // Solve the system and update the parameters.
      double x_1 = (b_1 - H_1_0_and_0_1 / H_0_0_LM * b_0) /
                   (H_1_1_LM - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0_LM);
      double x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0_LM;
      double test_x = cur_x - x_0;
      double test_y = cur_y - x_1;
      
      Matrix<double, 2, 2> ddxy_dxy;
      Matrix<double, 2, 1> distorted;
      camera.ProjectInnerPartWithJacobian(Vec2d(test_x, test_y), &distorted, &ddxy_dxy);  // TODO: No Jacobian required here
      dx = distorted.x() - distorted_point.x();
      dy = distorted.y() - distorted_point.y();
      double test_cost = dx * dx + dy * dy;
      
      if (test_cost < cost) {
        // Accept update
        cost = test_cost;
        cur_x = test_x;
        cur_y = test_y;
        lambda *= 0.1;
        update_found = true;
        break;
      } else {
        // Reject update
        lambda *= 10;
      }
    }
    
    // Test for convergence
    if (cost < kUndistortionEpsilon) {
      converged = true;
      break;
    } else if (!update_found) {
      break;
    }
  }
  
  if (!converged) {
    return false;
  }
  
  *direction = Vec3d(cur_x, cur_y, 1);
  return true;
}


class ParametricCameraModel : public CameraModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline ParametricCameraModel(
      int width, int height, CameraModel::Type type, int num_parameters)
      : CameraModel(width, height, 0, 0, width, height, type) {
    m_parameters.resize(num_parameters, Eigen::NoChange);
  }
  
  ParametricCameraModel(const ParametricCameraModel& other);
  
  
  /// Initializes the model to match as good as possible with the dense model
  /// provided. The dense model is given by an image of the same size as the
  /// camera images, where each pixel's value specifies the observation
  /// direction of that pixel. Pixels with NaN values in the dense model (more
  /// specifically, the x coordinate) are treated as invalid. Returns true if
  /// successful, false if an error ocurred.
  bool FitToDenseModel(
      const Image<Vec3d>& dense_model,
      Mat3d* parametric_r_dense,
      int subsample_step,
      bool print_progress = false);
  
  bool FitToDenseModel(
      const Image<Vec3d>& dense_model,
      int subsample_step,
      bool print_progress = false);
  
  /// Initializes a fit to the dense model for FitToDenseModel(). Does not need
  /// to be called separately, as it is called by FitToDenseModel() internally.
  virtual bool FitToDenseModelLinearly(
      const Image<Vec3d>& dense_model) = 0;
  
  
  inline const Matrix<double, Eigen::Dynamic, 1>& parameters() const { return m_parameters; }
  inline Matrix<double, Eigen::Dynamic, 1>& parameters() { return m_parameters; }
  
 protected:
  Matrix<double, Eigen::Dynamic, 1> m_parameters;
  
 friend struct ParametricStateWrapper;
 friend struct ParametricDirectionCostFunction;
};

}

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
#include "camera_calibration/models/parametric.h"

namespace vis {

// Base class for camera models used in the camera calibrator.
class CentralOpenCVModel : public ParametricCameraModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline CentralOpenCVModel(int width, int height)
      : ParametricCameraModel(width, height, Type::CentralOpenCV, 12) {}
  
  CentralOpenCVModel(const CentralOpenCVModel& other);
  
  virtual CameraModel* duplicate() override;
  
  
  virtual bool FitToDenseModelLinearly(
      const Image<Vec3d>& dense_model) override;
  
  
  virtual bool Project(
      const Vec3d& local_point,
      Vec2d* result) const override;
  
  virtual inline bool ProjectWithInitialEstimate(
      const Vec3d& local_point,
      Vec2d* result) const override {
    // Projection is not an optimization process for this model, so just ignore
    // the initial estimate.
    return Project(local_point, result);
  }
  
  bool ProjectInnerPartWithJacobian(
      const Vec2d& undistorted_point,
      Vec2d* result,
      Matrix<double, 2, 2>* jacobian) const;
  
  /// Unprojects the given pixel to a unit-length direction in local camera coordinates.
  virtual bool Unproject(double x, double y, Vec3d* direction) const override;
  
  inline virtual bool Unproject(double x, double y, Line3d* result) const override {
    if (!Unproject(x, y, &result->direction())) {
      return false;
    }
    result->origin() = Vec3d::Zero();
    return true;
  }
  
  
  inline virtual int update_parameter_count() const override {
    return 12;
  }
  
  
  template <typename Derived>
  inline void SubtractDelta(const MatrixBase<Derived>& delta) {
    m_parameters -= delta;
  }
  
  static constexpr const int IntrinsicsJacobianSize = 12;
  
  template <typename DerivedA, typename DerivedB>
  bool ProjectionJacobianWrtIntrinsics(
      const Vec3d& local_point,
      const Vec2d& /*projected_pixel*/,
      const Image<LineTangents>& /*tangents_image*/,
      double /*numerical_diff_delta*/,
      MatrixBase<DerivedA>* grid_update_indices,
      MatrixBase<DerivedB>* pixel_wrt_grid_updates) const {
    typedef double Scalar;
    
    const double& p_0 = local_point.x();
    const double& p_1 = local_point.y();
    const double& p_2 = local_point.z();
    
    const double& fx = m_parameters(0);
    const double& fy = m_parameters(1);
    const double& k1 = m_parameters(4);
    const double& k2 = m_parameters(5);
    const double& k3 = m_parameters(6);
    const double& k4 = m_parameters(7);
    const double& k5 = m_parameters(8);
    const double& k6 = m_parameters(9);
    const double& p1 = m_parameters(10);
    const double& p2 = m_parameters(11);
    
    const Scalar term0 = 1 / (p_2 * p_2);
    const Scalar term1 = 2*p_0*p_1*term0;
    const Scalar term2 = p_1*p_1*term0;
    const Scalar term3 = p_0*p_0*term0;
    const Scalar term4 = term2 + 3*term3;
    const Scalar term5 = 1.0/p_2;
    const Scalar term6 = term2 + term3;
    const Scalar term7 = term6*term6;
    const Scalar term8 = term7*term6;
    const Scalar term9 = k1*term6 + k2*term7 + k3*term8 + 1;
    const Scalar term10 = k4*term6 + k5*term7 + k6*term8 + 1;
    const Scalar term11 = 1.0/term10;
    const Scalar term12 = term11*term5*term9;
    const Scalar term13 = term11*term5*term6;
    const Scalar term14 = fx*p_0*term11*term5;
    const Scalar term15 = -term2 - term3;
    const Scalar term16 = 1 / (term10 * term10);
    const Scalar term17 = fx*p_0*term16*term5*term9;
    const Scalar term18 = 3*term2 + term3;
    const Scalar term19 = fy*p_1*term11*term5;
    const Scalar term20 = fy*p_1*term16*term5*term9;
    
    for (int i = 0; i < 12; ++ i) {
      (*grid_update_indices)(i) = i;
    }
    
    (*pixel_wrt_grid_updates)(0, 0) = p1*term1 + p2*term4 + p_0*term12;
    (*pixel_wrt_grid_updates)(0, 1) = 0;
    (*pixel_wrt_grid_updates)(0, 2) = 1;
    (*pixel_wrt_grid_updates)(0, 3) = 0;
    (*pixel_wrt_grid_updates)(0, 4) = fx*p_0*term13;
    (*pixel_wrt_grid_updates)(0, 5) = term14*term7;
    (*pixel_wrt_grid_updates)(0, 6) = term14*term8;
    (*pixel_wrt_grid_updates)(0, 7) = term15*term17;
    (*pixel_wrt_grid_updates)(0, 8) = -term17*term7;
    (*pixel_wrt_grid_updates)(0, 9) = -term17*term8;
    (*pixel_wrt_grid_updates)(0, 10) = fx*term1;
    (*pixel_wrt_grid_updates)(0, 11) = fx*term4;
    
    (*pixel_wrt_grid_updates)(1, 0) = 0;
    (*pixel_wrt_grid_updates)(1, 1) = p1*term18 + p2*term1 + p_1*term12;                                                                             
    (*pixel_wrt_grid_updates)(1, 2) = 0;                                                                                                             
    (*pixel_wrt_grid_updates)(1, 3) = 1;                                                                                                             
    (*pixel_wrt_grid_updates)(1, 4) = fy*p_1*term13;                                                                                                 
    (*pixel_wrt_grid_updates)(1, 5) = term19*term7;                                                                                                  
    (*pixel_wrt_grid_updates)(1, 6) = term19*term8;                                                                                                  
    (*pixel_wrt_grid_updates)(1, 7) = term15*term20;                                                                                                 
    (*pixel_wrt_grid_updates)(1, 8) = -term20*term7;                                                                                                 
    (*pixel_wrt_grid_updates)(1, 9) = -term20*term8;                                                                                                 
    (*pixel_wrt_grid_updates)(1, 10) = fy*term18;                                                                                                    
    (*pixel_wrt_grid_updates)(1, 11) = fy*term1;
    
    return true;
  }
};

}

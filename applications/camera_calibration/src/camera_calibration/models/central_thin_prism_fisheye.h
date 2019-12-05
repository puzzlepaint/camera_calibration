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

// Implements the Thin-Prism Fisheye model which is used by the ETH3D stereo
// benchmark.
// 
// Parameter ordering:
// fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1.
class CentralThinPrismFisheyeModel : public ParametricCameraModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline CentralThinPrismFisheyeModel(
      int width, int height,
      bool use_equidistant_projection)
      : ParametricCameraModel(width, height, Type::CentralThinPrismFisheye, 12),
        m_use_equidistant_projection(use_equidistant_projection) {}
  
  CentralThinPrismFisheyeModel(const CentralThinPrismFisheyeModel& other);
  
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
    const double& p1 = m_parameters(8);
    const double& p2 = m_parameters(9);
    const double& sx1 = m_parameters(10);
    const double& sy1 = m_parameters(11);
    
    for (int i = 0; i < 12; ++ i) {
      (*grid_update_indices)(i) = i;
    }
    
    if (local_point.z() <= 0) {
      return false;
    }
    Vec2d undistorted_point = Vec2d(local_point.x() / local_point.z(),
                                    local_point.y() / local_point.z());
    double r = undistorted_point.norm();
    const double kEpsilon = static_cast<double>(1e-6);
    if (m_use_equidistant_projection && r > kEpsilon) {
      // "Fisheye" case.
      const Scalar term0 = 1.0/p_2;
      const Scalar term1 = term0 * term0;
      const Scalar term2 = p_0*p_0*term1;
      const Scalar term3 = p_1*p_1*term1;
      const Scalar term4 = term2 + term3;
      const Scalar term5 = sqrt(term4);
      const Scalar term6 = 1.0/term5;
      const Scalar term7 = atan(term5);
      const Scalar term8 = term0*term6*term7;
      const Scalar term9 = p_0*term8;
      const Scalar term10 = 1.0/term4;
      const Scalar term11 = term7*term7;
      const Scalar term12 = 2*p_0*p_1*term1*term10*term11;
      const Scalar term13 = term10*term11;
      const Scalar term14 = term13*term2;
      const Scalar term15 = term13*term3;
      const Scalar term16 = term14 + term15;
      const Scalar term17 = 3*term14 + term15;
      const Scalar term18 = term16*term16;
      const Scalar term19 = term18*term16;
      const Scalar term20 = term19*term16;
      const Scalar term21 = k1*term16 + k2*term18 + k3*term19 + k4*term20;
      const Scalar term22 = fx*term16;
      const Scalar term23 = fx*p_0*term0*term6*term7;
      const Scalar term24 = p_1*term8;
      const Scalar term25 = term14 + 3*term15;
      const Scalar term26 = fy*term16;
      const Scalar term27 = fy*p_1*term0*term6*term7;
      
      (*pixel_wrt_grid_updates)(0, 0) = p1*term12 + p2*term17 + sx1*term16 + term21*term9 + term9;
      (*pixel_wrt_grid_updates)(0, 1) = 0;
      (*pixel_wrt_grid_updates)(0, 2) = 1;
      (*pixel_wrt_grid_updates)(0, 3) = 0;
      (*pixel_wrt_grid_updates)(0, 4) = term22*term9;
      (*pixel_wrt_grid_updates)(0, 5) = term18*term23;
      (*pixel_wrt_grid_updates)(0, 6) = term19*term23;
      (*pixel_wrt_grid_updates)(0, 7) = term20*term23;
      (*pixel_wrt_grid_updates)(0, 8) = fx*term12;
      (*pixel_wrt_grid_updates)(0, 9) = fx*term17;
      (*pixel_wrt_grid_updates)(0, 10) = term22;
      (*pixel_wrt_grid_updates)(0, 11) = 0;
      
      (*pixel_wrt_grid_updates)(1, 0) = 0;
      (*pixel_wrt_grid_updates)(1, 1) = p1*term25 + p2*term12 + sy1*term16 + term21*term24 + term24;
      (*pixel_wrt_grid_updates)(1, 2) = 0;
      (*pixel_wrt_grid_updates)(1, 3) = 1;
      (*pixel_wrt_grid_updates)(1, 4) = term24*term26;
      (*pixel_wrt_grid_updates)(1, 5) = term18*term27;
      (*pixel_wrt_grid_updates)(1, 6) = term19*term27;
      (*pixel_wrt_grid_updates)(1, 7) = term20*term27;
      (*pixel_wrt_grid_updates)(1, 8) = fy*term25;
      (*pixel_wrt_grid_updates)(1, 9) = fy*term12;
      (*pixel_wrt_grid_updates)(1, 10) = 0;
      (*pixel_wrt_grid_updates)(1, 11) = term26;
    } else {
      // "Non-fisheye" case.
      const Scalar term0 = 1.0/p_2;
      const Scalar term1 = p_0*term0;
      const Scalar term2 = term0 * term0;
      const Scalar term3 = 2*p_0*p_1*term2;
      const Scalar term4 = p_0*p_0*term2;
      const Scalar term5 = p_1*p_1*term2;
      const Scalar term6 = term4 + term5;
      const Scalar term7 = 3*term4 + term5;
      const Scalar term8 = term6*term6;
      const Scalar term9 = term8*term6;
      const Scalar term10 = term9*term6;
      const Scalar term11 = k1*term6 + k2*term8 + k3*term9 + k4*term10;
      const Scalar term12 = fx*term6;
      const Scalar term13 = fx*p_0*term0;
      const Scalar term14 = p_1*term0;
      const Scalar term15 = term4 + 3*term5;
      const Scalar term16 = fy*term6;
      const Scalar term17 = fy*p_1*term0;
      
      (*pixel_wrt_grid_updates)(0, 0) = p1*term3 + p2*term7 + sx1*term6 + term1*term11 + term1;
      (*pixel_wrt_grid_updates)(0, 1) = 0;
      (*pixel_wrt_grid_updates)(0, 2) = 1;
      (*pixel_wrt_grid_updates)(0, 3) = 0;
      (*pixel_wrt_grid_updates)(0, 4) = term1*term12;
      (*pixel_wrt_grid_updates)(0, 5) = term13*term8;
      (*pixel_wrt_grid_updates)(0, 6) = term13*term9;
      (*pixel_wrt_grid_updates)(0, 7) = term10*term13;
      (*pixel_wrt_grid_updates)(0, 8) = fx*term3;
      (*pixel_wrt_grid_updates)(0, 9) = fx*term7;
      (*pixel_wrt_grid_updates)(0, 10) = term12;
      (*pixel_wrt_grid_updates)(0, 11) = 0;
      
      (*pixel_wrt_grid_updates)(1, 0) = 0;
      (*pixel_wrt_grid_updates)(1, 1) = p1*term15 + p2*term3 + sy1*term6 + term11*term14 + term14;
      (*pixel_wrt_grid_updates)(1, 2) = 0;
      (*pixel_wrt_grid_updates)(1, 3) = 1;
      (*pixel_wrt_grid_updates)(1, 4) = term14*term16;
      (*pixel_wrt_grid_updates)(1, 5) = term17*term8;
      (*pixel_wrt_grid_updates)(1, 6) = term17*term9;
      (*pixel_wrt_grid_updates)(1, 7) = term10*term17;
      (*pixel_wrt_grid_updates)(1, 8) = fy*term15;
      (*pixel_wrt_grid_updates)(1, 9) = fy*term3;
      (*pixel_wrt_grid_updates)(1, 10) = 0;
      (*pixel_wrt_grid_updates)(1, 11) = term16;
    }
    return true;
  }
  
  inline bool use_equidistant_projection() const {
    return m_use_equidistant_projection;
  }
  
 private:
  bool m_use_equidistant_projection;
};

}

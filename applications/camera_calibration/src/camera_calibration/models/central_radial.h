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

// Implements a model based on the Thin-Prism Fisheye and OpenCV models where
// the radial distortion function has been replaced with a spline.
// 
// Parameter ordering:
// fx, fy, cx, cy, p1, p2, sx1, sy1, [spline parameters ...].
class CentralRadialModel : public CameraModel {
 friend struct RadialStateWrapper;
 friend struct RadialDirectionCostFunction;
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline CentralRadialModel(int width, int height, int spline_resolution)
      : CameraModel(width, height, 0, 0, width, height, Type::CentralRadial) {
    m_parameters.resize(8 + spline_resolution);
  }
  
  CentralRadialModel(const CentralRadialModel& other);
  
  virtual CameraModel* duplicate() override;
  
  
  bool FitToDenseModel(
      const Image<Vec3d>& dense_model,
      int subsample_step,
      bool print_progress = false);
  
  
  virtual bool Project(
      const Vec3d& local_point,
      Vec2d* result) const override;
  
  bool ProjectWithJacobian(
      const Vec3d& local_point,
      Vec2d* result,
      Matrix<double, 2, 3>* dresult_dlocalpoint) const;
  
  virtual inline bool ProjectWithInitialEstimate(
      const Vec3d& local_point,
      Vec2d* result) const override {
    // Projection is not an optimization process for this model, so just ignore
    // the initial estimate.
    return Project(local_point, result);
  }
  
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
    return m_parameters.rows();
  }
  
  
  template <typename Derived>
  inline void SubtractDelta(const MatrixBase<Derived>& delta) {
    m_parameters -= delta;
  }
  
  static constexpr const int IntrinsicsJacobianSize = 8 + 4;
  
  template <typename DerivedA, typename DerivedB>
  bool ProjectionJacobianWrtIntrinsics(
      const Vec3d& local_point,
      const Vec2d& /*projected_pixel*/,
      const Image<LineTangents>& /*tangents_image*/,
      double /*numerical_diff_delta*/,
      MatrixBase<DerivedA>* grid_update_indices,
      MatrixBase<DerivedB>* pixel_wrt_grid_updates) const {
    if (local_point.z() <= 0) {
      return false;
    }
    
    typedef double Scalar;
    
    const double& fx = m_parameters(0);
    const double& fy = m_parameters(1);
    const double& p1 = m_parameters(4);
    const double& p2 = m_parameters(5);
    const double& sx1 = m_parameters(6);
    const double& sy1 = m_parameters(7);
    
    const double& p_0 = local_point.x();
    const double& p_1 = local_point.y();
    const double& p_2 = local_point.z();
    
    double original_angle = acos(std::max(-1., std::min(1., Vec3d(0, 0, 1).dot(local_point.normalized()))));
    double pos_in_spline = 1. + (spline_resolution() - 3.) / (M_PI / 2) * original_angle;
    int chunk = std::max(1, std::min(spline_resolution() - 3, static_cast<int>(pos_in_spline)));
    double fraction = pos_in_spline - chunk;
    const double& spline_param0 = m_parameters(8 + chunk - 1);
    const double& spline_param1 = m_parameters(8 + chunk + 0);
    const double& spline_param2 = m_parameters(8 + chunk + 1);
    const double& spline_param3 = m_parameters(8 + chunk + 2);
    
    for (int i = 0; i < 8; ++ i) {
      (*grid_update_indices)(i) = i;
    }
    for (int i = 0; i < 4; ++ i) {
      (*grid_update_indices)(8 + i) = 8 + chunk - 1 + i;
    }
    
    const Scalar term0 = 1.0/p_2;
    const Scalar term1 = p_0*term0;
    const Scalar term2 = p_2*p_2;
    const Scalar term3 = 1.0/term2;
    const Scalar term4 = 2*p_0*p_1*term3;
    const Scalar term5 = p_0*p_0;
    const Scalar term6 = term3*term5;
    const Scalar term7 = p_1*p_1;
    const Scalar term8 = term3*term7;
    const Scalar term9 = term6 + term8;
    const Scalar term10 = 3*term6 + term8;
    const Scalar term11 = fraction;  // frac((0.636619772367581*spline_resolution() - 1.90985931710274)*acos(p_2/sqrt(term2 + term5 + term7)) + 1.0);
    const Scalar term12 = 0.166666666666667*term11*term11*term11;
    const Scalar term13b = term11 - 1.0;
    const Scalar term13 = 0.166666666666667*(-term11 + 1)*term13b*term13b;
    const Scalar term14 = term11 + 3.0;
    const Scalar term15 = term14*term14*(0.5*term11 + 1.5);
    const Scalar term16 = 19.5*term11 - term14*(5.5*term11 + 16.5) + term15 + 36.6666666666667;
    const Scalar term17 = -16*term11 + term14*(5*term11 + 15.0) - term15 - 31.3333333333333;
    const Scalar term18 = spline_param0*term13 + spline_param1*term16 + spline_param2*term17 + spline_param3*term12;
    const Scalar term19 = fx*p_0*term0;
    const Scalar term20 = p_1*term0;
    const Scalar term21 = term6 + 3*term8;
    const Scalar term22 = fy*p_1*term0;
    
    (*pixel_wrt_grid_updates)(0, 0) = p1*term4 + p2*term10 + sx1*term9 + term1*term18 + term1;
    (*pixel_wrt_grid_updates)(0, 1) = 0;
    (*pixel_wrt_grid_updates)(0, 2) = 1;
    (*pixel_wrt_grid_updates)(0, 3) = 0;
    (*pixel_wrt_grid_updates)(0, 4) = fx*term4;
    (*pixel_wrt_grid_updates)(0, 5) = fx*term10;
    (*pixel_wrt_grid_updates)(0, 6) = fx*term9;
    (*pixel_wrt_grid_updates)(0, 7) = 0;
    (*pixel_wrt_grid_updates)(0, 8) = term13*term19;
    (*pixel_wrt_grid_updates)(0, 9) = term16*term19;
    (*pixel_wrt_grid_updates)(0, 10) = term17*term19;
    (*pixel_wrt_grid_updates)(0, 11) = term12*term19;
    
    (*pixel_wrt_grid_updates)(1, 0) = 0;
    (*pixel_wrt_grid_updates)(1, 1) = p1*term21 + p2*term4 + sy1*term9 + term18*term20 + term20;
    (*pixel_wrt_grid_updates)(1, 2) = 0;
    (*pixel_wrt_grid_updates)(1, 3) = 1;
    (*pixel_wrt_grid_updates)(1, 4) = fy*term21;
    (*pixel_wrt_grid_updates)(1, 5) = fy*term4;
    (*pixel_wrt_grid_updates)(1, 6) = 0;
    (*pixel_wrt_grid_updates)(1, 7) = fy*term9;
    (*pixel_wrt_grid_updates)(1, 8) = term13*term22;
    (*pixel_wrt_grid_updates)(1, 9) = term16*term22;
    (*pixel_wrt_grid_updates)(1, 10) = term17*term22;
    (*pixel_wrt_grid_updates)(1, 11) = term12*term22;
    
    return true;
  }
  
  
  inline int spline_resolution() const {
    return m_parameters.size() - 8;
  }
  
  inline const Matrix<double, Eigen::Dynamic, 1>& parameters() const { return m_parameters; }
  inline Matrix<double, Eigen::Dynamic, 1>& parameters() { return m_parameters; }
  
 private:
  Matrix<double, Eigen::Dynamic, 1> m_parameters;
};

}

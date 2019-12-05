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

#include "camera_calibration/models/central_radial.h"

#include <libvis/lm_optimizer.h>
#include <sophus/so3.hpp>

#include "camera_calibration/b_spline.h"
#include "camera_calibration/local_parametrizations/direction_parametrization.h"
#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"

namespace vis {

struct RadialStateWrapper {
  RadialStateWrapper(
      CentralRadialModel* model)
      : model(model) {}
  
  inline int degrees_of_freedom() const {
    return model->update_parameter_count();
  }
  
  static constexpr bool is_reversible() { return true; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    model->m_parameters -= delta.template cast<double>();
  }
  
  CentralRadialModel* model;
};

struct RadialDirectionCostFunction {
  RadialDirectionCostFunction(
      int width,
      int height,
      const Image<Vec3d>* dense_model,
      int subsample_step)
      : width(width),
        height(height),
        dense_model(dense_model),
        step(subsample_step) {}
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const RadialStateWrapper& state,
      Accumulator* accumulator) const {
    const CentralRadialModel& model = *state.model;
    
    double dense_model_to_camera_x = width / (1.f * dense_model->width());
    double dense_model_to_camera_y = height / (1.f * dense_model->height());
    
    for (u32 y = 0; y < dense_model->height(); y += step) {
      for (u32 x = 0; x < dense_model->width(); x += step) {
        const Vec3d& target = dense_model->at(x, y);
        if (target.hasNaN()) {
          continue;
        }
        
        float cam_x = dense_model_to_camera_x * (x + 0.5f);
        float cam_y = dense_model_to_camera_y * (y + 0.5f);
        
        Vec3d direction;
        if (!model.Unproject(cam_x, cam_y, &direction)) {
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddInvalidResidual();
          }
          continue;
        }
        
        if (!compute_jacobians) {
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddResidual(direction(d) - target(d));
          }
          continue;
        }
        
        // Since the unprojection is the result of an optimization process,
        // use numerical derivatives.
        Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> indices;
        Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> unprojection_wrt_intrinsics;
        indices.resize(Eigen::NoChange, state.model->parameters().size());
        unprojection_wrt_intrinsics.resize(Eigen::NoChange, state.model->parameters().size());
        
        bool ok = true;
        for (int i = 0; i < unprojection_wrt_intrinsics.cols(); ++ i) {
          indices(i) = i;
          
          CentralRadialModel* mutable_model = const_cast<CentralRadialModel*>(state.model);
          double original_parameter = mutable_model->m_parameters(i);
          // TODO: Would it make sense to use different delta magnitudes for
          //       different components of the intrinsics, depending on how
          //       sensitive they are to changes?
          mutable_model->m_parameters(i) += delta;
          
          Vec3d offset_direction;
          ok = mutable_model->Unproject(cam_x, cam_y, &offset_direction);
          
          mutable_model->m_parameters(i) = original_parameter;
          
          if (!ok) {
            break;
          }
          for (int d = 0; d < 3; ++ d) {
            unprojection_wrt_intrinsics(d, i) = (offset_direction(d) - direction(d)) / delta;
          }
        }
        
        if (!ok) {
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddResidual(direction(d) - target(d));
          }
          continue;
        }
        
        for (int d = 0; d < 3; ++ d) {
          accumulator->AddResidualWithJacobian(
              direction(d) - target(d),
              indices,
              unprojection_wrt_intrinsics.row(d));
        }
      }
    }
  }
  
  inline void SetDelta(double delta) {
    this->delta = delta;
  }
  
 private:
  int width;
  int height;
  const Image<Vec3d>* dense_model;
  double delta;
  int step;
};


CentralRadialModel::CentralRadialModel(const CentralRadialModel& other)
    : CameraModel(other),
      m_parameters(other.m_parameters) {}

CameraModel* CentralRadialModel::duplicate() {
  return new CentralRadialModel(*this);
}

bool CentralRadialModel::FitToDenseModel(
    const Image<Vec3d>& dense_model,
    int subsample_step,
    bool print_progress) {
  // First, linearly fit a simple parametric model
  double k1, k2;
  if (!FitSimpleParametricToDenseModelLinearly(
      dense_model,
      m_width, m_height,
      /*use_equidistant_projection*/ false,
      &m_parameters(0), &m_parameters(1),
      &m_parameters(2), &m_parameters(3),
      &k1, &k2)) {
    return false;
  }
  
  m_parameters(4) = 0;
  m_parameters(5) = 0;
  m_parameters(6) = 0;
  m_parameters(7) = 0;
  
  // Roughly initialize the radial model based on k1, k2 from the simple parametric model
  for (int i = 0; i < spline_resolution(); ++ i) {
    // Compute the angle for this spline control point
    double angle = (M_PI / 2) / (spline_resolution() - 3.) * (i - 1);
    double clamped_angle = std::max(0., std::min(M_PI / 2, angle));
    
    // Convert the angle to a radius
    // (since M_PI/2 cannot be represented exactly, we cannot get inf).
    const double r = tan(clamped_angle);
    const double r2 = r * r;
    const double r4 = r2 * r2;
    
    const double radial = k1 * r2 + k2 * r4;
    m_parameters[8 + i] = radial;
  }
  
  // Non-linearly refine the radial model parameters to match the dense model.
  // TODO: Matching projections rather than unprojections should be massively faster
  //       (since we have to use numerical derivatives for the current case, and the
  //        number of parameters can be high when having many spline parameters)
  RadialDirectionCostFunction cost_function(
      m_width,
      m_height,
      &dense_model,
      subsample_step);
  LMOptimizer<double> optimizer;
  RadialStateWrapper state(this);
  double deltas[] = {1e-2, 1e-3, 1e-4, 1e-5};
  for (int i = 0; i < sizeof(deltas) / sizeof(deltas[0]); ++ i) {
    cost_function.SetDelta(deltas[i]);
    LOG(INFO) << "Optimizing with delta: " << deltas[i];
    optimizer.Optimize(
        &state,
        cost_function,
        /*max_iteration_count*/ 20,
        /*max_lm_attempts*/ 10,
        /*init_lambda*/ -1,
        /*init_lambda_factor*/ 0.001f,
        print_progress);
  }
  
  return true;
}

bool CentralRadialModel::Project(
    const Vec3d& local_point,
    Vec2d* result) const {
  if (local_point.z() <= 0) {
    return false;
  }
  
  // Compute the angle of the local point from the optical axis (0, 0, 1)^T.
  // This is in the range [0, M_PI/2[ (since z > 0 due to the filtering above).
  double original_angle = acos(std::max(-1., std::min(1., Vec3d(0, 0, 1).dot(local_point.normalized()))));
  
  // Compute the floating-point position within the spline parameters, in the range [1, spline_resolution - 1].
  double pos_in_spline = 1. + (spline_resolution() - 3.) / (M_PI / 2) * original_angle;
  
  int chunk = std::max(1, std::min(spline_resolution() - 3, static_cast<int>(pos_in_spline)));
  
  double fraction = pos_in_spline - chunk;
  
  double radial_factor = EvalUniformCubicBSpline(
      m_parameters(8 + chunk - 1),
      m_parameters(8 + chunk + 0),
      m_parameters(8 + chunk + 1),
      m_parameters(8 + chunk + 2),
      fraction + 3.);
  
  // Parametric part
  Vec2d undistorted_point = Vec2d(local_point.x() / local_point.z(),
                                  local_point.y() / local_point.z());
  const double x2 = undistorted_point.coeff(0) * undistorted_point.coeff(0);
  const double xy = undistorted_point.coeff(0) * undistorted_point.coeff(1);
  const double y2 = undistorted_point.coeff(1) * undistorted_point.coeff(1);
  const double r2 = x2 + y2;
  
  const double& fx = m_parameters(0);
  const double& fy = m_parameters(1);
  const double& cx = m_parameters(2);
  const double& cy = m_parameters(3);
  const double& p1 = m_parameters(4);
  const double& p2 = m_parameters(5);
  const double& sx1 = m_parameters(6);
  const double& sy1 = m_parameters(7);
  
  const double dx = static_cast<double>(2) * p1 * xy + p2 * (r2 + static_cast<double>(2) * x2) + sx1 * r2;
  const double dy = static_cast<double>(2) * p2 * xy + p1 * (r2 + static_cast<double>(2) * y2) + sy1 * r2;
  
  Vec2d distorted_point = Vec2d(undistorted_point.coeff(0) + radial_factor * undistorted_point.coeff(0) + dx,
                                undistorted_point.coeff(1) + radial_factor * undistorted_point.coeff(1) + dy);
  *result = Vec2d(fx * distorted_point.x() + cx,
                  fy * distorted_point.y() + cy);
  
  return result->x() >= 0 &&
         result->y() >= 0 &&
         result->x() < m_width &&
         result->y() < m_height;
}

bool CentralRadialModel::ProjectWithJacobian(
    const Vec3d& local_point,
    Vec2d* result,
    Matrix<double, 2, 3>* dresult_dlocalpoint) const {
  if (local_point.z() <= 0) {
    return false;
  }
  
  typedef double Scalar;
  
  const double& fx = m_parameters(0);
  const double& fy = m_parameters(1);
  const double& cx = m_parameters(2);
  const double& cy = m_parameters(3);
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
  
  // Auto-generated function and Jacobian computation.
  // opcount = 300
  // NOTE: Manually inserted std::max(1e-8, term13) in the computation of term36 to prevent NaNs for direction (0, 0, 1)^T
  const Scalar term0 = 1.0/p_2;
  const Scalar term1 = p_0*term0;
  const Scalar term2 = 2*p1;
  const Scalar term3 = p_2 * p_2;
  const Scalar term4 = 1.0/term3;
  const Scalar term5 = p_0*term4;
  const Scalar term6 = term2*term5;
  const Scalar term7 = p_0 * p_0;
  const Scalar term8 = term4*term7;
  const Scalar term9 = p_1 * p_1;
  const Scalar term10 = term4*term9;
  const Scalar term11 = term10 + term8;
  const Scalar term12 = 0.636619772367581*spline_resolution() - 1.90985931710274;
  const Scalar term13 = term7 + term9;
  const Scalar term14 = term13 + term3;
  const Scalar term15 = sqrt(term14);
  const Scalar term16 = 1.0/term15;
  const Scalar term17 = fraction;  // frac(term12*acos(p_2*term16) + 1.0);
  const Scalar term18 = -0.166666666666667*term17 + 0.166666666666667;
  const Scalar term19 = term17 - 1.0;
  const Scalar term20 = term19 * term19;
  const Scalar term21 = term17 + 3.0;
  const Scalar term22 = 5.5*term17 + 16.5;
  const Scalar term23 = 0.5*term17;
  const Scalar term24 = term23 + 1.5;
  const Scalar term25 = term21 * term21;
  const Scalar term26 = 5*term17 + 15.0;
  const Scalar term27 = -term23 - 1.5;
  const Scalar term28 = spline_param0*term18*term20 + spline_param1*(19.5*term17 - term21*term22 + term24*term25 + 36.6666666666667) - spline_param2*(16*term17 - term21*term26 - term25*term27 + 31.3333333333333) + 0.166666666666667*spline_param3*term17*term17*term17;
  const Scalar term29 = p_1*term0;
  const Scalar term30 = p2*p_0*term4;
  const Scalar term31 = 2*term30;
  const Scalar term32 = p_1*term4;
  const Scalar term33 = 2*p_0*term4;
  const Scalar term34 = term0*(term28 + 1);
  const Scalar term35 = 0.5*spline_param3*term17*term17;
  const Scalar term36 = 1 / sqrt(std::max(1e-8, term13));
  const Scalar term37 = 1.0/term14;
  const Scalar term38 = p_0*p_2*term12*term36*term37;
  const Scalar term39 = 0.166666666666667*spline_param0*term20;
  const Scalar term40 = 2*spline_param0*term18*term19;
  const Scalar term41 = p_0*p_2*term12*term21*term36*term37;
  const Scalar term42 = 0.5*term25;
  const Scalar term43 = term38*term42;
  const Scalar term44 = 2*p_0*p_2*term12*term21*term36*term37;
  const Scalar term45 = spline_param1*(-term22*term38 + term24*term44 + 19.5*term38 - 5.5*term41 + term43) + spline_param2*(term26*term38 + term27*term44 - 16*term38 + 5*term41 - term43) + term35*term38 - term38*term39 + term38*term40;
  const Scalar term46 = 2*p_1*term4;
  const Scalar term47 = p2*term46 + term6;
  const Scalar term48 = p_1*p_2*term12*term36*term37;
  const Scalar term49 = p_1*p_2*term12*term21*term36*term37;
  const Scalar term50 = term42*term48;
  const Scalar term51 = 2*p_1*p_2*term12*term21*term36*term37;
  const Scalar term52 = spline_param1*(-term22*term48 + term24*term51 + 19.5*term48 - 5.5*term49 + term50) + spline_param2*(term26*term48 + term27*term51 - 16*term48 + 5*term49 - term50) + term35*term48 - term39*term48 + term40*term48;
  const Scalar term53 = 1 / (p_2*p_2*p_2);
  const Scalar term54 = 4*p_0*p_1*term53;
  const Scalar term55 = term53*term7;
  const Scalar term56 = term53*term9;
  const Scalar term57 = -2*term56;
  const Scalar term58 = -2*term55;
  const Scalar term59 = term57 + term58;
  const Scalar term60b = sqrt(term14);
  const Scalar term60 = term16 - term3/(term60b*term60b*term60b);
  const Scalar term61 = term12*term15*term36*term60;
  const Scalar term62 = term12*term15*term21*term36*term60;
  const Scalar term63 = term42*term61;
  const Scalar term64 = 2*term12*term15*term21*term36*term60;
  const Scalar term65 = -spline_param1*(-term22*term61 + term24*term64 + 19.5*term61 - 5.5*term62 + term63) - spline_param2*(term26*term61 + term27*term64 - 16*term61 + 5*term62 - term63) - term35*term61 + term39*term61 - term40*term61;
  
  *result = Vec2d(cx + fx*(p2*(term10 + 3*term8) + p_1*term6 + sx1*term11 + term1*term28 + term1),
                  cy + fy*(p1*(3*term10 + term8) + p_1*term31 + sy1*term11 + term28*term29 + term29));
  (*dresult_dlocalpoint)(0, 0) = fx*(sx1*term33 + term1*term45 + term2*term32 + 6*term30 + term34);
  (*dresult_dlocalpoint)(0, 1) = fx*(sx1*term46 + term1*term52 + term47);
  (*dresult_dlocalpoint)(0, 2) = fx*(-p1*term54 + p2*(-6*term55 + term57) + sx1*term59 + term1*term65 - term28*term5 - term5);
  (*dresult_dlocalpoint)(1, 0) = fy*(sy1*term33 + term29*term45 + term47);
  (*dresult_dlocalpoint)(1, 1) = fy*(6*p1*term32 + sy1*term46 + term29*term52 + term31 + term34);
  (*dresult_dlocalpoint)(1, 2) = fy*(p1*(-6*term56 + term58) - p2*term54 + sy1*term59 - term28*term32 + term29*term65 - term32);
  
  return result->x() >= 0 &&
         result->y() >= 0 &&
         result->x() < m_width &&
         result->y() < m_height;
}

bool CentralRadialModel::Unproject(double x, double y, Vec3d* direction) const {
  // Use the optical axis direction as initial guess
  Vec3d cur_dir(0, 0, 1);
  
  // Levenberg-Marquardt optimization algorithm.
  const double kUndistortionEpsilon = 1e-10f;
  const usize kMaxIterations = 100;
  
  double lambda = -1;
  
  bool converged = false;
  for (usize i = 0; i < kMaxIterations; ++i) {
    // Compute projection and Jacobian
    Matrix<double, 2, 3> ddxy_ddir;
    Vec2d cur_projected;
    if (!ProjectWithJacobian(cur_dir, &cur_projected, &ddxy_ddir)) {
      return false;
    }
    
    Vec2d debug_projected;
    if (!Project(cur_dir, &debug_projected)) {
      return false;
    }
    
    DirectionTangents tangents;
    ComputeTangentsForDirectionOrLine(cur_dir, &tangents);
    Matrix<double, 3, 2> ddir_dupdate;
    DirectionJacobianWrtLocalUpdate(tangents, &ddir_dupdate);
    
    Matrix<double, 2, 2> jacobian = ddxy_ddir * ddir_dupdate;
    
    // (Non-squared) residuals.
    double dx = cur_projected.x() - x;
    double dy = cur_projected.y() - y;
    double cost = dx * dx + dy * dy;
    
    // Accumulate H and b.
    double H_0_0 = jacobian(0, 0) * jacobian(0, 0) + jacobian(1, 0) * jacobian(1, 0);
    double H_1_0_and_0_1 = jacobian(0, 0) * jacobian(0, 1) + jacobian(1, 0) * jacobian(1, 1);
    double H_1_1 = jacobian(0, 1) * jacobian(0, 1) + jacobian(1, 1) * jacobian(1, 1);
    double b_0 = dx * jacobian(0, 0) + dy * jacobian(1, 0);
    double b_1 = dx * jacobian(0, 1) + dy * jacobian(1, 1);
    
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
      
      Vec3d test_dir = cur_dir;
      ApplyLocalUpdateToDirection(
          &test_dir,
          tangents,
          -1 * x_0,
          -1 * x_1);
      
      Vec2d test_projected;
      bool projects = Project(test_dir, &test_projected);
      dx = test_projected.x() - x;
      dy = test_projected.y() - y;
      double test_cost = dx * dx + dy * dy;
      
      if (projects && test_cost < cost) {
        // Accept update
        cost = test_cost;
        cur_dir = test_dir;
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
  
  *direction = cur_dir;
  return true;
}

}

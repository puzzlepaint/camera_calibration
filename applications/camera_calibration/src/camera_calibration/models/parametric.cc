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

#include "camera_calibration/models/parametric.h"

#include <libvis/lm_optimizer.h>
#include <sophus/so3.hpp>

#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"

namespace vis {

struct ParametricStateWrapper {
  ParametricStateWrapper(
      ParametricCameraModel* model,
      Mat3d* parametric_r_dense)
      : model(model),
        parametric_r_dense(parametric_r_dense) {}
  
  inline int degrees_of_freedom() const {
    return model->m_parameters.rows() + (parametric_r_dense ? 3 : 0);
  }
  
  static constexpr bool is_reversible() { return true; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    model->m_parameters -= delta.template topRows(model->update_parameter_count()).template cast<double>();
    
    if (parametric_r_dense) {
      Quaterniond q(parametric_r_dense->cast<double>());
      q = ApplyLocalUpdateToQuaternion(
          q, -delta.template bottomRows<3>().template cast<double>());
      *parametric_r_dense = q.matrix().template cast<double>();
    }
  }
  
  ParametricCameraModel* model;
  Mat3d* parametric_r_dense;
};


struct ParametricDirectionCostFunction {
  ParametricDirectionCostFunction(
      int width,
      int height,
      const Image<Vec3d>* dense_model,
      int subsample_step,
      bool allow_rotation)
      : width(width),
        height(height),
        dense_model(dense_model),
        step(subsample_step),
        allow_rotation(allow_rotation) {}
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const ParametricStateWrapper& state,
      Accumulator* accumulator) const {
    const ParametricCameraModel& model = *state.model;
    Quaterniond q;
    if (state.parametric_r_dense) {
      q = Quaterniond(state.parametric_r_dense->cast<double>());
    }
    
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
        
        Vec3d rotated_target = state.parametric_r_dense ? (*state.parametric_r_dense * target) : target;
        
        Vec3d direction;
        if (!model.Unproject(cam_x, cam_y, &direction)) {
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddInvalidResidual();
          }
          continue;
        }
        
        if (!compute_jacobians) {
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddResidual(direction(d) - rotated_target(d));
          }
          continue;
        }
        
        // Since the unprojection is the result of an optimization process,
        // use numerical derivatives.
        constexpr int kNumVariables = 12;
        CHECK_EQ(kNumVariables, state.model->parameters().size()) << "This is currently hard-coded to 12 variables here.";
        Matrix<double, 3, kNumVariables, Eigen::RowMajor> unprojection_wrt_intrinsics;
        
        bool ok = true;
        for (int i = 0; i < kNumVariables; ++ i) {
          ParametricCameraModel* mutable_model = const_cast<ParametricCameraModel*>(state.model);
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
            accumulator->AddResidual(direction(d) - rotated_target(d));
          }
          continue;
        }
        
        if (allow_rotation) {
          // Derivative of the result wrt. local changes to parametric_r_dense:
          Matrix<double, 3, 3> rotated_point_wrt_update;
          ComputeRotatedPointJacobianWrtRotationUpdate<double>(
              q, target(0), target(1), target(2),
              &rotated_point_wrt_update);
          
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddResidualWithJacobian(
                direction(d) - rotated_target(d),
                0,
                unprojection_wrt_intrinsics.row(d),
                kNumVariables,
                -1 * rotated_point_wrt_update.row(d));
          }
        } else {
          for (int d = 0; d < 3; ++ d) {
            accumulator->AddResidualWithJacobian(
                direction(d) - rotated_target(d),
                0,
                unprojection_wrt_intrinsics.row(d));
          }
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
  bool allow_rotation;
};


bool FitSimpleParametricToDenseModelLinearly(
    const Image<Vec3d>& dense_model,
    int width, int height,
    bool use_equidistant_projection,
    double* fx, double* fy,
    double* cx, double* cy,
    double* k1, double* k2) {
  // Initialize the model by fitting a pinhole model to some points of the dense
  // model, and setting the rest of the parameters to zero.
  // Pinhole model mapping normalized image coordinates (nx, ny) to pixels:
  // px = fx * nx + cx;
  // py = fy * ny + cy;
  // We can thus create a linear equation system, where each pixel contributes
  // the following equations (for the pinhole-only case):
  // (nx  0 1 0)   (fx)   (px)
  // ( 0 ny 0 1) * (fy) = (py)
  //               (cx)
  //               (cy)
  // The equations for (fx, cx) and (fy, cy) are independent, so we can solve
  // for these individually.
  // 
  // For also taking k1 and k2 into account, we use the equation system(s):
  // (nx nx*r2 nx*r4   0     0     0 1 0)   (   fx)   (px)
  // ( 0     0     0  ny ny*r2 ny*r4 0 1) * (fx_k1) = (py)
  //                                        (fx_k2)
  //                                        (   fy)
  //                                        (fy_k1)
  //                                        (fy_k2)
  //                                        (   cx)
  //                                        (   cy)
  
  double dense_model_to_camera_x = width / (1.f * dense_model.width());
  double dense_model_to_camera_y = height / (1.f * dense_model.height());
  
  constexpr int kVariableCount = 4;
  constexpr int kCIndex = kVariableCount - 1;
  
  Matrix<double, Eigen::Dynamic, kVariableCount> matrix_x;
  matrix_x.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  Matrix<double, Eigen::Dynamic, kVariableCount> matrix_y;
  matrix_y.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  
  Matrix<double, Eigen::Dynamic, 1> pixels_x;
  pixels_x.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  Matrix<double, Eigen::Dynamic, 1> pixels_y;
  pixels_y.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  
  u32 row_x = 0;
  u32 row_y = 0;
  for (u32 y = 0; y < dense_model.height(); ++ y) {
    for (u32 x = 0; x < dense_model.width(); ++ x) {
      const Vec3d& direction = dense_model(x, y);
      if (direction.z() <= 0) {
        // We cannot represent directions that point backwards in this model.
        continue;
      }
      
      Vec2d undistorted_point = direction.hnormalized();  // divide (x, y) by z
      
      double r = undistorted_point.norm();
      
      Vec2d nxy;
      const double kEpsilon = static_cast<double>(1e-6);
      if (use_equidistant_projection && r > kEpsilon) {
        double theta_by_r = std::atan(r) / r;
        nxy.x() = theta_by_r * undistorted_point.coeff(0);
        nxy.y() = theta_by_r * undistorted_point.coeff(1);
      } else {
        nxy.x() = undistorted_point.coeff(0);
        nxy.y() = undistorted_point.coeff(1);
      }
      
      const double x2 = nxy.x() * nxy.x();
      const double y2 = nxy.y() * nxy.y();
      const double r2 = x2 + y2;
      const double r4 = r2 * r2;
      
      if (!std::isnan(nxy.x())) {
        matrix_x(row_x, 0) = nxy.x();
        matrix_x(row_x, 1) = nxy.x() * r2;
        matrix_x(row_x, 2) = nxy.x() * r4;
        matrix_x(row_x, kCIndex) = 1;
        pixels_x(row_x, 0) = dense_model_to_camera_x * (x + 0.5f);  // pixel-corner convention
        ++ row_x;
      }
      if (!std::isnan(nxy.y())) {
        matrix_y(row_y, 0) = nxy.y();
        matrix_y(row_y, 1) = nxy.y() * r2;
        matrix_y(row_y, 2) = nxy.y() * r4;
        matrix_y(row_y, kCIndex) = 1;
        pixels_y(row_y, 0) = dense_model_to_camera_y * (y + 0.5f);  // pixel-corner convention
        ++ row_y;
      }
    }
  }
  
  if (row_x < kVariableCount || row_y < kVariableCount) {
    LOG(ERROR) << "Not enough data to fit a simple parametric model";
    return false;
  }
  
  matrix_x.conservativeResize(row_x, Eigen::NoChange);
  pixels_x.conservativeResize(row_x);
  
  matrix_y.conservativeResize(row_y, Eigen::NoChange);
  pixels_y.conservativeResize(row_y);
  
  // Solve the (overdetermined) linear system of equations using the pseudoinverse
  Matrix<double, kVariableCount, 1> result_x = (matrix_x.transpose() * matrix_x).inverse() * (matrix_x.transpose() * pixels_x);
  Matrix<double, kVariableCount, 1> result_y = (matrix_y.transpose() * matrix_y).inverse() * (matrix_y.transpose() * pixels_y);
  
  // Extract the value of k1 from the results.
  *k1 = 0.5 * ((result_x(1) / result_x(0)) + (result_y(1) / result_y(0)));
  *k2 = 0.5 * ((result_x(2) / result_x(0)) + (result_y(2) / result_y(0)));
  
  // Initialize the parameters with the result
  *fx = result_x(0);
  *fy = result_y(0);
  *cx = result_x(kCIndex);
  *cy = result_y(kCIndex);
  
  return true;
}


bool FitPinholeToDenseModelLinearly(
    const Image<Vec3d>& dense_model,
    int width, int height,
    bool use_equidistant_projection,
    double* fx, double* fy,
    double* cx, double* cy) {
  // Initialize the model by fitting a pinhole model to some points of the dense
  // model, and setting the rest of the parameters to zero.
  // Pinhole model mapping normalized image coordinates (nx, ny) to pixels:
  // px = fx * nx + cx;
  // py = fy * ny + cy;
  // We can thus create a linear equation system, where each pixel contributes
  // the following equations (for the pinhole-only case):
  // (nx  0 1 0)   (fx)   (px)
  // ( 0 ny 0 1) * (fy) = (py)
  //               (cx)
  //               (cy)
  // The equations for (fx, cx) and (fy, cy) are independent, so we can solve
  // for these individually.
  
  double dense_model_to_camera_x = width / (1.f * dense_model.width());
  double dense_model_to_camera_y = height / (1.f * dense_model.height());
  
  constexpr int kVariableCount = 2;
  constexpr int kCIndex = kVariableCount - 1;
  
  Matrix<double, Eigen::Dynamic, kVariableCount> matrix_x;
  matrix_x.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  Matrix<double, Eigen::Dynamic, kVariableCount> matrix_y;
  matrix_y.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  
  Matrix<double, Eigen::Dynamic, 1> pixels_x;
  pixels_x.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  Matrix<double, Eigen::Dynamic, 1> pixels_y;
  pixels_y.resize(dense_model.width() * dense_model.height(), Eigen::NoChange);
  
  u32 row_x = 0;
  u32 row_y = 0;
  for (u32 y = 0; y < dense_model.height(); ++ y) {
    for (u32 x = 0; x < dense_model.width(); ++ x) {
      const Vec3d& direction = dense_model(x, y);
      if (direction.z() <= 0) {
        // We cannot represent directions that point backwards in this model.
        continue;
      }
      
      Vec2d undistorted_point = direction.hnormalized();  // divide (x, y) by z
      
      double r = undistorted_point.norm();
      
      Vec2d nxy;
      const double kEpsilon = static_cast<double>(1e-6);
      if (use_equidistant_projection && r > kEpsilon) {
        double theta_by_r = std::atan(r) / r;
        nxy.x() = theta_by_r * undistorted_point.coeff(0);
        nxy.y() = theta_by_r * undistorted_point.coeff(1);
      } else {
        nxy.x() = undistorted_point.coeff(0);
        nxy.y() = undistorted_point.coeff(1);
      }
      
      if (!std::isnan(nxy.x())) {
        matrix_x(row_x, 0) = nxy.x();
        matrix_x(row_x, kCIndex) = 1;
        pixels_x(row_x, 0) = dense_model_to_camera_x * (x + 0.5f);  // pixel-corner convention
        ++ row_x;
      }
      if (!std::isnan(nxy.y())) {
        matrix_y(row_y, 0) = nxy.y();
        matrix_y(row_y, kCIndex) = 1;
        pixels_y(row_y, 0) = dense_model_to_camera_y * (y + 0.5f);  // pixel-corner convention
        ++ row_y;
      }
    }
  }
  
  if (row_x < kVariableCount || row_y < kVariableCount) {
    LOG(ERROR) << "Not enough data to fit a simple parametric model";
    return false;
  }
  
  matrix_x.conservativeResize(row_x, Eigen::NoChange);
  pixels_x.conservativeResize(row_x);
  
  matrix_y.conservativeResize(row_y, Eigen::NoChange);
  pixels_y.conservativeResize(row_y);
  
  // Solve the (overdetermined) linear system of equations using the pseudoinverse
  Matrix<double, kVariableCount, 1> result_x = (matrix_x.transpose() * matrix_x).inverse() * (matrix_x.transpose() * pixels_x);
  Matrix<double, kVariableCount, 1> result_y = (matrix_y.transpose() * matrix_y).inverse() * (matrix_y.transpose() * pixels_y);
  
  // Initialize the parameters with the result
  *fx = result_x(0);
  *fy = result_y(0);
  *cx = result_x(kCIndex);
  *cy = result_y(kCIndex);
  
  return true;
}


ParametricCameraModel::ParametricCameraModel(const ParametricCameraModel& other)
    : CameraModel(other.width(), other.height(), 0, 0, other.width(), other.height(), other.type()),
      m_parameters(other.m_parameters) {}

bool ParametricCameraModel::FitToDenseModel(
    const Image<Vec3d>& dense_model,
    Mat3d* parametric_r_dense,
    int subsample_step,
    bool print_progress) {
  if (!FitToDenseModelLinearly(dense_model)) {
    return false;
  }
  
  // Optimize all parameters
  ParametricDirectionCostFunction cost_function(
      m_width,
      m_height,
      &dense_model,
      subsample_step,
      /*allow_rotation*/ true);
  LMOptimizer<double> optimizer;
  ParametricStateWrapper state(this, parametric_r_dense);
  double deltas[] = {1e-2, 1e-3, 1e-4, 1e-5};
  for (int i = 0; i < sizeof(deltas) / sizeof(deltas[0]); ++ i) {
    cost_function.SetDelta(deltas[i]);
    LOG(INFO) << "Optimizing with delta: " << deltas[i];
    optimizer.Optimize(
        &state,
        cost_function,
        /*max_iteration_count*/ 300,
        /*max_lm_attempts*/ 10,
        /*init_lambda*/ -1,
        /*init_lambda_factor*/ 0.001f,
        print_progress);
  }
  
  return true;
}

bool ParametricCameraModel::FitToDenseModel(
    const Image<Vec3d>& dense_model,
    int subsample_step,
    bool print_progress) {
  if (!FitToDenseModelLinearly(dense_model)) {
    return false;
  }
  
  // Optimize all parameters
  ParametricDirectionCostFunction cost_function(
      m_width,
      m_height,
      &dense_model,
      subsample_step,
      /*allow_rotation*/ false);
  LMOptimizer<double> optimizer;
  ParametricStateWrapper state(this, nullptr);
  double deltas[] = {1e-2, 1e-3, 1e-4, 1e-5};
  for (int i = 0; i < sizeof(deltas) / sizeof(deltas[0]); ++ i) {
    cost_function.SetDelta(deltas[i]);
    LOG(INFO) << "Optimizing with delta: " << deltas[i];
    optimizer.Optimize(
        &state,
        cost_function,
        /*max_iteration_count*/ 300,
        /*max_lm_attempts*/ 10,
        /*init_lambda*/ -1,
        /*init_lambda_factor*/ 0.001f,
        print_progress);
  }
  
  return true;
}

}

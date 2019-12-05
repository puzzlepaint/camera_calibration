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

#include "camera_calibration/models/noncentral_generic.h"

#include "libvis/lm_optimizer.h"

#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/local_parametrizations/line_parametrization.h"

// Include Jacobians implementation
#include "camera_calibration/models/noncentral_generic_jacobians.cc"

namespace vis {

/// Uses 5-dimensional local updates for lines in 3D space.
struct LineGridStateWithLocalUpdates {
  LineGridStateWithLocalUpdates(
      Image<Vec3d>* point_grid,
      Image<Vec3d>* direction_grid)
      : point_grid(*point_grid),
        direction_grid(*direction_grid) {}
  
  LineGridStateWithLocalUpdates(const LineGridStateWithLocalUpdates& other)
      : point_grid(other.point_grid),
        direction_grid(other.direction_grid) {}
  
  LineGridStateWithLocalUpdates& operator= (const LineGridStateWithLocalUpdates& other) {
    point_grid = other.point_grid;
    direction_grid = other.direction_grid;
    return *this;
  }
  
  inline int degrees_of_freedom() const {
    return 5 * point_grid.width() * point_grid.height();
  }
  
  /// Due to the way the local updates work, first subtracting delta and then
  /// subtracting minus delta will not lead to the initial value again. Thus,
  /// the state is not reversible.
  static constexpr bool is_reversible() { return false; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    int i = 0;
    for (u32 y = 0; y < point_grid.height(); ++ y) {
      for (u32 x = 0; x < point_grid.width(); ++ x) {
        // TODO: Cache tangents earlier, do not recompute here
        LineTangents tangents;
        ComputeTangentsForDirectionOrLine(direction_grid.at(x, y), &tangents);
        
        ParametrizedLine<double, 3> line(point_grid(x, y), direction_grid(x, y));
        
        ApplyLocalUpdateToLine(
            &line, tangents, -delta(i + 0), -delta(i + 1), -delta(i + 2),
            -delta(i + 3), -delta(i + 4));
        i += 5;
        
        point_grid(x, y) = line.origin();
        direction_grid(x, y) = line.direction();
      }
    }
    CHECK_EQ(i, degrees_of_freedom());
  }
  
  Image<Vec3d> point_grid;
  Image<Vec3d> direction_grid;
};

NoncentralGenericModel::NoncentralGenericModel()
    : CameraModel(-1, -1, -1, -1, -1, -1, CameraModel::Type::NoncentralGeneric) {}

NoncentralGenericModel::NoncentralGenericModel(
    int grid_resolution_x, int grid_resolution_y,
    int calibration_min_x, int calibration_min_y,
    int calibration_max_x, int calibration_max_y,
    int width, int height)
    : CameraModel(width, height,
                  calibration_min_x, calibration_min_y,
                  calibration_max_x, calibration_max_y,
                  CameraModel::Type::NoncentralGeneric) {
  m_point_grid.SetSize(grid_resolution_x, grid_resolution_y);
  m_direction_grid.SetSize(grid_resolution_x, grid_resolution_y);
}

NoncentralGenericModel::NoncentralGenericModel(const NoncentralGenericModel& other)
    : CameraModel(other.m_width, other.m_height,
                  other.m_calibration_min_x, other.m_calibration_min_y,
                  other.m_calibration_max_x, other.m_calibration_max_y,
                  CameraModel::Type::NoncentralGeneric) {
  m_point_grid = other.m_point_grid;
  m_direction_grid = other.m_direction_grid;
}

NoncentralGenericModel& NoncentralGenericModel::operator= (const NoncentralGenericModel& other) {
  m_width = other.m_width;
  m_height = other.m_height;
  m_calibration_min_x = other.m_calibration_min_x;
  m_calibration_min_y = other.m_calibration_min_y;
  m_calibration_max_x = other.m_calibration_max_x;
  m_calibration_max_y = other.m_calibration_max_y;
  m_point_grid = other.m_point_grid;
  m_direction_grid = other.m_direction_grid;
  return *this;
}

CameraModel* NoncentralGenericModel::duplicate() {
  return new NoncentralGenericModel(*this);
}

void NoncentralGenericModel::InitializeFromCentralGenericModel(const CentralGenericModel& other) {
  m_direction_grid = other.grid();
  m_point_grid.SetSize(m_direction_grid.size());
  m_point_grid.SetTo(Vec3d::Zero());
  m_calibration_min_x = other.calibration_min_x();
  m_calibration_min_y = other.calibration_min_y();
  m_calibration_max_x = other.calibration_max_x();
  m_calibration_max_y = other.calibration_max_y();
  m_width = other.width();
  m_height = other.height();
}

void NoncentralGenericModel::Scale(double factor) {
  for (u32 y = 0; y < m_point_grid.height(); ++ y) {
    for (u32 x = 0; x < m_point_grid.width(); ++ x) {
      m_point_grid(x, y) = factor * m_point_grid(x, y);
    }
  }
}

bool NoncentralGenericModel::ProjectWithInitialEstimate(const Vec3d& local_point, Vec2d* result) const {
  // Levenberg-Marquardt optimization algorithm.
  constexpr double kEpsilon = 1e-12;
  const usize kMaxIterations = 100;
  
  double lambda = -1;
  for (usize i = 0; i < kMaxIterations; ++i) {
    Matrix<double, 6, 2> dline_dxy;
    Line3d line;
    CHECK(UnprojectWithJacobian(result->x(), result->y(), &line, &dline_dxy));
    
    LineTangents tangents;
    ComputeTangentsForDirectionOrLine(line.direction(), &tangents);
    
    // (Non-squared) residuals.
    Vec3d point_to_origin = line.origin() - local_point;
    double d1 = tangents.t1.dot(point_to_origin);
    double d2 = tangents.t2.dot(point_to_origin);
    
    // Jacobian of residuals wrt. pixel x, y [2 x 2]
    Matrix<double, 6, 3> tangents_wrt_direction;
    TangentsJacobianWrtLineDirection(
        line.direction(),
        &tangents_wrt_direction);
    
    Matrix<double, 2, 9> d_wrt_t1_t2_origin;
    d_wrt_t1_t2_origin <<
        point_to_origin.x(), point_to_origin.y(), point_to_origin.z(), 0, 0, 0, tangents.t1.x(), tangents.t1.y(), tangents.t1.z(),
        0, 0, 0, point_to_origin.x(), point_to_origin.y(), point_to_origin.z(), tangents.t2.x(), tangents.t2.y(), tangents.t2.z();
    
    Matrix<double, 9, 2> t1_t2_origin_wrt_xy = Matrix<double, 9, 2>::Zero();
    t1_t2_origin_wrt_xy.block<6, 2>(0, 0) =
        tangents_wrt_direction * dline_dxy.block<3, 2>(0, 0);
    t1_t2_origin_wrt_xy.block<3, 2>(6, 0) =
        dline_dxy.block<3, 2>(3, 0);
    
    Matrix<double, 2, 2> residuals_wrt_xy =
        d_wrt_t1_t2_origin * t1_t2_origin_wrt_xy;
    
    double cost = d1 * d1 + d2 * d2;
    
    // Accumulate H and b.
    double H_0_0 = residuals_wrt_xy(0, 0) * residuals_wrt_xy(0, 0) + residuals_wrt_xy(1, 0) * residuals_wrt_xy(1, 0);
    double H_1_0_and_0_1 = residuals_wrt_xy(0, 0) * residuals_wrt_xy(0, 1) + residuals_wrt_xy(1, 0) * residuals_wrt_xy(1, 1);
    double H_1_1 = residuals_wrt_xy(0, 1) * residuals_wrt_xy(0, 1) + residuals_wrt_xy(1, 1) * residuals_wrt_xy(1, 1);
    double b_0 = d1 * residuals_wrt_xy(0, 0) + d2 * residuals_wrt_xy(1, 0);
    double b_1 = d1 * residuals_wrt_xy(0, 1) + d2 * residuals_wrt_xy(1, 1);
    
    if (lambda < 0) {
      constexpr double kInitialLambdaFactor = 0.01;
      lambda = kInitialLambdaFactor * 0.5 * (H_0_0 + H_1_1);
    }
    
    bool update_accepted = false;
    for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
      double H_0_0_LM = H_0_0 + lambda;
      double H_1_1_LM = H_1_1 + lambda;
      
      // Solve the system.
      double x_1 = (b_1 - H_1_0_and_0_1 / H_0_0_LM * b_0) /
                   (H_1_1_LM - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0_LM);
      double x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0_LM;
      
      // Compute the test state (constrained to the calibrated image area).
      Vec2d test_result(
          std::max<double>(m_calibration_min_x, std::min(m_calibration_max_x + 0.999, result->x() - x_0)),
          std::max<double>(m_calibration_min_y, std::min(m_calibration_max_y + 0.999, result->y() - x_1)));
      
      // Compute the test cost.
      double test_cost = numeric_limits<double>::infinity();
      Line3d test_line;
      if (Unproject(test_result.x(), test_result.y(), &test_line)) {
        LineTangents test_tangents;
        ComputeTangentsForDirectionOrLine(test_line.direction(), &test_tangents);
        
        // (Non-squared) residuals.
        Vec3d test_point_to_origin = test_line.origin() - local_point;
        double test_d1 = test_tangents.t1.dot(test_point_to_origin);
        double test_d2 = test_tangents.t2.dot(test_point_to_origin);
        
        test_cost = test_d1 * test_d1 + test_d2 * test_d2;
      }
      
      if (test_cost < cost) {
        lambda *= 0.5;
        *result = test_result;
        update_accepted = true;
        break;
      } else {
        lambda *= 2;
      }
    }
    
    if (!update_accepted) {
      // if (cost >= kEpsilon) {
      //   LOG(WARNING) << "No update found and not converged. Current state: " << result->transpose();
      // }
      
      return cost < kEpsilon;
    }
    
    if (cost < kEpsilon) {
      return true;
    }
  }
  
  // LOG(WARNING) << "Not converged. Current state: " << result->transpose();
  return false;
}

bool NoncentralGenericModel::UnprojectWithJacobian(double x, double y, Line3d* result, Matrix<double, 6, 2>* dresult_dxy) const {
  if (!IsInCalibratedArea(x, y)) {
    return false;
  }
  
  Vec2d grid_point = PixelCornerConvToGridPoint(x, y) + Vec2d(2, 2);
  
  int ix = std::floor(grid_point.x());
  int iy = std::floor(grid_point.y());
  
  double frac_x = grid_point.x() - (ix - 3);
  double frac_y = grid_point.y() - (iy - 3);
  
  Matrix<double, 6, 1> p[4][4];
  for (int y = 0; y < 4; ++ y) {
    for (int x = 0; x < 4; ++ x) {
      p[y][x].topRows<3>() = m_direction_grid(ix - 3 + x, iy - 3 + y);
      p[y][x].bottomRows<3>() = m_point_grid(ix - 3 + x, iy - 3 + y);
    }
  }
  
  NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, p, result, dresult_dxy);
  for (int i = 0; i < 6; ++ i) {
    (*dresult_dxy)(i, 0) = PixelScaleToGridScaleX((*dresult_dxy)(i, 0));
    (*dresult_dxy)(i, 1) = PixelScaleToGridScaleY((*dresult_dxy)(i, 1));
  }
  return true;
}

void NoncentralGenericModel::DebugVerifyGridValues() {
  for (u32 y = 0; y < m_direction_grid.height(); ++ y) {
    for (u32 x = 0; x < m_direction_grid.width(); ++ x) {
      if (fabs(1 - m_direction_grid(x, y).squaredNorm()) > 0.001f) {
        LOG(ERROR) << "Grid value at " << x << ", " << y << " is not normalized (length: " << m_direction_grid(x, y).norm() << ")";
        LOG(ERROR) << "Not reporting possible additional errors to avoid log spam.";
        return;
      }
    }
  }
}

}

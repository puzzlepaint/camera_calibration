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

#include "camera_calibration/models/central_generic.h"

#ifdef LIBVIS_HAVE_CUDA
#include <cuda_runtime.h>
#include <libvis/cuda/cuda_buffer.h>
#endif
#include <libvis/lm_optimizer.h>

#include "camera_calibration/local_parametrizations/direction_parametrization.h"

#ifdef LIBVIS_HAVE_CUDA
#include "camera_calibration/models/cuda_central_generic_model.cuh"
#endif

// Include Jacobians implementation
#include "camera_calibration/models/central_generic_jacobians.cc"

namespace vis {

/// Uses 2-dimensional local updates for directions on the unit sphere.
struct DirectionGridStateWithLocalUpdates {
  DirectionGridStateWithLocalUpdates(Image<Vec3d>* grid)
      : grid(*grid) {}
  
  DirectionGridStateWithLocalUpdates(const DirectionGridStateWithLocalUpdates& other)
      : grid(other.grid) {}
  
  DirectionGridStateWithLocalUpdates& operator= (const DirectionGridStateWithLocalUpdates& other) {
    grid = other.grid;
    return *this;
  }
  
  inline int degrees_of_freedom() const {
    return 2 * grid.width() * grid.height();
  }
  
  /// Due to the way the local updates work, first subtracting delta and then
  /// subtracting minus delta will not lead to the initial value again. Thus,
  /// the state is not reversible.
  static constexpr bool is_reversible() { return false; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    int i = 0;
    for (u32 y = 0; y < grid.height(); ++ y) {
      for (u32 x = 0; x < grid.width(); ++ x) {
        // TODO: Cache tangents earlier, do not recompute here
        DirectionTangents tangents;
        ComputeTangentsForDirectionOrLine(grid.at(x, y), &tangents);
        ApplyLocalUpdateToDirection(
            &grid.at(x, y), tangents,
            -delta(i + 0), -delta(i + 1));
        i += 2;
      }
    }
    CHECK_EQ(i, degrees_of_freedom());
  }
  
  Image<Vec3d> grid;
};


void CentralGenericModel::ComputeUnprojectedDirectionResidualAndJacobianWrtGridUpdates(
    const Vec3d& measurement,
    const Vec2d& grid_point,
    const Image<Vec3d>& grid,
    const Image<DirectionTangents>& tangents_image,
    Matrix<double, 3, 1>* residuals,
    Matrix<int, 2*16, 1>* local_indices,
    Matrix<double, 3, 2*16, Eigen::RowMajor>* local_jacobian) {
  int ix = std::floor(grid_point.x() + 2);
  int iy = std::floor(grid_point.y() + 2);
  
  double frac_x = grid_point.x() + 2 - (ix - 3);
  double frac_y = grid_point.y() + 2 - (iy - 3);
  
  Matrix<int, 16, 1> indices;  // sequential indices of the 4x4 interpolation points in the complete grid
  Matrix<double, 3, 16> values;  // each column holds the corresponding grid direction value
  for (int y = 0; y < 4; ++ y) {
    int gy = iy - 3 + y;
    for (int x = 0; x < 4; ++ x) {
      int gx = ix - 3 + x;
      indices[x + 4 * y] = gx + gy * grid.width();
      values.col(x + 4 * y) = grid.at(gx, gy);
    }
  }
  
  Matrix<double, 3, 3*16, Eigen::RowMajor> jacobian_wrt_directions;
  CentralGenericBSplineDirectionCostFunction_ComputeResidualAndJacobian<double>(
      frac_x, frac_y,
      measurement.x(), measurement.y(), measurement.z(),
      values.col(0).x(), values.col(0).y(), values.col(0).z(),
      values.col(1).x(), values.col(1).y(), values.col(1).z(),
      values.col(2).x(), values.col(2).y(), values.col(2).z(),
      values.col(3).x(), values.col(3).y(), values.col(3).z(),
      values.col(4).x(), values.col(4).y(), values.col(4).z(),
      values.col(5).x(), values.col(5).y(), values.col(5).z(),
      values.col(6).x(), values.col(6).y(), values.col(6).z(),
      values.col(7).x(), values.col(7).y(), values.col(7).z(),
      values.col(8).x(), values.col(8).y(), values.col(8).z(),
      values.col(9).x(), values.col(9).y(), values.col(9).z(),
      values.col(10).x(), values.col(10).y(), values.col(10).z(),
      values.col(11).x(), values.col(11).y(), values.col(11).z(),
      values.col(12).x(), values.col(12).y(), values.col(12).z(),
      values.col(13).x(), values.col(13).y(), values.col(13).z(),
      values.col(14).x(), values.col(14).y(), values.col(14).z(),
      values.col(15).x(), values.col(15).y(), values.col(15).z(),
      residuals->data(),
      jacobian_wrt_directions.row(0).data(),
      jacobian_wrt_directions.row(1).data(),
      jacobian_wrt_directions.row(2).data());
  
  for (int i = 0; i < 16; ++ i) {
    int sequential_index = indices(i);
    (*local_indices)(2 * i + 0) = 2 * sequential_index + 0;
    (*local_indices)(2 * i + 1) = 2 * sequential_index + 1;
    
    const DirectionTangents& tangents = tangents_image.data()[sequential_index];
    
    Matrix<double, 3, 2> direction_wrt_localupdate;
    DirectionJacobianWrtLocalUpdate(
        tangents, &direction_wrt_localupdate);
    
    // output_wrt_localupdate = output_wrt_direction * direction_wrt_localupdate
    local_jacobian->block<3, 2>(0, 2 * i) = jacobian_wrt_directions.block<3, 3>(0, 3 * i) * direction_wrt_localupdate;
  }
}


struct CentralGenericBSplineDirectionCostFunction {
  CentralGenericBSplineDirectionCostFunction(
      const vector<Vec2d>* grid_points,
      const vector<Vec3d>* directions,
      int calibration_min_x,
      int calibration_min_y,
      int calibration_max_x,
      int calibration_max_y,
      int image_width,
      int image_height)
      : grid_points(grid_points),
        directions(directions),
        calibration_min_x(calibration_min_x),
        calibration_min_y(calibration_min_y),
        calibration_max_x(calibration_max_x),
        calibration_max_y(calibration_max_y),
        image_width(image_width),
        image_height(image_height) {}
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const DirectionGridStateWithLocalUpdates& state,
      Accumulator* accumulator) const {
    CHECK_EQ(grid_points->size(), directions->size());
    
    CentralGenericModel state_model(
        state.grid.width(), state.grid.height(),
        calibration_min_x, calibration_min_y,
        calibration_max_x, calibration_max_y,
        image_width, image_height);
    state_model.SetGrid(state.grid);
    
    // Cache tangents. TODO: Do this earlier! When done here, it can be needlessly re-computed several times.
    Image<DirectionTangents> tangents_image(state.grid.size());
    for (u32 y = 0; y < state.grid.height(); ++ y) {
      for (u32 x = 0; x < state.grid.width(); ++ x) {
        ComputeTangentsForDirectionOrLine(
            state.grid.at(x, y),
            &tangents_image(x, y));
      }
    }
    
    // Compute cost and Jacobian.
    for (usize i = 0; i < grid_points->size(); ++ i) {
      const Vec3d& measurement = directions->at(i);
      const Vec2d& grid_point = grid_points->at(i);
      
      if (compute_jacobians) {
        Matrix<double, 3, 1> residuals;
        Matrix<int, 2*16, 1> local_indices;
        Matrix<double, 3, 2*16, Eigen::RowMajor> local_jacobian;
        CentralGenericModel::ComputeUnprojectedDirectionResidualAndJacobianWrtGridUpdates(
            measurement, grid_point, state.grid,
            tangents_image,
            &residuals, &local_indices, &local_jacobian);
        
        accumulator->AddResidualWithJacobian(residuals(0), local_indices, local_jacobian.row(0));
        accumulator->AddResidualWithJacobian(residuals(1), local_indices, local_jacobian.row(1));
        accumulator->AddResidualWithJacobian(residuals(2), local_indices, local_jacobian.row(2));
      } else {
        Vec3d unprojection = state_model.UnprojectFromGrid(grid_point.x(), grid_point.y());
        accumulator->AddResidual(unprojection.x() - measurement.x());
        accumulator->AddResidual(unprojection.y() - measurement.y());
        accumulator->AddResidual(unprojection.z() - measurement.z());
      }
    }
  }
  
 private:
  const vector<Vec2d>* grid_points;
  const vector<Vec3d>* directions;
  int calibration_min_x;
  int calibration_min_y;
  int calibration_max_x;
  int calibration_max_y;
  int image_width;
  int image_height;
};


CentralGenericModel::CentralGenericModel()
    : CentralGridModel(CameraModel::Type::CentralGeneric, 0, 0, -1, -1, -1, -1, -1, -1) {}

CentralGenericModel::CentralGenericModel(
    int grid_resolution_x, int grid_resolution_y,
    int calibration_min_x, int calibration_min_y,
    int calibration_max_x, int calibration_max_y,
    int width, int height)
    : CentralGridModel(CameraModel::Type::CentralGeneric,
                       grid_resolution_x, grid_resolution_y,
                       calibration_min_x, calibration_min_y,
                       calibration_max_x, calibration_max_y,
                       width, height) {}

CentralGenericModel::CentralGenericModel(const CentralGenericModel& other)
    : CentralGridModel(other) {}

CentralGenericModel& CentralGenericModel::operator= (const CentralGenericModel& other) {
  m_width = other.m_width;
  m_height = other.m_height;
  m_calibration_min_x = other.m_calibration_min_x;
  m_calibration_min_y = other.m_calibration_min_y;
  m_calibration_max_x = other.m_calibration_max_x;
  m_calibration_max_y = other.m_calibration_max_y;
  m_grid = other.m_grid;
  return *this;
}

CameraModel* CentralGenericModel::duplicate() {
  return new CentralGenericModel(*this);
}

CentralGenericModel::~CentralGenericModel() {}

bool CentralGenericModel::FitToDenseModel(const Image<Vec3d>& dense_model, int subsample_step, int max_iteration_count) {
  // Initialize the grid points by assigning to them the closest valid pixel in the dense model.
  double scale_x = dense_model.width() / static_cast<double>(m_width);
  double scale_y = dense_model.height() / static_cast<double>(m_height);
  
  bool have_nan = false;
  
  for (u32 grid_y = 0; grid_y < m_grid.height(); ++ grid_y) {
    for (u32 grid_x = 0; grid_x < m_grid.width(); ++ grid_x) {
      Vec2i center_pixel =
          Vec2d(scale_x, scale_y).cwiseProduct(GridPointToPixelCornerConv(grid_x, grid_y)).cast<int>();
      
      if (center_pixel.x() < 0 ||
          center_pixel.y() < 0 ||
          center_pixel.x() >= dense_model.width() ||
          center_pixel.y() >= dense_model.height()) {
        m_grid(grid_x, grid_y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
        have_nan = true;
        continue;
      }
      
      if (!std::isnan(dense_model(center_pixel).x())) {
        m_grid(grid_x, grid_y) = dense_model(center_pixel);
      } else {
        // Search around the center pixel.
        bool found = false;
        for (int radius = 1; radius < 5 /*std::min(dense_model.width(), dense_model.height())*/; ++ radius) {
          Vec2i min_pixel = center_pixel - Vec2i::Constant(radius);
          Vec2i max_pixel = center_pixel + Vec2i::Constant(radius);
          
          // Top and bottom
          for (int x = std::max(0, min_pixel.x()); x <= std::min<int>(dense_model.width() - 1, max_pixel.x()); ++ x) {
            if (min_pixel.y() >= 0 && !std::isnan(dense_model(x, min_pixel.y()).x())) {
              m_grid(grid_x, grid_y) = dense_model(x, min_pixel.y());
              found = true;
              break;
            }
            
            if (max_pixel.y() < dense_model.height() && !std::isnan(dense_model(x, max_pixel.y()).x())) {
              m_grid(grid_x, grid_y) = dense_model(x, max_pixel.y());
              found = true;
              break;
            }
          }
          if (found) {
            break;
          }
          
          // Left and right
          for (int y = std::max(0, min_pixel.y()); y <= std::min<int>(dense_model.height() - 1, max_pixel.y()); ++ y) {
            if (min_pixel.x() >= 0 && !std::isnan(dense_model(min_pixel.x(), y).x())) {
              m_grid(grid_x, grid_y) = dense_model(min_pixel.x(), y);
              found = true;
              break;
            }
            
            if (max_pixel.x() < dense_model.width() && !std::isnan(dense_model(max_pixel.x(), y).x())) {
              m_grid(grid_x, grid_y) = dense_model(max_pixel.x(), y);
              found = true;
              break;
            }
          }
          if (found) {
            break;
          }
        }
        if (!found) {
          m_grid(grid_x, grid_y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
          have_nan = true;
        }
      }
    }
  }
  
  // Fill in non-initialized values by doing simple linear steps from their neighbors
  for (int iteration = 0;
       have_nan &&
       iteration < dense_model.width() + dense_model.height();
       ++ iteration) {
    have_nan = false;
    for (u32 grid_y = 0; grid_y < m_grid.height(); ++ grid_y) {
      for (u32 grid_x = 0; grid_x < m_grid.width(); ++ grid_x) {
        if (m_grid(grid_x, grid_y).hasNaN()) {
          // Try to fill the value based on the neighbor values (average if multiple directions are valid)
          Vec3d sum = Vec3d::Zero();
          int count = 0;
          
          int directions[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
          for (int d = 0; d < 4; ++ d) {
            int nx1 = grid_x + 1 * directions[d][0];
            int ny1 = grid_y + 1 * directions[d][1];
            int nx2 = grid_x + 2 * directions[d][0];
            int ny2 = grid_y + 2 * directions[d][1];
            if (nx2 < 0 || ny2 < 0 ||
                nx2 >= m_grid.width() || ny2 >= m_grid.height()) {
              continue;
            }
            
            Vec3d v1 = m_grid(nx1, ny1);
            Vec3d v2 = m_grid(nx2, ny2);
            if (v1.hasNaN() || v2.hasNaN()) {
              continue;
            }
            
            sum += v1 + (v1 - v2);
            ++ count;
          }
          
          if (count > 0) {
            m_grid(grid_x, grid_y) = sum.normalized();
          } else {
            have_nan = true;
          }
        }
      }
    }
  }
  
  if (have_nan) {
    // Was not able to initialize all grid values.
    return false;
  }
  
  // Optimize the grid points such that the unprojected direction at each valid
  // pixel in dense_model matches the direction from that model as good as
  // possible.
  double model_to_camera_x = static_cast<double>(m_width) / dense_model.width();
  double model_to_camera_y = static_cast<double>(m_height) / dense_model.height();
  
  vector<Vec2d> grid_points;
  vector<Vec3d> directions;
  int expected_count = (dense_model.width() / subsample_step + 1) * (dense_model.height() / subsample_step + 1);
  grid_points.reserve(expected_count);
  directions.reserve(expected_count);
  for (u32 y = m_calibration_min_y; y <= m_calibration_max_y; y += subsample_step) {
    for (u32 x = m_calibration_min_x; x <= m_calibration_max_x; x += subsample_step) {
      int dense_model_x = scale_x * x;
      int dense_model_y = scale_y * y;
      
      const Vec3d& measurement = dense_model.at(dense_model_x, dense_model_y);
      if (measurement.hasNaN()) {
        continue;
      }
      
      grid_points.push_back(PixelCornerConvToGridPoint(model_to_camera_x * (dense_model_x + 0.5f),
                                                       model_to_camera_y * (dense_model_y + 0.5f)));
      directions.push_back(measurement);
    }
  }
  
  DebugVerifyGridValues();
  FitToPixelDirectionsImpl(grid_points, directions, max_iteration_count);
  DebugVerifyGridValues();
  
  return true;
}

void CentralGenericModel::FitToPixelDirections(const vector<Vec2d>& pixels, const vector<Vec3d>& directions, int max_iteration_count) {
  vector<Vec2d> grid_points(pixels.size());
  for (usize i = 0; i < pixels.size(); ++ i) {
    grid_points[i] = PixelCornerConvToGridPoint(pixels[i].x(), pixels[i].y());
  }
  
  FitToPixelDirectionsImpl(grid_points, directions, max_iteration_count);
}

bool CentralGenericModel::ProjectDirectionWithInitialEstimate(const Vec3d& point_direction, Vec2d* result) const {
  // Levenberg-Marquardt optimization algorithm.
  constexpr double kEpsilon = 1e-12;
  const usize kMaxIterations = 100;
  
  double lambda = -1;
  for (usize i = 0; i < kMaxIterations; ++i) {
    Matrix<double, 3, 2> ddxy_dxy;
    Vec3d direction;
    CHECK(UnprojectWithJacobian(result->x(), result->y(), &direction, &ddxy_dxy))
        << "result->x(): " << result->x() << ", result->y(): " << result->y()
        << ", calibration_min_x: " << m_calibration_min_x
        << ", calibration_min_y: " << m_calibration_min_y
        << ", calibration_max_x: " << m_calibration_max_x
        << ", calibration_max_y: " << m_calibration_max_y;
    
    // (Non-squared) residuals.
    double dx = direction.x() - point_direction.x();
    double dy = direction.y() - point_direction.y();
    double dz = direction.z() - point_direction.z();
    
    double cost = dx * dx + dy * dy + dz * dz;
    
    // Accumulate H and b.
    double H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0) + ddxy_dxy(2, 0) * ddxy_dxy(2, 0);
    double H_1_0_and_0_1 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1) + ddxy_dxy(2, 0) * ddxy_dxy(2, 1);
    double H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1) + ddxy_dxy(2, 1) * ddxy_dxy(2, 1);
    double b_0 = dx * ddxy_dxy(0, 0) + dy * ddxy_dxy(1, 0) + dz * ddxy_dxy(2, 0);
    double b_1 = dx * ddxy_dxy(0, 1) + dy * ddxy_dxy(1, 1) + dz * ddxy_dxy(2, 1);
    
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
      Vec3d test_direction;
      if (Unproject(test_result.x(), test_result.y(), &test_direction)) {
        double test_dx = test_direction.x() - point_direction.x();
        double test_dy = test_direction.y() - point_direction.y();
        double test_dz = test_direction.z() - point_direction.z();
        
        test_cost = test_dx * test_dx + test_dy * test_dy + test_dz * test_dz;
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

bool CentralGenericModel::UnprojectWithJacobian(double x, double y, Vec3d* result, Matrix<double, 3, 2>* dresult_dxy) const {
  if (!IsInCalibratedArea(x, y)) {
    return false;
  }
  
  Vec2d grid_point = PixelCornerConvToGridPoint(x, y) + Vec2d(2, 2);
  
  int ix = std::floor(grid_point.x());
  int iy = std::floor(grid_point.y());
  
  double frac_x = grid_point.x() - (ix - 3);
  double frac_y = grid_point.y() - (iy - 3);
  
  Vec3d p[4][4];
  for (int y = 0; y < 4; ++ y) {
    for (int x = 0; x < 4; ++ x) {
      p[y][x] = m_grid(ix - 3 + x, iy - 3 + y);
    }
  }
  
  CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, p, result, dresult_dxy);
  (*dresult_dxy)(0, 0) = PixelScaleToGridScaleX((*dresult_dxy)(0, 0));
  (*dresult_dxy)(0, 1) = PixelScaleToGridScaleY((*dresult_dxy)(0, 1));
  (*dresult_dxy)(1, 0) = PixelScaleToGridScaleX((*dresult_dxy)(1, 0));
  (*dresult_dxy)(1, 1) = PixelScaleToGridScaleY((*dresult_dxy)(1, 1));
  (*dresult_dxy)(2, 0) = PixelScaleToGridScaleX((*dresult_dxy)(2, 0));
  (*dresult_dxy)(2, 1) = PixelScaleToGridScaleY((*dresult_dxy)(2, 1));
  return true;
}

void CentralGenericModel::FitToPixelDirectionsImpl(const vector<Vec2d>& grid_points, const vector<Vec3d>& directions, int max_iteration_count) {
  CentralGenericBSplineDirectionCostFunction cost_function(
      &grid_points, &directions, m_calibration_min_x, m_calibration_min_y,
      m_calibration_max_x, m_calibration_max_y, m_width, m_height);
  LMOptimizer<double> optimizer;
  DirectionGridStateWithLocalUpdates state(&m_grid);
  
  optimizer.Optimize(
      &state,
      cost_function,
      max_iteration_count,
      /*max_lm_attempts*/ 10,
      /*init_lambda*/ -1,
      /*init_lambda_factor*/ 0.001f,
      /*print_progress*/ false);
  
  m_grid = state.grid;
}

Mat3d CentralGenericModel::ChooseNiceCameraOrientation() {
  // NOTE: It is important that this function obtains the same result for
  //       different calibrations of the same camera (to make them more easily
  //       comparable). For example, one thing to avoid is to make the location
  //       of the pixel(s) used for estimating the forward direction dependent
  //       on the calibrated image area rectangle extents, as those can differ
  //       among different calibrations of the same camera.
  
  Vec3d forward;
  bool ok = Unproject(
      0.5f * width(),
      0.5f * height(),
      &forward);
  if (!ok) {
    forward = Vec3d(0, 0, 1);
  }
  Mat3d forward_rotation = Quaterniond::FromTwoVectors(forward, Vec3d(0, 0, 1)).toRotationMatrix();
  
  const int right_min_x = std::min<int>(width() - 1, width() / 2 + 11);
  const int right_max_x = width() - 1;
  const int right_min_y = std::max<int>(0, height() / 2 - 10);
  const int right_max_y = std::min<int>(height() - 1, height() / 2 + 10);
  Vec3d right_sum = Vec3d::Zero();
  u32 right_count = 0;
  for (u32 y = right_min_y; y <= right_max_y; ++ y) {
    for (u32 x = right_min_x; x <= right_max_x; ++ x) {
      Vec3d direction;
      if (!Unproject(x + 0.5f, y + 0.5f, &direction)) {
        continue;
      }
      
      right_sum += direction.cast<double>();
      right_count += 1;
    }
  }
  
  Mat3d right_rotation;
  if (right_count > 0) {
    Vec3d forward_rotated_right = forward_rotation.cast<double>() * (right_sum / right_count);
    
    // We want to rotate forward_rotated_right around the forward vector (0, 0, 1) such as to maximize its x value.
    double angle = atan2(-forward_rotated_right.y(), forward_rotated_right.x());  // TODO: Is the minus here correct?
    right_rotation = AngleAxisd(angle, Vec3d(0, 0, 1)).toRotationMatrix();
  } else {
    right_rotation = Mat3d::Identity();
  }
  
  Mat3d rotation = right_rotation * forward_rotation;
  Rotate(rotation);
  
  return rotation;
}

CUDACameraModel* CentralGenericModel::CreateCUDACameraModel() {
  CUDACentralGenericModel* result = new CUDACentralGenericModel();
  
  result->m_width = m_width;
  result->m_height = m_height;
  
  result->m_calibration_min_x = m_calibration_min_x;
  result->m_calibration_min_y = m_calibration_min_y;
  result->m_calibration_max_x = m_calibration_max_x;
  result->m_calibration_max_y = m_calibration_max_y;
  
  m_cuda_grid.reset(new CUDABuffer<float3>(m_grid.height(), m_grid.width()));
  Image<float3> cpu_cuda_grid(m_grid.size());
  for (int y = 0; y < cpu_cuda_grid.height(); ++ y) {
    for (int x = 0; x < cpu_cuda_grid.width(); ++ x) {
      const Vec3d& v = m_grid(x, y);
      cpu_cuda_grid(x, y) = make_float3(v.x(), v.y(), v.z());
    }
  }
  
  m_cuda_grid->UploadAsync(/*stream*/ 0, cpu_cuda_grid);
  result->m_grid = m_cuda_grid->ToCUDA();
  
  return result;
}

}

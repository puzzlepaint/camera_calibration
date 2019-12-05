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

#include "camera_calibration/models/central_thin_prism_fisheye.h"

#include <libvis/lm_optimizer.h>
#include <sophus/so3.hpp>

#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"

namespace vis {

CentralThinPrismFisheyeModel::CentralThinPrismFisheyeModel(const CentralThinPrismFisheyeModel& other)
    : ParametricCameraModel(other),
      m_use_equidistant_projection(other.m_use_equidistant_projection) {}

CameraModel* CentralThinPrismFisheyeModel::duplicate() {
  return new CentralThinPrismFisheyeModel(*this);
}

bool CentralThinPrismFisheyeModel::FitToDenseModelLinearly(const Image<Vec3d>& dense_model) {
  for (int i = 6; i < 12; ++ i) {
    m_parameters(i) = 0;
  }
  return FitSimpleParametricToDenseModelLinearly(
      dense_model,
      m_width, m_height,
      m_use_equidistant_projection,
      &m_parameters(0), &m_parameters(1),
      &m_parameters(2), &m_parameters(3),
      &m_parameters(4), &m_parameters(5));
}

bool CentralThinPrismFisheyeModel::Project(const Vec3d& local_point, Vec2d* result) const {
  if (local_point.z() <= 0) {
    return false;
  }
  Vec2d undistorted_point = Vec2d(local_point.x() / local_point.z(),
                                  local_point.y() / local_point.z());
  
  double r = undistorted_point.norm();
  
  double fisheye_x, fisheye_y;
  const double kEpsilon = static_cast<double>(1e-6);
  if (m_use_equidistant_projection && r > kEpsilon) {
    double theta_by_r = std::atan(r) / r;
    fisheye_x = theta_by_r * undistorted_point.coeff(0);
    fisheye_y = theta_by_r * undistorted_point.coeff(1);
  } else {
    fisheye_x = undistorted_point.coeff(0);
    fisheye_y = undistorted_point.coeff(1);
  }
  
  const double x2 = fisheye_x * fisheye_x;
  const double xy = fisheye_x * fisheye_y;
  const double y2 = fisheye_y * fisheye_y;
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double r8 = r6 * r2;
  
  const double& fx = m_parameters(0);
  const double& fy = m_parameters(1);
  const double& cx = m_parameters(2);
  const double& cy = m_parameters(3);
  const double& k1 = m_parameters(4);
  const double& k2 = m_parameters(5);
  const double& k3 = m_parameters(6);
  const double& k4 = m_parameters(7);
  const double& p1 = m_parameters(8);
  const double& p2 = m_parameters(9);
  const double& sx1 = m_parameters(10);
  const double& sy1 = m_parameters(11);
  
  const double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
  const double dx = static_cast<double>(2) * p1 * xy + p2 * (r2 + static_cast<double>(2) * x2) + sx1 * r2;
  const double dy = static_cast<double>(2) * p2 * xy + p1 * (r2 + static_cast<double>(2) * y2) + sy1 * r2;
  
  Vec2d distorted_point = Vec2d(fisheye_x + radial * fisheye_x + dx,
                                fisheye_y + radial * fisheye_y + dy);
  *result = Vec2d(fx * distorted_point.x() + cx,
                  fy * distorted_point.y() + cy);
  
  return result->x() >= 0 &&
         result->y() >= 0 &&
         result->x() < m_width &&
         result->y() < m_height;
}

bool CentralThinPrismFisheyeModel::ProjectInnerPartWithJacobian(const Vec2d& undistorted_point, Vec2d* result, Matrix<double, 2, 2>* jacobian) const {
  const double x2 = undistorted_point.coeff(0) * undistorted_point.coeff(0);
  const double xy = undistorted_point.coeff(0) * undistorted_point.coeff(1);
  const double y2 = undistorted_point.coeff(1) * undistorted_point.coeff(1);
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double r8 = r6 * r2;
  
  const double& k1 = m_parameters(4);
  const double& k2 = m_parameters(5);
  const double& k3 = m_parameters(6);
  const double& k4 = m_parameters(7);
  const double& p1 = m_parameters(8);
  const double& p2 = m_parameters(9);
  const double& sx1 = m_parameters(10);
  const double& sy1 = m_parameters(11);
  
  const double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
  const double dx = static_cast<double>(2) * p1 * xy + p2 * (r2 + static_cast<double>(2) * x2) + sx1 * r2;
  const double dy = static_cast<double>(2) * p2 * xy + p1 * (r2 + static_cast<double>(2) * y2) + sy1 * r2;
  
  const double nx = undistorted_point.coeff(0);
  const double ny = undistorted_point.coeff(1);
  const double nx_ny = nx * ny;
  
  // NOTE: Could factor out more terms here which might improve performance.
  const double term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*r2;
  (*jacobian)(0, 0) = 2*k1*x2 + 4*k2*x2*r2 + 6*k3*x2*r4 + 8*k4*x2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*r2 + 1;
  (*jacobian)(0, 1)= 2*sx1*ny + term1;
  (*jacobian)(1, 0) = 2*sy1*nx + term1;
  (*jacobian)(1, 1) = 2*k1*y2 + 4*k2*y2*r2 + 6*k3*y2*r4 + 8*k4*y2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*r2 + 1;
  
  *result = Vec2d(undistorted_point.coeff(0) + radial * undistorted_point.coeff(0) + dx,
                  undistorted_point.coeff(1) + radial * undistorted_point.coeff(1) + dy);
  return true;
}

bool CentralThinPrismFisheyeModel::Unproject(double x, double y, Vec3d* direction) const {
  if (!UnprojectWithGaussNewton(x, y, *this, direction)) {
    return false;
  }

  const double theta = sqrtf(direction->x() * direction->x() + direction->y() * direction->y());
  const double theta_cos_theta = theta * cosf(theta);
  const double kEpsilon = 1e-6;
  if (m_use_equidistant_projection && theta_cos_theta > kEpsilon) {
    const double scale = sinf(theta) / theta_cos_theta;
    direction->x() *= scale;
    direction->y() *= scale;
  }
  
  direction->normalize();
  return true;
}

}

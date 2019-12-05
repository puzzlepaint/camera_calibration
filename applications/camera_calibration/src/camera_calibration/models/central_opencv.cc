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

#include "camera_calibration/models/central_opencv.h"

#include <libvis/lm_optimizer.h>
#include <sophus/so3.hpp>

#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"

namespace vis {

CentralOpenCVModel::CentralOpenCVModel(const CentralOpenCVModel& other)
    : ParametricCameraModel(other)  {}

CameraModel* CentralOpenCVModel::duplicate() {
  return new CentralOpenCVModel(*this);
}

bool CentralOpenCVModel::FitToDenseModelLinearly(const Image<Vec3d>& dense_model) {
  for (int i = 6; i < 12; ++ i) {
    m_parameters(i) = 0;
  }
  
  return FitSimpleParametricToDenseModelLinearly(
      dense_model,
      m_width, m_height,
      /*use_equidistant_projection*/ false,
      &m_parameters(0), &m_parameters(1),
      &m_parameters(2), &m_parameters(3),
      &m_parameters(4), &m_parameters(5));
}

bool CentralOpenCVModel::Project(const Vec3d& local_point, Vec2d* result) const {
  if (local_point.z() <= 0) {
    return false;
  }
  Vec2d undistorted_point = Vec2d(local_point.x() / local_point.z(),
                                  local_point.y() / local_point.z());
  
  const double x2 = undistorted_point.coeff(0) * undistorted_point.coeff(0);
  const double xy = undistorted_point.coeff(0) * undistorted_point.coeff(1);
  const double y2 = undistorted_point.coeff(1) * undistorted_point.coeff(1);
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  
  const double& fx = m_parameters(0);
  const double& fy = m_parameters(1);
  const double& cx = m_parameters(2);
  const double& cy = m_parameters(3);
  const double& k1 = m_parameters(4);
  const double& k2 = m_parameters(5);
  const double& k3 = m_parameters(6);
  const double& k4 = m_parameters(7);
  const double& k5 = m_parameters(8);
  const double& k6 = m_parameters(9);
  const double& p1 = m_parameters(10);
  const double& p2 = m_parameters(11);
  
  const double radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6);
  const double dx = static_cast<double>(2) * p1 * xy + p2 * (r2 + static_cast<double>(2) * x2);
  const double dy = static_cast<double>(2) * p2 * xy + p1 * (r2 + static_cast<double>(2) * y2);
  
  Vec2d distorted_point = Vec2d(undistorted_point.coeff(0) * radial + dx,
                                undistorted_point.coeff(1) * radial + dy);
  *result = Vec2d(fx * distorted_point.x() + cx,
                  fy * distorted_point.y() + cy);
  
  return result->x() >= 0 &&
         result->y() >= 0 &&
         result->x() < m_width &&
         result->y() < m_height;
}

bool CentralOpenCVModel::ProjectInnerPartWithJacobian(const Vec2d& undistorted_point, Vec2d* result, Matrix<double, 2, 2>* jacobian) const {
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
  const double& k5 = m_parameters(8);
  const double& k6 = m_parameters(9);
  const double& p1 = m_parameters(10);
  const double& p2 = m_parameters(11);
  
  const double radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6);
  const double dx = static_cast<double>(2) * p1 * xy + p2 * (r2 + static_cast<double>(2) * x2);
  const double dy = static_cast<double>(2) * p2 * xy + p1 * (r2 + static_cast<double>(2) * y2);
  
  const double nx = undistorted_point.coeff(0);
  const double ny = undistorted_point.coeff(1);
  
  // To derive with octave/Matlab:
  // 
  // pkg load symbolic;
  // syms k1 k2 k3 k4 k5 k6 p1 p2 nx ny;
  // r = nx * nx + ny * ny;
  // radial = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) / (1 + k4 * r^2 + k5 * r^4 + k6 * r^6);
  // dx = 2 * p1 * nx * ny + p2 * (r^2 + 2 * nx^2);
  // dy = 2 * p2 * nx * ny + p1 * (r^2 + 2 * ny^2);
  // ccode(simplify(diff(simplify(nx * radial + dx), nx)))
  // ccode(simplify(diff(simplify(nx * radial + dx), ny)))
  // ccode(simplify(diff(simplify(ny * radial + dy), nx)))
  // ccode(simplify(diff(simplify(ny * radial + dy), ny)))
  
  // TODO: Simplify this more
  (*jacobian)(0, 0) = (-4*nx*r2*(nx*(k1*r4 + k2*r6 + k3*r8 + 1) + (2*nx*ny*p1 + p2*(2*x2 + r4))*(k4*r4 + k5*r6+ k6*r8 + 1))*(k4 + 2*k5*r4 + 3*k6*r6) + (k4*r4 + k5*r6 + k6*r8 + 1)*(k1*r4 + k2*r6 + k3*r8 + 4*x2*r2*(k1 + 2*k2*r4 + 3*k3*r6) + 4*nx*r2*(2*nx*ny*p1 + p2*(2*x2 + r4))*(k4 + 2*k5*r4 + 3*k6*r6) + 2*(2*nx*p2*(r2 + 1) + ny*p1)*(k4*r4 +k5*r6 + k6*r8 + 1) + 1))/pow(k4*r4 + k5*r6 + k6*r8 + 1, 2);
  (*jacobian)(0, 1)= 2*(-2*ny*r2*(nx*(k1*r4 + k2*r6 + k3*r8 + 1) + (2*nx*ny*p1 + p2*(2*x2 + r4))*(k4*r4 + k5*r6 + k6*r8 + 1))*(k4 + 2*k5*r4 + 3*k6*r6) + (2*nx*ny*r2*(k1 + 2*k2*r4 + 3*k3*r6) + 2*ny*r2*(2*nx*ny*p1 + p2*(2*x2 + r4))*(k4 + 2*k5*r4 + 3*k6*r6) + (nx*p1 + 2*ny*p2*r2)*(k4*r4 + k5*r6 + k6*r8 + 1))*(k4*r4 + k5*r6 + k6*r8 + 1))/pow(k4*r4 +k5*r6 + k6*r8 + 1, 2);
  (*jacobian)(1, 0) = 2*(-2*nx*r2*(ny*(k1*r4 + k2*r6 + k3*r8 + 1) + (2*nx*ny*p2 + p1*(2*y2 + r4))*(k4*r4 + k5*r6 + k6*r8 + 1))*(k4 + 2*k5*r4 + 3*k6*r6) + (2*nx*ny*r2*(k1 + 2*k2*r4 + 3*k3*r6) + 2*nx*r2*(2*nx*ny*p2 + p1*(2*y2 + r4))*(k4 + 2*k5*r4 + 3*k6*r6) + (2*nx*p1*r2 + ny*p2)*(k4*r4 + k5*r6 + k6*r8 + 1))*(k4*r4 + k5*r6 + k6*r8 + 1))/pow(k4*r4 +k5*r6 + k6*r8 + 1, 2);
  (*jacobian)(1, 1) = (-4*ny*r2*(ny*(k1*r4 + k2*r6 + k3*r8 + 1) + (2*nx*ny*p2 + p1*(2*y2 + r4))*(k4*r4 + k5*r6+ k6*r8 + 1))*(k4 + 2*k5*r4 + 3*k6*r6) + (k4*r4 + k5*r6 + k6*r8 + 1)*(k1*r4 + k2*r6 + k3*r8 + 4*y2*r2*(k1 + 2*k2*r4 + 3*k3*r6) + 4*ny*r2*(2*nx*ny*p2 + p1*(2*y2 + r4))*(k4 + 2*k5*r4 + 3*k6*r6) + 2*(nx*p2 + 2*ny*p1*(r2 + 1))*(k4*r4 +k5*r6 + k6*r8 + 1) + 1))/pow(k4*r4 + k5*r6 + k6*r8 + 1, 2);
  
  *result = Vec2d(undistorted_point.coeff(0) * radial + dx,
                  undistorted_point.coeff(1) * radial + dy);
  return true;
}

bool CentralOpenCVModel::Unproject(double x, double y, Vec3d* direction) const {
  if (!UnprojectWithGaussNewton(x, y, *this, direction)) {
    return false;
  }
  direction->normalize();
  return true;
}

}

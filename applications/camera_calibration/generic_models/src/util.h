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

#include <vector>

template <typename T>
T EvalUniformCubicBSpline(const T& a, const T& b, const T& c, const T& d, double x) {
  // x must be in [3, 4[.
  
  // i == 3
  double x_for_d = x - 3;
  double d_factor = 1./6. * x_for_d * x_for_d * x_for_d;
  
  // i == 2
  double c_factor = -1./2.*x*x*x + 5*x*x - 16*x + 50./3.;
  
  // i == 1
  double b_factor = 1./2.*x*x*x - 11./2.*x*x + (39./2.)*x - 131./6.;
  
  // i == 0
  double a_factor =  -1./6. * (x - 4) * (x - 4) * (x - 4);
  
  return a_factor * a + b_factor * b + c_factor * c + d_factor * d;
}

template <typename T>
T EvalUniformCubicBSplineSurface(const std::vector<T>& control_points, int grid_width, double x, double y) {
  // Make it such that the "surrounding" control points are used for interpolation, not the previous ones
  x += 2;
  y += 2;
  
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  
  // The coefficients for x-interpolation are always the same, take advantage of that.
  double frac_x = x - (ix - 3);
  
  // i == 3
  double x_for_d = frac_x - 3;
  double d_factor = 1./6. * x_for_d * x_for_d * x_for_d;
  
  // i == 2
  double c_factor = -1./2.*frac_x*frac_x*frac_x + 5*frac_x*frac_x - 16*frac_x + 50./3.;
  
  // i == 1
  double b_factor = 1./2.*frac_x*frac_x*frac_x - 11./2.*frac_x*frac_x + (39./2.)*frac_x - 131./6.;
  
  // i == 0
  double a_factor =  -1./6. * (frac_x - 4) * (frac_x - 4) * (frac_x - 4);
  
  
  int ky = iy - 3;
  T a = a_factor * control_points[(ix - 3) + ky * grid_width] + b_factor * control_points[(ix - 2) + ky * grid_width] + c_factor * control_points[(ix - 1) + ky * grid_width] + d_factor * control_points[(ix - 0) + ky * grid_width];
  
  ky = iy - 2;
  T b = a_factor * control_points[(ix - 3) + ky * grid_width] + b_factor * control_points[(ix - 2) + ky * grid_width] + c_factor * control_points[(ix - 1) + ky * grid_width] + d_factor * control_points[(ix - 0) + ky * grid_width];
  
  ky = iy - 1;
  T c = a_factor * control_points[(ix - 3) + ky * grid_width] + b_factor * control_points[(ix - 2) + ky * grid_width] + c_factor * control_points[(ix - 1) + ky * grid_width] + d_factor * control_points[(ix - 0) + ky * grid_width];
  
  ky = iy - 0;
  T d = a_factor * control_points[(ix - 3) + ky * grid_width] + b_factor * control_points[(ix - 2) + ky * grid_width] + c_factor * control_points[(ix - 1) + ky * grid_width] + d_factor * control_points[(ix - 0) + ky * grid_width];
  
  return EvalUniformCubicBSpline(a, b, c, d, y - (iy - 3));
}

template <typename T>
void EvalTwoUniformCubicBSplineSurfaces(const std::vector<T>& control_points_a, const std::vector<T>& control_points_b, int grid_width, double x, double y, T* result_a, T* result_b) {
  // Make it such that the "surrounding" control points are used for interpolation, not the previous ones
  x += 2;
  y += 2;
  
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  
  // The coefficients for x-interpolation are always the same, take advantage of that.
  double frac_x = x - (ix - 3);
  
  // i == 3
  double x_for_d = frac_x - 3;
  double d_factor = 1./6. * x_for_d * x_for_d * x_for_d;
  
  // i == 2
  double c_factor = -1./2.*frac_x*frac_x*frac_x + 5*frac_x*frac_x - 16*frac_x + 50./3.;
  
  // i == 1
  double b_factor = 1./2.*frac_x*frac_x*frac_x - 11./2.*frac_x*frac_x + (39./2.)*frac_x - 131./6.;
  
  // i == 0
  double a_factor =  -1./6. * (frac_x - 4) * (frac_x - 4) * (frac_x - 4);
  
  
  int ky = iy - 3;
  T a_a = a_factor * control_points_a[ix - 3 + ky * grid_width] + b_factor * control_points_a[ix - 2 + ky * grid_width] + c_factor * control_points_a[ix - 1 + ky * grid_width] + d_factor * control_points_a[ix - 0 + ky * grid_width];
  T a_b = a_factor * control_points_b[ix - 3 + ky * grid_width] + b_factor * control_points_b[ix - 2 + ky * grid_width] + c_factor * control_points_b[ix - 1 + ky * grid_width] + d_factor * control_points_b[ix - 0 + ky * grid_width];
  
  ky = iy - 2;
  T b_a = a_factor * control_points_a[ix - 3 + ky * grid_width] + b_factor * control_points_a[ix - 2 + ky * grid_width] + c_factor * control_points_a[ix - 1 + ky * grid_width] + d_factor * control_points_a[ix - 0 + ky * grid_width];
  T b_b = a_factor * control_points_b[ix - 3 + ky * grid_width] + b_factor * control_points_b[ix - 2 + ky * grid_width] + c_factor * control_points_b[ix - 1 + ky * grid_width] + d_factor * control_points_b[ix - 0 + ky * grid_width];
  
  ky = iy - 1;
  T c_a = a_factor * control_points_a[ix - 3 + ky * grid_width] + b_factor * control_points_a[ix - 2 + ky * grid_width] + c_factor * control_points_a[ix - 1 + ky * grid_width] + d_factor * control_points_a[ix - 0 + ky * grid_width];
  T c_b = a_factor * control_points_b[ix - 3 + ky * grid_width] + b_factor * control_points_b[ix - 2 + ky * grid_width] + c_factor * control_points_b[ix - 1 + ky * grid_width] + d_factor * control_points_b[ix - 0 + ky * grid_width];
  
  ky = iy - 0;
  T d_a = a_factor * control_points_a[ix - 3 + ky * grid_width] + b_factor * control_points_a[ix - 2 + ky * grid_width] + c_factor * control_points_a[ix - 1 + ky * grid_width] + d_factor * control_points_a[ix - 0 + ky * grid_width];
  T d_b = a_factor * control_points_b[ix - 3 + ky * grid_width] + b_factor * control_points_b[ix - 2 + ky * grid_width] + c_factor * control_points_b[ix - 1 + ky * grid_width] + d_factor * control_points_b[ix - 0 + ky * grid_width];
  
  // Take advantage of the equal y-factors as well
  double frac_y = y - (iy - 3);
  
  // i == 3
  double y_for_d = frac_y - 3;
  d_factor = 1./6. * y_for_d * y_for_d * y_for_d;
  
  // i == 2
  c_factor = -1./2.*frac_y*frac_y*frac_y + 5*frac_y*frac_y - 16*frac_y + 50./3.;
  
  // i == 1
  b_factor = 1./2.*frac_y*frac_y*frac_y - 11./2.*frac_y*frac_y + (39./2.)*frac_y - 131./6.;
  
  // i == 0
  a_factor =  -1./6. * (frac_y - 4) * (frac_y - 4) * (frac_y - 4);
  
  *result_a = a_factor * a_a + b_factor * b_a + c_factor * c_a + d_factor * d_a;
  *result_b = a_factor * a_b + b_factor * b_b + c_factor * c_b + d_factor * d_b;
}

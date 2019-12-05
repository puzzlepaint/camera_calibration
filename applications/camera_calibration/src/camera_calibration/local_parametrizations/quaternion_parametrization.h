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
#include <libvis/libvis.h>

namespace vis {

// These functions provide a local 3-dimensional Euclidean parametrization for
// quaternions which can be used to optimize them.

template <typename T, typename Derived>
Quaternion<T> ApplyLocalUpdateToQuaternion(
    const Quaternion<T>& q,
    const MatrixBase<Derived>& update) {
  const float norm_update =
      sqrt(update[0] * update[0] +
           update[1] * update[1] +
           update[2] * update[2]);
  if (norm_update == 0) {
    return q;
  }
  
  const float sin_update_by_update = sin(norm_update) / norm_update;
  
  // Note, in the constructor w is first.
  Quaternion<T> update_q(
      cos(norm_update),
      sin_update_by_update * update[0],
      sin_update_by_update * update[1],
      sin_update_by_update * update[2]);
  return update_q * q;
}

/// Derived in derive_jacobians.py.
template <typename T>
inline void QuaternionJacobianWrtLocalUpdate(
    const Quaternion<T>& q,
    Matrix<T, 4, 3>* jacobian) {
  (*jacobian) <<
      -q.x(), -q.y(), -q.z(),
       q.w(),  q.z(), -q.y(),
      -q.z(),  q.w(),  q.x(),
       q.y(), -q.x(),  q.w();
}

template <typename Scalar>
inline void ComputeRotatedPointJacobianWrtRotationUpdate(
    const Quaterniond& q,
    Scalar px, Scalar py, Scalar pz,
    Matrix<double, 3, 3>* rotated_point_wrt_update) {
  const Scalar term0 = 2*q.y();
  const Scalar term1 = pz*term0;
  const Scalar term2 = q.z()*py;
  const Scalar term3 = 2*term2;
  const Scalar term4 = py*term0;
  const Scalar term5 = 2*q.z()*pz;
  const Scalar term6 = 2*q.w();
  const Scalar term7 = pz*term6;
  const Scalar term8 = 2*q.x();
  const Scalar term9 = py*term8;
  const Scalar term10 = 4*q.y();
  const Scalar term11 = py*term6;
  const Scalar term12 = pz*term8;
  const Scalar term13 = q.z()*px;
  const Scalar term20 = 2*term13;
  const Scalar term21 = 4*q.x();
  const Scalar term22 = px*term0;
  const Scalar term23 = px*term8;
  const Scalar term24 = px*term6;
  
  Matrix<double, 3, 4> rotated_point_wrt_quaternion_values;
  rotated_point_wrt_quaternion_values(0, 0) = term1 - term3;
  rotated_point_wrt_quaternion_values(0, 1) = term4 + term5;
  rotated_point_wrt_quaternion_values(0, 2) = -px*term10 + term7 + term9;
  rotated_point_wrt_quaternion_values(0, 3) = -term11 + term12 - 4*term13;
  rotated_point_wrt_quaternion_values(1, 0) = -term12 + term20;
  rotated_point_wrt_quaternion_values(1, 1) = -py*term21 + term22 - term7;
  rotated_point_wrt_quaternion_values(1, 2) = term23 + term5;
  rotated_point_wrt_quaternion_values(1, 3) = term1 - 4*term2 + term24;
  rotated_point_wrt_quaternion_values(2, 0) = -term22 + term9;
  rotated_point_wrt_quaternion_values(2, 1) = -pz*term21 + term11 + term20;
  rotated_point_wrt_quaternion_values(2, 2) = -pz*term10 - term24 + term3;
  rotated_point_wrt_quaternion_values(2, 3) = term23 + term4;
  
  Matrix<double, 4, 3> quaternion_wrt_update;
  QuaternionJacobianWrtLocalUpdate(q, &quaternion_wrt_update);
  
  *rotated_point_wrt_update = rotated_point_wrt_quaternion_values * quaternion_wrt_update;
}

}

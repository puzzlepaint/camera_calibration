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


#include "camera_calibration/local_parametrizations/line_parametrization.h"

namespace vis {

// These functions provide a local 2-dimensional Euclidean parametrization for
// directions in 3D space (defined on the unit sphere) which can be used to
// optimize such directions using 2-dimensional updates.

typedef LineTangents DirectionTangents;  // We use the same struct for direction and line tangents // TODO: Would this definition be cleaner as a union {}?

template <typename Scalar>
inline void ApplyLocalUpdateToDirection(
    Matrix<Scalar, 3, 1>* direction,
    const DirectionTangents& tangents,
    double offset1,
    double offset2) {
  // Projection onto the sphere in the direction towards the origin.
  // NOTE: We could theoretically divide by sqrt(1 + offset1 * offset1 + offset2 * offset2) to normalize here,
  //       but this accumulates error. So we do a full renormalization instead.
  *direction = (*direction + offset1 * tangents.t1 + offset2 * tangents.t2).normalized();
}

inline void DirectionJacobianWrtLocalUpdate(
    const DirectionTangents& tangents,
    Matrix<double, 3, 2>* jacobian) {
  // To compute this in octave or Matlab (for the top-left entry; it is however
  // also easy to see directly):
  // 
  // pkg load symbolic  % if using octave
  // syms dx dy dz t1x t1y t1z t2x t2y t2z
  // syms o1 o2
  // subs(diff((dx + o1 * t1x + o2 * t2x) / sqrt(1 + o1*o1 + o2*o2), o1), [o1 o2], [0 0])
  *jacobian << tangents.t1.x(), tangents.t2.x(),
               tangents.t1.y(), tangents.t2.y(),
               tangents.t1.z(), tangents.t2.z();
}

inline void ConvertDirectionToLocalUpdate(
    const Vec3d& base_direction,
    const Vec3d& target_direction,
    const DirectionTangents& tangents,
    double* offset1,
    double* offset2) {
  // Factor that target_direction needs to be multiplied with to get its
  // intersection with the tangent plane that goes through the point
  // base_direction.
  double factor = 1 / base_direction.dot(target_direction);
  
  // Offset of that intersection point in the plane from base_direction.
  Vec3d offset = (factor * target_direction) - base_direction;
  
  *offset1 = tangents.t1.dot(offset);
  *offset2 = tangents.t2.dot(offset);
}

/// base_direction must be normalized.
inline void LocalUpdateJacobianWrtDirection(
    const Vec3d& base_direction,
    const DirectionTangents& tangents,
    Matrix<double, 2, 3>* jacobian) {
  // Derivation can be done with derive_jacobians.py.
  // Additional simplification was applied, knowing that
  // base_direction.squaredNorm() == 1.
  const double term5 = base_direction.x() * base_direction.y();
  const double term6 = base_direction.x() * base_direction.z();
  const double term9 = base_direction.y() * base_direction.z();
  const double term8 = 1 - (base_direction.x() * base_direction.x());
  const double term10 = 1 - (base_direction.y() * base_direction.y());
  const double term11 = 1 - (base_direction.z() * base_direction.z());
  
  const auto& t1 = tangents.t1;
  const auto& t2 = tangents.t2;
  
  *jacobian <<
      t1.x()*term8 - t1.y()*term5 - t1.z()*term6,
      -t1.x()*term5 + t1.y()*term10 - t1.z()*term9,
      -t1.x()*term6 - t1.y()*term9 + t1.z()*term11,
      t2.x()*term8 - t2.y()*term5 - t2.z()*term6,
      -t2.x()*term5 + t2.y()*term10 - t2.z()*term9,
      -t2.x()*term6 - t2.y()*term9 + t2.z()*term11;
}

}

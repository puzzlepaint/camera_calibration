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

#include <Eigen/Geometry>
#include <libvis/eigen.h>
#include <libvis/libvis.h>

namespace vis {

// These functions provide a local 5-dimensional Euclidean parametrization for
// lines in 3D space (defined by a point and a unit direction vector) which can
// be used to optimize such lines. Normally lines would only require 4
// dimensions, but we represent them by an origin and a direction and account
// for changes of the origin in the line direction. While this does not affect
// the line itself, it may affect interpolated lines within a grid of such lines.
// There should ideally be some regularization on that direction if optimizing
// lines to prevent it from remaining unconstrained.

struct LineTangents {
  Vec3d t1;
  Vec3d t2;
};

/// Computes tangent vectors to the direction which are used to define the
/// local parametrization. Should ideally be computed once for a direction and
/// be cached to use it for the other functions for best performance.
template <typename Derived>
inline void ComputeTangentsForDirectionOrLine(
    const MatrixBase<Derived>& direction,
    LineTangents* tangents) {
  tangents->t1 = direction.cross((fabs(direction.x()) > 0.9f) ? Vec3d(0, 1, 0) : Vec3d(1, 0, 0)).normalized();
  tangents->t2 = direction.cross(tangents->t1);  // is already normalized
}

template <typename Derived>
inline void TangentsJacobianWrtLineDirection(
    const MatrixBase<Derived>& direction,
    Matrix<double, 6, 3>* jacobian) {
  if (fabs(direction.x()) > 0.9f) {
    const double term0 = direction.x() * direction.x();
    const double term1 = direction.z() * direction.z();
    const double term2 = term0 + term1;
    const double term7 = 1. / sqrt(term2);
    const double term3 = term7 * term7 * term7;
    const double term4 = direction.x()*direction.z()*term3;
    const double term5 = term0*term3;
    const double term6 = term1*term3;
    const double term8 = direction.x()*term7;
    const double term9 = -direction.y()*term4;
    const double term10 = direction.z()*term7;
    
    *jacobian << term4, 0, -term5,
                 0, 0, 0,
                 term6, 0, -term4,
                 direction.y()*term6, term8, term9,
                 -term8, 0, -term10,
                 term9, term10, direction.y()*term5;
  } else {
    const double term0 = direction.y() * direction.y();
    const double term1 = direction.z() * direction.z();
    const double term2 = term0 + term1;
    const double term7 = 1. / sqrt(term2);
    const double term3 = term7 * term7 * term7;
    const double term4 = direction.y()*direction.z()*term3;
    const double term5 = term0*term3;
    const double term6 = term1*term3;
    const double term8 = direction.y()*term7;
    const double term9 = direction.z()*term7;
    const double term10 = -direction.x()*term4;
    
    *jacobian << 0, 0, 0,
                 0, -term4, term5,
                 0, -term6, term4,
                 0, -term8, -term9,
                 term8, direction.x()*term6, term10,
                 term9, term10, direction.x()*term5;
  }
}

inline void ApplyLocalUpdateToLine(
    ParametrizedLine<double, 3>* line,
    const LineTangents& tangents,
    double offset1,
    double offset2,
    double offset3,
    double offset4,
    double origin_offset_along_line) {
  line->origin() = line->origin() + offset3 * tangents.t1 + offset4 * tangents.t2 + origin_offset_along_line * line->direction();
  // Projection onto the sphere in the direction towards the origin.
  // NOTE: We could theoretically divide by sqrt(1 + offset1 * offset1 + offset2 * offset2) to normalize here,
  //       but this would accumulate error. So we do a full renormalization instead.
  line->direction() = (line->direction() + offset1 * tangents.t1 + offset2 * tangents.t2).normalized();
}

/// The first three dimensions are the direction, the last three are the origin.
inline void LineJacobianWrtLocalUpdate(
    const ParametrizedLine<double, 3>& line,
    const LineTangents& tangents,
    Matrix<double, 6, 5>* jacobian) {
  const Vec3d& t1 = tangents.t1;
  const Vec3d& t2 = tangents.t2;
  *jacobian << t1.x(), t2.x(),      0,      0,                    0,
               t1.y(), t2.y(),      0,      0,                    0,
               t1.z(), t2.z(),      0,      0,                    0,
                    0,      0, t1.x(), t2.x(), line.direction().x(),
                    0,      0, t1.y(), t2.y(), line.direction().y(),
                    0,      0, t1.z(), t2.z(), line.direction().z();
}

}

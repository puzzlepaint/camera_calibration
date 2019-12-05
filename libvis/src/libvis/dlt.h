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

#include "libvis/eigen.h"
#include "libvis/statistics.h"

namespace vis {

// Implements the DLT algorithm to compute a homography H from at least four pairs
// of points (a[i], b[i]), such that: b = H * a. The number of points is given by n.
// Notice that DLT() can be numerically unstable and thus in practice, one should
// usually use NormalizedDLT() instead. To multiply a 2D point with the resulting
// matrix using homogeneous coordinates, do (in case T = float):
// Vec3f(H * point.homogeneous()).hnormalized()
template <typename T>
Matrix<T, 3, 3> DLT(const Matrix<T, 2, 1>* a, const Matrix<T, 2, 1>* b, int n) {
  // TODO: Should assert(n >= 4)
  
  Matrix<T, Eigen::Dynamic, 9> A;
  A.resize(2 * n, Eigen::NoChange);
  
  for (int i = 0; i < n; ++ i) {
    A.row(2 * i + 0) << 0, 0, 0, -a[i].x(), -a[i].y(), -1, b[i].y() * a[i].x(), b[i].y() * a[i].y(), b[i].y();
    A.row(2 * i + 1) << a[i].x(), a[i].y(), 1, 0, 0, 0, -b[i].x() * a[i].x(), -b[i].x() * a[i].y(), -b[i].x();
  }
  
  JacobiSVD<Matrix<T, Eigen::Dynamic, 9>> svd(A, ComputeFullV);
  const auto& h = svd.matrixV().template rightCols<1>();
  return (Matrix<T, 3, 3>() << h(0), h(1), h(2),
                               h(3), h(4), h(5),
                               h(6), h(7), h(8)).finished();
}

// Version of DLT() which normalizes the input points.
template <typename T>
Matrix<T, 3, 3> NormalizedDLT(const Matrix<T, 2, 1>* a, const Matrix<T, 2, 1>* b, int n) {
  // TODO: Should assert(n >= 4)
  
  SinglePassMeanAndVariance<T> mv_a_x;
  SinglePassMeanAndVariance<T> mv_a_y;
  SinglePassMeanAndVariance<T> mv_b_x;
  SinglePassMeanAndVariance<T> mv_b_y;
  
  for (int i = 0; i < n; ++ i) {
    mv_a_x.AddData(a[i].x());
    mv_a_y.AddData(a[i].y());
    mv_b_x.AddData(b[i].x());
    mv_b_y.AddData(b[i].y());
  }
  
  Matrix<T, 2, 1> a_mean = Matrix<T, 2, 1>(mv_a_x.mean(), mv_a_y.mean());
  Matrix<T, 2, 1> b_mean = Matrix<T, 2, 1>(mv_b_x.mean(), mv_b_y.mean());
  Matrix<T, 2, 1> a_factors = Matrix<T, 2, 1>(mv_a_x.ComputeVariance(), mv_a_y.ComputeVariance()).cwiseSqrt().cwiseInverse();
  Matrix<T, 2, 1> b_factors = Matrix<T, 2, 1>(mv_b_x.ComputeVariance(), mv_b_y.ComputeVariance()).cwiseSqrt().cwiseInverse();
  
  Matrix<T, Eigen::Dynamic, 9> A;
  A.resize(2 * n, Eigen::NoChange);
  
  for (int i = 0; i < n; ++ i) {
    Matrix<T, 2, 1> na = (a[i] - a_mean).cwiseProduct(a_factors);
    Matrix<T, 2, 1> nb = (b[i] - b_mean).cwiseProduct(b_factors);
    A.row(2 * i + 0) << 0, 0, 0, -na.x(), -na.y(), -1, nb.y() * na.x(), nb.y() * na.y(), nb.y();
    A.row(2 * i + 1) << na.x(), na.y(), 1, 0, 0, 0, -nb.x() * na.x(), -nb.x() * na.y(), -nb.x();
  }
  
  JacobiSVD<Matrix<T, Eigen::Dynamic, 9>> svd(A, ComputeFullV);
  const auto& h = svd.matrixV().template rightCols<1>();
  
  Matrix<T, 3, 3> a_norm;
  a_norm << a_factors.x(), 0, -a_factors.x() * a_mean.x(),
            0, a_factors.y(), -a_factors.y() * a_mean.y(),
            0, 0, 1;
  
  Matrix<T, 3, 3> b_denorm;
  b_denorm << 1 / b_factors.x(), 0, b_mean.x(),
              0, 1 / b_factors.y(), b_mean.y(),
              0, 0, 1;
  
  return b_denorm * (Matrix<T, 3, 3>() << h(0), h(1), h(2),
                                          h(3), h(4), h(5),
                                          h(6), h(7), h(8)).finished() * a_norm;
}

// TODO: Implement non-linear refinement of homography according to the geometrical error
//       (instead of the algebraical error used by DLT).
//       --> rename this header to homography_estimation.h or similar?

}

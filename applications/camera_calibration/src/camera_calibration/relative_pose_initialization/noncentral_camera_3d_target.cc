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

#include "camera_calibration/relative_pose_initialization/algorithms.h"

namespace vis {

bool NonCentralCamera3DCalibrationObjectRelativePose(const Point3fCloud clouds[3], SE3d cloud2_tr_cloud[2]) {
  usize num_points = clouds[0].size();
  // * 2 constraints per point triple
  // * 53 unknowns in system of homogeneous equations (... = 0)
  // * solution only possible up to scale
  // --> 26 point triples required
  if (num_points < 26) {
    return false;
  }
  
  // Assemble coefficients matrices.
  Eigen::Matrix<double, Eigen::Dynamic, 7 + 2 * 23> C;
  C.resize(2 * num_points, Eigen::NoChange);
  
  for (usize point = 0; point < num_points; ++ point) {
    // Q corresponds to clouds[2][point] (the cloud with fixed pose).
    const Vec3f& Q = clouds[2][point].position();
    // Q' corresponds to clouds[0][point].
    const Vec3f& Qp = clouds[0][point].position();
    // Q'' corresponds to clouds[1][point].
    const Vec3f& Qpp = clouds[1][point].position();
    
    // Row 1
    int r = 2 * point + 0;
    
    C(r,  1 - 1) = Q.x() * Qp.x() * 1;
    C(r,  2 - 1) = Q.x() * Qp.y() * 1;
    C(r,  3 - 1) = Q.x() * Qp.z() * 1;
    C(r,  4 - 1) = Q.x() * 1 * Qpp.x();
    C(r,  5 - 1) = Q.x() * 1 * Qpp.y();
    C(r,  6 - 1) = Q.x() * 1 * Qpp.z();
    C(r,  7 - 1) = Q.x() * 1 * 1;
    
    C(r,  8 - 1) = Q.z() * Qp.x() * 1;
    C(r,  9 - 1) = Q.z() * Qp.y() * 1;
    C(r, 10 - 1) = Q.z() * Qp.z() * 1;
    C(r, 11 - 1) = Q.z() * 1 * Qpp.x();
    C(r, 12 - 1) = Q.z() * 1 * Qpp.y();
    C(r, 13 - 1) = Q.z() * 1 * Qpp.z();
    C(r, 14 - 1) = Q.z() * 1 * 1;
    C(r, 15 - 1) = 1 * Qp.x() * Qpp.x();
    C(r, 16 - 1) = 1 * Qp.x() * Qpp.y();
    C(r, 17 - 1) = 1 * Qp.x() * Qpp.z();
    C(r, 18 - 1) = 1 * Qp.x() * 1;
    C(r, 19 - 1) = 1 * Qp.y() * Qpp.x();
    C(r, 20 - 1) = 1 * Qp.y() * Qpp.y();
    C(r, 21 - 1) = 1 * Qp.y() * Qpp.z();
    C(r, 22 - 1) = 1 * Qp.y() * 1;
    C(r, 23 - 1) = 1 * Qp.z() * Qpp.x();
    C(r, 24 - 1) = 1 * Qp.z() * Qpp.y();
    C(r, 25 - 1) = 1 * Qp.z() * Qpp.z();
    C(r, 26 - 1) = 1 * Qp.z() * 1;
    C(r, 27 - 1) = 1 * 1 * Qpp.x();
    C(r, 28 - 1) = 1 * 1 * Qpp.y();
    C(r, 29 - 1) = 1 * 1 * Qpp.z();
    C(r, 30 - 1) = 1 * 1 * 1;
    
    for (int c = 31 - 1; c < C.cols(); ++ c) {
      C(r, c) = 0;
    }
    
    // Row 2
    r = 2 * point + 1;
    
    C(r,  1 - 1) = Q.y() * Qp.x() * 1;
    C(r,  2 - 1) = Q.y() * Qp.y() * 1;
    C(r,  3 - 1) = Q.y() * Qp.z() * 1;
    C(r,  4 - 1) = Q.y() * 1 * Qpp.x();
    C(r,  5 - 1) = Q.y() * 1 * Qpp.y();
    C(r,  6 - 1) = Q.y() * 1 * Qpp.z();
    C(r,  7 - 1) = Q.y() * 1 * 1;
    
    for (int c = 8 - 1; c < 31 - 1; ++ c) {
      C(r, c) = 0;
    }
    
    C(r, 31 - 1) = Q.z() * Qp.x() * 1;
    C(r, 32 - 1) = Q.z() * Qp.y() * 1;
    C(r, 33 - 1) = Q.z() * Qp.z() * 1;
    C(r, 34 - 1) = Q.z() * 1 * Qpp.x();
    C(r, 35 - 1) = Q.z() * 1 * Qpp.y();
    C(r, 36 - 1) = Q.z() * 1 * Qpp.z();
    C(r, 37 - 1) = Q.z() * 1 * 1;
    C(r, 38 - 1) = 1 * Qp.x() * Qpp.x();
    C(r, 39 - 1) = 1 * Qp.x() * Qpp.y();
    C(r, 40 - 1) = 1 * Qp.x() * Qpp.z();
    C(r, 41 - 1) = 1 * Qp.x() * 1;
    C(r, 42 - 1) = 1 * Qp.y() * Qpp.x();
    C(r, 43 - 1) = 1 * Qp.y() * Qpp.y();
    C(r, 44 - 1) = 1 * Qp.y() * Qpp.z();
    C(r, 45 - 1) = 1 * Qp.y() * 1;
    C(r, 46 - 1) = 1 * Qp.z() * Qpp.x();
    C(r, 47 - 1) = 1 * Qp.z() * Qpp.y();
    C(r, 48 - 1) = 1 * Qp.z() * Qpp.z();
    C(r, 49 - 1) = 1 * Qp.z() * 1;
    C(r, 50 - 1) = 1 * 1 * Qpp.x();
    C(r, 51 - 1) = 1 * 1 * Qpp.y();
    C(r, 52 - 1) = 1 * 1 * Qpp.z();
    C(r, 53 - 1) = 1 * 1 * 1;
  }
  
  // Compute solution vector V (up to scale).
  JacobiSVD<MatrixXd> svd_V(C, ComputeFullV);
  Matrix<double, 53, 1> V = svd_V.matrixV().col(52);
  
  // Extract the motion parameters.
  // V(1-1)..V(7-1) corresponds to both V(8-8)..V(14-8) and W(1-1)..W(7-1)
  // V(8-1)..V(30-1) corresponds to W(15-8)..W(37-8)
  // V(31-1)..V(53-1) corresponds to V(15-8)..V(37-8)
  
  // Estimate scale factor based on the orthonormality of R', up to sign.
  double lambda = sqrt(max(0., V(1-1)*V(1-1) + V(2-1)*V(2-1) + V(3-1)*V(3-1)));
  V = V / lambda;
  
  // Corresponds to R'.
  Mat3d R0;
  R0 << -V( 8-1), -V( 9-1), -V(10-1),
        -V(31-1), -V(32-1), -V(33-1),
         V( 1-1),  V( 2-1),  V( 3-1);
  
  // Corresponds to R''.
  Mat3d R1;
  R1 <<  V(11-1),  V(12-1),  V(13-1),
         V(34-1),  V(35-1),  V(36-1),
        -V( 4-1), -V( 5-1), -V( 6-1);
  
  if (R0.determinant() < 0) {
    V = -V;
    R0 = -R0;
    R1 = -R1;
  }
  
  // Get closest orthonormal matrices to noisy R0 and R1
  JacobiSVD<MatrixXd> svd_R0(R0, ComputeFullU | ComputeFullV);
  R0 = svd_R0.matrixU() * svd_R0.matrixV().transpose();
  
  JacobiSVD<MatrixXd> svd_R1(R1, ComputeFullU | ComputeFullV);
  R1 = svd_R1.matrixU() * svd_R1.matrixV().transpose();
  
  // Assemble matrix for solving for the translations.
  Eigen::Matrix<double, 15, 6> A = Eigen::Matrix<double, 15, 6>::Zero();
  
  A(2, 0) = -1;
  A(10, 0) = R1(2, 0);
  A(12, 0) = R1(2, 1);
  A(14, 0) = R1(2, 2);
  
  A(1, 1) = -1;
  A( 9, 1) = R1(2, 0);
  A(11, 1) = R1(2, 1);
  A(13, 1) = R1(2, 2);
  
  A(0, 2) = 1;
  A( 9, 2) = -R1(1, 0);
  A(10, 2) = -R1(0, 0);
  A(11, 2) = -R1(1, 1);
  A(12, 2) = -R1(0, 1);
  A(13, 2) = -R1(1, 2);
  A(14, 2) = -R1(0, 2);
  
  A(2, 3) = 1;
  A(10-6, 3) = -R0(2, 0);
  A(12-6, 3) = -R0(2, 1);
  A(14-6, 3) = -R0(2, 2);
  
  A(1, 4) = 1;
  A( 9-6, 4) = -R0(2, 0);
  A(11-6, 4) = -R0(2, 1);
  A(13-6, 4) = -R0(2, 2);
  
  A(0, 5) = -1;
  A( 9-6, 5) = R0(1, 0);
  A(10-6, 5) = R0(0, 0);
  A(11-6, 5) = R0(1, 1);
  A(12-6, 5) = R0(0, 1);
  A(13-6, 5) = R0(1, 2);
  A(14-6, 5) = R0(0, 2);
  
  // V(1-1)..V(7-1) corresponds to both V(8-8)..V(14-8) and W(1-1)..W(7-1)
  // V(8-1)..V(30-1) corresponds to W(15-8)..W(37-8)
  // V(31-1)..V(53-1) corresponds to V(15-8)..V(37-8)
  
  Eigen::Matrix<double, 15, 1> b;
  b << V( 7 - 1),
       V(37 - 1),
       V(14 - 1),
       V(41 - 1),
       V(18 - 1),
       V(45 - 1),
       V(22 - 1),
       V(49 - 1),
       V(26 - 1),
       V(50 - 1),
       V(27 - 1),
       V(51 - 1),
       V(28 - 1),
       V(52 - 1),
       V(29 - 1);
  
  Matrix<double, 6, 1> t01 = (A.transpose() * A).inverse() * (A.transpose() * b);
  
  cloud2_tr_cloud[0] = SE3d(R0, t01.topRows<3>());
  cloud2_tr_cloud[1] = SE3d(R1, t01.bottomRows<3>());
  
  return true;
}

}

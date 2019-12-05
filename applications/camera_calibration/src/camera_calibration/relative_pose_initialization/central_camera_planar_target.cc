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

// Implemented from Sec. 5.3 in S. Ramalingam's PhD thesis.
bool CentralCameraPlanarCalibrationObjectRelativePose(const Point3fCloud clouds[3], SE3d cloud2_tr_cloud[2], Vec3d* optical_center) {
  usize num_points = clouds[0].size();
  // * 2 constraints per point triple
  // * 9 unknowns in system of homogeneous equations (... = 0)
  // * solution only possible up to scale
  // --> 4 point triples required
  if (num_points < 4) {
    return false;
  }
  
  // Normalize position and scaling of homogeneous points.
  vector<Vec2d> normalized_clouds[3];
  Vec2d sum = Vec2d::Zero();
  for (int i = 0; i < 3; ++ i) {
    for (usize r = 0; r < num_points; ++ r) {
      const Vec3f& p = clouds[i][r].position();
      if (fabs(p.z()) > 1e-6) {
        std::cout << "ERROR: The z coordinate of all input points must be zero!\n";
        return false;
      }
      
      sum += Vec2d(p.x(), p.y());
    }
  }
  
  Vec2d mean = sum / (3 * num_points);
  double dist_sum = 0;
  for (int i = 0; i < 3; ++ i) {
    normalized_clouds[i].resize(num_points);
    for (usize r = 0; r < num_points; ++ r) {
      const Vec3f& p = clouds[i][r].position();
      normalized_clouds[i][r] = Vec2d(p.x(), p.y()) - mean;
      dist_sum += normalized_clouds[i][r].norm();
    }
  }
  
  double mean_dist = dist_sum / (3 * num_points);
  double norm_factor = sqrtf(2) / mean_dist;
  for (int i = 0; i < 3; ++ i) {
    for (usize r = 0; r < num_points; ++ r) {
      normalized_clouds[i][r] = norm_factor * normalized_clouds[i][r];
    }
  }
  
  // Assemble coefficients matrices. Each point triple gives one row.
  Eigen::Matrix<double, Eigen::Dynamic, 9> C4;
  Eigen::Matrix<double, Eigen::Dynamic, 9> C3;
  C4.resize(2 * num_points, Eigen::NoChange);
  C3.resize(2 * num_points, Eigen::NoChange);
  
  for (usize point = 0; point < num_points; ++ point) {
    if (fabs(clouds[0][point].position().z()) > 1e-6 ||
        fabs(clouds[1][point].position().z()) > 1e-6 ||
        fabs(clouds[2][point].position().z()) > 1e-6) {
      std::cout << "ERROR: The z coordinate of all input points must be zero!\n";
      return false;
    }
    
    // Q corresponds to clouds[2][point] (the cloud with fixed pose).
    const Vec2d& Q = normalized_clouds[2][point];
    // Q' corresponds to clouds[0][point].
    const Vec2d& Qp = normalized_clouds[0][point];
    // Q'' corresponds to clouds[1][point].
    const Vec2d& Qpp = normalized_clouds[1][point];
    
    // C4 - row 1 (for V):
    int r = 2 * point + 0;
    
    C4(r, 1 - 1) = Q.y() * Qp.x();
    C4(r, 2 - 1) = Q.y() * Qp.y();
    C4(r, 3 - 1) = Q.y() * 1 /*Qp.w()*/;
    
    C4(r, 4 - 1) = 1 /*Q.w()*/ * Qp.x();
    C4(r, 5 - 1) = 1 /*Q.w()*/ * Qp.y();
    C4(r, 6 - 1) = 1 /*Q.w()*/ * 1 /*Qp.w()*/;
    
    C4(r, 7 - 1) = 0;
    C4(r, 8 - 1) = 0;
    C4(r, 9 - 1) = 0;
    
    // C4 - row 2 (for W):
    r = 2 * point + 1;
    
    C4(r, 1 - 1) = Q.x() * Qp.x();
    C4(r, 2 - 1) = Q.x() * Qp.y();
    C4(r, 3 - 1) = Q.x() * 1 /*Qp.w()*/;
    
    C4(r, 4 - 1) = 0;
    C4(r, 5 - 1) = 0;
    C4(r, 6 - 1) = 0;
    
    C4(r, 7 - 1) = 1 /*Q.w()*/ * Qp.x();
    C4(r, 8 - 1) = 1 /*Q.w()*/ * Qp.y();
    C4(r, 9 - 1) = 1 /*Q.w()*/ * 1 /*Qp.w()*/;
    
    // C3 - row 1 (for M):
    r = 2 * point + 0;
    
    C3(r, 1 - 1) = Q.y() * Qpp.x();
    C3(r, 2 - 1) = Q.y() * Qpp.y();
    C3(r, 3 - 1) = Q.y() * 1 /*Qpp.w()*/;
    
    C3(r, 4 - 1) = 1 /*Q.w()*/ * Qpp.x();
    C3(r, 5 - 1) = 1 /*Q.w()*/ * Qpp.y();
    C3(r, 6 - 1) = 1 /*Q.w()*/ * 1 /*Qpp.w()*/;
    
    C3(r, 7 - 1) = 0;
    C3(r, 8 - 1) = 0;
    C3(r, 9 - 1) = 0;
    
    // C4 - row 2 (for N):
    r = 2 * point + 1;
    
    C3(r, 1 - 1) = Q.x() * Qpp.x();
    C3(r, 2 - 1) = Q.x() * Qpp.y();
    C3(r, 3 - 1) = Q.x() * 1 /*Qpp.w()*/;
    
    C3(r, 4 - 1) = 0;
    C3(r, 5 - 1) = 0;
    C3(r, 6 - 1) = 0;
    
    C3(r, 7 - 1) = 1 /*Q.w()*/ * Qpp.x();
    C3(r, 8 - 1) = 1 /*Q.w()*/ * Qpp.y();
    C3(r, 9 - 1) = 1 /*Q.w()*/ * 1 /*Qpp.w()*/;
  }
  
  // Compute solution vector U (up to scale), containing the values of V and W from the original algorithm.
  JacobiSVD<MatrixXd> svd_U(C4, ComputeFullV);
  Matrix<double, 9, 1> U = svd_U.matrixV().col(8);
  
  // U(1-1)..U(3-1) corresponds to both V(4-4)..V(6-4) and W(1-1)..W(3-1).
  // U(4-1)..U(6-1) corresponds to V(7-4)..V(9-4).
  // U(7-1)..U(9-1) corresponds to W(7-4)..W(9-4).
  
  // Compute solution vector L (up to scale), containing the values of M and N from the original algorithm.
  JacobiSVD<MatrixXd> svd_L(C3, ComputeFullV);
  Matrix<double, 9, 1> L = svd_L.matrixV().col(8);
  
  // L(1-1)..L(3-1) corresponds to both M(4-4)..M(6-4) and N(1-1)..N(3-1).
  // L(4-1)..L(6-1) corresponds to M(7-4)..M(9-4).
  // L(7-1)..L(9-1) corresponds to N(7-4)..N(9-4).
  
  
  // Extract the motion parameters.
  // Set up the matrix A_{12 \times 14}.
  // NOTE: In the PhD thesis, A contains some entries without a prime (').
  //       I think this is a mistake, as we cannot know these at this point
  //       since we do not know the scaling factors \lambda_1 / 2.
  // NOTE: In the last row of A there is -N'_6 according to the PhD thesis,
  //       but that would be zero. It seems that -M'_6 is intended instead.
  //       In the same row of b, M / N also seem to be flipped.
  Matrix<double, 12, 14> A = Matrix<double, 12, 14>::Zero();
  
  A(0, 1) = -1 * U(1 - 1);
  A(0, 6) = 1;
  A(1, 1) = -1 * U(2 - 1);
  A(1, 7) = 1;
  A(2, 1) = -1 * U(3 - 1);
  A(2, 3) = 1;
  
  A(3, 0) = -1 * U(1 - 1);
  A(3, 4) = 1;
  A(4, 0) = -1 * U(2 - 1);
  A(4, 5) = 1;
  A(5, 0) = -1 * U(3 - 1);
  A(5, 2) = 1;
  
  A(6, 1) = -1 * L(1 - 1);
  A(6, 12) = 1;
  A(7, 1) = -1 * L(2 - 1);
  A(7, 13) = 1;
  A(8, 1) = -1 * L(3 - 1);
  A(8, 9) = 1;
  
  A(9, 0) = -1 * L(1 - 1);
  A(9, 10) = 1;
  A(10, 0) = -1 * L(2 - 1);
  A(10, 11) = 1;
  A(11, 0) = -1 * L(3 - 1);
  A(11, 8) = 1;
  
  Matrix<double, 12, 1> A_b;
  A_b << U(4 - 1),
         U(5 - 1),
         U(6 - 1),
         U(7 - 1),
         U(8 - 1),
         U(9 - 1),
         L(4 - 1),
         L(5 - 1),
         L(6 - 1),
         L(7 - 1),
         L(8 - 1),
         L(9 - 1);
  
  // Solve the system A * u = b as a linear combination of three new vectors
  // a, b, c, where a satisfies the equation and b, c are the two null vectors
  // of A. The solution can then be expressed as a + l1 * b + l2 * c for
  // arbitrary factors l1 and l2. This uses Algorithm A5.2 in Appendix 4 of
  // "Multiple View Geometry in Computer Vision".
  // With the notation from this algorithm:
  // m = 12, r = 12, n = 14.
  // In the SVD, U is 12x12, V is 14x14.
  JacobiSVD<MatrixXd> svd_A(A, ComputeFullU | ComputeFullV);
  Matrix<double, 12, 1> b_prime = svd_A.matrixU().transpose() * A_b;
  Matrix<double, 14, 1> y;
  for (int i = 0; i < 12; ++ i) {
    y[i] = b_prime[i] / svd_A.singularValues()[i];
  }
  y[12] = 0;
  y[13] = 0;
  // The solution vectors are named as in Ramalingam's algorithm.
  Matrix<double, 14, 1> solution_a = svd_A.matrixV() * y;
  Matrix<double, 14, 1> solution_b = svd_A.matrixV().col(12);
  Matrix<double, 14, 1> solution_c = svd_A.matrixV().col(13);
  
  // Compose the next linear equation system, variables A_{12}, A_{345}, A_{678}
  // in the PhD thesis. Use some lambda functions to conveniently index the
  // vectors with 1-based indices, as in the thesis.
  auto a = [&](int i){return solution_a[i-1];};
  auto b = [&](int i){return solution_b[i-1];};
  auto c = [&](int i){return solution_c[i-1];};
  
  Matrix<double, 6, 8> A8;
  
  A8(0, 0) = a(5)*b(6) + b(5)*a(6) + a(7)*b(8) + b(7)*a(8);
  A8(1, 0) = a(11)*b(12) + b(11)*a(12) + a(13)*b(14) + b(13)*a(14);
  A8(2, 0) = 2*a(5)*b(5) + 2*a(7)*b(7);
  A8(3, 0) = 2*a(6)*b(6) + 2*a(8)*b(8);
  A8(4, 0) = 2*a(11)*b(11) + 2*a(13)*b(13);
  A8(5, 0) = 2*a(12)*b(12) + 2*a(14)*b(14);
  
  A8(0, 1) = a(5)*c(6) + c(5)*a(6) + a(7)*c(8) + c(7)*a(8);
  A8(1, 1) = a(11)*c(12) + c(11)*a(12) + a(13)*c(14) + c(13)*a(14);
  A8(2, 1) = 2*a(5)*c(5) + 2*a(7)*c(7);
  A8(3, 1) = 2*a(6)*c(6) + 2*a(8)*c(8);
  A8(4, 1) = 2*a(11)*c(11) + 2*a(13)*c(13);
  A8(5, 1) = 2*a(12)*c(12) + 2*a(14)*c(14);
  
  A8(0, 2) = b(5)*c(6) + c(5)*b(6) + b(7)*c(8) + c(7)*b(8);
  A8(1, 2) = b(11)*c(12) + c(11)*b(12) + b(13)*c(14) + c(13)*b(14);
  A8(2, 2) = 2*b(5)*c(5) + 2*b(7)*c(7);
  A8(3, 2) = 2*b(6)*c(6) + 2*b(8)*c(8);
  A8(4, 2) = 2*b(11)*c(11) + 2*b(13)*c(13);
  A8(5, 2) = 2*b(12)*c(12) + 2*b(14)*c(14);
  
  A8(0, 3) = b(5)*b(6) + b(7)*b(8);
  A8(1, 3) = b(11)*b(12) + b(13)*b(14);
  A8(2, 3) = b(5)*b(5) + b(7)*b(7);
  A8(3, 3) = b(6)*b(6) + b(8)*b(8);
  A8(4, 3) = b(11)*b(11) + b(13)*b(13);
  A8(5, 3) = b(12)*b(12) + b(14)*b(14);
  
  A8(0, 4) = c(5)*c(6) + c(7)*c(8);
  A8(1, 4) = c(11)*c(12) + c(13)*c(14);
  A8(2, 4) = c(5)*c(5) + c(7)*c(7);
  A8(3, 4) = c(6)*c(6) + c(8)*c(8);
  A8(4, 4) = c(11)*c(11) + c(13)*c(13);
  A8(5, 4) = c(12)*c(12) + c(14)*c(14);
  
  A8(0, 5) = U(1 - 1) * U(2 - 1);
  A8(1, 5) = L(1 - 1) * L(2 - 1);
  A8(2, 5) = U(1 - 1) * U(1 - 1);
  A8(3, 5) = U(2 - 1) * U(2 - 1);
  A8(4, 5) = L(1 - 1) * L(1 - 1);
  A8(5, 5) = L(2 - 1) * L(2 - 1);
  
  A8(0, 6) = 0;
  A8(1, 6) = 0;
  A8(2, 6) = -1;
  A8(3, 6) = -1;
  A8(4, 6) = 0;
  A8(5, 6) = 0;
  
  A8(0, 7) = 0;
  A8(1, 7) = 0;
  A8(2, 7) = 0;
  A8(3, 7) = 0;
  A8(4, 7) = -1;
  A8(5, 7) = -1;
  
  // Compose b8. NOTE: This is missing in the PhD thesis.
  Matrix<double, 6, 1> b8;
  b8 << -a(5)*a(6) - a(7)*a(8),
        -a(11)*a(12) - a(13)*a(14),
        -a(5)*a(5) - a(7)*a(7),
        -a(6)*a(6) - a(8)*a(8),
        -a(11)*a(11) - a(13)*a(13),
        -a(12)*a(12) - a(14)*a(14);
  
  // Solve the system A8 * x = b8.
  // A8 has rank 5 (according to the PhD thesis), so the solution x takes the
  // form : (d + m1 * e + m2 * f + m3 * g) with vectors d, e, f, g and unknown scalars
  // m1, m2, m3 (NOTE: This is wrong in the thesis, it shows only two additional vectors).
  // Here, the sizes are:
  // m = 6, r = 5, n = 8.
  // U is 6x6, V is 8x8.
  JacobiSVD<MatrixXd> svd_A8(A8, ComputeFullU | ComputeFullV);
  Matrix<double, 6, 1> b8_prime = svd_A8.matrixU().transpose() * b8;
  Matrix<double, 8, 1> y8;
  for (int i = 0; i < 5; ++ i) {
    y8[i] = b8_prime[i] / svd_A8.singularValues()[i];
  }
  y8[5] = 0;
  y8[6] = 0;
  y8[7] = 0;
  Matrix<double, 8, 1> solution_d = svd_A8.matrixV() * y8;
  // Matrix<double, 8, 1> solution_e = svd_A8.matrixV().col(7);
  // Matrix<double, 8, 1> solution_f = svd_A8.matrixV().col(6);
  // Matrix<double, 8, 1> solution_g = svd_A8.matrixV().col(5);
  
  // Compute u using the resulting values for l1 and l2.
  Matrix<double, 14, 1> solution_u =
      solution_a +
      solution_d[0] * solution_b +
      solution_d[1] * solution_c;
  auto u = [&](int i){return solution_u[i-1];};
  
  // Compute motion parameters.
  Vec3d O;
  O.x() = u(1);
  O.y() = u(2);
  double denom_V = -1 * U(1 - 1) * U(2 - 1);
  double denom_M = -1 * L(1 - 1) * L(2 - 1);
  if (fabs(denom_V) > fabs(denom_M)) {
    double temp = (u(5)*u(6) + u(7)*u(8)) / denom_V;
    if (temp < -1e-3) {
      std::cout << "WARNING: Negative value in sqrt(): " << temp << endl;
      return false;
    }
    O.z() = sqrt(std::max(0., temp));
  } else {
    double temp = (u(11)*u(12) + u(13)*u(14)) / denom_M;
    if (temp < -1e-3) {
      std::cout << "WARNING: Negative value in sqrt(): " << temp << endl;
      return false;
    }
    O.z() = sqrt(std::max(0., temp));
  }
  
  // We assume the calibration target points to be specified such that
  // cameras will always observe it from negative (instead of positive) z.
  // I.e., x goes to the right and y goes to the bottom in the target's
  // coordinate system.
  if (O.z() > 0) {
    O.z() = -O.z();
  }
  
  if (optical_center) {
    *optical_center = O;
  }
  
  // There is a flipping ambiguity here: we do not know the sign of the first
  // two columns in R0. This corresponds to the decision of whether the pattern
  // is in front of the camera, or a flipped pattern is behind the camera (mirrored
  // at the optical center). We choose it such that it is in the same direction
  // as the fixed cloud2.
  for (int lambda_1_sign = -1; lambda_1_sign <= 1; lambda_1_sign += 2) {
    double lambda_1 = lambda_1_sign * sqrt(u(5)*u(5) + u(7)*u(7) + U(1-1)*U(1-1)*O.z()*O.z()) / O.z();
    
    Mat3d R0;  // corresponds to R'
    
    R0(0, 0) = u(5) / (O.z() * lambda_1);
    R0(1, 0) = u(7) / (O.z() * lambda_1);
    R0(2, 0) = U(1-1) / lambda_1;
    
    R0(0, 1) = u(6) / (O.z() * lambda_1);
    R0(1, 1) = u(8) / (O.z() * lambda_1);
    R0(2, 1) = U(2-1) / lambda_1;
    
    R0.col(2) = R0.col(0).cross(R0.col(1));
    
    Vec3d t0;  // corresponds to t'
    // TODO: Can the denominators here be near-zero?
    t0.x() = (u(3) + O.x() * O.z() * lambda_1) / (O.z() * lambda_1);
    t0.y() = (u(4) + O.y() * O.z() * lambda_1) / (O.z() * lambda_1);
    t0.z() = (O.z() * t0.y() - U(6-1) / lambda_1) / O.y();
    
    // Test whether the transformed points are on the same side of the optical center as the fixed cloud2 points.
    if (lambda_1_sign == -1) {
      usize num_same_side = 0;
      usize num_opposite_side = 0;
      for (usize i = 0; i < 3; ++ i) {  // NOTE: Testing a single point would probably be sufficient
        Vec3d tp = R0 * clouds[0].at(i).position().cast<double>() + t0;
        bool same_side = (tp - O).dot(clouds[2].at(i).position().cast<double>() - O) > 0;
        if (same_side) {
          ++ num_same_side;
        } else {
          ++ num_opposite_side;
        }
      }
      
      if (num_opposite_side > num_same_side) {
        continue;
      }
    }
    
    cloud2_tr_cloud[0] = SE3d(R0, t0);
    break;
  }
  
  for (int lambda_2_sign = -1; lambda_2_sign <= 1; lambda_2_sign += 2) {
    double lambda_2 = lambda_2_sign * sqrt(u(11)*u(11) + u(13)*u(13) + L(1-1)*L(1-1)*O.z()*O.z()) / O.z();
    
    Mat3d R1;  // corresponds to R''
    
    R1(0, 0) = u(11) / (O.z() * lambda_2);
    R1(1, 0) = u(13) / (O.z() * lambda_2);
    R1(2, 0) = L(1-1) / lambda_2;
    
    R1(0, 1) = u(12) / (O.z() * lambda_2);
    R1(1, 1) = u(11) / (O.z() * lambda_2);
    R1(2, 1) = L(2-1) / lambda_2;
    
    R1.col(2) = R1.col(0).cross(R1.col(1));
    
    Vec3d t1;  // corresponds to t''
    t1.x() = (u(9) + O.x() * O.z() * lambda_2) / (O.z() * lambda_2);
    t1.y() = (u(10) + O.y() * O.z() * lambda_2) / (O.z() * lambda_2);
    t1.z() = (O.z() * t1.y() - L(6-1) / lambda_2) / O.y();
    
    // Test whether the transformed points are on the same side of the optical center as the fixed cloud2 points.
    if (lambda_2_sign == -1) {
      usize num_same_side = 0;
      usize num_opposite_side = 0;
      for (usize i = 0; i < 3; ++ i) {  // NOTE: Testing a single point would probably be sufficient
        Vec3d tp = R1 * clouds[1].at(i).position().cast<double>() + t1;
        bool same_side = (tp - O).dot(clouds[2].at(i).position().cast<double>() - O) > 0;
        if (same_side) {
          ++ num_same_side;
        } else {
          ++ num_opposite_side;
        }
      }
      
      if (num_opposite_side > num_same_side) {
        continue;
      }
    }
    
    cloud2_tr_cloud[1] = SE3d(R1, t1);
    break;
  }
  
  // NOTE: In the original description of the algorithm, they also compute
  //       estimates for ray directions here. This is not implemented here.
  
  // De-normalize the solution.
  *optical_center /= norm_factor;
  *optical_center += Vec3d(mean.x(), mean.y(), 0);
  
  cloud2_tr_cloud[0].translation() /= norm_factor;
  cloud2_tr_cloud[1].translation() /= norm_factor;
  
  cloud2_tr_cloud[0] = SE3d(Quaterniond::Identity(), Vec3d(mean.x(), mean.y(), 0)) *
                       cloud2_tr_cloud[0] *
                       SE3d(Quaterniond::Identity(), Vec3d(-mean.x(), -mean.y(), 0));
  cloud2_tr_cloud[1] = SE3d(Quaterniond::Identity(), Vec3d(mean.x(), mean.y(), 0)) *
                       cloud2_tr_cloud[1] *
                       SE3d(Quaterniond::Identity(), Vec3d(-mean.x(), -mean.y(), 0));
  
  return true;
}

}

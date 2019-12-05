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

bool NonCentralCameraPlanarCalibrationObjectRelativePose(const Point3fCloud clouds[3], SE3d cloud2_tr_cloud[2], SE3d gt_cloud2_tr_cloud[2]) {
  usize num_points = clouds[0].size();
  // * 2 constraints per point triple
  // * 23 unknowns in system of homogeneous equations (... = 0)
  // * solution only possible up to scale
  // --> 11 point triples required
  if (num_points < 11) {
    return false;
  }
  
  Mat3d gt_R0 = gt_cloud2_tr_cloud[0].rotationMatrix();
//   Vec3f gt_t0 = gt_cloud2_tr_cloud[0].translation();
  
//   Mat3f gt_R1 = gt_cloud2_tr_cloud[1].rotationMatrix();
//   Vec3f gt_t1 = gt_cloud2_tr_cloud[1].translation();
  
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
  
  // Assemble coefficients matrices.
  Eigen::Matrix<double, Eigen::Dynamic, 23> C;
  C.resize(2 * num_points, Eigen::NoChange);
  
  for (usize point = 0; point < num_points; ++ point) {
    // Q corresponds to clouds[2][point] (the cloud with fixed pose).
    const Vec2d& Q = normalized_clouds[2][point];
    // Q' corresponds to clouds[0][point].
    const Vec2d& Qp = normalized_clouds[0][point];
    // Q'' corresponds to clouds[1][point].
    const Vec2d& Qpp = normalized_clouds[1][point];
    
    // Row 1 (for V):
    int r = 2 * point + 0;
    
    C(r,  1 - 1) = Q.y() * Qp.x() * 1;
    C(r,  2 - 1) = Q.y() * Qp.y() * 1;
    C(r,  3 - 1) = Q.y() * 1 * Qpp.x();
    C(r,  4 - 1) = Q.y() * 1 * Qpp.y();
    C(r,  5 - 1) = Q.y() * 1 * 1;
    
    C(r,  6 - 1) = 1 * Qp.x() * Qpp.x();
    C(r,  7 - 1) = 1 * Qp.x() * Qpp.y();
    C(r,  8 - 1) = 1 * Qp.x() * 1;
    C(r,  9 - 1) = 1 * Qp.y() * Qpp.x();
    C(r, 10 - 1) = 1 * Qp.y() * Qpp.y();
    C(r, 11 - 1) = 1 * Qp.y() * 1;
    C(r, 12 - 1) = 1 * 1 * Qpp.x();
    C(r, 13 - 1) = 1 * 1 * Qpp.y();
    C(r, 14 - 1) = 1 * 1 * 1;
    
    for (int i = 15; i < 15 + 9; ++ i) {
      C(r, i - 1) = 0;
    }
    
    // Row 2 (for W):
    r = 2 * point + 1;
    
    C(r,  1 - 1) = Q.x() * Qp.x() * 1;
    C(r,  2 - 1) = Q.x() * Qp.y() * 1;
    C(r,  3 - 1) = Q.x() * 1 * Qpp.x();
    C(r,  4 - 1) = Q.x() * 1 * Qpp.y();
    C(r,  5 - 1) = Q.x() * 1 * 1;
    
    for (int i = 6; i < 6 + 9; ++ i) {
      C(r, i - 1) = 0;
    }
    
    C(r, 15 - 1) = 1 * Qp.x() * Qpp.x();
    C(r, 16 - 1) = 1 * Qp.x() * Qpp.y();
    C(r, 17 - 1) = 1 * Qp.x() * 1;
    C(r, 18 - 1) = 1 * Qp.y() * Qpp.x();
    C(r, 19 - 1) = 1 * Qp.y() * Qpp.y();
    C(r, 20 - 1) = 1 * Qp.y() * 1;
    C(r, 21 - 1) = 1 * 1 * Qpp.x();
    C(r, 22 - 1) = 1 * 1 * Qpp.y();
    C(r, 23 - 1) = 1 * 1 * 1;
  }
  
  // Compute solution vector U (up to scale), containing the values of V and W from the original algorithm.
  JacobiSVD<MatrixXd> svd_U(C, ComputeFullV);
  Matrix<double, 23, 1> U = svd_U.matrixV().col(22);
  
  // U(1-1)..U(5-1) corresponds to both V(6-6)..V(10-6) and W(1-1)..W(5-1).
  // U(6-1)..U(14-1) corresponds to V(11-6)..V(19-6).
  // U(15-1)..U(23-1) corresponds to W(11-6)..W(19-6).
  
  
  
  // DEBUG
//   std::cout << "gt_R0(2, 0): " << gt_R0(2, 0) << " vs. " << gt_lambda_1 * U(1-1) << endl;
//   std::cout << "t'3: " << gt_t0.z() << " vs. " << gt_lambda_1 * up << endl;
//   std::cout << "t''3: " << gt_t1.z() << " vs. " << gt_lambda_1 * upp << endl;
  
  // Set up first system to extract some rotation components.
  Matrix<double, 4, 4> A;
  A << -U(3-1), 0, -U(1-1), 0,
       -U(4-1), 0, 0, -U(1-1),
       0, -U(3-1), -U(2-1), 0,
       0, -U(4-1), 0, -U(2-1);
  
  Matrix<double, 4, 1> A1_b;
  A1_b << U(6-1),
          U(7-1),
          U(9-1),
          U(10-1);
  
  // DEBUG
//   std::cout << "b from gt:\n" << (A * Matrix<double, 4, 1>(gt_R0(1, 0), gt_R0(1, 1), gt_R1(1, 0), gt_R1(1, 1))) << endl;
//   std::cout << "assembled b:\n" << A1_b << endl;
  
  // Obtain solution with one free parameter.
  // m = 4, r = 3, n = 4.
  // In the SVD, U is 4x4, V is 4x4.
  JacobiSVD<MatrixXd> svd_A1(A, ComputeFullU | ComputeFullV);
  Matrix<double, 4, 1> b1_prime = svd_A1.matrixU().transpose() * A1_b;
  Matrix<double, 4, 1> y1;
  for (int i = 0; i < 3; ++ i) {
    y1[i] = b1_prime[i] / svd_A1.singularValues()[i];
  }
  y1[3] = 0;
  Matrix<double, 4, 1> solution_a1 = svd_A1.matrixV() * y1;
//   Matrix<double, 4, 1> solution_b1 = svd_A1.matrixV().col(3);
  
  // Set up second system to extract some rotation components.
  Matrix<double, 4, 1> A2_b;
  A2_b << U(15-1),
          U(16-1),
          U(18-1),
          U(19-1);
  
  // DEBUG
//   std::cout << "b2 from gt:\n" << (A * Matrix<double, 4, 1>(gt_R0(0, 0), gt_R0(0, 1), gt_R1(0, 0), gt_R1(0, 1))) << endl;
//   std::cout << "assembled b2:\n" << A2_b << endl;
  
  // Obtain solution with one free parameter.
  // m = 4, r = 3, n = 4.
  // In the SVD, U is 4x4, V is 4x4.
  JacobiSVD<MatrixXd> svd_A2(A, ComputeFullU | ComputeFullV);
  Matrix<double, 4, 1> b2_prime = svd_A2.matrixU().transpose() * A2_b;
  Matrix<double, 4, 1> y2;
  for (int i = 0; i < 3; ++ i) {
    y2[i] = b2_prime[i] / svd_A2.singularValues()[i];
  }
  y2[3] = 0;
  Matrix<double, 4, 1> solution_a2 = svd_A2.matrixV() * y2;
//   Matrix<double, 4, 1> solution_b2 = svd_A2.matrixV().col(3);
  
  // Set up system to extract the scaling of the previous solutions.
  // Use the same indexing as in S. Ramalingam's PhD thesis for convenience.
  auto a = [&](int i){return (i <= 4) ? solution_a1[i-1] : solution_a2[i - 5];};
//   auto b = [&](int i){return (i <= 4) ? solution_b1[i-1] : solution_b2[i - 5];};
  
  // TODO: Use closed-form expressions for a as well. Those are not given in
  //       the paper "due to lack of space".
  // Using the approach from "A unifying model for camera calibration" instead
  // of the more complicated one from the PhD thesis. Index mapping:
  // thesis - paper
  // V6 - V8
  // V7 - V9
  // V8 - V11
  // V9 - V12
  Matrix<double, 6, 3> A3;
  
  A3(0, 0) = a(1)*U(2-1) + U(1-1)*a(2);
  A3(1, 0) = -a(3)*U(4-1) - U(3-1)*a(4);
  A3(2, 0) = 2*a(1)*U(1-1);
  A3(3, 0) = 2*a(2)*U(2-1);
  A3(4, 0) = -2*a(3)*U(3-1);
  A3(5, 0) = -2*a(4)*U(4-1);
  
  A3(0, 1) = a(5)*U(2-1) + U(1-1)*a(6);
  A3(1, 1) = -a(7)*U(4-1) - U(3-1)*a(8);
  A3(2, 1) = 2*a(5)*U(1-1);
  A3(3, 1) = 2*a(6)*U(2-1);
  A3(4, 1) = -2*a(7)*U(3-1);
  A3(5, 1) = -2*a(8)*U(4-1);
  
  A3(0, 2) = U(1-1)*U(2-1);
  A3(1, 2) = U(3-1)*U(4-1);
  A3(2, 2) = U(1-1)*U(1-1);
  A3(3, 2) = U(2-1)*U(2-1);
  A3(4, 2) = U(3-1)*U(3-1);
  A3(5, 2) = U(4-1)*U(4-1);
  
  Matrix<double, 6, 1> A5_b;
  A5_b << -a(5)*a(6) - a(1)*a(2),
          -a(7)*a(8) - a(3)*a(4),
          1 - a(1)*a(1) - a(5)*a(5),
          1 - a(2)*a(2) - a(6)*a(6),
          1 - a(3)*a(3) - a(7)*a(7),
          1 - a(4)*a(4) - a(8)*a(8);
  
  // Solve the system.
  Matrix<double, 3, 1> solution_d = (A3.transpose() * A3).inverse() * (A3.transpose() * A5_b);
  double l1 = solution_d[0];
  double l2 = solution_d[1];
  // TODO: Only computing the solution for +lambda here!
  // TODO: Use the ground truth to decide for the sign of lambda
  double lambda_squared = 1 / (solution_d[2] - l1*l1 - l2*l2);
  if (lambda_squared < -1e-3) {
    std::cout << "WARNING: Negative value in sqrt(): " << lambda_squared << endl;
  }
  double lambda = sqrt(std::max(0., lambda_squared));
  
  Mat3d R0, R1;
  
  R0(1, 0) = a(1) + l1 * U(1-1);
  R0(1, 1) = a(2) + l1 * U(2-1);
  R1(1, 0) = a(3) + l1 * -U(3-1);
  R1(1, 1) = a(4) + l1 * -U(4-1);
  
  R0(0, 0) = a(5) + l2 * U(1-1);
  R0(0, 1) = a(6) + l2 * U(2-1);
  R1(0, 0) = a(7) + l2 * -U(3-1);
  R1(0, 1) = a(8) + l2 * -U(4-1);
  
  // HACK: Use the ground truth to decide for the sign of lambda
  if (gt_R0(2, 0) * (U(1-1) / lambda) < 0) {
    lambda = -lambda;
  }
  
  R0(2, 0) = U(1-1) / lambda;
  R0(2, 1) = U(2-1) / lambda;
  
  R1(2, 0) = -U(3-1) / lambda;
  R1(2, 1) = -U(4-1) / lambda;
  
  R0.col(2) = R0.col(0).cross(R0.col(1));
  R1.col(2) = R1.col(0).cross(R1.col(1));
  
  Vec3d t0, t1;
  double denom1 = -U(4-1)*U(6-1) + U(3-1)*U(7-1);
  double denom2 = U(4-1)*U(15-1) - U(3-1)*U(16-1);
  double up;  // corresponds to u'
  if (fabs(denom1) > fabs(denom2)) {
    up = ((-U(4-1)*U(12-1) + U(3-1)*U(13-1)) * U(1-1)) / denom1;
  } else {
    up = ((U(4-1)*U(21-1) - U(3-1)*U(22-1)) * U(1-1)) / denom2;
  }
  
  double upp = up - U(5-1);  // corresponds to u''
  t0.z() = up / lambda;
  t1.z() = upp / lambda;
  
  if (fabs(U(3-1)) > fabs(U(4-1))) {
    t0.x() = (U(21-1) + R1(0, 0) * up) / (-U(3-1));
    t0.y() = (U(12-1) + R1(1, 0) * up) / (-U(3-1));
  } else {
    t0.x() = (U(22-1) + R1(0, 1) * up) / (-U(4-1));
    t0.y() = (U(13-1) + R1(1, 1) * up) / (-U(4-1));
  }
  
  if (fabs(U(1-1)) > fabs(U(2-1))) {
    t1.x() = (R0(0, 0) * upp - U(17-1)) / U(1-1);
    t1.y() = (R0(1, 0) * upp - U(8-1)) / U(1-1);
  } else {
    t1.x() = (R0(0, 1) * upp - U(20-1)) / U(2-1);
    t1.y() = (R0(1, 1) * upp - U(11-1)) / U(2-1);
  }
  
  // Try to improve numerical stability a bit more by going over V_19 / W_19
  // in case either t1.x/y or t2.x/y are very unstable.
  // V19 = t'2 t''3 - t''2 t'3
  // W19 = t'1 t''3 - t''1 t'3
  constexpr double kEpsilon = 1e-5f;
  if (fabs(U(3-1)) < kEpsilon &&
      fabs(U(4-1)) < kEpsilon &&
      fabs(t1.z()) >= kEpsilon) {
    t0.x() = (U(23-1) / lambda + t1.x() * t0.z()) / t1.z();
    t0.y() = (U(14-1) / lambda + t1.y() * t0.z()) / t1.z();
  } else if (fabs(U(1-1)) < kEpsilon &&
             fabs(U(2-1)) < kEpsilon &&
             fabs(t0.z()) >= kEpsilon) {
    t1.x() = (t0.x() * t1.z() - U(23-1) / lambda) / t0.z();
    t1.y() = (t0.y() * t1.z() - U(14-1) / lambda) / t0.z();
  }
  
  cloud2_tr_cloud[0] = SE3d(R0, t0);
  cloud2_tr_cloud[1] = SE3d(R1, t1);
  
  // De-normalize the solution.
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

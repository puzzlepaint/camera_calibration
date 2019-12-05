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

bool CentralCamera3DCalibrationObjectRelativePose(const Point3fCloud clouds[2], SE3d cloud1_tr_cloud[1], Vec3d* optical_center) {
  usize num_points = clouds[0].size();
  // * 2 constraints per point triple
  // * 20 unknowns in system of homogeneous equations (... = 0)
  // * solution only possible up to scale
  // --> 10 point triples required (actually 9.5)
  if (num_points < 10) {
    return false;
  }
  
  // Assemble coefficients matrix. Each point triple gives one row.
  Eigen::Matrix<double, Eigen::Dynamic, 20> C;
  C.resize(2 * num_points, Eigen::NoChange);
  
  for (usize point = 0; point < num_points; ++ point) {
    // Q corresponds to clouds[1][point] (the cloud with fixed pose).
    const Vec3f& Q = clouds[1][point].position();
    // Q' corresponds to clouds[0][point].
    const Vec3f& Qp = clouds[0][point].position();
    
    // Row 1 (for V):
    int r = 2 * point + 0;
    
    C(r, 1 - 1) = Q.y() * Qp.x();
    C(r, 2 - 1) = Q.y() * Qp.y();
    C(r, 3 - 1) = Q.y() * Qp.z();
    C(r, 4 - 1) = Q.y() * 1;
    
    C(r, 5 - 1) = Q.z() * Qp.x();
    C(r, 6 - 1) = Q.z() * Qp.y();
    C(r, 7 - 1) = Q.z() * Qp.z();
    C(r, 8 - 1) = Q.z() * 1;
    C(r, 9 - 1) = 1 * Qp.x();
    C(r, 10 - 1) = 1 * Qp.y();
    C(r, 11 - 1) = 1 * Qp.z();
    C(r, 12 - 1) = 1 * 1;
    
    for (int i = 13; i < 13 + 8; ++ i) {
      C(r, i - 1) = 0;
    }
    
    
    // Row 2 (for W):
    r = 2 * point + 1;
    
    C(r, 1 - 1) = Q.x() * Qp.x();
    C(r, 2 - 1) = Q.x() * Qp.y();
    C(r, 3 - 1) = Q.x() * Qp.z();
    C(r, 4 - 1) = Q.x() * 1;
    
    for (int i = 5; i < 5 + 8; ++ i) {
      C(r, i - 1) = 0;
    }
    
    C(r, 13 - 1) = Q.z() * Qp.x();
    C(r, 14 - 1) = Q.z() * Qp.y();
    C(r, 15 - 1) = Q.z() * Qp.z();
    C(r, 16 - 1) = Q.z() * 1;
    C(r, 17 - 1) = 1 * Qp.x();
    C(r, 18 - 1) = 1 * Qp.y();
    C(r, 19 - 1) = 1 * Qp.z();
    C(r, 20 - 1) = 1 * 1;
  }
  
  // Compute solution vector U (up to scale), containing the values of V and W from the original algorithm.
  JacobiSVD<MatrixXd> svd_U(C, ComputeFullV);
  Matrix<double, 20, 1> U = svd_U.matrixV().col(19);
  
  // U(1-1)..U(4-1) corresponds to both V(5-5)..V(8-5) and W(1-1)..W(4-1).
  // U(5-1)..U(12-1) corresponds to V(9-5)..V(16-5).
  // U(13-1)..U(20-1) corresponds to W(9-5)..W(16-5).
  
  // Extract the motion parameters.
  // Scale U.
  double lambda = sqrt(U(1 - 1) * U(1 - 1) +
                       U(2 - 1) * U(2 - 1) +
                       U(3 - 1) * U(3 - 1));
  U /= lambda;
  
  // Extract R.
  Mat3d R;
  R << -U(13 - 1), -U(14 - 1), -U(15 - 1),
       -U(5 - 1), -U(6 - 1), -U(7 - 1),
        U(1 - 1),  U(2 - 1),  U(3 - 1);
  
  // Fix sign in case det(R) == -1 (instead of 1).
  double det = R.determinant();
  if (det < 0) {
    U = -U;
    R = -R;
  }
  
  // Compute optical center.
  // NOTE: The coefficients of O are written in the wrong order (z, y, x) in the paper.
  // 
  // NOTE: The equations below are from the paper and should be correct, but
  //       seem to be unstable. We use the same scheme as for O.z() instead.
  //   O.y() = (O.z() * R(2 - 1, 1 - 1) - U(9 - 1)) / R(3 - 1, 1 - 1);
  //   O.x() = (O.z() * R(1 - 1, 1 - 1) - U(17 - 1)) / R(3 - 1, 1 - 1);
  // 
  // There are different possibilities of how to compute the entries of O.
  // Since they involve divisions through elements of R, care has to be taken
  // to choose a variant which does not divide by near-zero values depending
  // on how R looks like.
  
  Vec3d O;
  
  // X
  double denomW13W14 = R(3 - 1, 1 - 1) * R(1 - 1, 2 - 1) - R(3 - 1, 2 - 1) * R(1 - 1, 1 - 1);
  double denomW13W15 = R(3 - 1, 1 - 1) * R(1 - 1, 3 - 1) - R(3 - 1, 3 - 1) * R(1 - 1, 1 - 1);
  double denomW14W15 = R(3 - 1, 2 - 1) * R(1 - 1, 3 - 1) - R(3 - 1, 3 - 1) * R(1 - 1, 2 - 1);
  
  if (fabs(denomW13W14) > fabs(denomW14W15)) {
    if (fabs(denomW13W14) > fabs(denomW13W15)) {
      O.x() = -(U(17 - 1) * R(1 - 1, 2 - 1) - U(18 - 1) * R(1 - 1, 1 - 1)) / denomW13W14;  // O.x() from W13 and W14
    } else {
      O.x() = -(U(17 - 1) * R(1 - 1, 3 - 1) - U(19 - 1) * R(1 - 1, 1 - 1)) / denomW13W15;  // O.y() from W13 and W15
    }
  } else {
    if (fabs(denomW14W15) > fabs(denomW13W15)) {
      O.x() = -(U(18 - 1) * R(1 - 1, 3 - 1) - U(19 - 1) * R(1 - 1, 2 - 1)) / denomW14W15;  // O.y() from W14 and W15
    } else {
      O.x() = -(U(17 - 1) * R(1 - 1, 3 - 1) - U(19 - 1) * R(1 - 1, 1 - 1)) / denomW13W15;  // O.y() from W13 and W15
    }
  }
  
  // Y
  double denom_y_V13V14 = R(3 - 1, 1 - 1) * R(2 - 1, 2 - 1) - R(3 - 1, 2 - 1) * R(2 - 1, 1 - 1);
  double denom_y_V13V15 = R(3 - 1, 1 - 1) * R(2 - 1, 3 - 1) - R(3 - 1, 3 - 1) * R(2 - 1, 1 - 1);
  double denom_y_V14V15 = R(3 - 1, 2 - 1) * R(2 - 1, 3 - 1) - R(3 - 1, 3 - 1) * R(2 - 1, 2 - 1);
  
  if (fabs(denom_y_V13V14) > fabs(denom_y_V14V15)) {
    if (fabs(denom_y_V13V14) > fabs(denom_y_V13V15)) {
      O.y() = -(U(9 - 1) * R(2 - 1, 2 - 1) - U(10 - 1) * R(2 - 1, 1 - 1)) / denom_y_V13V14;  // O.y() from V13 and V14
    } else {
      O.y() = -(U(9 - 1) * R(2 - 1, 3 - 1) - U(11 - 1) * R(2 - 1, 1 - 1)) / denom_y_V13V15;  // O.y() from V13 and V15
    }
  } else {
    if (fabs(denom_y_V14V15) > fabs(denom_y_V13V15)) {
      O.y() = -(U(10 - 1) * R(2 - 1, 3 - 1) - U(11 - 1) * R(2 - 1, 2 - 1)) / denom_y_V14V15;  // O.y() from V14 and V15
    } else {
      O.y() = -(U(9 - 1) * R(2 - 1, 3 - 1) - U(11 - 1) * R(2 - 1, 1 - 1)) / denom_y_V13V15;  // O.y() from V13 and V15
    }
  }
  
  // Z
  double denomV13V14 = R(2 - 1, 1 - 1) * R(3 - 1, 2 - 1) - R(2 - 1, 2 - 1) * R(3 - 1, 1 - 1);
  double denomV13V15 = R(2 - 1, 1 - 1) * R(3 - 1, 3 - 1) - R(2 - 1, 3 - 1) * R(3 - 1, 1 - 1);
  double denomV14V15 = R(2 - 1, 2 - 1) * R(3 - 1, 3 - 1) - R(2 - 1, 3 - 1) * R(3 - 1, 2 - 1);
  
  if (fabs(denomV13V14) > fabs(denomV14V15)) {
    if (fabs(denomV13V14) > fabs(denomV13V15)) {
      O.z() = (U(9 - 1) * R(3 - 1, 2 - 1) - U(10 - 1) * R(3 - 1, 1 - 1)) / denomV13V14;  // O.z() from V13 and V14
    } else {
      O.z() = (U(9 - 1) * R(3 - 1, 3 - 1) - U(11 - 1) * R(3 - 1, 1 - 1)) / denomV13V15;  // O.z() from V13 and V15
    }
  } else {
    if (fabs(denomV14V15) > fabs(denomV13V15)) {
      O.z() = (U(10 - 1) * R(3 - 1, 3 - 1) - U(11 - 1) * R(3 - 1, 2 - 1)) / denomV14V15;  // O.z() from V14 and V15
    } else {
      O.z() = (U(9 - 1) * R(3 - 1, 3 - 1) - U(11 - 1) * R(3 - 1, 1 - 1)) / denomV13V15;  // O.z() from V13 and V15
    }
  }
  
  // Extract t.
  Vec3d t;
  t.x() = O.x() - U(16 - 1);
  t.y() = O.y() - U(8 - 1);
  t.z() = U(4 - 1) + O.z();
  
  // Return result.
  if (optical_center) {
    *optical_center = O;
  }
  cloud1_tr_cloud[0] = SE3d(R, t);
  return true;
}

}

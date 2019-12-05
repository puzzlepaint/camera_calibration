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

#include "libvis/cuda/test/cholesky_solver.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>

#include "libvis/cuda/cuda_util.h"

using namespace vis;

namespace {
template <int N>
bool TestCholeskySolver() {
  // LOG(INFO) << "Test with N = " << N << " ...";
  
  // Choose random H and b such that H is positive definite
  Eigen::Matrix<float, N, N> H = Eigen::Matrix<float, N, N>::Identity();
  Eigen::Matrix<float, N, 1> b = Eigen::Matrix<float, N, 1>::Random();
  
  Eigen::Matrix<float, N, 1> random = Eigen::Matrix<float, N, 1>::Random();
  H += random * random.transpose();
  
  // Solve with Eigen
  Eigen::Matrix<float, N, 1> x_eigen = H.ldlt().solve(b);
  
  // Solve with SolveWithParallelCholesky()
  vector<float> H_continuous(N * (N + 1) / 2);
  int index = 0;
  for (int y = 0; y < N; ++ y) {
    for (int x = y; x < N; ++ x) {
      H_continuous[index] = H(y, x);
      ++ index;
    }
  }
  CHECK_EQ(index, H_continuous.size());
  
  CHECK_CUDA_NO_ERROR();
  
  float* H_cuda;
  cudaMalloc(&H_cuda, H_continuous.size() * sizeof(float));
  CHECK_CUDA_NO_ERROR();
  cudaMemcpy(H_cuda, H_continuous.data(), H_continuous.size() * sizeof(float), cudaMemcpyHostToDevice);
  CHECK_CUDA_NO_ERROR();
  
  float* b_cuda;
  cudaMalloc(&b_cuda, N * sizeof(float));
  CHECK_CUDA_NO_ERROR();
  cudaMemcpy(b_cuda, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  CHECK_CUDA_NO_ERROR();
  
  if (N <= 7) {
    CallCholeskySolverKernelForNMax7(N, H_cuda, b_cuda);
  } else {
    CallCholeskySolverKernel(N, H_cuda, b_cuda);
  }
  
  Eigen::Matrix<float, N, 1> x_cuda;
  cudaMemcpy(x_cuda.data(), b_cuda, N * sizeof(float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_NO_ERROR();
  
  // Compare the results
  constexpr float kEpsilon = 1e-6f;
  
  bool success = true;
  for (int i = 0; i < N; ++ i) {
    EXPECT_NEAR(x_eigen(i), x_cuda(i), kEpsilon) << "for i == " << i;
    if (fabs(x_eigen(i) - x_cuda(i)) > kEpsilon) {
      success = false;
    }
  }
  
  return success;
}
}

TEST(CholeskySolver, ComparisonToEigen) {
  srand(0);
  
  constexpr int kTestIterations = 100;
  for (int i = 0; i < kTestIterations; ++ i) {
    // Testing the variant with N <= 7:
    ASSERT_TRUE(TestCholeskySolver<1>());
    ASSERT_TRUE(TestCholeskySolver<2>());
    ASSERT_TRUE(TestCholeskySolver<3>());
    ASSERT_TRUE(TestCholeskySolver<4>());
    ASSERT_TRUE(TestCholeskySolver<5>());
    ASSERT_TRUE(TestCholeskySolver<6>());
    ASSERT_TRUE(TestCholeskySolver<7>());
    
    // Testing the variant with N >= 8:
    ASSERT_TRUE(TestCholeskySolver<8>());
    ASSERT_TRUE(TestCholeskySolver<16>());
  }
}

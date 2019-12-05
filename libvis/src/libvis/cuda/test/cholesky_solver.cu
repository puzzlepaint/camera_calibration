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

#include "libvis/cuda/cholesky_solver.h"
#include "libvis/cuda/cuda_util.h"
#include "libvis/logging.h"

namespace vis {

template <int N>
__global__ void CholeskySolverKernelForNMax7(
    float* H, float* b) {
  __shared__ float H_shared[N * (N + 1) / 2];
  __shared__ float b_shared[N];
  
  // Read inputs from global memory into shared memory.
  // Note: cannot do all reads in one step for N == 7 with only 32 threads.
  if (threadIdx.x < N) {
    b_shared[threadIdx.x] = b[threadIdx.x];
  }
  if (threadIdx.x < N * (N + 1) / 2) {
    H_shared[threadIdx.x] = H[threadIdx.x];
  }
  __syncthreads();
  
  // Compute the results
  SolveWithParallelCholeskyForNMax7<N>(H_shared, b_shared);
  
  // Read results from shared memory into global memory
  if (threadIdx.x < N) {
    b[threadIdx.x] = b_shared[threadIdx.x];
  }
  if (threadIdx.x < N * (N + 1) / 2) {
    H[threadIdx.x] = H_shared[threadIdx.x];
  }
}

void CallCholeskySolverKernelForNMax7(int N, float* H_cuda, float* b_cuda) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(32, 1, 1);
  cudaStream_t stream = 0;
  if (N == 1) {
    CholeskySolverKernelForNMax7<1><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 2) {
    CholeskySolverKernelForNMax7<2><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 3) {
    CholeskySolverKernelForNMax7<3><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 4) {
    CholeskySolverKernelForNMax7<4><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 5) {
    CholeskySolverKernelForNMax7<5><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 6) {
    CholeskySolverKernelForNMax7<6><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 7) {
    CholeskySolverKernelForNMax7<7><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else {
    LOG(FATAL) << "Value of N not supported: " << N;
  }
  CHECK_CUDA_NO_ERROR();
}

template <int N>
__global__ void CholeskySolverKernel(
    float* H, float* b) {
  const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ float H_shared[N * (N + 1) / 2];
  __shared__ float b_shared[N];
  
  // Read inputs from global memory into shared memory.
  if (thread_index < N) {
    b_shared[thread_index] = b[thread_index];
  } else if (thread_index < N + N * (N + 1) / 2) {
    H_shared[thread_index - N] = H[thread_index - N];
  }
  __syncthreads();
  
  // Compute the results
  SolveWithParallelCholesky<N>(H_shared, b_shared);
  
  // Read results from shared memory into global memory
  if (thread_index < N) {
    b[thread_index] = b_shared[thread_index];
  } else if (thread_index < N + N * (N + 1) / 2) {
    H[thread_index - N] = H_shared[thread_index - N];
  }
}

void CallCholeskySolverKernel(int N, float* H_cuda, float* b_cuda) {
  constexpr int kBlockDim = 256;
  CHECK_GE(kBlockDim, N + N * (N + 1) / 2);
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(kBlockDim, 1, 1);
  cudaStream_t stream = 0;
  if (N == 8) {
    CholeskySolverKernel<8><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else if (N == 16) {
    CholeskySolverKernel<16><<<grid_dim, block_dim, 0, stream>>>(H_cuda, b_cuda);
  } else {
    LOG(FATAL) << "Value of N not supported, please add it here to fix: " << N;
  }
  CHECK_CUDA_NO_ERROR();
}

}

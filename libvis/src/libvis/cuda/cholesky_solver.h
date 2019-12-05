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

#include <cuda_runtime.h>

#include "libvis/libvis.h"

// TODO: Shouldn't this file's extension be .cuh?

namespace vis {

/// Solves H * x = b for x, given H and b, using a parallel Cholesky
/// decomposition of H (i.e., H must be positive definite symmetric).
/// 
/// The composition is done in-place, and the result is returned in b. H is of
/// size kDim x kDim. kDim can be at most 7.
/// H_shared and b_shared must point to shared memory. H must be stored
/// row-by-row or column-by-column while leaving out redundant symmetric values,
/// so its size is (kDim * (kDim + 1) / 2).
/// 
/// Every thread in a given warp must call this function.
/// The warp shape must be [32 x 1 x 1].
/// 
/// With Eigen on the CPU, the corresponding code would be:
/// Eigen::Matrix<float, kDim, 1> x = H.selfadjointView<Eigen::Upper>().ldlt().solve(b);
template<int kDim>
__device__ void SolveWithParallelCholeskyForNMax7(
    float* H_shared,
    float* b_shared) {
  const unsigned int& thread_index = threadIdx.x;
  
  // Compute row / column for each thread. TODO: Can this be sped up?
  int row;
  int column;
  int counter = 0;
  #pragma unroll
  for (int i = 0; i < kDim; ++ i) {
    if (thread_index >= counter && thread_index < counter + kDim - i) {
      row = i + thread_index - counter;
      column = i;
    }
    counter += kDim - i;
  }
  
  counter = 0;
  float product_sum = 0;
  #pragma unroll
  for (int i = 0; i < kDim; ++ i) {
    // Index of current diagonal entry for this iteration:
    int cur_diag_index = (i * (kDim + (kDim + 1 - i))) / 2;
    // Compute the diagonal entry (only correct for corresponding thread).
    float new_value = sqrtf(::max(0.f, H_shared[cur_diag_index] - product_sum));
    // Distribute correct new diagonal value to all other threads.
    new_value = __shfl_sync(0xffff, new_value, cur_diag_index);
    
    // Compute the beyond-diagonal entries in the same column / row and
    // store the results (as well as the diagonal result) in H_shared.
    if (thread_index >= cur_diag_index && thread_index < cur_diag_index + kDim - i) {
      if (thread_index > cur_diag_index) {
        new_value = 1.0f / new_value * (H_shared[thread_index] - product_sum);
      }
      H_shared[thread_index] = new_value;
    }
    
    // Distribute the new off-diagonal values to the threads that need them.
    // Each thread gets the new value with the same row, and the new value
    // with the same column (in the symmetric view of the matrix).
    float value_1 = __shfl_up_sync(0xffff, new_value, thread_index - (row - i + counter));
    float value_2 = __shfl_up_sync(0xffff, new_value, thread_index - (column - i + counter));
    
    // Accumulate the beyond-diagonal entries as required for the remaining
    // entries.
    product_sum += value_1 * value_2;
    
    counter += kDim - i;
  }
  __syncwarp();  // make H_shared changes visible to all threads
  
//    // DEBUG: Output result
//    if (thread_index < kDim * (kDim + 1) / 2) {
//      printf("L %i (%i, %i): %f\n", thread_index, row, column, H_shared[thread_index]);
//    }
//    // END DEBUG
  
  // With H = L * L^T,
  // H * x = b becomes L * L^T * x = b .
  // We define y = L^T * x ,
  // and first solve for L * y = b (in-place):
  // y(0) = 1 / H(0, 0) * (b(0))
  // y(1) = 1 / H(1, 1) * (b(1) - H(1, 0) * y(0))
  // y(2) = 1 / H(2, 2) * (b(2) - H(2, 0) * y(0) - H(2, 1) * y(1))
  // ...
  counter = 0;
  product_sum = 0;
  #pragma unroll
  for (int i = 0; i < kDim; ++ i) {
    // Index of current diagonal entry for this iteration:
    int cur_diag_index = (i * (kDim + (kDim + 1 - i))) / 2;
    // Compute the y entry (only correct for corresponding thread).
    float new_value = 1.0f / H_shared[cur_diag_index] * (b_shared[i] - product_sum);
    // Distribute correct new y value to all other threads.
    new_value = __shfl_sync(0xffff, new_value, i);
    
    if (thread_index == i) {
      b_shared[i] = new_value;
    }
    
    product_sum += H_shared[::min(counter + thread_index - i, kDim * (kDim + 1) / 2 - 1)] * new_value;
    
    counter += kDim - i;
  }
  __syncwarp();
  
//    // DEBUG
//    if (thread_index < kDim) {
//      printf("y[%i] = %f\n", thread_index, b_shared[thread_index]);
//    }
//    // END DEBUG
  
  // Then solve for L^T * x = y (in-place):
  // x(5) = 1 / H(5, 5) * (y(5))
  // x(4) = 1 / H(4, 4) * (y(4) - H(4, 5) * x(5))
  // x(3) = 1 / H(3, 3) * (y(3) - H(3, 5) * x(5) - H(3, 4) * x(4))
  // ...
  product_sum = 0;
  #pragma unroll
  for (int i = kDim - 1; i >= 0; -- i) {
    // Index of current diagonal entry for this iteration:
    int cur_diag_index = (i * (kDim + (kDim + 1 - i))) / 2;
    // Compute the y entry (only correct for corresponding thread).
    float new_value = 1.0f / H_shared[cur_diag_index] * (b_shared[i] - product_sum);
    // Distribute correct new y value to all other threads.
    new_value = __shfl_sync(0xffff, new_value, i);
    
    if (thread_index == i) {
      b_shared[i] = new_value;
    }
    
    product_sum += H_shared[::min(kDim * (kDim + 1) / 2 - 1, ::max(0, ((thread_index+1) * (kDim + (kDim + 1 - (thread_index+1)))) / 2 - 1 + (i - (kDim - 1))))] * new_value;
  }
  __syncwarp();
  
//    // DEBUG
//    if (thread_index < kDim) {
//      printf("x[%i] = %f\n", thread_index, b_shared[thread_index]);
//    }
//    // END DEBUG
}

/// Solves H * x = b for x, given H and b, using a parallel Cholesky
/// decomposition of H (i.e., H must be positive definite symmetric).
/// 
/// The composition is done in-place, and the result is returned in b. H is of
/// size kDim x kDim.
/// H_shared and b_shared must point to shared memory. H must be stored
/// row-by-row or column-by-column while leaving out redundant symmetric values,
/// so its size is (kDim * (kDim + 1) / 2).
/// 
/// Every thread in a given block must call this function.
/// The block shape must be [N x 1 x 1] with N >= kDim.
/// 
/// With Eigen on the CPU, the corresponding code would be:
/// Eigen::Matrix<float, kDim, 1> x = H.selfadjointView<Eigen::Upper>().ldlt().solve(b);
template<int kDim>
__device__ void SolveWithParallelCholesky(
    float* H_shared,
    float* b_shared) {
  const unsigned int& thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ float shared_new_value;
  
  // Compute row / column for each thread. TODO: Can this be sped up?
  int row = 0;
  int column = 0;
  int counter = 0;
  for (int i = 0; i < kDim; ++ i) {
    if (thread_index >= counter && thread_index < counter + kDim - i) {
      row = i + thread_index - counter;
      column = i;
    }
    counter += kDim - i;
  }
  
  counter = 0;
  float product_sum = 0;
  for (int i = 0; i < kDim; ++ i) {
    // Index of current diagonal entry for this iteration:
    int cur_diag_index = (i * (kDim + (kDim + 1 - i))) / 2;
    // Compute the diagonal entry (only correct for corresponding thread).
    if (thread_index == cur_diag_index) {
      shared_new_value = sqrtf(::max(0.f, H_shared[cur_diag_index] - product_sum));
    }
    __syncthreads();
    
    // Compute the beyond-diagonal entries in the same column / row and
    // store the results (as well as the diagonal result) in H_shared.
    if (thread_index >= cur_diag_index && thread_index < cur_diag_index + kDim - i) {
      float value = shared_new_value;
      if (thread_index > cur_diag_index) {
        value = 1.0f / value * (H_shared[thread_index] - product_sum);
      }
      H_shared[thread_index] = value;
    }
    __syncthreads();
    
    // Accumulate the beyond-diagonal entries as required for the remaining
    // entries.
    product_sum += H_shared[row - i + counter] * H_shared[column - i + counter];
    
    counter += kDim - i;
  }
  
//    // DEBUG: Output result
//    if (thread_index < kDim * (kDim + 1) / 2) {
//      printf("L %i (%i, %i): %f\n", thread_index, row, column, H_shared[thread_index]);
//    }
//    // END DEBUG
  
  // With H = L * L^T,
  // H * x = b becomes L * L^T * x = b .
  // We define y = L^T * x ,
  // and first solve for L * y = b (in-place):
  // y(0) = 1 / H(0, 0) * (b(0))
  // y(1) = 1 / H(1, 1) * (b(1) - H(1, 0) * y(0))
  // y(2) = 1 / H(2, 2) * (b(2) - H(2, 0) * y(0) - H(2, 1) * y(1))
  // ...
  counter = 0;
  product_sum = 0;
  for (int i = 0; i < kDim; ++ i) {
    // Index of current diagonal entry for this iteration:
    int cur_diag_index = (i * (kDim + (kDim + 1 - i))) / 2;
    // Compute the y entry (only correct for corresponding thread).
    if (thread_index == i) {
      shared_new_value = 1.0f / H_shared[cur_diag_index] * (b_shared[i] - product_sum);
      b_shared[i] = shared_new_value;
    }
    __syncthreads();
    if (i == kDim - 1) {
      break;
    }
    
    product_sum += H_shared[::min(counter + thread_index - i, kDim * (kDim + 1) / 2 - 1)] * shared_new_value;
    
    counter += kDim - i;
    __syncthreads();  // wait for all threads to read shared_new_value before it may be overwritten by the next iteration of this loop (TODO: avoid this by using two shared values?)
  }
  
//    // DEBUG
//    if (thread_index < kDim) {
//      printf("y[%i] = %f\n", thread_index, b_shared[thread_index]);
//    }
//    // END DEBUG
  
  // Then solve for L^T * x = y (in-place):
  // x(5) = 1 / H(5, 5) * (y(5))
  // x(4) = 1 / H(4, 4) * (y(4) - H(4, 5) * x(5))
  // x(3) = 1 / H(3, 3) * (y(3) - H(3, 5) * x(5) - H(3, 4) * x(4))
  // ...
  product_sum = 0;
  for (int i = kDim - 1; i >= 0; -- i) {
    // Index of current diagonal entry for this iteration:
    int cur_diag_index = (i * (kDim + (kDim + 1 - i))) / 2;
    // Compute the y entry (only correct for corresponding thread).
    if (thread_index == i) {
      shared_new_value = 1.0f / H_shared[cur_diag_index] * (b_shared[i] - product_sum);
      b_shared[i] = shared_new_value;
    }
    __syncthreads();
    if (i == 0) {
      break;
    }
    
    product_sum += H_shared[::min(kDim * (kDim + 1) / 2 - 1, ::max(0, ((thread_index+1) * (kDim + (kDim + 1 - (thread_index+1)))) / 2 - 1 + (i - (kDim - 1))))] * shared_new_value;
    __syncthreads();  // wait for all threads to read shared_new_value before it may be overwritten by the next iteration of this loop (TODO: avoid this by using two shared values?)
  }
  
//    // DEBUG
//    if (thread_index < kDim) {
//      printf("x[%i] = %f\n", thread_index, b_shared[thread_index]);
//    }
//    // END DEBUG
}

}

// Copyright 2019 ETH Zurich, Thomas Schöps
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

// Avoid warnings in Eigen includes with CUDA compiler
#pragma diag_suppress code_is_unreachable

#include "camera_calibration/feature_detection/cuda_refinement_by_matching.cuh"

#include <cub/cub.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_util.h>
#include <libvis/logging.h>
#include <math_constants.h>

#include "camera_calibration/feature_detection/cuda_util.cuh"

namespace vis {

struct UpdateEquationCoefficients4 {
  float H_0_0;
  float H_0_1;
  float H_0_2;
  float H_0_3;
  
  float H_1_1;
  float H_1_2;
  float H_1_3;
  
  float H_2_2;
  float H_2_3;
  
  float H_3_3;
  
  float b_0;
  float b_1;
  float b_2;
  float b_3;
  
  float cost;
  
  __forceinline__ __device__ void SetZero() {
    H_0_0 = 0;
    H_0_1 = 0;
    H_0_2 = 0;
    H_0_3 = 0;
    
    H_1_1 = 0;
    H_1_2 = 0;
    H_1_3 = 0;
    
    H_2_2 = 0;
    H_2_3 = 0;
    
    H_3_3 = 0;
    
    b_0 = 0;
    b_1 = 0;
    b_2 = 0;
    b_3 = 0;
    
    cost = 0;
  }
};

/// Returns the pattern intensity (0 for black, 1 for white, 0.5 for ill-defined
/// positions) at the given position within the pattern. The pattern is supposed
/// to have endless extent, feature positions are at integer coordinates, and
/// (0, 0) is supposed to correspond to a feature location.
__forceinline__ __device__ float PatternIntensityAt(float x, float y, int num_star_segments) {
  // Have coordinates in [-0.5, 0.5].
  float c_x = x - (x > 0 ? 1 : -1) * static_cast<int>(::fabs(x) + 0.5f);
  float c_y = y - (y > 0 ? 1 : -1) * static_cast<int>(::fabs(y) + 0.5f);
  
  if (c_x * c_x + c_y * c_y < 1e-8f) {
    return 0.5f;
  }
  
  float angle = ::atan2(c_y, c_x) - 0.5f * M_PI;
  if (angle < 0) {
    angle += 2 * M_PI;
  }
  return (static_cast<int>(num_star_segments * angle / (2 * M_PI)) % 2 == 0) ? 1.f : 0.f;
}

__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ 1024, /*minBlocksPerMultiprocessor*/ 1)
RefineFeatureByMatchingKernel_RenderSamples(
    int num_star_segments,
    int num_samples,
    CUDABuffer_<float2> samples,
    CUDABuffer_<float> local_pattern_tr_pixel_buffer,
    int window_half_size,
    CUDABuffer_<float> rendered_samples) {
  constexpr int kNumAntiAliasSamples = 16;
  
  unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_index >= num_samples) {
    return;
  }
  
  // Load the homography (column-major storage, as is Eigen's default)
  __shared__ float h[9];
  if (threadIdx.x < 9) {
    h[threadIdx.x] = local_pattern_tr_pixel_buffer(0, 9 * blockIdx.z + threadIdx.x);
  }
  __syncthreads();
  
  // Loop over the anti-alias samples
  float sum = 0;
  for (int s = 0; s < kNumAntiAliasSamples; ++ s) {
    // Samples spread in [-0.5, 0.5], i.e., within the range of one pixel.
    float pixel_offset_x = window_half_size * samples(0, sample_index).x +
                           -0.5 + 1 / 8.f + 1 / 4.f * (s % 4);
    float pixel_offset_y = window_half_size * samples(0, sample_index).y +
                           -0.5 + 1 / 8.f + 1 / 4.f * (s / 4);
    
    float pattern_offset_factor = 1.f / (h[2] * pixel_offset_x + h[5] * pixel_offset_y + h[8]);
    float pattern_offset_x = (h[0] * pixel_offset_x + h[3] * pixel_offset_y + h[6]) * pattern_offset_factor;
    float pattern_offset_y = (h[1] * pixel_offset_x + h[4] * pixel_offset_y + h[7]) * pattern_offset_factor;
    
    sum += PatternIntensityAt(pattern_offset_x, pattern_offset_y, num_star_segments);
  }
  
  // Normalization by kNumSubpixelSamples is not necessary here since an
  // affine intensity transformation is optimized for later.
  rendered_samples(blockIdx.z, sample_index) = sum;
}

void CallRefineFeatureByMatchingKernel_RenderSamples(
    cudaStream_t stream,
    int feature_count,
    int num_star_segments,
    int num_samples,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& local_pattern_tr_pixel_buffer,
    int window_half_size,
    const CUDABuffer_<float>& rendered_samples) {
  #define CALL_KERNEL(block_width_value) \
      constexpr int block_width = block_width_value; \
      dim3 grid_dim(GetBlockCount(num_samples, block_width), 1, feature_count); \
      dim3 block_dim(block_width, 1, 1); \
      RefineFeatureByMatchingKernel_RenderSamples \
      <<<grid_dim, block_dim, 0, stream>>>( \
          num_star_segments, num_samples, sample_positions, local_pattern_tr_pixel_buffer, \
          window_half_size, rendered_samples);
  if (num_samples > 512) {
    CALL_KERNEL(1024);
  } else if (num_samples > 256) {
    CALL_KERNEL(512);
  } else if (num_samples > 128) {
    CALL_KERNEL(256);
  } else {
    CALL_KERNEL(128);
  }
  #undef CALL_KERNEL
  CHECK_CUDA_NO_ERROR();
}


template <int block_width>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ 1024, /*minBlocksPerMultiprocessor*/ 1)
RefineFeatureByMatchingKernel_InitFactorAndBias(
    int num_samples,
    CUDABuffer_<float2> samples,
    CUDABuffer_<float> rendered_samples,
    cudaTextureObject_t image,
    float4* states,
    int window_half_size) {
  constexpr int block_height = 1;
  
  float position_x = states[blockIdx.z].x;
  float position_y = states[blockIdx.z].y;
  
  float sum_qp = 0;
  float sum_p = 0;
  float sum_q = 0;
  float sum_pp = 0;
  
  for (int sample_index = /*blockIdx.x * block_width +*/ threadIdx.x; sample_index < num_samples; sample_index += block_width) {
    float sample_x = position_x + window_half_size * samples(0, sample_index).x + 0.5f;  // convert pixel center to pixel corner conv
    float sample_y = position_y + window_half_size * samples(0, sample_index).y + 0.5f;  // convert pixel center to pixel corner conv
//     if (!ImageContainsPixelCornerConv(sample_x, sample_y, image_width, image_height)) {
//       return CUDART_INF_F;
//     }
    float p = tex2D<float>(image, sample_x, sample_y);
    float q = rendered_samples(blockIdx.z, sample_index);
    
    sum_qp += q * p;
    sum_p += p;
    sum_q += q;
    sum_pp += p * p;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage temp_storage;
  
  // TODO: Would it be a good idea for performance to remove some of the
  //       __syncthreads below by using multiple separate shared temp_storage
  //       memory buffers instead of re-using one?
  
  sum_qp = BlockReduceFloat(temp_storage).Sum(sum_qp);
  __syncthreads();
  sum_p = BlockReduceFloat(temp_storage).Sum(sum_p);
  __syncthreads();
  sum_q = BlockReduceFloat(temp_storage).Sum(sum_q);
  __syncthreads();
  sum_pp = BlockReduceFloat(temp_storage).Sum(sum_pp);
  
  if (threadIdx.x == 0) {
    float denominator = sum_pp - (sum_p * sum_p / num_samples);
    if (fabs(denominator) > 1e-6f) {
      /*factor*/ states[blockIdx.z].z = (sum_qp - (sum_p / num_samples) * sum_q) / denominator;
    } else {
      /*factor*/ states[blockIdx.z].z = 1.f;
    }
    /*bias*/ states[blockIdx.z].w = (1.f / num_samples) * (sum_q - states[blockIdx.z].z * sum_p);
  }
}

void CallRefineFeatureByMatchingKernel_InitFactorAndBias(
    cudaStream_t stream,
    int feature_count,
    int num_samples,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& rendered_samples,
    cudaTextureObject_t image,
    float4* states,
    int window_half_size) {
  #define CALL_KERNEL(block_width_value) \
      constexpr int block_width = block_width_value; \
      dim3 grid_dim(1, 1, feature_count); \
      dim3 block_dim(block_width, 1, 1); \
      RefineFeatureByMatchingKernel_InitFactorAndBias<block_width> \
      <<<grid_dim, block_dim, 0, stream>>>( \
          num_samples, sample_positions, rendered_samples, \
          image, states, window_half_size);
  if (num_samples > 512) {
    CALL_KERNEL(1024);
  } else if (num_samples > 256) {
    CALL_KERNEL(512);
  } else if (num_samples > 128) {
    CALL_KERNEL(256);
  } else {
    CALL_KERNEL(128);
  }
  #undef CALL_KERNEL
  CHECK_CUDA_NO_ERROR();
}


__forceinline__ __device__ static void AddCornerRefinementAgainstPatternCostAndJacobian(
    float position_x,
    float position_y,
    float factor,
    float bias,
    int sample_index,
    int feature_index,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& rendered_samples,
    float window_half_size,
    int image_width,
    int image_height,
    cudaTextureObject_t image,
    UpdateEquationCoefficients4* out) {
  float2 sample = sample_positions(0, sample_index);
  
  // Transform pixel center to pixel corner coordinates
  position_x += 0.5f;
  position_y += 0.5f;
  
  float sample_pos_x = position_x + window_half_size * sample.x;
  float sample_pos_y = position_y + window_half_size * sample.y;
  // if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
  //   return CUDART_INF_F;
  // }
  
  int ix = static_cast<int>(::max(0.f, sample_pos_x - 0.5f));
  int iy = static_cast<int>(::max(0.f, sample_pos_y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, sample_pos_x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, sample_pos_y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = tex2D<float>(image, ix + 0.5f, iy + 0.5f);
  float top_right = tex2D<float>(image, ix + 1.5f, iy + 0.5f);
  float bottom_left = tex2D<float>(image, ix + 0.5f, iy + 1.5f);
  float bottom_right = tex2D<float>(image, ix + 1.5f, iy + 1.5f);
  float intensity = tex2D<float>(image, sample_pos_x, sample_pos_y);
  
  float dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  float residual = factor * intensity + bias - rendered_samples(feature_index, sample_index);
  float jac_0 = factor * dx;  // Jac. wrt. position_x
  float jac_1 = factor * dy;  // Jac. wrt. position_y
  float jac_2 = intensity;  // Jac. wrt. factor
  constexpr float jac_3 = 1;  // Jac. wrt. bias
  
  out->H_0_0 += jac_0 * jac_0;
  out->H_0_1 += jac_0 * jac_1;
  out->H_0_2 += jac_0 * jac_2;
  out->H_0_3 += jac_0 * jac_3;
  
  out->H_1_1 += jac_1 * jac_1;
  out->H_1_2 += jac_1 * jac_2;
  out->H_1_3 += jac_1 * jac_3;
  
  out->H_2_2 += jac_2 * jac_2;
  out->H_2_3 += jac_2 * jac_3;
  
  out->H_3_3 += jac_3 * jac_3;
  
  out->b_0 += jac_0 * residual;
  out->b_1 += jac_1 * residual;
  out->b_2 += jac_2 * residual;
  out->b_3 += jac_3 * residual;
  
  // Should actually be: 0.5f * residual * residual. However, we don't care
  // about (positive) scaling here.
  out->cost += residual * residual;
}

__forceinline__ __device__ static float ComputeCornerRefinementAgainstPatternCost(
    float position_x,
    float position_y,
    float factor,
    float bias,
    int sample_index,
    int feature_index,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& rendered_samples,
    float window_half_size,
    int image_width,
    int image_height,
    cudaTextureObject_t image) {
  float2 sample = sample_positions(0, sample_index);
  
  // Transform pixel center to pixel corner coordinates
  position_x += 0.5f;
  position_y += 0.5f;
  
  float sample_pos_x = position_x + window_half_size * sample.x;
  float sample_pos_y = position_y + window_half_size * sample.y;
  if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
    return CUDART_INF_F;
  }
  
  float intensity = tex2D<float>(image, sample_pos_x, sample_pos_y);
  
  float residual = factor * intensity + bias - rendered_samples(feature_index, sample_index);
  
  // Should actually be: 0.5f * residual * residual. However, we don't care
  // about (positive) scaling here.
  return residual * residual;
}

template <int block_width>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ 1024, /*minBlocksPerMultiprocessor*/ 1)
RefineFeatureByMatchingKernel_Refine(
    int num_samples,
    CUDABuffer_<float2> sample_positions,
    CUDABuffer_<float> rendered_samples,
    cudaTextureObject_t image_texture,
    float4* states,
    float* final_cost,
    int window_half_size,
    int image_width,
    int image_height) {
  constexpr int block_height = 1;
  
  __shared__ float test_position_x;
  __shared__ float test_position_y;
  __shared__ float test_factor;
  __shared__ float test_bias;
  __shared__ float test_cost_shared;
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage temp_storage;
  
  
  float lambda = -1;
  float last_step_squared_norm = -1;
  
  float original_position_x = states[blockIdx.z].x;
  if (::isnan(original_position_x)) {
    return;
  }
  float original_position_y = states[blockIdx.z].y;
  __shared__ float position_x;
  __shared__ float position_y;
  __shared__ float factor;
  __shared__ float bias;
  if (threadIdx.x == 0) {
    position_x = original_position_x;
    position_y = original_position_y;
    factor = states[blockIdx.z].z;
    bias = states[blockIdx.z].w;
  }
  
  __shared__ bool applied_update;
  
  constexpr int kMaxIterationCount = 50;
  for (int iteration = 0; iteration < kMaxIterationCount; ++ iteration) {
    // Compute cost and Jacobian
    UpdateEquationCoefficients4 coeffs;
    coeffs.SetZero();
    __syncthreads();  // for position_x/y and for BlockReduce
    for (int sample_index = /*blockIdx.x * block_width +*/ threadIdx.x; sample_index < num_samples; sample_index += block_width) {
      AddCornerRefinementAgainstPatternCostAndJacobian(
          position_x, position_y, factor, bias, sample_index, blockIdx.z,
          sample_positions, rendered_samples, window_half_size, image_width, image_height, image_texture, &coeffs);
    }
    
    // TODO: Test whether it would be helpful for performance to group some of
    //       these together, resulting in higher shared memory usage but less
    //       reductions and less __syncthreads(). Trying to use the whole
    //       UpdateEquationCoefficients4 struct for accumulation resulted in
    //       exceeding the available shared memory.
    coeffs.H_0_0 = BlockReduceFloat(temp_storage).Sum(coeffs.H_0_0);
    __syncthreads();
    coeffs.H_0_1 = BlockReduceFloat(temp_storage).Sum(coeffs.H_0_1);
    __syncthreads();
    coeffs.H_0_2 = BlockReduceFloat(temp_storage).Sum(coeffs.H_0_2);
    __syncthreads();
    coeffs.H_0_3 = BlockReduceFloat(temp_storage).Sum(coeffs.H_0_3);
    __syncthreads();
    coeffs.H_1_1 = BlockReduceFloat(temp_storage).Sum(coeffs.H_1_1);
    __syncthreads();
    coeffs.H_1_2 = BlockReduceFloat(temp_storage).Sum(coeffs.H_1_2);
    __syncthreads();
    coeffs.H_1_3 = BlockReduceFloat(temp_storage).Sum(coeffs.H_1_3);
    __syncthreads();
    coeffs.H_2_2 = BlockReduceFloat(temp_storage).Sum(coeffs.H_2_2);
    __syncthreads();
    coeffs.H_2_3 = BlockReduceFloat(temp_storage).Sum(coeffs.H_2_3);
    __syncthreads();
    coeffs.H_3_3 = BlockReduceFloat(temp_storage).Sum(coeffs.H_3_3);
    __syncthreads();
    coeffs.b_0 = BlockReduceFloat(temp_storage).Sum(coeffs.b_0);
    __syncthreads();
    coeffs.b_1 = BlockReduceFloat(temp_storage).Sum(coeffs.b_1);
    __syncthreads();
    coeffs.b_2 = BlockReduceFloat(temp_storage).Sum(coeffs.b_2);
    __syncthreads();
    coeffs.b_3 = BlockReduceFloat(temp_storage).Sum(coeffs.b_3);
    __syncthreads();
    coeffs.cost = BlockReduceFloat(temp_storage).Sum(coeffs.cost);
    
    // if (threadIdx.x == 0) {
    //   printf("Iteration %i | cost: %f\n", iteration, coeffs.cost);
    // }
    
    // Initialize lambda?
    if (lambda < 0) {
      lambda = 0.001f * 0.5f * (coeffs.H_0_0 + coeffs.H_1_1 + coeffs.H_2_2 + coeffs.H_3_3);
    }
    
    applied_update = false;
    for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
      float H_0_0 = coeffs.H_0_0 + lambda;
      float H_1_1 = coeffs.H_1_1 + lambda;
      float H_2_2 = coeffs.H_2_2 + lambda;
      float H_3_3 = coeffs.H_3_3 + lambda;
      
      // Solve for the update.
      // Perform in-place Cholesky decomposition of H:
      // https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky%E2%80%93Banachiewicz_and_Cholesky%E2%80%93Crout_algorithms
      // Compared to the algorithm in Wikipedia, the matrix is transposed here,
      // and zero-based indexing is used, so the formulas are:
      // 
      // H_j_j = sqrtf(H_j_j - sum_{k=0}^{j-1} (H_k_j * H_k_j))       // for diagonal items
      // H_j_i = ( H_j_i - sum_{k=0}^{j-1} (H_k_i * H_k_j) ) / H_j_j  // for off-diagonal items
      H_0_0 = sqrtf(H_0_0);
      
      float H_0_1 = (coeffs.H_0_1) / H_0_0;
      H_1_1 = sqrtf(H_1_1 - H_0_1 * H_0_1);
      
      float H_0_2 = (coeffs.H_0_2) / H_0_0;
      float H_1_2 = (coeffs.H_1_2 - H_0_2 * H_0_1) / H_1_1;
      H_2_2 = sqrtf(H_2_2 - H_0_2 * H_0_2 - H_1_2 * H_1_2);
      
      float H_0_3 = (coeffs.H_0_3) / H_0_0;
      float H_1_3 = (coeffs.H_1_3 - H_0_3 * H_0_1) / H_1_1;
      float H_2_3 = (coeffs.H_2_3 - H_0_3 * H_0_2 - H_1_3 * H_1_2) / H_2_2;
      H_3_3 = sqrtf(H_3_3 - H_0_3 * H_0_3 - H_1_3 * H_1_3 - H_2_3 * H_2_3);
      
      // Solve H * x = b for x.
      //
      // (H_0_0     0     0     0)   (H_0_0 H_0_1 H_0_2 H_0_3)   (x0)   (b0)
      // (H_0_1 H_1_1     0     0) * (    0 H_1_1 H_1_2 H_1_3) * (x1) = (b1)
      // (H_0_2 H_1_2 H_2_2     0)   (    0     0 H_2_2 H_2_3)   (x2)   (b2)
      // (H_0_3 H_1_3 H_2_3 H_3_3)   (    0     0     0 H_3_3)   (x3)   (b3)
      //
      // Naming the result of the second multiplication y, we get:
      //
      // (H_0_0     0     0     0)   (y0)   (b0)
      // (H_0_1 H_1_1     0     0) * (y1) = (b1)
      // (H_0_2 H_1_2 H_2_2     0)   (y2)   (b2)
      // (H_0_3 H_1_3 H_2_3 H_3_3)   (y3)   (b3)
      // 
      // and:
      // 
      // (H_0_0 H_0_1 H_0_2 H_0_3)   (x0)   (y0)
      // (    0 H_1_1 H_1_2 H_1_3) * (x1) = (y1)
      // (    0     0 H_2_2 H_2_3)   (x2) = (y2)
      // (    0     0     0 H_3_3)   (x3) = (y3)
      
      float y0 = (coeffs.b_0) / H_0_0;
      float y1 = (coeffs.b_1 - H_0_1 * y0) / H_1_1;
      float y2 = (coeffs.b_2 - H_0_2 * y0 - H_1_2 * y1) / H_2_2;
      float y3 = (coeffs.b_3 - H_0_3 * y0 - H_1_3 * y1 - H_2_3 * y2) / H_3_3;
      
      float x3 = (y3) / H_3_3;
      float x2 = (y2 - H_2_3 * x3) / H_2_2;
      float x1 = (y1 - H_1_3 * x3 - H_1_2 * x2) / H_1_1;
      float x0 = (y0 - H_0_3 * x3 - H_0_2 * x2 - H_0_1 * x1) / H_0_0;
      
      // Test whether the update improves the cost.
      if (threadIdx.x == 0) {
        test_position_x = position_x - x0;
        test_position_y = position_y - x1;
        test_factor = factor - x2;
        test_bias = bias - x3;
      }
      __syncthreads();  // for test_<...> and BlockReduceFloat and applied_update
      
      float test_cost_local = 0;
      for (int sample_index = /*blockIdx.x * block_width +*/ threadIdx.x; sample_index < num_samples; sample_index += block_width) {
        test_cost_local += ComputeCornerRefinementAgainstPatternCost(
            test_position_x, test_position_y, test_factor, test_bias, sample_index, blockIdx.z,
            sample_positions, rendered_samples, window_half_size, image_width, image_height, image_texture);
      }
      const float test_cost = BlockReduceFloat(reinterpret_cast<typename BlockReduceFloat::TempStorage&>(temp_storage)).Sum(test_cost_local);
      // if (threadIdx.x == 0) {
      //   printf("  LM iteration %i | lambda: %f, x_0: %f, x_1: %f, test cost: %f\n", lm_iteration, lambda, x_0, x_1, test_cost);
      // }
      
      if (threadIdx.x == 0) {
        test_cost_shared = test_cost;
        
        if (test_cost < coeffs.cost) {
          last_step_squared_norm = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
          position_x = test_position_x;
          position_y = test_position_y;
          factor = test_factor;
          bias = test_bias;
          lambda *= 0.5f;
          applied_update = true;
        } else {
          lambda *= 2.f;
        }
      }
      
      __syncthreads();  // for applied_update, position_x/y, test_cost_shared
      if (::isinf(test_cost_shared)) {
        // Position went out of bounds
        // if (threadIdx.x == 0) {
        //   printf("  Position out of bounds\n");
        // }
        states[blockIdx.z].x = CUDART_NAN_F;
        return;
      }
      if (applied_update) {
        break;
      }
    }
    
    if (!applied_update) {
      // Cannot find an update that improves the cost. Treat this as converged.
      states[blockIdx.z] = make_float4(position_x, position_y, factor, bias);
      if (final_cost) {
        final_cost[blockIdx.z] = test_cost_shared;
      }
      return;
    }
    
    // Check for divergence.
    if (fabs(original_position_x - position_x) >= window_half_size ||
        fabs(original_position_y - position_y) >= window_half_size) {
      // The result is probably not the originally intended corner,
      // since it is not within the original search window.
      // if (threadIdx.x == 0) {
      //   printf("  Position too far away from start. original_position_x: %f, position_x: %f, original_position_y: %f, position_y: %f, window_half_size: %i\n",
      //           original_position_x, position_x, original_position_y, position_y, window_half_size);
      // }
      states[blockIdx.z].x = CUDART_NAN_F;
      return;
    }
  }
  
  if (threadIdx.x == 0) {
    if (last_step_squared_norm >= 1e-8) {
      // Not converged
      // printf("  Not converged\n");
      states[blockIdx.z].x = CUDART_NAN_F;
    } else {
      // Converged
      states[blockIdx.z] = make_float4(position_x, position_y, factor, bias);
      if (final_cost) {
        final_cost[blockIdx.z] = test_cost_shared;
      }
    }
  }
}

void CallRefineFeatureByMatchingKernel_Refine(
    cudaStream_t stream,
    int feature_count,
    int num_samples,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& rendered_samples,
    cudaTextureObject_t image_texture,
    float4* states,
    float* final_cost,
    int window_half_size,
    int image_width,
    int image_height) {
  #define CALL_KERNEL(block_width_value) \
      constexpr int block_width = block_width_value; \
      dim3 grid_dim(1, 1, feature_count); \
      dim3 block_dim(block_width, 1, 1); \
      RefineFeatureByMatchingKernel_Refine<block_width> \
      <<<grid_dim, block_dim, 0, stream>>>( \
          num_samples, sample_positions, rendered_samples, image_texture, states, \
          final_cost, window_half_size, image_width, image_height);
  if (num_samples > 512) {
    CALL_KERNEL(1024);
  } else if (num_samples > 256) {
    CALL_KERNEL(512);
  } else if (num_samples > 128) {
    CALL_KERNEL(256);
  } else {
    CALL_KERNEL(128);
  }
  #undef CALL_KERNEL
  CHECK_CUDA_NO_ERROR();
}

}

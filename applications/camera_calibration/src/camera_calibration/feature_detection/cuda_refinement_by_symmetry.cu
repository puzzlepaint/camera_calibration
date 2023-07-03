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

#include "camera_calibration/feature_detection/cuda_refinement_by_symmetry.cuh"

#include <cub/cub.cuh>
#include <libvis/cuda/cholesky_solver.h>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_util.h>
#include <libvis/logging.h>
#include <math_constants.h>

#include "camera_calibration/feature_detection/cuda_util.cuh"

namespace vis {

template <int block_width>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ 1024, /*minBlocksPerMultiprocessor*/ 1)
TransformSamplesToPatternSpaceKernel(
    int num_samples,
    CUDABuffer_<float2> sample_positions,
    float window_half_size,
    CUDABuffer_<float2> pattern_sample_positions,
    CUDABuffer_<float> local_pattern_tr_pixel_buffer) {
  // Load the homography (column-major storage, as is Eigen's default)
  __shared__ float h[9];
  if (threadIdx.x < 9) {
    h[threadIdx.x] = local_pattern_tr_pixel_buffer(0, 9 * blockIdx.z + threadIdx.x);
  }
  __syncthreads();
  
  // Transform the samples (scaling with window_half_size and multiplication with local_pattern_tr_pixel)
  for (int sample_index = /*blockIdx.x * block_width +*/ threadIdx.x; sample_index < num_samples; sample_index += block_width) {
    float2 sample = sample_positions(0, sample_index);
    float2 scaled_sample = make_float2(window_half_size * sample.x, window_half_size * sample.y);
    
    float pattern_offset_factor = 1.f / (h[2] * scaled_sample.x + h[5] * scaled_sample.y + h[8]);
    float pattern_offset_x = (h[0] * scaled_sample.x + h[3] * scaled_sample.y + h[6]) * pattern_offset_factor;
    float pattern_offset_y = (h[1] * scaled_sample.x + h[4] * scaled_sample.y + h[7]) * pattern_offset_factor;
    
    pattern_sample_positions(blockIdx.z, sample_index) = make_float2(pattern_offset_x, pattern_offset_y);
  }
}

void CallTransformSamplesToPatternSpaceKernel(
    cudaStream_t stream,
    int feature_count,
    int num_samples,
    const CUDABuffer_<float2>& sample_positions,
    float window_half_size,
    CUDABuffer_<float2>* pattern_sample_positions,
    CUDABuffer_<float> local_pattern_tr_pixel_buffer) {
  #define CALL_KERNEL(block_width_value) \
      constexpr int block_width = block_width_value; \
      dim3 grid_dim(1, 1, feature_count); \
      dim3 block_dim(block_width, 1, 1); \
      TransformSamplesToPatternSpaceKernel<block_width> \
      <<<grid_dim, block_dim, 0, stream>>>( \
          num_samples, sample_positions, window_half_size, *pattern_sample_positions, local_pattern_tr_pixel_buffer);
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


struct GradientsXYCostFunction {
  __forceinline__ __device__ static float ComputeCornerRefinementCost(
      float* pixel_tr_pattern_samples,
      int sample_index,
      const CUDABuffer_<float2>& pattern_sample_positions,
      int image_width,
      int image_height,
      cudaTextureObject_t image) {
    const float& H00 = pixel_tr_pattern_samples[0];
    const float& H01 = pixel_tr_pattern_samples[3];
    const float& H02 = pixel_tr_pattern_samples[6];
    const float& H10 = pixel_tr_pattern_samples[1];
    const float& H11 = pixel_tr_pattern_samples[4];
    const float& H12 = pixel_tr_pattern_samples[7];
    const float& H20 = pixel_tr_pattern_samples[2];
    const float& H21 = pixel_tr_pattern_samples[5];
    
    float2 sample = pattern_sample_positions(blockIdx.z, sample_index);
    
    // Get sample in one direction
    float sample_factor = 1.f / (H20 * sample.x + H21 * sample.y + /*H22*/ 1);
    float sample_pos_x = (H00 * sample.x + H01 * sample.y + H02) * sample_factor + 0.5f;
    float sample_pos_y = (H10 * sample.x + H11 * sample.y + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      return CUDART_NAN_F;
    }
    
    float2 intensity_a = tex2D<float2>(image, sample_pos_x, sample_pos_y);
    
    // Get sample in opposite direction
    sample_factor = 1.f / (H20 * (-sample.x) + H21 * (-sample.y) + /*H22*/ 1);
    sample_pos_x = (H00 * (-sample.x) + H01 * (-sample.y) + H02) * sample_factor + 0.5f;
    sample_pos_y = (H10 * (-sample.x) + H11 * (-sample.y) + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      return CUDART_NAN_F;
    }
    
    float2 intensity_b = tex2D<float2>(image, sample_pos_x, sample_pos_y);
    
    float2 intensity_sum = make_float2(
        intensity_a.x + intensity_b.x,
        intensity_a.y + intensity_b.y);
    return intensity_sum.x * intensity_sum.x + intensity_sum.y * intensity_sum.y;
  }
  
  __forceinline__ __device__ static void GetImageSampleAndJacobian(
      float sample_pos_x,
      float sample_pos_y,
      cudaTextureObject_t image,
      float2* intensity,
      float2* dx,
      float2* dy) {
    *intensity = tex2D<float2>(image, sample_pos_x, sample_pos_y);
    
    int ix = static_cast<int>(::max(0.f, sample_pos_x - 0.5f));
    int iy = static_cast<int>(::max(0.f, sample_pos_y - 0.5f));
    float tx = ::max(0.f, ::min(1.f, sample_pos_x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
    float ty = ::max(0.f, ::min(1.f, sample_pos_y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
    
    float2 top_left = tex2D<float2>(image, ix + 0.5f, iy + 0.5f);
    float2 top_right = tex2D<float2>(image, ix + 1.5f, iy + 0.5f);
    float2 bottom_left = tex2D<float2>(image, ix + 0.5f, iy + 1.5f);
    float2 bottom_right = tex2D<float2>(image, ix + 1.5f, iy + 1.5f);
    
    *dx = make_float2(
        (bottom_right.x - bottom_left.x) * ty + (top_right.x - top_left.x) * (1 - ty),
        (bottom_right.y - bottom_left.y) * ty + (top_right.y - top_left.y) * (1 - ty));
    *dy = make_float2(
        (bottom_right.x - top_right.x) * tx + (bottom_left.x - top_left.x) * (1 - tx),
        (bottom_right.y - top_right.y) * tx + (bottom_left.y - top_left.y) * (1 - tx));
  }
  
  __forceinline__ __device__ static void AddCornerRefinementCostAndJacobian(
      float* pixel_tr_pattern_samples,
      int sample_index,
      const CUDABuffer_<float2>& pattern_sample_positions,
      int image_width,
      int image_height,
      cudaTextureObject_t image,
      float* H,
      float* b,
      float* cost) {
    const float& H00 = pixel_tr_pattern_samples[0];
    const float& H01 = pixel_tr_pattern_samples[3];
    const float& H02 = pixel_tr_pattern_samples[6];
    const float& H10 = pixel_tr_pattern_samples[1];
    const float& H11 = pixel_tr_pattern_samples[4];
    const float& H12 = pixel_tr_pattern_samples[7];
    const float& H20 = pixel_tr_pattern_samples[2];
    const float& H21 = pixel_tr_pattern_samples[5];
    
    float2 sample = pattern_sample_positions(blockIdx.z, sample_index);
    
    // Get sample in one direction
    float sample_factor = 1.f / (H20 * sample.x + H21 * sample.y + /*H22*/ 1);
    float sample_pos_x = (H00 * sample.x + H01 * sample.y + H02) * sample_factor + 0.5f;
    float sample_pos_y = (H10 * sample.x + H11 * sample.y + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      *cost = CUDART_NAN_F;
      return;
    }
    float2 intensity_a, dx_a, dy_a;
    GetImageSampleAndJacobian(sample_pos_x, sample_pos_y, image, &intensity_a, &dx_a, &dy_a);
    
    // Get sample in opposite direction
    sample_factor = 1.f / (H20 * (-sample.x) + H21 * (-sample.y) + /*H22*/ 1);
    sample_pos_x = (H00 * (-sample.x) + H01 * (-sample.y) + H02) * sample_factor + 0.5f;
    sample_pos_y = (H10 * (-sample.x) + H11 * (-sample.y) + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      *cost = CUDART_NAN_F;
      return;
    }
    float2 intensity_b, dx_b, dy_b;
    GetImageSampleAndJacobian(sample_pos_x, sample_pos_y, image, &intensity_b, &dx_b, &dy_b);
    
    float2 residuals = make_float2(
        intensity_a.x + intensity_b.x,
        intensity_a.y + intensity_b.y);
    
    // Sample in first direction
    float term0 = 1 / (H20 * sample.x + H21 * sample.y + 1);
    float term1 = -1 * term0 * term0;
    float term2 = (H00 * sample.x + H01 * sample.y + H02) * term1;
    float term3 = (H10 * sample.x + H11 * sample.y + H12) * term1;
    
    // position_wrt_homography_a[2 x 8] = ...
    const float jac_row0_0 = sample.x * term0;
    const float jac_row0_1 = sample.y * term0;
    const float jac_row0_2 = term0;
    constexpr float jac_row0_3 = 0;
    constexpr float jac_row0_4 = 0;
    constexpr float jac_row0_5 = 0;
    const float jac_row0_6 = sample.x * term2;
    const float jac_row0_7 = sample.y * term2;
    
    constexpr float jac_row1_0 = 0;
    constexpr float jac_row1_1 = 0;
    constexpr float jac_row1_2 = 0;
    const float jac_row1_3 = jac_row0_0;
    const float jac_row1_4 = jac_row0_1;
    const float jac_row1_5 = jac_row0_2;
    const float jac_row1_6 = sample.x * term3;
    const float jac_row1_7 = sample.y * term3;
    
    // Sample in opposite direction
    term0 = 1 / (H20 * (-sample.x) + H21 * (-sample.y) + 1);
    term1 = -1 * term0 * term0;
    term2 = (H00 * (-sample.x) + H01 * (-sample.y) + H02) * term1;
    term3 = (H10 * (-sample.x) + H11 * (-sample.y) + H12) * term1;
    
    // position_wrt_homography_a[2 x 8] = ...
    const float jac2_row0_0 = (-sample.x) * term0;
    const float jac2_row0_1 = (-sample.y) * term0;
    const float jac2_row0_2 = term0;
    constexpr float jac2_row0_3 = 0;
    constexpr float jac2_row0_4 = 0;
    constexpr float jac2_row0_5 = 0;
    const float jac2_row0_6 = (-sample.x) * term2;
    const float jac2_row0_7 = (-sample.y) * term2;
    
    constexpr float jac2_row1_0 = 0;
    constexpr float jac2_row1_1 = 0;
    constexpr float jac2_row1_2 = 0;
    const float jac2_row1_3 = jac2_row0_0;
    const float jac2_row1_4 = jac2_row0_1;
    const float jac2_row1_5 = jac2_row0_2;
    const float jac2_row1_6 = (-sample.x) * term3;
    const float jac2_row1_7 = (-sample.y) * term3;
    
    // Computing gradient_a * position_wrt_homography_a + gradient_b * position_wrt_homography_b.
    // Calling the gradient_a matrix G (and the gradient_b matrix F):
    // [dx_a.x dy_a.x]
    // [dx_a.y dy_a.y]
    const float& G00 = dx_a.x;
    const float& G01 = dy_a.x;
    const float& G10 = dx_a.y;
    const float& G11 = dy_a.y;
    
    const float& F00 = dx_b.x;
    const float& F01 = dy_b.x;
    const float& F10 = dx_b.y;
    const float& F11 = dy_b.y;
    
    constexpr int kDim = 8;
    float jac_row[kDim];
    
    constexpr float weight = 1;
    
    // Row 0 (re-ordering the terms from row-wise to column-wise):
    jac_row[0] = G00 * jac_row0_0 + G01 * jac_row1_0 + F00 * jac2_row0_0 + F01 * jac2_row1_0;
    jac_row[3] = G00 * jac_row0_1 + G01 * jac_row1_1 + F00 * jac2_row0_1 + F01 * jac2_row1_1;
    jac_row[6] = G00 * jac_row0_2 + G01 * jac_row1_2 + F00 * jac2_row0_2 + F01 * jac2_row1_2;
    jac_row[1] = G00 * jac_row0_3 + G01 * jac_row1_3 + F00 * jac2_row0_3 + F01 * jac2_row1_3;
    jac_row[4] = G00 * jac_row0_4 + G01 * jac_row1_4 + F00 * jac2_row0_4 + F01 * jac2_row1_4;
    jac_row[7] = G00 * jac_row0_5 + G01 * jac_row1_5 + F00 * jac2_row0_5 + F01 * jac2_row1_5;
    jac_row[2] = G00 * jac_row0_6 + G01 * jac_row1_6 + F00 * jac2_row0_6 + F01 * jac2_row1_6;
    jac_row[5] = G00 * jac_row0_7 + G01 * jac_row1_7 + F00 * jac2_row0_7 + F01 * jac2_row1_7;
    
    float* cur_H = H;
    #pragma unroll
    for (int i = 0; i < kDim; ++ i) {
      #pragma unroll
      for (int k = i; k < kDim; ++ k) {
        *cur_H += jac_row[i] * weight * jac_row[k];
        ++ cur_H;
      }
      
      b[i] += residuals.x * weight * jac_row[i];
    }
    
    // Row 1 (re-ordering the terms from row-wise to column-wise):
    jac_row[0] = G10 * jac_row0_0 + G11 * jac_row1_0 + F10 * jac2_row0_0 + F11 * jac2_row1_0;
    jac_row[3] = G10 * jac_row0_1 + G11 * jac_row1_1 + F10 * jac2_row0_1 + F11 * jac2_row1_1;
    jac_row[6] = G10 * jac_row0_2 + G11 * jac_row1_2 + F10 * jac2_row0_2 + F11 * jac2_row1_2;
    jac_row[1] = G10 * jac_row0_3 + G11 * jac_row1_3 + F10 * jac2_row0_3 + F11 * jac2_row1_3;
    jac_row[4] = G10 * jac_row0_4 + G11 * jac_row1_4 + F10 * jac2_row0_4 + F11 * jac2_row1_4;
    jac_row[7] = G10 * jac_row0_5 + G11 * jac_row1_5 + F10 * jac2_row0_5 + F11 * jac2_row1_5;
    jac_row[2] = G10 * jac_row0_6 + G11 * jac_row1_6 + F10 * jac2_row0_6 + F11 * jac2_row1_6;
    jac_row[5] = G10 * jac_row0_7 + G11 * jac_row1_7 + F10 * jac2_row0_7 + F11 * jac2_row1_7;
    
    cur_H = H;
    #pragma unroll
    for (int i = 0; i < kDim; ++ i) {
      #pragma unroll
      for (int k = i; k < kDim; ++ k) {
        *cur_H += jac_row[i] * weight * jac_row[k];
        ++ cur_H;
      }
      
      b[i] += residuals.y * weight * jac_row[i];
    }
    
    *cost += residuals.x * residuals.x + residuals.y * residuals.y;  // Actually: 0.5 times this. However, we don't care about (positive) scaling here.
  }
};


struct IntensitiesCostFunction {
  __forceinline__ __device__ static float ComputeCornerRefinementCost(
      float* pixel_tr_pattern_samples,
      int sample_index,
      const CUDABuffer_<float2>& pattern_sample_positions,
      int image_width,
      int image_height,
      cudaTextureObject_t image) {
    const float& H00 = pixel_tr_pattern_samples[0];
    const float& H01 = pixel_tr_pattern_samples[3];
    const float& H02 = pixel_tr_pattern_samples[6];
    const float& H10 = pixel_tr_pattern_samples[1];
    const float& H11 = pixel_tr_pattern_samples[4];
    const float& H12 = pixel_tr_pattern_samples[7];
    const float& H20 = pixel_tr_pattern_samples[2];
    const float& H21 = pixel_tr_pattern_samples[5];
    
    float2 sample = pattern_sample_positions(blockIdx.z, sample_index);
    
    // Get sample in one direction
    float sample_factor = 1.f / (H20 * sample.x + H21 * sample.y + /*H22*/ 1);
    float sample_pos_x = (H00 * sample.x + H01 * sample.y + H02) * sample_factor + 0.5f;
    float sample_pos_y = (H10 * sample.x + H11 * sample.y + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      return CUDART_NAN_F;
    }
    
    float intensity_a = tex2D<float>(image, sample_pos_x, sample_pos_y);
    
    // Get sample in opposite direction
    sample_factor = 1.f / (H20 * (-sample.x) + H21 * (-sample.y) + /*H22*/ 1);
    sample_pos_x = (H00 * (-sample.x) + H01 * (-sample.y) + H02) * sample_factor + 0.5f;
    sample_pos_y = (H10 * (-sample.x) + H11 * (-sample.y) + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      return CUDART_NAN_F;
    }
    
    float intensity_b = tex2D<float>(image, sample_pos_x, sample_pos_y);
    
    float intensity_diff = intensity_a - intensity_b;
    return intensity_diff * intensity_diff;
  }
  
  __forceinline__ __device__ static void GetImageSampleAndJacobian(
      float sample_pos_x,
      float sample_pos_y,
      cudaTextureObject_t image,
      float* intensity,
      float* dx,
      float* dy) {
    *intensity = tex2D<float>(image, sample_pos_x, sample_pos_y);
    
    int ix = static_cast<int>(::max(0.f, sample_pos_x - 0.5f));
    int iy = static_cast<int>(::max(0.f, sample_pos_y - 0.5f));
    float tx = ::max(0.f, ::min(1.f, sample_pos_x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
    float ty = ::max(0.f, ::min(1.f, sample_pos_y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
    
    float top_left = tex2D<float>(image, ix + 0.5f, iy + 0.5f);
    float top_right = tex2D<float>(image, ix + 1.5f, iy + 0.5f);
    float bottom_left = tex2D<float>(image, ix + 0.5f, iy + 1.5f);
    float bottom_right = tex2D<float>(image, ix + 1.5f, iy + 1.5f);
    
    *dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
    *dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  }
  
  __forceinline__ __device__ static void AddCornerRefinementCostAndJacobian(
      float* pixel_tr_pattern_samples,
      int sample_index,
      const CUDABuffer_<float2>& pattern_sample_positions,
      int image_width,
      int image_height,
      cudaTextureObject_t image,
      float* H,
      float* b,
      float* cost) {
    const float& H00 = pixel_tr_pattern_samples[0];
    const float& H01 = pixel_tr_pattern_samples[3];
    const float& H02 = pixel_tr_pattern_samples[6];
    const float& H10 = pixel_tr_pattern_samples[1];
    const float& H11 = pixel_tr_pattern_samples[4];
    const float& H12 = pixel_tr_pattern_samples[7];
    const float& H20 = pixel_tr_pattern_samples[2];
    const float& H21 = pixel_tr_pattern_samples[5];
    
    float2 sample = pattern_sample_positions(blockIdx.z, sample_index);
    
    // Get sample in one direction
    float sample_factor = 1.f / (H20 * sample.x + H21 * sample.y + /*H22*/ 1);
    float sample_pos_x = (H00 * sample.x + H01 * sample.y + H02) * sample_factor + 0.5f;
    float sample_pos_y = (H10 * sample.x + H11 * sample.y + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      *cost = CUDART_NAN_F;
      return;
    }
    float intensity_a, dx_a, dy_a;
    GetImageSampleAndJacobian(sample_pos_x, sample_pos_y, image, &intensity_a, &dx_a, &dy_a);
    
    // Get sample in opposite direction
    sample_factor = 1.f / (H20 * (-sample.x) + H21 * (-sample.y) + /*H22*/ 1);
    sample_pos_x = (H00 * (-sample.x) + H01 * (-sample.y) + H02) * sample_factor + 0.5f;
    sample_pos_y = (H10 * (-sample.x) + H11 * (-sample.y) + H12) * sample_factor + 0.5f;
    if (!ImageContainsPixelCornerConv(sample_pos_x, sample_pos_y, image_width, image_height)) {
      *cost = CUDART_NAN_F;
      return;
    }
    float intensity_b, dx_b, dy_b;
    GetImageSampleAndJacobian(sample_pos_x, sample_pos_y, image, &intensity_b, &dx_b, &dy_b);
    
    float residual = intensity_a - intensity_b;
    
    // Sample in first direction
    float term0 = 1 / (H20 * sample.x + H21 * sample.y + 1);
    float term1 = -1 * term0 * term0;
    float term2 = (H00 * sample.x + H01 * sample.y + H02) * term1;
    float term3 = (H10 * sample.x + H11 * sample.y + H12) * term1;
    
    // position_wrt_homography_a[2 x 8] = ...
    const float jac_row0_0 = sample.x * term0;
    const float jac_row0_1 = sample.y * term0;
    const float jac_row0_2 = term0;
    constexpr float jac_row0_3 = 0;
    constexpr float jac_row0_4 = 0;
    constexpr float jac_row0_5 = 0;
    const float jac_row0_6 = sample.x * term2;
    const float jac_row0_7 = sample.y * term2;
    
    constexpr float jac_row1_0 = 0;
    constexpr float jac_row1_1 = 0;
    constexpr float jac_row1_2 = 0;
    const float jac_row1_3 = jac_row0_0;
    const float jac_row1_4 = jac_row0_1;
    const float jac_row1_5 = jac_row0_2;
    const float jac_row1_6 = sample.x * term3;
    const float jac_row1_7 = sample.y * term3;
    
    // Sample in opposite direction
    term0 = 1 / (H20 * (-sample.x) + H21 * (-sample.y) + 1);
    term1 = -1 * term0 * term0;
    term2 = (H00 * (-sample.x) + H01 * (-sample.y) + H02) * term1;
    term3 = (H10 * (-sample.x) + H11 * (-sample.y) + H12) * term1;
    
    // position_wrt_homography_a[2 x 8] = ...
    const float jac2_row0_0 = (-sample.x) * term0;
    const float jac2_row0_1 = (-sample.y) * term0;
    const float jac2_row0_2 = term0;
    constexpr float jac2_row0_3 = 0;
    constexpr float jac2_row0_4 = 0;
    constexpr float jac2_row0_5 = 0;
    const float jac2_row0_6 = (-sample.x) * term2;
    const float jac2_row0_7 = (-sample.y) * term2;
    
    constexpr float jac2_row1_0 = 0;
    constexpr float jac2_row1_1 = 0;
    constexpr float jac2_row1_2 = 0;
    const float jac2_row1_3 = jac2_row0_0;
    const float jac2_row1_4 = jac2_row0_1;
    const float jac2_row1_5 = jac2_row0_2;
    const float jac2_row1_6 = (-sample.x) * term3;
    const float jac2_row1_7 = (-sample.y) * term3;
    
    // Computing gradient_a * position_wrt_homography_a + gradient_b * position_wrt_homography_b.
    // Calling the gradient_a matrix G (and the gradient_b matrix F):
    // [dx_a.x dy_a.x]
    // [dx_a.y dy_a.y]
    const float& G00 = dx_a;
    const float& G01 = dy_a;
    
    const float& F00 = dx_b;
    const float& F01 = dy_b;
    
    constexpr int kDim = 8;
    float jac_row[kDim];
    
    constexpr float weight = 1;
    
    // (Re-ordering the terms from row-wise to column-wise):
    jac_row[0] = G00 * jac_row0_0 + G01 * jac_row1_0 - F00 * jac2_row0_0 - F01 * jac2_row1_0;
    jac_row[3] = G00 * jac_row0_1 + G01 * jac_row1_1 - F00 * jac2_row0_1 - F01 * jac2_row1_1;
    jac_row[6] = G00 * jac_row0_2 + G01 * jac_row1_2 - F00 * jac2_row0_2 - F01 * jac2_row1_2;
    jac_row[1] = G00 * jac_row0_3 + G01 * jac_row1_3 - F00 * jac2_row0_3 - F01 * jac2_row1_3;
    jac_row[4] = G00 * jac_row0_4 + G01 * jac_row1_4 - F00 * jac2_row0_4 - F01 * jac2_row1_4;
    jac_row[7] = G00 * jac_row0_5 + G01 * jac_row1_5 - F00 * jac2_row0_5 - F01 * jac2_row1_5;
    jac_row[2] = G00 * jac_row0_6 + G01 * jac_row1_6 - F00 * jac2_row0_6 - F01 * jac2_row1_6;
    jac_row[5] = G00 * jac_row0_7 + G01 * jac_row1_7 - F00 * jac2_row0_7 - F01 * jac2_row1_7;
    
    float* cur_H = H;
    #pragma unroll
    for (int i = 0; i < kDim; ++ i) {
      #pragma unroll
      for (int k = i; k < kDim; ++ k) {
        *cur_H += jac_row[i] * weight * jac_row[k];
        ++ cur_H;
      }
      
      b[i] += residual * weight * jac_row[i];
    }
    
    *cost += residual * residual;  // Actually: 0.5 times this. However, we don't care about (positive) scaling here.
  }
};


template <int block_width, typename CostFunction>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ 512, /*minBlocksPerMultiprocessor*/ 1)
RefineCheckerboardCornerPositionCUDAKernel_Refine(
    int num_samples,
    CUDABuffer_<float2> pattern_sample_positions,
    cudaTextureObject_t image_texture,
    CUDABuffer_<float> pixel_tr_pattern_samples,
    float* final_cost,
    int window_half_size,
    int image_width,
    int image_height) {
  // #define LOG_CONDITION (threadIdx.x == 0 && blockIdx.z == 0 /*pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6) > 110 && pixel_tr_pattern_samples(0, 9 * blockIdx.z + 7) > 140 && pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6) < 120 && pixel_tr_pattern_samples(0, 9 * blockIdx.z + 7) < 150*/)
  
  constexpr int block_height = 1;
  
  // Number of optimized variables
  constexpr int kDim = 8;
  // Half a kDim x kDim matrix, including the diagonal
  constexpr int kHSize = kDim * (kDim + 1) / 2;
  
  if (::isnan(/*original_position_x*/ pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6))) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  __shared__ union {
    typename BlockReduceFloat::TempStorage temp_storage;
    struct {
      float H[kHSize];
      float b[kDim];
    } h_and_b;
  } shared;
  
  __shared__ float test_cost_shared;
  __shared__ float buffer1[kDim];
  __shared__ float buffer2[kDim];
  float* cur_pixel_tr_pattern_samples = buffer1;
  float* test_pixel_tr_pattern_samples = buffer2;
  if (threadIdx.x < kDim) {
    cur_pixel_tr_pattern_samples[threadIdx.x] = pixel_tr_pattern_samples(0, 9 * blockIdx.z + threadIdx.x);
  }
  
  float lambda = -1;
  float last_step_squared_norm = -1;
  __shared__ bool applied_update;
  
  float H_local[kHSize];
  float b_local[kDim];
  float cost_local;
  __shared__ float temp[2];
  
  constexpr int kMaxIterationCount = 30;
  for (int iteration = 0; iteration < kMaxIterationCount; ++ iteration) {
    // Clear accumulation buffers
    #pragma unroll
    for (int i = 0; i < kDim; ++ i) {
      b_local[i] = 0;
    }
    #pragma unroll
    for (int i = 0; i < kHSize; ++ i) {
      H_local[i] = 0;
    }
    cost_local = 0;
    
    // Compute cost and Jacobian
    __syncthreads();  // for cur_pixel_tr_pattern_samples and for BlockReduceCoeffs
//     if (LOG_CONDITION) {
//       printf("INPUT CUDA:\n%f, %f, %f,\n%f, %f, %f,\n%f, %f, %f;\n",
//              cur_pixel_tr_pattern_samples[0], cur_pixel_tr_pattern_samples[3], cur_pixel_tr_pattern_samples[6],
//              cur_pixel_tr_pattern_samples[1], cur_pixel_tr_pattern_samples[4], cur_pixel_tr_pattern_samples[7],
//              cur_pixel_tr_pattern_samples[2], cur_pixel_tr_pattern_samples[5], pixel_tr_pattern_samples(0, 9 * blockIdx.z + 8));
//     }
    
    for (int sample_index = threadIdx.x; sample_index < num_samples; sample_index += block_width) {
      CostFunction::AddCornerRefinementCostAndJacobian(
          cur_pixel_tr_pattern_samples, sample_index,
          pattern_sample_positions, image_width, image_height, image_texture, H_local, b_local, &cost_local);
    }
    
    #pragma unroll
    for (int i = 0; i < kDim; ++ i) {
      float result = BlockReduceFloat(shared.temp_storage).Sum(b_local[i]);
      if (threadIdx.x == 0) {
        temp[i & 1] = result;
      }
      __syncthreads();
      b_local[i] = temp[i & 1];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kHSize; ++ i) {
      float result = BlockReduceFloat(shared.temp_storage).Sum(H_local[i]);
      if (threadIdx.x == 0) {
        temp[i & 1] = result;
      }
      __syncthreads();
      H_local[i] = temp[i & 1];
    }
    // NOTE: cost_local will only be correct for threadIdx.x == 0.
    cost_local = BlockReduceFloat(shared.temp_storage).Sum(cost_local);
    __syncthreads();  // for re-use of shared below
    
//     if (LOG_CONDITION) {
//       printf("OUTPUT CUDA:\ncost = %f\n", cost_local);
//       printf("b:\n");
//       for (int i = 0; i < kDim; ++ i) {
//         printf("%f\n", b_local[i]);
//       }
//       printf("H:\n");
//       for (int i = 0; i < kHSize; ++ i) {
//         printf("%f\n", H_local[i]);
//       }
//     }
    
//     if (LOG_CONDITION) {
//       printf("[%i] Iteration %i | cost: %f\n", blockIdx.z, iteration, cost_local);
//     }
    
    // Initialize lambda from the values of H on the diagonal?
    if (lambda < 0) {
      lambda = 0.001f * (1.f / kDim) * (
          H_local[0] +
          H_local[8] +
          H_local[8 + 7] +
          H_local[8 + 7 + 6] +
          H_local[8 + 7 + 6 + 5] +
          H_local[8 + 7 + 6 + 5 + 4] +
          H_local[8 + 7 + 6 + 5 + 4 + 3] +
          H_local[8 + 7 + 6 + 5 + 4 + 3 + 2]);
    }
    
    applied_update = false;
    float old_lambda = 0;
    for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
      // TODO: Split this up among threads?
      H_local[0] += (lambda - old_lambda);
      H_local[8] += (lambda - old_lambda);
      H_local[8 + 7] += (lambda - old_lambda);
      H_local[8 + 7 + 6] += (lambda - old_lambda);
      H_local[8 + 7 + 6 + 5] += (lambda - old_lambda);
      H_local[8 + 7 + 6 + 5 + 4] += (lambda - old_lambda);
      H_local[8 + 7 + 6 + 5 + 4 + 3] += (lambda - old_lambda);
      H_local[8 + 7 + 6 + 5 + 4 + 3 + 2] += (lambda - old_lambda);
      old_lambda = lambda;
      
      if (threadIdx.x < kDim) {
        shared.h_and_b.b[threadIdx.x] = b_local[threadIdx.x];
      } else if (threadIdx.x < kDim + kHSize) {
        shared.h_and_b.H[threadIdx.x - kDim] = H_local[threadIdx.x - kDim];
      }
      __syncthreads();  // make all threads see the updated shared variables in SolveWithParallelCholesky()
      
      // Solve for the update (in-place; the result is in shared.h_and_b.b afterwards).
      SolveWithParallelCholesky<kDim>(shared.h_and_b.H, shared.h_and_b.b);
      
      // Test whether the update improves the cost.
      if (threadIdx.x < kDim) {
        test_pixel_tr_pattern_samples[threadIdx.x] = cur_pixel_tr_pattern_samples[threadIdx.x] - shared.h_and_b.b[threadIdx.x];
      }
      float dx = -shared.h_and_b.b[6];
      float dy = -shared.h_and_b.b[7];
      __syncthreads();  // for test_pixel_tr_pattern_samples and BlockReduceFloat and applied_update
      
      float test_cost = 0;
      for (int sample_index = threadIdx.x; sample_index < num_samples; sample_index += block_width) {
        test_cost += CostFunction::ComputeCornerRefinementCost(
            test_pixel_tr_pattern_samples, sample_index,
            pattern_sample_positions, image_width, image_height, image_texture);
      }
      // NOTE: test_cost will only be correct for threadIdx.x == 0.
      test_cost = BlockReduceFloat(shared.temp_storage).Sum(test_cost);
//       if (LOG_CONDITION) {
//         printf("  [%i] LM iteration %i | lambda: %f, dx: %f, dy: %f, test cost: %f\n", blockIdx.z, lm_iteration, lambda, dx, dy, test_cost);
//       }
      
      if (threadIdx.x == 0) {
        test_cost_shared = test_cost;
        if (test_cost < cost_local) {
          applied_update = true;
        }
      }
      
      __syncthreads();  // for applied_update, test_cost_shared
      
      if (::isnan(test_cost_shared)) {
        // Position went out of bounds
//         if (LOG_CONDITION) {
//           printf("  [%i] Position out of bounds\n", blockIdx.z);
//         }
        pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6) = CUDART_NAN_F;
        return;
      } else if (applied_update) {
        lambda *= 0.5f;
        last_step_squared_norm = dx * dx + dy * dy;
        
        // Swap cur_pixel_tr_pattern_samples and test_pixel_tr_pattern_samples
        // to accept the test values
        float* temp = cur_pixel_tr_pattern_samples;
        cur_pixel_tr_pattern_samples = test_pixel_tr_pattern_samples;
        test_pixel_tr_pattern_samples = temp;
        
        break;
      } else {
        lambda *= 2.f;
      }
    }
    
    if (!applied_update) {
      // Cannot find an update that improves the cost. Treat this as converged.
//       if (LOG_CONDITION) {
//         printf("  [%i] Cannot find update, assuming convergence. Position: (%f, %f)\n", blockIdx.z, cur_pixel_tr_pattern_samples[6], cur_pixel_tr_pattern_samples[7]);
//       }
      if (threadIdx.x < kDim) {
        pixel_tr_pattern_samples(0, 9 * blockIdx.z + threadIdx.x) = cur_pixel_tr_pattern_samples[threadIdx.x];
      }
      if (final_cost && threadIdx.x == 32) {
        final_cost[blockIdx.z] = test_cost_shared;
      }
      return;
    }
    
    // NOTE: This does not necessarily need to be checked, as the CPU code will
    //       later use a stricter threshold anyway.
//     // Check for divergence.
//     if (fabs(pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6) - cur_pixel_tr_pattern_samples[6]) >= window_half_size ||
//         fabs(pixel_tr_pattern_samples(0, 9 * blockIdx.z + 7) - cur_pixel_tr_pattern_samples[7]) >= window_half_size) {
//       // The result is probably not the originally intended corner,
//       // since it is not within the original search window.
//       if (LOG_CONDITION) {
//         printf("  [%i] Position too far away from start. original_position_x: %f, position_x: %f, original_position_y: %f, position_y: %f, window_half_size: %i\n",
//                blockIdx.z, pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6), cur_pixel_tr_pattern_samples[6], pixel_tr_pattern_samples(0, 9 * blockIdx.z + 7), cur_pixel_tr_pattern_samples[7], window_half_size);
//       }
//       pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6) = CUDART_NAN_F;
//       return;
//     }
  }
  
  if (last_step_squared_norm >= 1e-4f) {
    // Not converged
//     if (LOG_CONDITION) {
//       printf("  [%i] Not converged\n", blockIdx.z);
//     }
    if (threadIdx.x == 0) {
      pixel_tr_pattern_samples(0, 9 * blockIdx.z + 6) = CUDART_NAN_F;
    }
  } else {
    // Converged
//     if (LOG_CONDITION) {
//       printf("  [%i] Converged. Position: (%f, %f)\n", blockIdx.z, cur_pixel_tr_pattern_samples[6], cur_pixel_tr_pattern_samples[7]);
//     }
    if (threadIdx.x < kDim) {
      pixel_tr_pattern_samples(0, 9 * blockIdx.z + threadIdx.x) = cur_pixel_tr_pattern_samples[threadIdx.x];
    }
    if (final_cost && threadIdx.x == 32) {
      final_cost[blockIdx.z] = test_cost_shared;
    }
  }
}

void CallRefineCheckerboardCornerPositionCUDAKernel_Refine(
    cudaStream_t stream,
    int feature_count,
    int num_samples,
    const CUDABuffer_<float2>& pattern_sample_positions,
    cudaTextureObject_t image_texture,
    FeatureRefinement refinement_type,
    CUDABuffer_<float>* pixel_tr_pattern_samples,
    float* final_cost,
    int window_half_size,
    int image_width,
    int image_height) {
  if (refinement_type == FeatureRefinement::GradientsXY) {
    #define CALL_KERNEL(block_width_value) \
        constexpr int block_width = block_width_value; \
        dim3 grid_dim(1, 1, feature_count); \
        dim3 block_dim(block_width, 1, 1); \
        RefineCheckerboardCornerPositionCUDAKernel_Refine<block_width, GradientsXYCostFunction> \
        <<<grid_dim, block_dim, 0, stream>>>( \
            num_samples, pattern_sample_positions, image_texture, *pixel_tr_pattern_samples, \
            final_cost, window_half_size, image_width, image_height);
    // TODO: This did not work because too many resources were requested (most
    //       likely shared memory).
    /*if (num_samples > 512) {
      CALL_KERNEL(1024);
    } else*/ if (num_samples > 256) {
      CALL_KERNEL(512);
    } else if (num_samples > 128) {
      CALL_KERNEL(256);
    } else {
      CALL_KERNEL(128);
    }
    #undef CALL_KERNEL
  } else if (refinement_type == FeatureRefinement::Intensities) {
    #define CALL_KERNEL(block_width_value) \
        constexpr int block_width = block_width_value; \
        dim3 grid_dim(1, 1, feature_count); \
        dim3 block_dim(block_width, 1, 1); \
        RefineCheckerboardCornerPositionCUDAKernel_Refine<block_width, IntensitiesCostFunction> \
        <<<grid_dim, block_dim, 0, stream>>>( \
            num_samples, pattern_sample_positions, image_texture, *pixel_tr_pattern_samples, \
            final_cost, window_half_size, image_width, image_height);
    // TODO: This did not work because too many resources were requested (most
    //       likely shared memory).
    /*if (num_samples > 512) {
      CALL_KERNEL(1024);
    } else*/ if (num_samples > 256) {
      CALL_KERNEL(512);
    } else if (num_samples > 128) {
      CALL_KERNEL(256);
    } else {
      CALL_KERNEL(128);
    }
    #undef CALL_KERNEL
  } else {
    LOG(FATAL) << "This refinement type is not supported here yet.";
  }
  CHECK_CUDA_NO_ERROR();
}

}

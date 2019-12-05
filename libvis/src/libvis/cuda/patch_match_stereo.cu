// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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

#include "libvis/cuda/patch_match_stereo.cuh"

#include <math_constants.h>

#include "libvis/cuda/cuda_auto_tuner.h"
#include "libvis/cuda/cuda_unprojection_lookup.cuh"
#include "libvis/cuda/cuda_util.cuh"
#include "libvis/cuda/cuda_util.h"
#include "libvis/cuda/patch_match_stereo_cost.cuh"

namespace vis {

__global__ void MedianFilterDepthMap3x3CUDAKernel(
    int context_radius,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out,
    CUDABuffer_<float> costs,
    CUDABuffer_<float> costs_out,
    CUDABuffer_<float> second_best_costs,
    CUDABuffer_<float> second_best_costs_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;  // TODO: De-duplicate with above
  
  if (x >= context_radius && y >= context_radius &&
      x < inv_depth_map.width() - context_radius && y < inv_depth_map.height() - context_radius) {
    // Collect valid depth values of 3x3 neighborhood
    int count = 1;
    float inv_depths[9];
    float cost[9];
    float second_best_cost[9];
    
    inv_depths[0] = inv_depth_map(y, x);
    if (inv_depths[0] == kInvalidInvDepth) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      costs_out(y, x) = CUDART_NAN_F;
      second_best_costs_out(y, x) = CUDART_NAN_F;
      return;
    }
    cost[0] = costs(y, x);
    second_best_cost[0] = second_best_costs(y, x);
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      if (y + dy < context_radius || y + dy >= inv_depth_map.height() - context_radius) {
        continue;
      }
      
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if (dy == 0 && dx == 0) {
          continue;
        }
        
        if (x + dx < context_radius || x + dx >= inv_depth_map.width() - context_radius) {
          continue;
        }
        
        float inv_depth = inv_depth_map(y + dy, x + dx);
        if (inv_depth != kInvalidInvDepth) {
          inv_depths[count] = inv_depth;
          cost[count] = costs(y + dy, x + dx);
          second_best_cost[count] = second_best_costs(y + dy, x + dx);
          ++ count;
        }
      }
    }
    
    // Sort depth values up to the middle of the maximum count
    for (int i = 0; i <= 4; ++ i) {
      for (int k = i + 1; k < 9; ++ k) {
        if (k < count && inv_depths[i] > inv_depths[k]) {
          // Swap.
          float temp = inv_depths[i];
          inv_depths[i] = inv_depths[k];
          inv_depths[k] = temp;
          
          temp = cost[i];
          cost[i] = cost[k];
          cost[k] = temp;
          
          temp = second_best_cost[i];
          second_best_cost[i] = second_best_cost[k];
          second_best_cost[k] = temp;
        }
      }
    }
    
    // Assign the median
    if (count % 2 == 1) {
      inv_depth_map_out(y, x) = inv_depths[count / 2];
      costs_out(y, x) = cost[count / 2];
      second_best_costs_out(y, x) = second_best_cost[count / 2];
    } else {
      // For disambiguation in the even-count case, use the value which is
      // closer to the average of the two middle values.
      float average = 0.5f * (inv_depths[count / 2 - 1] + inv_depths[count / 2]);
      if (fabs(average - inv_depths[count / 2 - 1]) <
          fabs(average - inv_depths[count / 2])) {
        inv_depth_map_out(y, x) = inv_depths[count / 2 - 1];
        costs_out(y, x) = cost[count / 2 - 1];
        second_best_costs_out(y, x) = second_best_cost[count / 2 - 1];
      } else {
        inv_depth_map_out(y, x) = inv_depths[count / 2];
        costs_out(y, x) = cost[count / 2];
        second_best_costs_out(y, x) = second_best_cost[count / 2];
      }
    }
  } else if (x < inv_depth_map_out.width() && y < inv_depth_map_out.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
    costs_out(y, x) = CUDART_NAN_F;
    second_best_costs_out(y, x) = CUDART_NAN_F;
  }
}

void MedianFilterDepthMap3x3CUDA(
    cudaStream_t stream,
    int context_radius,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<float>* costs,
    CUDABuffer_<float>* costs_out,
    CUDABuffer_<float>* second_best_costs,
    CUDABuffer_<float>* second_best_costs_out) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      MedianFilterDepthMap3x3CUDAKernel,
      32, 32,
      inv_depth_map->width(), inv_depth_map->height(),
      0, stream,
      /* kernel parameters */
      context_radius,
      *inv_depth_map,
      *inv_depth_map_out,
      *costs,
      *costs_out,
      *second_best_costs,
      *second_best_costs_out);
  CHECK_CUDA_NO_ERROR();
}


__global__ void BilateralFilterCUDAKernel(
    float denom_xy,
    float denom_value,
    int radius,
    int radius_squared,
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;  // TODO: De-duplicate with above
  
  if (x < inv_depth_map_out.width() && y < inv_depth_map_out.height()) {
    const float center_value = inv_depth_map(y, x);
    if (center_value == kInvalidInvDepth) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    // Bilateral filtering.
    float sum = 0;
    float weight = 0;
    
    const int min_y = max(static_cast<int>(0), static_cast<int>(y - radius));
    const int max_y = min(static_cast<int>(inv_depth_map_out.height() - 1), static_cast<int>(y + radius));
    for (int sample_y = min_y; sample_y <= max_y; ++ sample_y) {
      const int dy = sample_y - y;
      
      const int min_x = max(static_cast<int>(0), static_cast<int>(x - radius));
      const int max_x = min(static_cast<int>(inv_depth_map_out.width() - 1), static_cast<int>(x + radius));
      for (int sample_x = min_x; sample_x <= max_x; ++ sample_x) {
        const int dx = sample_x - x;
        
        const int grid_distance_squared = dx * dx + dy * dy;
        if (grid_distance_squared > radius_squared) {
          continue;
        }
        
        const float sample = inv_depth_map(sample_y, sample_x);
        if (sample == kInvalidInvDepth) {
          continue;
        }
        
        float value_distance_squared = center_value - sample;
        value_distance_squared *= value_distance_squared;
        float w = exp(-grid_distance_squared / denom_xy + -value_distance_squared / denom_value);
        sum += w * sample;
        weight += w;
      }
    }
    
    inv_depth_map_out(y, x) = (weight == 0) ? kInvalidInvDepth : (sum / weight);
  }
}

void BilateralFilterCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value,
    float radius_factor,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out) {
  CHECK_CUDA_NO_ERROR();
  
  int radius = radius_factor * sigma_xy + 0.5f;
  
  CUDA_AUTO_TUNE_2D(
      BilateralFilterCUDAKernel,
      32, 32,
      inv_depth_map_out->width(), inv_depth_map_out->height(),
      0, stream,
      /* kernel parameters */
      2.0f * sigma_xy * sigma_xy,
      2.0f * sigma_value * sigma_value,
      radius,
      radius * radius,
      inv_depth_map,
      *inv_depth_map_out);
  CHECK_CUDA_NO_ERROR();
}


__global__ void FillHolesCUDAKernel(
    CUDABuffer_<float> inv_depth_map,
    CUDABuffer_<float> inv_depth_map_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;  // TODO: De-duplicate with above
  
  if (x < inv_depth_map_out.width() && y < inv_depth_map_out.height()) {
    const float center_inv_depth = inv_depth_map(y, x);
    if (center_inv_depth != kInvalidInvDepth ||
        x < 1 ||
        y < 1 ||
        x >= inv_depth_map.width() - 1 ||
        y >= inv_depth_map.height() - 1) {
      inv_depth_map_out(y, x) = center_inv_depth;
      return;
    }
    
    // Get the average depth of the neighbor pixels.
    float sum = 0;
    int count = 0;
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        
        float inv_depth = inv_depth_map(y + dy, x + dx);
        if (inv_depth != kInvalidInvDepth) {
          sum += inv_depth;
          ++ count;
        }
      }
    }
    
    float avg_inv_depth = sum / count;
    
    // Fill in this pixel if there are at least a minimum number of valid
    // neighbor pixels nearby which have similar depth.
    constexpr float kSimilarDepthFactorThreshold = 1.01f;  // TODO: Make parameter
    constexpr int kMinSimilarPixelsForFillIn = 6;  // TODO: Make parameter
    
    sum = 0;
    count = 0;
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        
        float inv_depth = inv_depth_map(y + dy, x + dx);
        if (inv_depth != kInvalidInvDepth) {
          float factor = inv_depth / avg_inv_depth;
          if (factor < 1) {
            factor = 1 / factor;
          }
          
          if (factor <= kSimilarDepthFactorThreshold) {
            sum += inv_depth;
            ++ count;
          }
        }
      }
    }
    
    inv_depth_map_out(y, x) = (count >= kMinSimilarPixelsForFillIn) ? (sum / count) : kInvalidInvDepth;
  }
}

void FillHolesCUDA(
    cudaStream_t stream,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      FillHolesCUDAKernel,
      32, 32,
      inv_depth_map_out->width(), inv_depth_map_out->height(),
      0, stream,
      /* kernel parameters */
      inv_depth_map,
      *inv_depth_map_out);
  CHECK_CUDA_NO_ERROR();
}

}

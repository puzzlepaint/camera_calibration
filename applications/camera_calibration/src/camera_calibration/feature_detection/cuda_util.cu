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

#include "camera_calibration/feature_detection/cuda_util.cuh"

#include <cub/cub.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_util.h>
#include <libvis/logging.h>
#include <math_constants.h>

namespace vis {

template <int block_width, int block_height>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ 32 * 32, /*minBlocksPerMultiprocessor*/ 1)
ComputeGradientImageCUDAKernel(
    CUDABuffer_<u8> image,
    int width,
    int height,
    CUDABuffer_<float2> gradmag_image) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < width &&
      y < height) {
    int mx = ::max(0, x - 1);
    int px = ::min(width - 1, x + 1);
    
    int my = ::max(0, y - 1);
    int py = ::min(height - 1, y + 1);
    
    float dx = (image(y, px) - static_cast<float>(image(y, mx))) / (px - mx);
    float dy = (image(py, x) - static_cast<float>(image(my, x))) / (py - my);
    
    gradmag_image(y, x) = make_float2(dx, dy);
  }
}

void ComputeGradientImageCUDA(
    const CUDABuffer_<u8>& image,
    int width,
    int height,
    CUDABuffer_<float2>* gradmag_image) {
  CHECK_CUDA_NO_ERROR();
  
  constexpr cudaStream_t stream = 0;
  
  constexpr int block_width = 32;
  constexpr int block_height = 32;
  
  dim3 grid_dim(GetBlockCount(width, block_width),
                GetBlockCount(height, block_height));
  dim3 block_dim(block_width, block_height);
  ComputeGradientImageCUDAKernel<block_width, block_height>
  <<<grid_dim, block_dim, 0, stream>>>(
      image,
      width,
      height,
      *gradmag_image);
  CHECK_CUDA_NO_ERROR();
}

}

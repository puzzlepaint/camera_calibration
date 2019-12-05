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
#include <cub/util_type.cuh>
#include <libvis/cuda/cuda_buffer.cuh>
#include <libvis/libvis.h>

namespace vis {

void CallRefineFeatureByMatchingKernel_RenderSamples(
    cudaStream_t stream,
    int feature_count,
    int num_star_segments,
    int num_samples,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& local_pattern_tr_pixel_buffer,
    int window_half_size,
    const CUDABuffer_<float>& rendered_samples);

void CallRefineFeatureByMatchingKernel_InitFactorAndBias(
    cudaStream_t stream,
    int feature_count,
    int num_samples,
    const CUDABuffer_<float2>& sample_positions,
    const CUDABuffer_<float>& rendered_samples,
    cudaTextureObject_t image,
    float4* states,
    int window_half_size);

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
    int image_height);

}

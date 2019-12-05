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

#include "camera_calibration/feature_detection/cuda_refinement_by_matching.h"

#include "camera_calibration/feature_detection/cuda_refinement_by_matching.cuh"

namespace vis {

void RefineFeatureByMatchingCUDA(
    cudaStream_t stream,
    int num_star_segments,
    int num_samples,
    const CUDABuffer<float2>& sample_positions,
    cudaTextureObject_t texture,
    int window_half_size,
    int image_width,
    int image_height,
    const vector<FeatureDetection>& predicted_features,
    vector<Vec2f>* refined_positions,
    int cost_buffer_size,
    CUDABuffer<float4>* state_buffer,
    CUDABuffer<float>* rendered_samples,  // must be of width sample_positions, height feature_count
    CUDABuffer<float>* local_pattern_tr_pixel_buffer) {  // must be at least of size 9 * feature_count
  int feature_count = predicted_features.size();
  CHECK_GE(cost_buffer_size, feature_count);
  
  // Upload the predicted positions and the local homographies to the GPU
  vector<float4> state_cpu(feature_count);
  vector<Mat3f> homographies(feature_count);
  for (int i = 0; i < feature_count; ++ i) {
    state_cpu[i] = make_float4(predicted_features[i].position.x(), predicted_features[i].position.y(), -1, -1);
    homographies[i] = predicted_features[i].local_pixel_tr_pattern.inverse();
  }
  state_buffer->UploadPartAsync(0, feature_count * sizeof(float4), stream, state_cpu.data());
  local_pattern_tr_pixel_buffer->UploadPartAsync(0, feature_count * 9 * sizeof(float), stream, reinterpret_cast<float*>(homographies.data()));
  
  // Using the local homography, render the known pattern at the samples. Each
  // sample defines a pixel offset from the feature position.
  CallRefineFeatureByMatchingKernel_RenderSamples(
      stream,
      feature_count,
      num_star_segments,
      num_samples,
      sample_positions.ToCUDA(),
      local_pattern_tr_pixel_buffer->ToCUDA(),
      window_half_size,
      rendered_samples->ToCUDA());
  
  // Initialize factor and bias.
  CallRefineFeatureByMatchingKernel_InitFactorAndBias(
      stream,
      feature_count,
      num_samples,
      sample_positions.ToCUDA(),
      rendered_samples->ToCUDA(),
      texture,
      state_buffer->ToCUDA().address(),
      window_half_size);
  
  // Locally optimize
  CallRefineFeatureByMatchingKernel_Refine(
      stream, feature_count, num_samples, sample_positions.ToCUDA(),
      rendered_samples->ToCUDA(), texture, state_buffer->ToCUDA().address(),
      nullptr, window_half_size, image_width, image_height);
  
  // Output result
  refined_positions->resize(feature_count);
  state_buffer->DownloadPartAsync(0, feature_count * sizeof(float4), stream, state_cpu.data());
  for (int i = 0; i < feature_count; ++ i) {
    if (state_cpu[i].z <= 0) {
      refined_positions->at(i).x() = numeric_limits<float>::quiet_NaN();
    } else {
      refined_positions->at(i) = Vec2f(state_cpu[i].x, state_cpu[i].y);
    }
  }
}

}

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

#include "camera_calibration/feature_detection/cuda_refinement_by_symmetry.h"

#include "camera_calibration/feature_detection/cuda_refinement_by_symmetry.cuh"

namespace vis {

void RefineFeatureBySymmetryCUDA(
    cudaStream_t stream,
    int num_samples,
    const CUDABuffer<float2>& sample_positions,
    CUDABuffer<float2>* pattern_sample_positions,
    cudaTextureObject_t texture,
    FeatureRefinement refinement_type,
    int window_half_size,
    int image_width,
    int image_height,
    const vector<FeatureDetection>& predicted_features,
    const vector<Vec2f>& predicted_positions,
    vector<FeatureDetection>* refined_features,
    int cost_buffer_size,
    cub::KeyValuePair<int, float>* cost_buffer,
    CUDABuffer<float>* local_pattern_tr_pixel_buffer) {  // must be of size 9 * feature_count and contain the data already
  int feature_count = predicted_positions.size();
  CHECK_EQ(predicted_features.size(), feature_count);
  CHECK_GE(cost_buffer_size, feature_count);
  
  // Transform the samples from pixel space to pattern space (with (0, 0) being
  // the feature whose position is estimated) using the initial homography estimate.
  CallTransformSamplesToPatternSpaceKernel(
      stream, feature_count, num_samples, sample_positions.ToCUDA(), window_half_size, &pattern_sample_positions->ToCUDA(), local_pattern_tr_pixel_buffer->ToCUDA());
  
  // Optimize the homography locally that maps the local pattern coordinates to
  // the pixel coordinates.
  // TODO: Avoid repeated upload of the homographies to the GPU: Can we upload them
  //       in this "transformed" version in the first place (in
  //       cuda_refinement_by_matching.cc) and keep in local_pattern_tr_pixel_buffer?
  vector<Mat3f> homographies(feature_count);
  for (int i = 0; i < feature_count; ++ i) {
    Mat3f pixel_tr_pattern_samples = predicted_features[i].local_pixel_tr_pattern;
    Mat3f local_to_global_mapping = Mat3f::Identity();
    local_to_global_mapping(0, 2) = predicted_positions[i].x();
    local_to_global_mapping(1, 2) = predicted_positions[i].y();
    pixel_tr_pattern_samples = local_to_global_mapping * pixel_tr_pattern_samples;
    // Normalize pixel_tr_pattern_samples such that its bottom-right element is one.
    homographies[i] = pixel_tr_pattern_samples / pixel_tr_pattern_samples(2, 2);
  }
  local_pattern_tr_pixel_buffer->UploadPartAsync(0, feature_count * 9 * sizeof(float), stream, reinterpret_cast<float*>(homographies.data()));
  
  // Locally optimize the positions
  CallRefineCheckerboardCornerPositionCUDAKernel_Refine(
      stream, feature_count, num_samples, pattern_sample_positions->ToCUDA(), texture,
      refinement_type, &local_pattern_tr_pixel_buffer->ToCUDA(),
      reinterpret_cast<float*>(cost_buffer),
      window_half_size, image_width, image_height);
  
  // Output result
  // TODO: Do not read back the complete homographies, but only the translation parts, if possible. Maybe each homography item should have its own row in the CUDA buffer.
  vector<Mat3f> output_homographies(feature_count);
  local_pattern_tr_pixel_buffer->DownloadPartAsync(0, feature_count * 9 * sizeof(float), stream, reinterpret_cast<float*>(output_homographies.data()));
  
  float final_cost[feature_count];
  cudaMemcpyAsync(final_cost, cost_buffer, feature_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  for (int i = 0; i < feature_count; ++ i) {
    (*refined_features)[i].final_cost = final_cost[i];
    (*refined_features)[i].position = Vec2f(output_homographies[i](0, 2), output_homographies[i](1, 2));
  }
}

}

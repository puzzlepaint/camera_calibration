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
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/eigen.h>
#include <libvis/libvis.h>

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"
#include "camera_calibration/feature_detection/feature_refinement.h"

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
    CUDABuffer<float>* local_pattern_tr_pixel_buffer);  // must be of size 9 * feature_count and contain the data already

}

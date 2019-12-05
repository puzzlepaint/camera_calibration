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
#include "libvis/cuda/patch_match_stereo_util.cuh"

namespace vis {

__constant__ float kSamplesCUDA[kNumSamples][2];

void InitPatchMatchSamples() {
  static bool initialized = false;
  if (!initialized) {
    // Shift the samples such that they have zero mean (such that we actually
    // estimate depths for the centers of pixels rather than for the random mean
    // of the samples).
    // TODO: Make sure that the original data is already centered.
    float sum_x = 0;
    float sum_y = 0;
    for (int s = 0; s < kNumSamples; ++ s) {
      sum_x += kSamples[s][0];
      sum_y += kSamples[s][1];
    }
    float mean_x = sum_x / kNumSamples;
    float mean_y = sum_y / kNumSamples;
    
    float centered_samples[kNumSamples][2];
    for (int s = 0; s < kNumSamples; ++ s) {
      centered_samples[s][0] = kSamples[s][0] - mean_x;
      centered_samples[s][1] = kSamples[s][1] - mean_y;
    }
    
    cudaMemcpyToSymbol(kSamplesCUDA, centered_samples, kNumSamples * 2 * sizeof(float));
    cudaDeviceSynchronize();
    
    initialized = true;
  }
}

}

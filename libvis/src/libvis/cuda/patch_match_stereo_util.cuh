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


#pragma once

#include <cuda_runtime.h>

#include "libvis/cuda/cuda_buffer.cuh"
#include "libvis/cuda/cuda_matrix.cuh"
#include "libvis/cuda/cuda_unprojection_lookup.cuh"
#include "libvis/cuda/pixel_corner_projector.cuh"
#include "libvis/cuda/patch_match_stereo.cuh"
#include "libvis/cuda/patch_match_stereo_samples.cuh"
#include "libvis/libvis.h"

namespace vis {

/// Groups common parameters to CUDA kernels for stereo estimation, for a single stereo image.
struct StereoParametersSingleCUDA {
  StereoParametersSingleCUDA(const StereoParametersSingle& p)
      : context_radius(p.context_radius),
        reference_unprojection_lookup(p.reference_unprojection_lookup),
        reference_image(p.reference_image),
        reference_texture(p.reference_texture),
        stereo_camera(p.stereo_camera),
        mask(p.mask),
        stereo_tr_reference(p.stereo_tr_reference),
        stereo_image(p.stereo_image),
        inv_depth_map(p.inv_depth_map),
        normals(p.normals),
        costs(p.costs) {}
  
  int context_radius;
  CUDAUnprojectionLookup2D_ reference_unprojection_lookup;
  CUDABuffer_<u8> reference_image;
  cudaTextureObject_t reference_texture;
  PixelCornerProjector_ stereo_camera;
  CUDABuffer_<u8> mask;
  
  CUDAMatrix3x4 stereo_tr_reference;
  cudaTextureObject_t stereo_image;
  
  CUDABuffer_<float> inv_depth_map;
  CUDABuffer_<char2> normals;
  CUDABuffer_<float> costs;
};

/// Groups common parameters to CUDA kernels for stereo estimation, for multiple stereo images.
struct StereoParametersMultiCUDA {
  StereoParametersMultiCUDA(const StereoParametersMulti& p)
      : context_radius(p.context_radius),
        reference_unprojection_lookup(p.reference_unprojection_lookup),
        reference_image(p.reference_image),
        reference_texture(p.reference_texture),
        stereo_camera(p.stereo_camera),
        mask(p.mask),
        num_stereo_images(p.num_stereo_images),
        stereo_tr_reference(p.stereo_tr_reference),
        stereo_images(p.stereo_images),
        inv_depth_map(p.inv_depth_map),
        normals(p.normals) {}
  
  int context_radius;
  CUDAUnprojectionLookup2D_ reference_unprojection_lookup;
  CUDABuffer_<u8> reference_image;
  cudaTextureObject_t reference_texture;
  PixelCornerProjector_ stereo_camera;
  CUDABuffer_<u8> mask;
  
  int num_stereo_images;
  CUDAMatrix3x4* stereo_tr_reference;
  cudaTextureObject_t* stereo_images;
  
  CUDABuffer_<float> inv_depth_map;
  CUDABuffer_<char2> normals;
};

}

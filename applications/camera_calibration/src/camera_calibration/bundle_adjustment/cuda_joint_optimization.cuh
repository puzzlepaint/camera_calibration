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
#include <libvis/cuda/cuda_buffer.cuh>
#include <libvis/cuda/cuda_matrix.cuh>
#include <libvis/libvis.h>

namespace vis {

typedef float PCGScalar;

/// Groups the dataset and state parameters to shorten the parameter lists for
/// CUDA kernel calls. All values relate to a single camera; if there are
/// multiple cameras, then multiple CUDADatasetAndState structs are created.
/// Only images that are used (image_used is true in the Dataset) are included here.
/// All pointers here point to device (GPU) memory.
struct CUDADatasetAndState {
  /// Index of the camera that this struct refers to.
  u32 camera_index;
  
  /// Delta for numeric derivatives.
  PCGScalar numerical_diff_delta;
  
  // --- Dataset ---
  
  /// Number of feature observations (summed over all images of this camera).
  int num_features;
  
  /// For each feature observation, indicates the index of the image which it comes from. Indexed by: [feature_observation_index].
  u16* features_image;
  
  /// For each feature observation, gives its x coordinate in the image (in pixel corner coordinate origin convention). Indexed by: [feature_observation_index].
  float* features_x;
  
  /// For each feature observation, gives its y coordinate in the image (in pixel corner coordinate origin convention). Indexed by: [feature_observation_index].
  float* features_y;
  
  /// For each feature observation, gives the index of its corresponding 3D point in the points array. Indexed by: [feature_observation_index].
  u16* features_index;
  
  /// For each feature observation, stores its residual value in x-direction. Indexed by: [feature_observation_index].
  float* features_residual_x;
  
  /// For each feature observation, stores its residual value in y-direction. Indexed by: [feature_observation_index].
  float* features_residual_y;
  
  // --- State ---
  
  /// Coordinates of each 3D point in the state. Indexed by: [point_index].
  float* points[3];
  
  /// Pose of each used image. Indexed by: [sequential_image_index].
  CUDAMatrix3x4* image_tr_global;
  
  /// Rig pose quaternions (4 parameters per entry). Indexed by: [camera_index].
  float* camera_q_rig;
  
  /// Camera poses (4 quaternion + 3 translation parameters per entry). Indexed by: [sequential_image_index].
  float* rig_tr_global;
  
  /// First index of the first rig_tr_global unknown in the state.
  u32 rig_tr_global_start_index;
  
  /// Whether the camera_tr_rig poses are included in the state (and thus are optimized).
  bool are_camera_tr_rig_in_state;
  
  /// First index of the first camera_tr_rig unknown in the state.
  u32 camera_tr_rig_start_index;
  
  /// First index of the first point unknown in the state.
  u32 points_start_index;
  
  /// First index of the first intrinsics unknown in the state, for the camera that this struct refers to.
  /// NOTE: This thus does *not* necessarily refer to the first intrinsics index for all cameras!
  u32 intrinsic_start_index;
};

template <class Model>
void PCGCompareCostCUDA(
    cudaStream_t stream,
    const CUDADatasetAndState& s,
    const Model& model,
    CUDABuffer_<PCGScalar>* pcg_cost,
    CUDABuffer_<PCGScalar>* pcg_relative_cost);

template <class Model>
void PCGInitCUDA(
    cudaStream_t stream,
    const CUDADatasetAndState& s,
    const Model& model,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M,
    CUDABuffer_<PCGScalar>* pcg_cost);

void PCGInit2CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    PCGScalar lambda,
    const CUDABuffer_<PCGScalar>& pcg_r,
    const CUDABuffer_<PCGScalar>& pcg_M,
    CUDABuffer_<PCGScalar>* pcg_delta,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n);

template <class Model>
void PCGStep1CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    const CUDADatasetAndState& s,
    const Model& model,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_alpha_d);

void PCGStep2CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    PCGScalar lambda,
    CUDABuffer_<PCGScalar>* pcg_r,
    const CUDABuffer_<PCGScalar>& pcg_M,
    CUDABuffer_<PCGScalar>* pcg_delta,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n,
    CUDABuffer_<PCGScalar>* pcg_alpha_d,
    CUDABuffer_<PCGScalar>* pcg_beta_n);

void PCGStep3CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n,
    CUDABuffer_<PCGScalar>* pcg_beta_n);

}

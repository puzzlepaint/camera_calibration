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

#include "camera_calibration/bundle_adjustment/cuda_joint_optimization.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_util.h>
#include <math_constants.h>

#include "camera_calibration/bundle_adjustment/joint_optimization_jacobians.h"
#include "camera_calibration/cuda/cuda_matrix.cuh"
#include "camera_calibration/cuda/cuda_util.cuh"
#include "camera_calibration/models/cuda_central_generic_model.cuh"

namespace vis {

/*
 * Schema for accumulator classes:
 * 
 * struct Accumulator {
 *   /// Called if the residuals belonging to this thread are invalid.
 *   /// This is only called once and before any SetJacobianComponent() call,
 *   /// except for calls to SetJacobianComponent_AllThreadsSameIndex() with valid == false.
 *   __forceinline__ __device__ void SetResidualsInvalid(PCGScalar* features_residual_x, u32 feature_index);
 *   
 *   /// Called if the residuals belonging to this thread are valid.
 *   /// This is only called once and before any SetJacobianComponent() call,
 *   /// except for calls to SetJacobianComponent_AllThreadsSameIndex() with valid == false.
 *   __forceinline__ __device__ void SetResiduals(PCGScalar residual_x, PCGScalar residual_y, PCGScalar* features_residual_x, PCGScalar* features_residual_y, u32 feature_index);
 *   
 *   /// Sets the values of one column in the [2 x N] Jacobian of the pixel position
 *   /// wrt. the unknowns. I.e., value_x corresponds to the x-residual (row 0) and value_y
 *   /// to the y-residual (row 1).
 *   /// 
 *   /// This version is called if there are no possible conflicts between
 *   /// different threads in the kernel call, i.e., for a given thread, no other
 *   /// thread in the thread grid possibly writes to the same index.
 *   /// 
 *   /// In case the residuals are invalid (SetResidualsInvalid() has been called
 *   /// before), this function must not be called.
 *   __forceinline__ __device__ void SetJacobianComponent_ThreadConflictFree(u32 index, PCGScalar jac_x, PCGScalar jac_y);
 *   
 *   /// See SetJacobianComponent_ThreadConflictFree(). This version is used
 *   /// if all threads in the thread block write to the same index.
 *   /// 
 *   /// This variant of SetJacobianComponent() is called both if the residuals
 *   /// are valid and if they are invalid to enable block-wide operations.
 *   /// 
 *   /// NOTE: If the temp_storage is used before, a __syncthreads() has to be done.
 *   __forceinline__ __device__ void SetJacobianComponent_AllThreadsSameIndex(u32 index, PCGScalar jac_x, PCGScalar jac_y, bool valid);
 *   
 *   /// See SetJacobianComponent_ThreadConflictFree(). This version is used
 *   /// if none of the other two versions applies.
 *   /// 
 *   /// In case the residuals are invalid (SetResidualsInvalid() has been called
 *   /// before), this function must not be called.
 *   __forceinline__ __device__ void SetJacobianComponent_RandomThreadConflicts(u32 index, PCGScalar jac_x, PCGScalar jac_y);
 * };
 */

constexpr PCGScalar kHuberWeight = 1.0;  // TODO: Make parameter

constexpr int kResidualJacobianBlockSize = 256;

template<int block_width, int block_height, bool compute_jacobians, bool are_camera_tr_rig_in_state, class Model, class Accumulator>
__device__ void ComputeResidualAndJacobian(
    bool valid,
    u32 feature_index,
    CUDADatasetAndState& s,
    Model& model,
    Accumulator& accumulator) {
  u16 point_index = s.features_index[feature_index];
  float3 point = make_float3(s.points[0][point_index], s.points[1][point_index], s.points[2][point_index]);
  
  u16 image_index = s.features_image[feature_index];
  float3 local_point = s.image_tr_global[image_index] * point;
  
  float2 pixel = make_float2(0.5f * (model.calibration_min_x() + model.calibration_max_x() + 1),
                             0.5f * (model.calibration_min_y() + model.calibration_max_y() + 1));
  if (!model.ProjectWithInitialEstimate(local_point, &pixel)) {
    if (valid) {
      accumulator.SetResidualsInvalid(s.features_residual_x, feature_index);
    }
    valid = false;
  }
  
  if (valid) {
    accumulator.SetResiduals(
        pixel.x - s.features_x[feature_index],
        pixel.y - s.features_y[feature_index],
        s.features_residual_x, s.features_residual_y, feature_index);
  }
  
  if (compute_jacobians) {
    // Compute Jacobian wrt. image pose, optionally camera_tr_rig, and point position [2 x (6 + (rig ? 6 : 0) + 3)].
    // Residual: Project(exp(delta) * image_tr_pattern * pattern_point) - measurement
    
    // Compute Jacobian as follows:
    //   (d pixel) / (d local_point)                  [2 x 3], numerical
    // * (d local_point) / (d pose and global_point)  [3 x (7 + (rig ? 7 : 0) + 3)], analytical
    
    // Numerical part:
    CUDAMatrix<PCGScalar, 2, 3> pixel_wrt_local_point;
    const PCGScalar kDelta = s.numerical_diff_delta * (model.is_central_camera_model() ? Norm(local_point) : 0.1);
    #pragma unroll
    for (int dimension = 0; dimension < 3; ++ dimension) {
      float3 offset_point = local_point;
      *(&offset_point.x + dimension) += kDelta;
      float2 offset_pixel = pixel;
      if (!model.ProjectWithInitialEstimate(offset_point, &offset_pixel)) {
        valid = false;
        break;
      }
      
      pixel_wrt_local_point(0, dimension) = (offset_pixel.x - pixel.x) / kDelta;
      pixel_wrt_local_point(1, dimension) = (offset_pixel.y - pixel.y) / kDelta;
    }
    
    // Analytical part:
    CUDAMatrix<PCGScalar, 3, 7 + 7 + 3> local_point_wrt_poses_and_global_point;
    if (are_camera_tr_rig_in_state) {
      ComputeRigJacobian(
          s.camera_q_rig[4 * s.camera_index + 0], s.camera_q_rig[4 * s.camera_index + 1], s.camera_q_rig[4 * s.camera_index + 2], s.camera_q_rig[4 * s.camera_index + 3],
          point.x, point.y, point.z,
          s.rig_tr_global[7 * image_index + 0], s.rig_tr_global[7 * image_index + 1], s.rig_tr_global[7 * image_index + 2], s.rig_tr_global[7 * image_index + 3],
          s.rig_tr_global[7 * image_index + 4], s.rig_tr_global[7 * image_index + 5], s.rig_tr_global[7 * image_index + 6],
          local_point_wrt_poses_and_global_point.row(0),
          local_point_wrt_poses_and_global_point.row(1),
          local_point_wrt_poses_and_global_point.row(2));
    } else {
      // NOTE: The first row expects image_q_global values. Thus, here we assume
      //       that rig_q_global == image_q_global, i.e., the camera_q_rig
      //       transformation is identity.
      ComputeJacobian(
          s.rig_tr_global[7 * image_index + 0], s.rig_tr_global[7 * image_index + 1], s.rig_tr_global[7 * image_index + 2], s.rig_tr_global[7 * image_index + 3],
          point.x, point.y, point.z,
          local_point_wrt_poses_and_global_point.row(0),
          local_point_wrt_poses_and_global_point.row(1),
          local_point_wrt_poses_and_global_point.row(2));
    }
    
    CUDAMatrix<PCGScalar, 2, 6> pose_jacobian;
    CUDAMatrix<PCGScalar, 2, 6> rig_jacobian;
    CUDAMatrix<PCGScalar, 2, 3> point_jacobian;
    
    if (are_camera_tr_rig_in_state) {
      // local_point_wrt_poses_and_global_point contains the Jacobian wrt.:
      // - rig_tr_global (indices 0 .. 6)
      // - camera_tr_rig (indices 7 .. 13)
      // - global_point (indices 14 .. 16)
      
      CUDAMatrix<PCGScalar, 4, 3> camera_q_rig_wrt_update;
      QuaternionJacobianWrtLocalUpdate(s.camera_q_rig[4 * s.camera_index + 0], s.camera_q_rig[4 * s.camera_index + 1], s.camera_q_rig[4 * s.camera_index + 2], s.camera_q_rig[4 * s.camera_index + 3], &camera_q_rig_wrt_update);
      
      CUDAMatrix<PCGScalar, 4, 3> rig_q_global_wrt_update;
      QuaternionJacobianWrtLocalUpdate(s.rig_tr_global[7 * image_index + 0], s.rig_tr_global[7 * image_index + 1], s.rig_tr_global[7 * image_index + 2], s.rig_tr_global[7 * image_index + 3], &rig_q_global_wrt_update);
      
      CUDAMatrix<PCGScalar, 2, 4> temp;
      MatrixMultiply(temp, pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<4>(0));
      MatrixMultiply(pose_jacobian.cols<3>(0), temp, rig_q_global_wrt_update);
      MatrixMultiply(pose_jacobian.cols<3>(3), pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<3>(0 + 4));
      
      MatrixMultiply(temp, pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<4>(7));
      MatrixMultiply(rig_jacobian.cols<3>(0), temp, camera_q_rig_wrt_update);
      MatrixMultiply(rig_jacobian.cols<3>(3), pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<3>(7 + 4));
      
      MatrixMultiply(point_jacobian, pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<3>(14));
    } else {
      // local_point_wrt_poses_and_global_point contains the Jacobian wrt.:
      // - rig_tr_global (indices 0 .. 6)
      // - global_point (indices 7 .. 9)
      
      // NOTE: Here, we assume that rig_q_global == image_q_global, i.e., the
      //       camera_q_rig transformation is identity.
      CUDAMatrix<PCGScalar, 4, 3> quaternion_wrt_update;  // derived in derive_jacobians.py
      QuaternionJacobianWrtLocalUpdate(s.rig_tr_global[7 * image_index + 0], s.rig_tr_global[7 * image_index + 1], s.rig_tr_global[7 * image_index + 2], s.rig_tr_global[7 * image_index + 3], &quaternion_wrt_update);
      
      CUDAMatrix<PCGScalar, 2, 4> temp;
      MatrixMultiply(temp, pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<4>(0));
      MatrixMultiply(pose_jacobian.cols<3>(0), temp, quaternion_wrt_update);
      MatrixMultiply(pose_jacobian.cols<3>(3), pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<3>(4));
      
      MatrixMultiply(point_jacobian, pixel_wrt_local_point, local_point_wrt_poses_and_global_point.cols<3>(7));
    }
    
    // Get the model Jacobian
    constexpr int num_intrinsic_variables = Model::IntrinsicsJacobianSize;
    CUDAMatrix<u32, num_intrinsic_variables, 1> grid_update_indices;
    CUDAMatrix<PCGScalar, 2, num_intrinsic_variables> intrinsic_jac;
    if (!model.ProjectionJacobianWrtIntrinsics(
        local_point,
        pixel,
        s.numerical_diff_delta,
        grid_update_indices.data(),
        intrinsic_jac.row(0),
        intrinsic_jac.row(1))) {
      valid = false;
    }
    
    // Accumulate Jacobians:
    if (are_camera_tr_rig_in_state) {
      if (valid) {
        for (int i = 0; valid && i < 6; ++ i) {
          accumulator.SetJacobianComponent_RandomThreadConflicts(
              s.rig_tr_global_start_index + 6 * image_index + i,
              pose_jacobian(0, i),
              pose_jacobian(1, i));
        }
      }
      
      for (int i = 0; i < 6; ++ i) {
        accumulator.SetJacobianComponent_AllThreadsSameIndex(
            s.camera_tr_rig_start_index + s.camera_index * 6 + i,
            rig_jacobian(0, i),
            rig_jacobian(1, i),
            valid);
      }
      
      if (valid) {
        for (int i = 0; i < 3; ++ i) {
          accumulator.SetJacobianComponent_RandomThreadConflicts(
              s.points_start_index + point_index * 3 + i,
              point_jacobian(0, i),
              point_jacobian(1, i));
        }
      }
    } else {
      if (valid) {
        for (int i = 0; i < 6; ++ i) {
          accumulator.SetJacobianComponent_RandomThreadConflicts(
              s.rig_tr_global_start_index + 6 * image_index + i,
              pose_jacobian(0, i),
              pose_jacobian(1, i));
        }
        
        for (int i = 0; i < 3; ++ i) {
          accumulator.SetJacobianComponent_RandomThreadConflicts(
              s.points_start_index + point_index * 3 + i,
              point_jacobian(0, i),
              point_jacobian(1, i));
        }
      }
    }
    
    if (valid) {
      for (int i = 0; i < num_intrinsic_variables; ++ i) {
        accumulator.SetJacobianComponent_RandomThreadConflicts(
            s.intrinsic_start_index + grid_update_indices(i),
            intrinsic_jac(0, i),
            intrinsic_jac(1, i));
      }
    }
  }
}


template <int block_width, int block_height>
struct PCGCompareCostAccumulator {
  __forceinline__ __device__ PCGCompareCostAccumulator(
      typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* temp_storage)
      : temp_storage_(temp_storage) {}
  
  __forceinline__ __device__ void SetResidualsInvalid(PCGScalar* features_residual_x, u32 feature_index) {
    // No need to do anything, as these residuals do not matter for the comparison
  }
  
  __forceinline__ __device__ void SetResiduals(PCGScalar residual_x, PCGScalar residual_y, PCGScalar* features_residual_x, PCGScalar* features_residual_y, u32 feature_index) {
    PCGScalar this_cost = ComputeHuberCost(residual_x, residual_y, kHuberWeight);
    cost_ += this_cost;
    
    if (::isnan(features_residual_x[feature_index])) {
      // These residuals were invalid for the other cost, so ignore them for the comparison
      return;
    }
    
    // Both in the old and the new state, the residuals are valid. Compare them.
    PCGScalar other_cost = ComputeHuberCost(features_residual_x[feature_index], features_residual_y[feature_index], kHuberWeight);
    relative_cost_ += this_cost - other_cost;
  }
  
  __forceinline__ __device__ void SetJacobianComponent_ThreadConflictFree(u32 /*index*/, PCGScalar /*jac_x*/, PCGScalar /*jac_y*/) {}
  
  __forceinline__ __device__ void SetJacobianComponent_AllThreadsSameIndex(u32 /*index*/, PCGScalar /*jac_x*/, PCGScalar /*jac_y*/, bool /*valid*/) {}
  
  __forceinline__ __device__ void SetJacobianComponent_RandomThreadConflicts(u32 /*index*/, PCGScalar /*jac_x*/, PCGScalar /*jac_y*/) {}
  
  PCGScalar cost_ = 0;
  PCGScalar relative_cost_ = 0;
  
  typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* temp_storage_;
};

template<int block_width, class Model, bool are_camera_tr_rig_in_state>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ kResidualJacobianBlockSize, /*minBlocksPerMultiprocessor*/ 1)
PCGCompareCostCUDAKernel(
    CUDADatasetAndState s,
    Model model,
    CUDABuffer_<PCGScalar> pcg_cost,
    CUDABuffer_<PCGScalar> pcg_relative_cost) {
  unsigned int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = feature_index < s.num_features;
  if (!valid) {
    feature_index = 0;
  }
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage temp_storage;
  
  PCGCompareCostAccumulator<block_width, block_height> accumulator(&temp_storage);
  ComputeResidualAndJacobian<block_width, block_height, /*compute_jacobians*/ false, are_camera_tr_rig_in_state>(valid, feature_index, s, model, accumulator);
  
  __syncthreads();
  BlockedAtomicSum<block_width, block_height>(
      &pcg_cost(0, 0),
      accumulator.cost_,
      valid,
      &temp_storage);
  __syncthreads();
  BlockedAtomicSum<block_width, block_height>(
      &pcg_relative_cost(0, 0),
      accumulator.relative_cost_,
      valid,
      &temp_storage);
}

template <class Model>
void PCGCompareCostCUDA(
    cudaStream_t stream,
    const CUDADatasetAndState& s,
    const Model& model,
    CUDABuffer_<PCGScalar>* pcg_cost,
    CUDABuffer_<PCGScalar>* pcg_relative_cost) {
  CHECK_CUDA_NO_ERROR();
  if (s.num_features == 0) {
    return;
  }
  
  constexpr int block_width = kResidualJacobianBlockSize;
  dim3 grid_dim(GetBlockCount(s.num_features, block_width));
  dim3 block_dim(block_width);
  bool are_camera_tr_rig_in_state = s.are_camera_tr_rig_in_state;
  COMPILE_OPTION(
      are_camera_tr_rig_in_state,
      PCGCompareCostCUDAKernel<block_width, Model, _are_camera_tr_rig_in_state>
      <<<grid_dim, block_dim, 0, stream>>>(
          s,
          model,
          *pcg_cost,
          *pcg_relative_cost););
  CHECK_CUDA_NO_ERROR();
}

template
void PCGCompareCostCUDA<CUDACentralGenericModel>(
    cudaStream_t stream,
    const CUDADatasetAndState& s,
    const CUDACentralGenericModel& model,
    CUDABuffer_<PCGScalar>* pcg_cost,
    CUDABuffer_<PCGScalar>* pcg_relative_cost);


template <int block_width, int block_height>
struct PCGInitAccumulator {
  __forceinline__ __device__ PCGInitAccumulator(
      CUDABuffer_<PCGScalar>* pcg_r,
      CUDABuffer_<PCGScalar>* pcg_M,
      typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* temp_storage)
      : pcg_r_(pcg_r),
        pcg_M_(pcg_M),
        temp_storage_(temp_storage) {}
  
  __forceinline__ __device__ void SetResidualsInvalid(PCGScalar* features_residual_x, u32 feature_index) {
    features_residual_x[feature_index] = CUDART_NAN_F;
    // features_residual_y[feature_index] = CUDART_NAN_F;
  }
  
  __forceinline__ __device__ void SetResiduals(PCGScalar residual_x, PCGScalar residual_y, PCGScalar* features_residual_x, PCGScalar* features_residual_y, u32 feature_index) {
    features_residual_x[feature_index] = residual_x;
    features_residual_y[feature_index] = residual_y;
    
    // Cache residuals and weights
    weight_ = ComputeHuberWeight(residual_x, residual_y, kHuberWeight);
    weighted_residual_x_ = weight_ * residual_x;
    weighted_residual_y_ = weight_ * residual_y;
    
    cost_ += ComputeHuberCost(residual_x, residual_y, kHuberWeight);
  }
  
  __forceinline__ __device__ void SetJacobianComponent_ThreadConflictFree(u32 index, PCGScalar jac_x, PCGScalar jac_y) {
    (*pcg_r_)(0, index) -= jac_x * weighted_residual_x_ + jac_y * weighted_residual_y_;
    (*pcg_M_)(0, index) += jac_x * weight_ * jac_x + jac_y * weight_ * jac_y;
  }
  
  __forceinline__ __device__ void SetJacobianComponent_AllThreadsSameIndex(u32 index, PCGScalar jac_x, PCGScalar jac_y, bool valid) {
    BlockedAtomicSum<block_width, block_height>(
        &(*pcg_r_)(0, index),
        -1 * jac_x * weighted_residual_x_ +
        -1 * jac_y * weighted_residual_y_,
        valid, temp_storage_);
    
    __syncthreads();
    
    BlockedAtomicSum<block_width, block_height>(
        &(*pcg_M_)(0, index),
        jac_x * weight_ * jac_x +
        jac_y * weight_ * jac_y,
        valid, temp_storage_);
    
    __syncthreads();
  }
  
  __forceinline__ __device__ void SetJacobianComponent_RandomThreadConflicts(u32 index, PCGScalar jac_x, PCGScalar jac_y) {
    atomicAddFloatOrDouble(
        &(*pcg_r_)(0, index),
        -1 * jac_x * weighted_residual_x_ +
        -1 * jac_y * weighted_residual_y_);
    
    atomicAddFloatOrDouble(
        &(*pcg_M_)(0, index),
        jac_x * weight_ * jac_x +
        jac_y * weight_ * jac_y);
  }
  
  PCGScalar weight_;
  PCGScalar weighted_residual_x_;
  PCGScalar weighted_residual_y_;
  
  PCGScalar cost_ = 0;
  
  CUDABuffer_<PCGScalar>* pcg_r_;
  CUDABuffer_<PCGScalar>* pcg_M_;
  typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* temp_storage_;
};

template<int block_width, class Model, bool are_camera_tr_rig_in_state>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ kResidualJacobianBlockSize, /*minBlocksPerMultiprocessor*/ 1)
PCGInitCUDAKernel(
    CUDADatasetAndState s,
    Model model,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_M,
    CUDABuffer_<PCGScalar> pcg_cost) {
  unsigned int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = feature_index < s.num_features;
  if (!valid) {
    feature_index = 0;
  }
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage temp_storage;
  
  PCGInitAccumulator<block_width, block_height> accumulator(&pcg_r, &pcg_M, &temp_storage);
  ComputeResidualAndJacobian<block_width, block_height, /*compute_jacobians*/ true, are_camera_tr_rig_in_state>(valid, feature_index, s, model, accumulator);
  
  __syncthreads();
  BlockedAtomicSum<block_width, block_height>(
      &pcg_cost(0, 0),
      accumulator.cost_,
      valid,
      &temp_storage);
}

template <class Model>
void PCGInitCUDA(
    cudaStream_t stream,
    const CUDADatasetAndState& s,
    const Model& model,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M,
    CUDABuffer_<PCGScalar>* pcg_cost) {
  CHECK_CUDA_NO_ERROR();
  if (s.num_features == 0) {
    return;
  }
  
  constexpr int block_width = kResidualJacobianBlockSize;
  dim3 grid_dim(GetBlockCount(s.num_features, block_width));
  dim3 block_dim(block_width);
  bool are_camera_tr_rig_in_state = s.are_camera_tr_rig_in_state;
  COMPILE_OPTION(
      are_camera_tr_rig_in_state,
      PCGInitCUDAKernel<block_width, Model, _are_camera_tr_rig_in_state>
      <<<grid_dim, block_dim, 0, stream>>>(
          s,
          model,
          *pcg_r,
          *pcg_M,
          *pcg_cost););
  CHECK_CUDA_NO_ERROR();
}

template
void PCGInitCUDA<CUDACentralGenericModel>(
    cudaStream_t stream,
    const CUDADatasetAndState& s,
    const CUDACentralGenericModel& model,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M,
    CUDABuffer_<PCGScalar>* pcg_cost);


template<int block_width>
__global__ void PCGInit2CUDAKernel(
    u32 unknown_count,
    PCGScalar lambda,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_M,
    CUDABuffer_<PCGScalar> pcg_delta,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_n) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage temp_storage;
  
  PCGScalar alpha_term;
  
  if (unknown_index < unknown_count) {
    pcg_g(0, unknown_index) = 0;
    
    // p_0 = M^-1 r_0
    // The addition of lambda is also handled here.
    PCGScalar r_value = pcg_r(0, unknown_index);
    PCGScalar p_value = r_value / (pcg_M(0, unknown_index) + lambda);
    pcg_p(0, unknown_index) = p_value;
    
    // delta_0 = 0
    pcg_delta(0, unknown_index) = 0;
    
    // alpha_n_0 = r_0^T p_0
    alpha_term = r_value * p_value;
  }
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_alpha_n(0, 0),
      alpha_term,
      unknown_index < unknown_count,
      &temp_storage);
}

void PCGInit2CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    PCGScalar lambda,
    const CUDABuffer_<PCGScalar>& pcg_r,
    const CUDABuffer_<PCGScalar>& pcg_M,
    CUDABuffer_<PCGScalar>* pcg_delta,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n) {
  CHECK_CUDA_NO_ERROR();
  if (unknown_count == 0) {
    return;
  }
  
  cudaMemsetAsync(pcg_alpha_n->address(), 0, 1 * sizeof(PCGScalar), stream);
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGInit2CUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      lambda,
      pcg_r,
      pcg_M,
      *pcg_delta,
      *pcg_g,
      *pcg_p,
      *pcg_alpha_n);
  CHECK_CUDA_NO_ERROR();
}


template <int block_width, int block_height>
struct PCGStep1SumAccumulator {
  __forceinline__ __device__ PCGStep1SumAccumulator(const CUDABuffer_<PCGScalar>& pcg_p)
      : pcg_p_(pcg_p) {}
  
  __forceinline__ __device__ void SetResidualsInvalid(PCGScalar* features_residual_x, u32 feature_index) {
    // Do nothing
  }
  
  __forceinline__ __device__ void SetResiduals(PCGScalar residual_x, PCGScalar residual_y, PCGScalar* features_residual_x, PCGScalar* features_residual_y, u32 feature_index) {
    // Cache weights
    weight_ = ComputeHuberWeight(residual_x, residual_y, kHuberWeight);
  }
  
  __forceinline__ __device__ void SetJacobianComponent_ThreadConflictFree(u32 index, PCGScalar jac_x, PCGScalar jac_y) {
    PCGScalar p = pcg_p_(0, index);
    sum_x_ += jac_x * p;
    sum_y_ += jac_y * p;
  }
  
  __forceinline__ __device__ void SetJacobianComponent_AllThreadsSameIndex(u32 index, PCGScalar jac_x, PCGScalar jac_y, bool valid) {
    if (valid) {
      PCGScalar p = pcg_p_(0, index);
      sum_x_ += jac_x * p;
      sum_y_ += jac_y * p;
    }
  }
  
  __forceinline__ __device__ void SetJacobianComponent_RandomThreadConflicts(u32 index, PCGScalar jac_x, PCGScalar jac_y) {
    PCGScalar p = pcg_p_(0, index);
    sum_x_ += jac_x * p;
    sum_y_ += jac_y * p;
  }
  
  PCGScalar sum_x_ = 0;  // holds the result of (J * p) for the row of the first residual.
  PCGScalar sum_y_ = 0;  // holds the result of (J * p) for the row of the second residual.
  
  PCGScalar weight_ = 0;
  
  const CUDABuffer_<PCGScalar>& pcg_p_;
};

template <int block_width, int block_height>
struct PCGStep1ResolveAccumulator {
  __forceinline__ __device__ PCGStep1ResolveAccumulator(
      PCGScalar sum_x,
      PCGScalar sum_y,
      CUDABuffer_<PCGScalar>* pcg_g,
      typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* temp_storage)
      : sum_x_(sum_x),
        sum_y_(sum_y),
        pcg_g_(pcg_g),
        temp_storage_(temp_storage) {}
  
  __forceinline__ __device__ void SetResidualsInvalid(PCGScalar* features_residual_x, u32 feature_index) {
    // Do nothing
  }
  
  __forceinline__ __device__ void SetResiduals(PCGScalar residual_x, PCGScalar residual_y, PCGScalar* features_residual_x, PCGScalar* features_residual_y, u32 feature_index) {
    // Do nothing
  }
  
  __forceinline__ __device__ void SetJacobianComponent_ThreadConflictFree(u32 index, PCGScalar jac_x, PCGScalar jac_y) {
    (*pcg_g_)(0, index) += jac_x * sum_x_ + jac_y * sum_y_;
  }
  
  __forceinline__ __device__ void SetJacobianComponent_AllThreadsSameIndex(u32 index, PCGScalar jac_x, PCGScalar jac_y, bool valid) {
    BlockedAtomicSum<block_width, block_height>(
        &(*pcg_g_)(0, index),
        jac_x * sum_x_ + jac_y * sum_y_,
        valid, temp_storage_);
    __syncthreads();
  }
  
  __forceinline__ __device__ void SetJacobianComponent_RandomThreadConflicts(u32 index, PCGScalar jac_x, PCGScalar jac_y) {
    atomicAddFloatOrDouble(
        &(*pcg_g_)(0, index),
        jac_x * sum_x_ + jac_y * sum_y_);
  }
  
  PCGScalar sum_x_;
  PCGScalar sum_y_;
  
  CUDABuffer_<PCGScalar>* pcg_g_;
  typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* temp_storage_;
};

template<int block_width, class Model, bool are_camera_tr_rig_in_state>
__global__ void
__launch_bounds__(/*maxThreadsPerBlock*/ kResidualJacobianBlockSize, /*minBlocksPerMultiprocessor*/ 1)
PCGStep1CUDAKernel(
    CUDADatasetAndState s,
    Model model,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_alpha_d) {
  unsigned int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = feature_index < s.num_features;
  if (!valid) {
    feature_index = 0;
  }
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage temp_storage;
  
  PCGScalar weight;
  PCGScalar sum_x;
  PCGScalar sum_y;
  {
    PCGStep1SumAccumulator<block_width, block_height> accumulator(pcg_p);
    ComputeResidualAndJacobian<block_width, block_height, /*compute_jacobians*/ true, are_camera_tr_rig_in_state>(valid, feature_index, s, model, accumulator);
    
    weight = accumulator.weight_;
    sum_x = accumulator.sum_x_;
    sum_y = accumulator.sum_y_;
  }
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_alpha_d(0, 0), sum_x * weight * sum_x + sum_y * weight * sum_y, valid, &temp_storage);
  sum_x *= weight;
  sum_y *= weight;
  __syncthreads();
  
  // TODO: Try storing sum_x and sum_y in global memory here and moving the
  //       part below into its own kernel. It might be faster since it might be
  //       possible to run one of the two resulting kernels with higher
  //       parallelism than the current large kernel.
  {
    PCGStep1ResolveAccumulator<block_width, block_height> accumulator(sum_x, sum_y, &pcg_g, &temp_storage);
    ComputeResidualAndJacobian<block_width, block_height, /*compute_jacobians*/ true, are_camera_tr_rig_in_state>(valid, feature_index, s, model, accumulator);
  }
}

template<int block_width>
__global__ void AddAlphaDEpsilonTermsCUDAKernel(
    u32 unknown_count,
    PCGScalar lambda,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_d) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = unknown_index < unknown_count;
  if (!valid) {
    unknown_index = unknown_count - 1;
  }
  
  PCGScalar p_value = pcg_p(0, unknown_index);
  PCGScalar term = lambda * p_value * p_value;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage temp_storage;
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_alpha_d(0, 0), term, valid, &temp_storage);
}

template <class Model>
void PCGStep1CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    const CUDADatasetAndState& s,
    const Model& model,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_alpha_d) {
  CHECK_CUDA_NO_ERROR();
  if (s.num_features == 0) {
    return;
  }
  
  constexpr int block_width = kResidualJacobianBlockSize;
  dim3 grid_dim(GetBlockCount(s.num_features, block_width));
  dim3 block_dim(block_width);
  bool are_camera_tr_rig_in_state = s.are_camera_tr_rig_in_state;
  COMPILE_OPTION(
      are_camera_tr_rig_in_state,
      PCGStep1CUDAKernel<block_width, Model, _are_camera_tr_rig_in_state>
      <<<grid_dim, block_dim, 0, stream>>>(
          s,
          model,
          *pcg_p,
          *pcg_g,
          *pcg_alpha_d););
  CHECK_CUDA_NO_ERROR();
}

template
void PCGStep1CUDA<CUDACentralGenericModel>(
    cudaStream_t stream,
    u32 unknown_count,
    const CUDADatasetAndState& s,
    const CUDACentralGenericModel& model,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_alpha_d);


template<int block_width>
__global__ void PCGStep2CUDAKernel(
    u32 unknown_count,
    PCGScalar lambda,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_M,
    CUDABuffer_<PCGScalar> pcg_delta,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_n,
    CUDABuffer_<PCGScalar> pcg_alpha_d,
    CUDABuffer_<PCGScalar> pcg_beta_n) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  PCGScalar beta_term;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage temp_storage;
  
  if (unknown_index < unknown_count) {
    // TODO: Default to 1 or to 0 if denominator is near-zero? Stop optimization if that happens?
    PCGScalar alpha =
        (pcg_alpha_d(0, 0) >= 1e-35f) ? (pcg_alpha_n(0, 0) / pcg_alpha_d(0, 0)) : 0;
    
    PCGScalar p_value = pcg_p(0, unknown_index);
    pcg_delta(0, unknown_index) += alpha * p_value;
    
    PCGScalar r_value = pcg_r(0, unknown_index);
    r_value -= alpha * (pcg_g(0, unknown_index) + lambda * p_value);
    pcg_r(0, unknown_index) = r_value;
    
    // This is called z in the Opt paper, but stored in g here to save memory.
    PCGScalar z_value = r_value / (pcg_M(0, unknown_index) + lambda);
    pcg_g(0, unknown_index) = z_value;
    
    beta_term = z_value * r_value;
  }
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_beta_n(0, 0),
      beta_term,
      unknown_index < unknown_count,
      &temp_storage);
}

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
    CUDABuffer_<PCGScalar>* pcg_beta_n) {
  CHECK_CUDA_NO_ERROR();
  if (unknown_count == 0) {
    return;
  }
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      AddAlphaDEpsilonTermsCUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      lambda,
      *pcg_p,
      *pcg_alpha_d);
  CHECK_CUDA_NO_ERROR();
  
  cudaMemsetAsync(pcg_beta_n->address(), 0, 1 * sizeof(PCGScalar), stream);
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGStep2CUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      lambda,
      *pcg_r,
      pcg_M,
      *pcg_delta,
      *pcg_g,
      *pcg_p,
      *pcg_alpha_n,
      *pcg_alpha_d,
      *pcg_beta_n);
  CHECK_CUDA_NO_ERROR();
}


template<int block_width>
__global__ void PCGStep3CUDAKernel(
    u32 unknown_count,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_n,
    CUDABuffer_<PCGScalar> pcg_beta_n) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (unknown_index < unknown_count) {
    // TODO: Default to 1 or to 0 if denominator is near-zero? Stop optimization if that happens?
    PCGScalar beta =
        (pcg_alpha_n(0, 0) >= 1e-35f) ? (pcg_beta_n(0, 0) / pcg_alpha_n(0, 0)) : 0;
    
    pcg_p(0, unknown_index) = pcg_g/*z*/(0, unknown_index) + beta * pcg_p(0, unknown_index);
  }
}

void PCGStep3CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n,
    CUDABuffer_<PCGScalar>* pcg_beta_n) {
  CHECK_CUDA_NO_ERROR();
  if (unknown_count == 0) {
    return;
  }
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGStep3CUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      *pcg_g,
      *pcg_p,
      *pcg_alpha_n,
      *pcg_beta_n);
  CHECK_CUDA_NO_ERROR();
}

}

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

#include "camera_calibration/bundle_adjustment/cuda_joint_optimization.h"
#include "camera_calibration/bundle_adjustment/cuda_joint_optimization.cuh"

#include <cuda_runtime.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/cuda/cuda_matrix.cuh>
#include <libvis/cuda/cuda_util.h>
#include <libvis/lm_optimizer.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"
#include "camera_calibration/models/all_models.h"
#include "camera_calibration/models/cuda_camera_model.cuh"
#include "camera_calibration/models/cuda_central_generic_model.cuh"

namespace vis {

void PrintGPUMemoryUsage() {
  size_t free_bytes;
  size_t total_bytes;
  CUDA_CHECKED_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t used_bytes = total_bytes - free_bytes;
  
  constexpr double kBytesToMiB = 1.0 / (1024.0 * 1024.0);
  LOG(INFO) << "GPU memory used: " <<
               kBytesToMiB * used_bytes << " MiB, free: " <<
               kBytesToMiB * free_bytes << " MiB";
}

void TransferStateToGPU(
    const BAState& state,
    const vector<int>& seq_to_original_index,
    float* points[3],
    vector<CUDAMatrix3x4*>* image_tr_global,
    float* camera_q_rig,
    float* rig_tr_global,
    vector<shared_ptr<CUDACameraModel>>* cuda_camera_models) {
  // points:
  for (int i = 0; i < 3; ++ i) {
    vector<float> cpu_points(state.points.size());
    for (int p = 0; p < cpu_points.size(); ++ p) {
      cpu_points[p] = state.points[p](i);
    }
    CUDA_CHECKED_CALL(cudaMemcpy(points[i], cpu_points.data(), cpu_points.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  
  // image_tr_global:
  for (int camera_index = 0; camera_index < state.num_cameras(); ++ camera_index) {
    vector<CUDAMatrix3x4> cpu_image_tr_global(seq_to_original_index.size());
    for (int i = 0; i < seq_to_original_index.size(); ++ i) {
      SE3d image_tr_global = state.camera_tr_rig[camera_index] * state.rig_tr_global[seq_to_original_index[i]];
      cpu_image_tr_global[i] = CUDAMatrix3x4(image_tr_global.matrix3x4());
    }
    
    CUDA_CHECKED_CALL(cudaMemcpy((*image_tr_global)[camera_index], cpu_image_tr_global.data(), seq_to_original_index.size() * sizeof(CUDAMatrix3x4), cudaMemcpyHostToDevice));
  }
  
  // camera_q_rig:
  vector<float> cpu_camera_q_rig(4 * state.camera_tr_rig.size());
  for (int i = 0; i < state.camera_tr_rig.size(); ++ i) {
    const auto& q = state.camera_tr_rig[i].unit_quaternion();
    cpu_camera_q_rig[4 * i + 0] = q.w();
    cpu_camera_q_rig[4 * i + 1] = q.x();
    cpu_camera_q_rig[4 * i + 2] = q.y();
    cpu_camera_q_rig[4 * i + 3] = q.z();
  }
  CUDA_CHECKED_CALL(cudaMemcpy(camera_q_rig, cpu_camera_q_rig.data(), cpu_camera_q_rig.size() * sizeof(float), cudaMemcpyHostToDevice));
  cpu_camera_q_rig = vector<float>();  // free memory
  
  // rig_tr_global:
  vector<float> cpu_rig_tr_global(7 * seq_to_original_index.size());
  for (int i = 0; i < seq_to_original_index.size(); ++ i) {
    const auto& tr = state.rig_tr_global[seq_to_original_index[i]];
    const auto& q = tr.unit_quaternion();
    const auto& t = tr.translation();
    cpu_rig_tr_global[7 * i + 0] = q.w();
    cpu_rig_tr_global[7 * i + 1] = q.x();
    cpu_rig_tr_global[7 * i + 2] = q.y();
    cpu_rig_tr_global[7 * i + 3] = q.z();
    cpu_rig_tr_global[7 * i + 4] = t.x();
    cpu_rig_tr_global[7 * i + 5] = t.y();
    cpu_rig_tr_global[7 * i + 6] = t.z();
  }
  CUDA_CHECKED_CALL(cudaMemcpy(rig_tr_global, cpu_rig_tr_global.data(), cpu_rig_tr_global.size() * sizeof(float), cudaMemcpyHostToDevice));
  cpu_rig_tr_global = vector<float>();  // free memory
  
  // intrinsics:
  for (int camera_index = 0; camera_index < state.num_cameras(); ++ camera_index) {
    (*cuda_camera_models)[camera_index].reset(state.intrinsics[camera_index]->CreateCUDACameraModel());
  }
}

OptimizationReport CudaOptimizeJointly(
    Dataset& dataset,
    BAState* state,
    int max_iteration_count,
    int max_inner_iterations,
    double init_lambda,
    double numerical_diff_delta,
    double regularization_weight,
    double* final_lambda,
    bool debug_verify_cost,
    bool debug_fix_points,
    bool debug_fix_poses,
    bool debug_fix_rig_poses,
    bool debug_fix_intrinsics,
    bool print_progress) {
  constexpr cudaStream_t stream = 0;
  
  constexpr int max_lm_attempts = 10;
  
  CHECK_EQ(state->image_used.size(), state->rig_tr_global.size());
  
  // NOTE: Increasing numerical_diff_delta to have a higher value compared to
  //       the CPU version called with the same value.
  // TODO: Don't do this here, this makes it be hidden
  numerical_diff_delta *= 10;
  
  if (regularization_weight > 0) {
    LOG(ERROR) << "Regularization is not supported by CUDA-based optimization yet. Ignoring it.";
  }
  if (debug_verify_cost) {
    LOG(ERROR) << "debug_verify_cost is not supported by CUDA-based optimization yet. Ignoring it.";
  }
  
  OptimizationReport report;
  report.num_iterations_performed = 0;
  report.cost_and_jacobian_evaluation_time = 0;
  report.solve_time = 0;
  
  // For debugging:
  // LOG(INFO) << "Memory usage before optimization:";
  // PrintGPUMemoryUsage();
  
  // Create sequential indexing for poses for the state
  vector<int> seq_to_original_index;
  seq_to_original_index.clear();
  for (usize i = 0; i < state->rig_tr_global.size(); ++ i) {
    if (state->image_used[i]) {
      seq_to_original_index.push_back(i);
    }
  }
  
  // Transfer dataset and state data to the GPU.
  vector<usize> num_features(dataset.num_cameras());
  
  // Dataset: Imagesets with point features (xy coordinates in image, and index of 3D point).
  //   For each camera, store arrays where items with the same index form one feature observation:
  //   - u16: Array of image_index
  //   - 2 x float: Array of xy
  //   - u16: Array of 3D point
  vector<u16*> features_image(dataset.num_cameras());
  vector<float*> features_x(dataset.num_cameras());
  vector<float*> features_y(dataset.num_cameras());
  vector<u16*> features_index(dataset.num_cameras());
  vector<float*> features_residual_x(dataset.num_cameras());
  vector<float*> features_residual_y(dataset.num_cameras());
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    usize& n_features = num_features[camera_index];
    n_features = 0;
    for (int i = 0; i < seq_to_original_index.size(); ++ i) {
      n_features += dataset.GetImageset(seq_to_original_index[i])->FeaturesOfCamera(camera_index).size();
    }
    
    vector<u16> cpu_features_image(n_features);
    vector<float> cpu_features_x(n_features);
    vector<float> cpu_features_y(n_features);
    vector<u16> cpu_features_index(n_features);
    
    usize index = 0;
    for (int i = 0; i < seq_to_original_index.size(); ++ i) {
      const auto& features = dataset.GetImageset(seq_to_original_index[i])->FeaturesOfCamera(camera_index);
      
      for (int f = 0; f < features.size(); ++ f) {
        cpu_features_image[index] = i;
        cpu_features_x[index] = features[f].xy.x();
        cpu_features_y[index] = features[f].xy.y();
        cpu_features_index[index] = features[f].index;
        ++ index;
      }
    }
    CHECK_EQ(index, n_features);
    
    CUDA_CHECKED_CALL(cudaMalloc(&features_image[camera_index], n_features * sizeof(u16)));
    CUDA_CHECKED_CALL(cudaMemcpy(features_image[camera_index], cpu_features_image.data(), n_features * sizeof(u16), cudaMemcpyHostToDevice));
    CUDA_CHECKED_CALL(cudaMalloc(&features_x[camera_index], n_features * sizeof(float)));
    CUDA_CHECKED_CALL(cudaMemcpy(features_x[camera_index], cpu_features_x.data(), n_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECKED_CALL(cudaMalloc(&features_y[camera_index], n_features * sizeof(float)));
    CUDA_CHECKED_CALL(cudaMemcpy(features_y[camera_index], cpu_features_y.data(), n_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECKED_CALL(cudaMalloc(&features_index[camera_index], n_features * sizeof(u16)));
    CUDA_CHECKED_CALL(cudaMemcpy(features_index[camera_index], cpu_features_index.data(), n_features * sizeof(u16), cudaMemcpyHostToDevice));
    
    CUDA_CHECKED_CALL(cudaMalloc(&features_residual_x[camera_index], n_features * sizeof(float)));
    CUDA_CHECKED_CALL(cudaMalloc(&features_residual_y[camera_index], n_features * sizeof(float)));
  }
  
  // State:
  //   - 3 x float: Array of 3D point xyz
  //   - SE3d: Array of camera_tr_rig
  //   - SE3d: Array of rig_tr_global
  //   - CameraModel: Array of intrinsics
  bool are_camera_tr_rig_in_state = state->camera_tr_rig.size() > 1;
  
  if (!are_camera_tr_rig_in_state) {
    // Ensure that the camera_tr_rig transformation(s) are identity by applying
    // camera_tr_rig[0] to all poses. This allows us to assume:
    // camera_tr_global == rig_tr_global.
    for (int i = 0; i < seq_to_original_index.size(); ++ i) {
      int original_index = seq_to_original_index[i];
      state->rig_tr_global[original_index] = state->camera_tr_rig[0] * state->rig_tr_global[original_index];
    }
    state->camera_tr_rig[0] = SE3d();
  }
  
  float* points[3];
  for (int i = 0; i < 3; ++ i) {
    CUDA_CHECKED_CALL(cudaMalloc(&points[i], state->points.size() * sizeof(float)));
  }
  
  // Cache image_tr_global matrices for each image
  vector<CUDAMatrix3x4*> image_tr_global(dataset.num_cameras());
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    CUDA_CHECKED_CALL(cudaMalloc(&image_tr_global[camera_index], seq_to_original_index.size() * sizeof(CUDAMatrix3x4)));
  }
  
  // camera_q_rig (4 quaternion parameters)
  float* camera_q_rig;
  CUDA_CHECKED_CALL(cudaMalloc(&camera_q_rig, 4 * state->camera_tr_rig.size() * sizeof(float)));
  
  // rig_tr_global (4 quaternion parameters + 3 translation parameters)
  float* rig_tr_global;
  CUDA_CHECKED_CALL(cudaMalloc(&rig_tr_global, 7 * seq_to_original_index.size() * sizeof(float)));
  
  // intrinsics (CUDA-compatible camera objects)
  vector<shared_ptr<CUDACameraModel>> cuda_camera_models(dataset.num_cameras());
  
  // Count the unknowns, remember the index ranges corresponding to the different unknowns, and allocate the corresponding variables.
  const usize rig_tr_global_start_index = 0;
  const usize camera_tr_rig_start_index = rig_tr_global_start_index + SE3f::DoF * state->rig_tr_global.size();
  const usize points_start_index = camera_tr_rig_start_index + (are_camera_tr_rig_in_state ? (SE3f::DoF * state->camera_tr_rig.size()) : 0);
  const usize intrinsics_start_index = points_start_index + 3 * state->points.size();
  
  usize num_unknowns = intrinsics_start_index;
  vector<usize> intrinsic_start_index(dataset.num_cameras());
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    intrinsic_start_index[camera_index] = num_unknowns;
    num_unknowns += state->intrinsics[camera_index]->update_parameter_count();
  }
  
  CUDABuffer<PCGScalar> pcg_initial_r(1, num_unknowns);
  CUDABuffer<PCGScalar> pcg_r(1, num_unknowns);
  CUDABuffer<PCGScalar> pcg_M(1, num_unknowns);
  CUDABuffer<PCGScalar> pcg_delta(1, num_unknowns);
  CUDABuffer<PCGScalar> pcg_final_delta(1, num_unknowns);
  CUDABuffer<PCGScalar> pcg_g(1, num_unknowns);
  CUDABuffer<PCGScalar> pcg_p(1, num_unknowns);
  shared_ptr<CUDABuffer<PCGScalar>> pcg_alpha_n(new CUDABuffer<PCGScalar>(1, 1));
  shared_ptr<CUDABuffer<PCGScalar>> pcg_alpha_d(new CUDABuffer<PCGScalar>(1, 1));
  shared_ptr<CUDABuffer<PCGScalar>> pcg_beta_n(new CUDABuffer<PCGScalar>(1, 1));
  shared_ptr<CUDABuffer<PCGScalar>> pcg_cost(new CUDABuffer<PCGScalar>(1, 1));
  shared_ptr<CUDABuffer<PCGScalar>> pcg_relative_cost(new CUDABuffer<PCGScalar>(1, 1));
  
  // Cache parameter structs
  vector<CUDADatasetAndState> cuda_states(dataset.num_cameras());
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    CUDADatasetAndState& s = cuda_states[camera_index];
    s.camera_index = camera_index;
    s.numerical_diff_delta = numerical_diff_delta;
    s.num_features = num_features[camera_index];
    s.features_image = features_image[camera_index];
    s.features_x = features_x[camera_index];
    s.features_y = features_y[camera_index];
    s.features_index = features_index[camera_index];
    s.features_residual_x = features_residual_x[camera_index];
    s.features_residual_y = features_residual_y[camera_index];
    s.points[0] = points[0];
    s.points[1] = points[1];
    s.points[2] = points[2];
    s.image_tr_global = image_tr_global[camera_index];
    s.camera_q_rig = camera_q_rig;
    s.rig_tr_global = rig_tr_global;
    s.rig_tr_global_start_index = rig_tr_global_start_index;
    s.are_camera_tr_rig_in_state = are_camera_tr_rig_in_state;
    s.camera_tr_rig_start_index = camera_tr_rig_start_index;
    s.points_start_index = points_start_index;
    s.intrinsic_start_index = intrinsic_start_index[camera_index];
  }
  
  // Transfer the initial state from the CPU to the GPU
  TransferStateToGPU(
      *state,
      seq_to_original_index,
      points,
      &image_tr_global,
      camera_q_rig,
      rig_tr_global,
      &cuda_camera_models);
  
  // For debugging:
  // LOG(INFO) << "Memory usage during optimization:";
  // PrintGPUMemoryUsage();
  
  // Iterate over optimization iterations
  double lambda = (init_lambda <= 0) ? 1 : init_lambda;  // NOTE: This differs from the CPU implementation, which initializes lambda based on the diagonal H values
  bool applied_update = true;
  PCGScalar last_cost = numeric_limits<PCGScalar>::quiet_NaN();
  int iteration;
  for (iteration = 0; iteration < max_iteration_count; ++ iteration) {
    // Run PCG Init
    cudaMemsetAsync(pcg_initial_r.ToCUDA().address(), 0, num_unknowns * sizeof(PCGScalar), stream);
    cudaMemsetAsync(pcg_M.ToCUDA().address(), 0, num_unknowns * sizeof(PCGScalar), stream);
    cudaMemsetAsync(pcg_cost->ToCUDA().address(), 0, 1 * sizeof(PCGScalar), stream);
    for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
      PCGInitCUDA(
          stream,
          cuda_states[camera_index],
          *reinterpret_cast<CUDACentralGenericModel*>(cuda_camera_models[camera_index].get()),  // TODO: use IDENTIFY_CUDA_CAMERA?
          &pcg_initial_r.ToCUDA(),
          &pcg_M.ToCUDA(),
          &pcg_cost->ToCUDA());
    }
    PCGScalar cpu_cost;
    pcg_cost->DebugDownload(&cpu_cost);
    last_cost = cpu_cost;
    if (iteration == 0) {
      report.initial_cost = last_cost;
    }
    
    if (print_progress) {
      if (iteration == 0) {
        LOG(INFO) << "CudaOptimizeJointly: [0] Initial cost: " << cpu_cost;
      } else {
        LOG(1) << "CudaOptimizeJointly: [" << iteration << "] cost: " << cpu_cost;
      }
    }
    
    applied_update = false;
    
    // Iterate over Levenberg-Marquardt iterations
    for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++ lm_iteration) {
      pcg_r.SetTo(pcg_initial_r, stream);
      
      // Init 2 depends on lambda, thus run it within the LM loop
      PCGInit2CUDA(
          stream,
          num_unknowns,
          lambda,
          pcg_r.ToCUDA(),
          pcg_M.ToCUDA(),
          &pcg_delta.ToCUDA(),
          &pcg_g.ToCUDA(),
          &pcg_p.ToCUDA(),
          &pcg_alpha_n->ToCUDA());
      
      // Run PCG inner iterations to determine pcg_delta
      PCGScalar prev_r_norm = numeric_limits<PCGScalar>::infinity();
      int num_iterations_without_improvement = 0;
      
      PCGScalar initial_r_norm = numeric_limits<PCGScalar>::quiet_NaN();
      PCGScalar smallest_r_norm = numeric_limits<PCGScalar>::infinity();
      for (int step = 0; step < max_inner_iterations; ++ step) {
        if (step > 0) {
          // Set pcg_alpha_n_ to pcg_beta_n_ by swapping the pointers (since we
          // don't need to preserve pcg_beta_n_).
          // NOTE: This is wrong in the Opt paper, it says "beta" only instead of
          //       "beta_n" which is something different.
          std::swap(pcg_alpha_n, pcg_beta_n);
          
          // This is cleared by PCGInit2CUDA() for the first iteration
          cudaMemsetAsync(pcg_g.ToCUDA().address(), 0, num_unknowns * sizeof(PCGScalar), stream);
        }
        
        // Run PCG step 1 & 2
        cudaMemsetAsync(pcg_alpha_d->ToCUDA().address(), 0, 1 * sizeof(PCGScalar), stream);
        for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
          PCGStep1CUDA(
              stream,
              num_unknowns,
              cuda_states[camera_index],
              *reinterpret_cast<CUDACentralGenericModel*>(cuda_camera_models[camera_index].get()),  // TODO: use IDENTIFY_CUDA_CAMERA?
              &pcg_p.ToCUDA(),
              &pcg_g.ToCUDA(),
              &pcg_alpha_d->ToCUDA());
        }
        PCGStep2CUDA(
            stream,
            num_unknowns,
            lambda,
            &pcg_r.ToCUDA(),
            pcg_M.ToCUDA(),
            &pcg_delta.ToCUDA(),
            &pcg_g.ToCUDA(),
            &pcg_p.ToCUDA(),
            &pcg_alpha_n->ToCUDA(),
            &pcg_alpha_d->ToCUDA(),
            &pcg_beta_n->ToCUDA());
        
        // Check for convergence of the inner iterations
        PCGScalar r_norm;
        pcg_beta_n->DownloadAsync(stream, &r_norm);
        cudaStreamSynchronize(stream);
        r_norm = sqrt(r_norm);
        if (step == 0) {
          initial_r_norm = r_norm;
        }
        if (r_norm < smallest_r_norm) {
          smallest_r_norm = r_norm;
          pcg_final_delta.SetTo(pcg_delta, stream);
          if (r_norm == 0) {
            break;
          }
        }
        
        // if (print_progress) {
        //   LOG(1) << "  r_norm: " << r_norm << "; advancement: " << (prev_r_norm - r_norm);
        // }
        
        if (r_norm < prev_r_norm - 1e-3) {  // TODO: Make this threshold a parameter
          num_iterations_without_improvement = 0;
        } else {
          ++ num_iterations_without_improvement;
          if (num_iterations_without_improvement >= 3) {
            break;
          }
        }
        prev_r_norm = r_norm;
        
        // This (and some computations from step 2) is not necessary in the last
        // iteration since the result is already computed in pcg_final_delta.
        // NOTE: For best speed, could make a special version of step 2 (templated)
        //       which excludes the unnecessary operations. Probably not very relevant though.
        if (step < max_inner_iterations - 1) {
          PCGStep3CUDA(
              stream,
              num_unknowns,
              &pcg_g.ToCUDA(),
              &pcg_p.ToCUDA(),
              &pcg_alpha_n->ToCUDA(),
              &pcg_beta_n->ToCUDA());
        }
      }  // end loop over PCG inner iterations
      
      if (print_progress) {
        LOG(1) << "  r norm: " << initial_r_norm << " --> " << smallest_r_norm;
      }
      
      // Update the variables with pcg_final_delta to form a test state.
      // TODO: Move the update code into BAState and drop the separate state class of the CPU optimization as it is somewhat redundant
      BAState test_state(*state);
      
      vector<PCGScalar> delta_cpu;
      
      if (!debug_fix_poses) {
        delta_cpu.resize(SE3d::DoF * seq_to_original_index.size());
        pcg_final_delta.DownloadPartAsync(rig_tr_global_start_index * sizeof(PCGScalar), SE3d::DoF * seq_to_original_index.size() * sizeof(PCGScalar), stream, delta_cpu.data());
        cudaStreamSynchronize(stream);
        for (usize pose_index = 0; pose_index < seq_to_original_index.size(); ++ pose_index) {
          auto& tr = test_state.rig_tr_global[seq_to_original_index[pose_index]];
          
          usize delta_index = SE3d::DoF * pose_index;
          tr = SE3d(ApplyLocalUpdateToQuaternion(
                        tr.unit_quaternion(),
                        Vec3d(delta_cpu[delta_index + 0], delta_cpu[delta_index + 1], delta_cpu[delta_index + 2])),
                    tr.translation() + Vec3d(delta_cpu[delta_index + 3], delta_cpu[delta_index + 4], delta_cpu[delta_index + 5]));
        }
      }
      
      if (are_camera_tr_rig_in_state && !debug_fix_rig_poses) {
        delta_cpu.resize(SE3d::DoF * test_state.camera_tr_rig.size());
        pcg_final_delta.DownloadPartAsync(camera_tr_rig_start_index * sizeof(PCGScalar), SE3d::DoF * test_state.camera_tr_rig.size() * sizeof(PCGScalar), stream, delta_cpu.data());
        cudaStreamSynchronize(stream);
        for (usize pose_index = 0; pose_index < test_state.camera_tr_rig.size(); ++ pose_index) {
          auto& tr = test_state.camera_tr_rig[pose_index];
          
          usize delta_index = SE3d::DoF * pose_index;
          tr = SE3d(ApplyLocalUpdateToQuaternion(
                        tr.unit_quaternion(),
                        Vec3d(delta_cpu[delta_index + 0], delta_cpu[delta_index + 1], delta_cpu[delta_index + 2])),
                    tr.translation() + Vec3d(delta_cpu[delta_index + 3], delta_cpu[delta_index + 4], delta_cpu[delta_index + 5]));
        }
      }
      
      if (!debug_fix_points) {
        delta_cpu.resize(3 * test_state.points.size());
        pcg_final_delta.DownloadPartAsync(points_start_index * sizeof(PCGScalar), 3 * test_state.points.size() * sizeof(PCGScalar), stream, delta_cpu.data());
        cudaStreamSynchronize(stream);
        for (usize point_index = 0; point_index < test_state.points.size(); ++ point_index) {
          usize delta_index = 3 * point_index;
          test_state.points[point_index] += Vec3d(delta_cpu[delta_index + 0], delta_cpu[delta_index + 1], delta_cpu[delta_index + 2]);
        }
      }
      
      if (!debug_fix_intrinsics) {
        delta_cpu.resize(num_unknowns - intrinsics_start_index);
        pcg_final_delta.DownloadPartAsync(intrinsics_start_index * sizeof(PCGScalar), (num_unknowns - intrinsics_start_index) * sizeof(PCGScalar), stream, delta_cpu.data());
        cudaStreamSynchronize(stream);
        usize delta_index = 0;
        for (int i = 0; i < test_state.intrinsics.size(); ++ i) {
          CameraModel& model_ref = *test_state.intrinsics[i];
          IDENTIFY_CAMERA_MODEL(model_ref,
            _model_ref.SubtractDelta(-1 * Eigen::Matrix<PCGScalar, Eigen::Dynamic, 1>::Map(&delta_cpu[delta_index], _model_ref.update_parameter_count()).cast<double>());
          )
          delta_index += test_state.intrinsics[i]->update_parameter_count();
        }
        CHECK_EQ(delta_index, num_unknowns - intrinsics_start_index);
      }
      
      // Transfer the test state to the GPU
      TransferStateToGPU(
          test_state,
          seq_to_original_index,
          points,
          &image_tr_global,
          camera_q_rig,
          rig_tr_global,
          &cuda_camera_models);
      
      // Compare the cost of the test state to the initial state
      cudaMemsetAsync(pcg_cost->ToCUDA().address(), 0, 1 * sizeof(PCGScalar), stream);
      cudaMemsetAsync(pcg_relative_cost->ToCUDA().address(), 0, 1 * sizeof(PCGScalar), stream);
      for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
        PCGCompareCostCUDA(
            stream,
            cuda_states[camera_index],
            *reinterpret_cast<CUDACentralGenericModel*>(cuda_camera_models[camera_index].get()),  // TODO: use IDENTIFY_CUDA_CAMERA?
            &pcg_cost->ToCUDA(),
            &pcg_relative_cost->ToCUDA());
      }
      PCGScalar cpu_cost;
      pcg_cost->DebugDownload(&cpu_cost);
      PCGScalar cpu_relative_cost;
      pcg_relative_cost->DebugDownload(&cpu_relative_cost);
      
      // Take over the new state and decrease lambda,
      // or revert back to the old state and increase lambda?
      if (cpu_relative_cost < 0) {
        // Take over the update.
        if (print_progress && lm_iteration > 0) {
          LOG(1) << "CudaOptimizeJointly:   [" << (iteration + 1) << "] update accepted";
        }
        *state = test_state;
        lambda = 0.5f * lambda;
        applied_update = true;
        report.num_iterations_performed += 1;
        last_cost = cpu_cost;
        break;
      } else {
        // Revert to the old state on the GPU
        TransferStateToGPU(
            *state,
            seq_to_original_index,
            points,
            &image_tr_global,
            camera_q_rig,
            rig_tr_global,
            &cuda_camera_models);
        
        lambda = 2.f * lambda;
        if (print_progress) {
          LOG(1) << "CudaOptimizeJointly:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << max_lm_attempts
                  << "] update rejected (bad cost: " << cpu_cost
                  << "), new lambda: " << lambda;
        }
      }
    }  // end loop over Levenberg-Marquardt iterations
    
    if (!applied_update || last_cost == 0) {
      if (print_progress) {
        if (last_cost == 0) {
          LOG(INFO) << "CudaOptimizeJointly: Reached zero cost, stopping.";
        } else {
          LOG(INFO) << "CudaOptimizeJointly: Cannot find an update which decreases the cost, aborting.";
        }
      }
      iteration += 1;  // For correct display only.
      break;
    }
  }  // end loop over optimization iterations
  
  *final_lambda = lambda;
  
  // Free used GPU memory
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    CUDA_CHECKED_CALL(cudaFree(features_image[camera_index]));
    CUDA_CHECKED_CALL(cudaFree(features_x[camera_index]));
    CUDA_CHECKED_CALL(cudaFree(features_y[camera_index]));
    CUDA_CHECKED_CALL(cudaFree(features_index[camera_index]));
    
    CUDA_CHECKED_CALL(cudaFree(features_residual_x[camera_index]));
    CUDA_CHECKED_CALL(cudaFree(features_residual_y[camera_index]));
  }
  for (int i = 0; i < 3; ++ i) {
    CUDA_CHECKED_CALL(cudaFree(points[i]));
  }
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    cudaFree(image_tr_global[camera_index]);
  }
  cudaFree(camera_q_rig);
  cudaFree(rig_tr_global);
  
  // For debugging
  // LOG(INFO) << "Memory usage after optimization (should be the same as before, otherwise there might be a memory leak):";
  // PrintGPUMemoryUsage();
  
  if (print_progress) {
    if (applied_update) {
      LOG(INFO) << "CudaOptimizeJointly: Maximum iteration count reached, stopping.";
    }
    LOG(INFO) << "CudaOptimizeJointly: [" << iteration << "] Final cost:   " << last_cost;  // length matches with "Initial cost: "
  }
  
  report.final_cost = last_cost;
  return report;
}

}

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

#include <fstream>
#include <memory>
#include <unordered_map>

#include "libvis/logging.h"

#include "libvis/cuda/cuda_util.h"
#include "libvis/libvis.h"

namespace vis {

// Finds the best block size parameters for CUDA kernels by exhaustive search.
class CUDAAutoTuner {
public:
  // TODO: How does that behave when accessed in an executable vs. accessed from the library when linked statically and dynamically on Windows and on Linux?
  //       Any situation where more than one instance could be created?
  static CUDAAutoTuner& Instance() {
    static CUDAAutoTuner singleton_instance;
    return singleton_instance;
  }
  
  inline int tuning_iteration() const {
    return tuning_iteration_;
  }
  
  // iteration must be in [0, 6].
  inline void SetTuningIteration(int iteration) {
    tuning_iteration_ = iteration;
  }
  
  void GetParametersForKernel(const char* kernel_name, int dimensions,
                              int tuning_iteration,
                              int default_block_width, int default_block_height,
                              int* out_block_width, int* out_block_height) {
    if (tuning_iteration == -1) {
      // Using the final or default values.
      KernelParameters* parameters;
      
      auto it = parameter_map_1_.find(kernel_name);
      if (it != parameter_map_1_.end()) {
        parameters = it->second.get();
      } else {
        auto it2 = parameter_map_2_.find(kernel_name);
        if (it2 != parameter_map_2_.end()) {
          parameters = it2->second.get();
        } else {
          *out_block_width = default_block_width;
          if (out_block_height) {
            *out_block_height = default_block_height;
          }
          return;
        }
      }
      
      *out_block_width = parameters->block_width;
      if (out_block_height) {
        *out_block_height = parameters->block_height;
      }
    } else {
      // Using the values of the current tuning iteration.
      // NOTE: These values must be ordered from small to large, since if a
      //       configuration turns out not to work due to too many resources
      //       requested, values at smaller indices will be tried (and it should
      //       be ensured that a working configuration can be found).
      if (dimensions == 1) {
        const int tuning_values[] = {32, 64, 128, 256, 512, 1024, 1024};
        *out_block_width = tuning_values[tuning_iteration];
      } else if (dimensions == 2) {
        const int tuning_values_width[] = {8, 16, 8, 16, 32, 16, 32};
        const int tuning_values_height[] = {8, 8, 16, 16, 16, 32, 32};
        *out_block_width = tuning_values_width[tuning_iteration];
        *out_block_height = tuning_values_height[tuning_iteration];
      }
    }
  }
  
  void AddTuningMeasurement(const char* kernel_name,
                            int width, int height,
                            double runtime) {
    KernelParameters* parameters;
    
    auto it = parameter_map_1_.find(kernel_name);
    if (it != parameter_map_1_.end()) {
      parameters = it->second.get();
    } else {
      auto it2 = parameter_map_2_.find(kernel_name);
      if (it2 != parameter_map_2_.end()) {
        parameter_map_1_.insert(make_pair(kernel_name, it2->second));
        parameters = it2->second.get();
      } else {
        shared_ptr<KernelParameters> new_parameters(new KernelParameters());
        parameter_map_1_.insert(make_pair(kernel_name, new_parameters));
        parameter_map_2_.insert(make_pair(kernel_name, new_parameters));
        parameters = new_parameters.get();
        
        parameters->kernel_name = kernel_name;
        parameters->block_width = width;
        parameters->block_height = height;
        parameters->runtime = 0;
      }
    }
    
    CHECK_EQ(parameters->block_width, width);
    CHECK_EQ(parameters->block_height, height);
    
    parameters->runtime += runtime;
  }
  
  bool SaveTuningFile(const char* file_path) {
    ofstream out_file(file_path, std::ios::out);
    if (!out_file) {
      return false;
    }
    
    for (const auto& item : parameter_map_1_) {
      const KernelParameters& parameters = *item.second;
      
      out_file << parameters.kernel_name << " "
               << parameters.block_width << " "
               << parameters.block_height << " "
               << parameters.runtime << std::endl;
    }
    
    return true;
  }
  
  bool LoadParametersFile(const char* file_path) {
    ifstream in_file(file_path, std::ios::in);
    if (!in_file) {
      return false;
    }
    
    while (!in_file.eof() && !in_file.bad()) {
      std::string line;
      std::getline(in_file, line);
      if (line.size() == 0 || line[0] == '#') {
        continue;
      }
      
      shared_ptr<KernelParameters> new_parameters(new KernelParameters());
      
      std::istringstream line_stream(line);
      line_stream >> new_parameters->kernel_name
                  >> new_parameters->block_width
                  >> new_parameters->block_height
                  >> new_parameters->runtime;
      
      parameter_map_2_.insert(make_pair(new_parameters->kernel_name, new_parameters));
    }
    
    return true;
  }
  
  inline bool tuning_active() const {
    return tuning_iteration_ >= 0;
  }
  
private:
  struct KernelParameters {
    int block_width;
    int block_height;
    
    // Required during tuning only:
    string kernel_name;
    double runtime;
  };
  
  
  unordered_map<const char*, shared_ptr<KernelParameters>> parameter_map_1_;
  unordered_map<string, shared_ptr<KernelParameters>> parameter_map_2_;
  
  int tuning_iteration_ = -1;
};

// NOTE: The following macros use static and are not thread-safe for tuning.
//       However, in the tuned state, they should be thread-safe since the
//       static values are only assigned in the first run. This was necessary
//       to get any speed benefit from the tuning; it seems that the overhead
//       of GetParametersForKernel() was too large otherwise.

// Version of CUDA_AUTO_TUNE_2D() accounting for border sizes that must be
// subtracted from each block for the grid size calculation.
#define CUDA_AUTO_TUNE_2D_BORDER( \
    kernel_name, \
    default_block_width, default_block_height, \
    border_width, border_height, \
    domain_width, domain_height, \
    shared_memory_size, stream, \
    ...) \
    do { \
    vis::CUDAAutoTuner& tuner = vis::CUDAAutoTuner::Instance(); \
    static bool block_size_works = !tuner.tuning_active(); \
    do { \
      static const char* const_kernel_name = #kernel_name; \
      static int tuning_iteration = tuner.tuning_iteration(); \
      static cudaEvent_t tuning_pre_event = 0; \
      static cudaEvent_t tuning_post_event = 0; \
      static int var_block_width = 0; \
      static int var_block_height = 0; \
      if (var_block_width == 0) { \
        tuner.GetParametersForKernel(const_kernel_name, 2, \
                                     tuning_iteration, \
                                     default_block_width, default_block_height, \
                                     &var_block_width, &var_block_height); \
      } \
      dim3 grid_dim(GetBlockCount(domain_width, var_block_width - border_width), \
                    GetBlockCount(domain_height, var_block_height - border_height)); \
      dim3 block_dim(var_block_width, var_block_height); \
      \
      if (tuner.tuning_active()) { \
        if (tuning_pre_event == 0) { \
          cudaEventCreate(&tuning_pre_event); \
          cudaEventCreate(&tuning_post_event); \
          cudaGetLastError(); \
        } \
        cudaEventRecord(tuning_pre_event, stream); \
      } \
      \
      kernel_name<<<grid_dim, block_dim, shared_memory_size, stream>>>( \
          __VA_ARGS__); \
      \
      if (tuner.tuning_active()) { \
        cudaEventRecord(tuning_post_event, stream); \
        cudaEventSynchronize(tuning_post_event); \
        \
        if (!block_size_works) { \
          cudaError_t last_error = cudaGetLastError(); \
          if (last_error == cudaErrorLaunchOutOfResources || last_error == cudaErrorInvalidConfiguration) { \
            -- tuning_iteration; \
            var_block_width = 0; \
            CHECK_GE(tuning_iteration, 0) << "CUDA auto tuner: Could not find any working configuration for this kernel."; \
            continue; \
          } \
          block_size_works = true; \
        } \
        \
        float elapsed_milliseconds; \
        cudaEventElapsedTime(&elapsed_milliseconds, tuning_pre_event, tuning_post_event); \
        tuner.AddTuningMeasurement(const_kernel_name, \
                                   var_block_width, var_block_height, \
                                   0.001 * elapsed_milliseconds); \
      } \
    } while (!block_size_works); \
    } while (false)

// The kernel name must be given directly (not as a string). Template parameters
// may be included in the kernel name as long as they do not depend on the
// chosen block size (if they do, use CUDA_AUTO_TUNE_2D_TEMPLATED()).
// Tests the following block sizes:
//  8,  8
//  8, 16
// 16,  8
// 16, 16
// 16, 32
// 32, 16
// 32, 32
#define CUDA_AUTO_TUNE_2D( \
    kernel_name, \
    default_block_width, default_block_height, \
    domain_width, domain_height, \
    shared_memory_size, stream, \
    ...) \
    CUDA_AUTO_TUNE_2D_BORDER(kernel_name, default_block_width, default_block_height, 0, 0, domain_width, domain_height, shared_memory_size, stream, __VA_ARGS__)

// The kernel name must be given directly (not as a string).
// The block width and height can be used in the template parameters as
// block_width and block_height.
// Tests the following block sizes:
//  8,  8
//  8, 16
// 16,  8
// 16, 16
// 16, 32
// 32, 16
// 32, 32
#define CUDA_AUTO_TUNE_2D_BORDER_TEMPLATED( \
    kernel_name, \
    default_block_width, default_block_height, \
    border_width, border_height, \
    domain_width, domain_height, \
    shared_memory_size, stream, \
    template_parameters, \
    ...) \
    do { \
    vis::CUDAAutoTuner& tuner = vis::CUDAAutoTuner::Instance(); \
    static bool block_size_works = !tuner.tuning_active(); \
    do { \
      static const char* const_kernel_name = #kernel_name; \
      static int tuning_iteration = tuner.tuning_iteration(); \
      static cudaEvent_t tuning_pre_event; \
      static cudaEvent_t tuning_post_event; \
      static int var_block_width = 0; \
      static int var_block_height = 0; \
      if (var_block_width == 0) { \
        tuner.GetParametersForKernel(const_kernel_name, 2, \
                                     tuning_iteration, \
                                     default_block_width, default_block_height, \
                                     &var_block_width, &var_block_height); \
      } \
      dim3 grid_dim(GetBlockCount(domain_width, var_block_width - border_width), \
                    GetBlockCount(domain_height, var_block_height - border_height)); \
      dim3 block_dim(var_block_width, var_block_height); \
      \
      if (tuner.tuning_active()) { \
        if (tuning_pre_event == 0) { \
          cudaEventCreate(&tuning_pre_event); \
          cudaEventCreate(&tuning_post_event); \
          cudaGetLastError(); \
        } \
        cudaEventRecord(tuning_pre_event, stream); \
      } \
      \
      if (var_block_width == 8 && var_block_height == 8) { \
        constexpr int block_width = 8; \
        constexpr int block_height = 8; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 8 && var_block_height == 16) { \
        constexpr int block_width = 8; \
        constexpr int block_height = 16; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 16 && var_block_height == 8) { \
        constexpr int block_width = 16; \
        constexpr int block_height = 8; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 16 && var_block_height == 16) { \
        constexpr int block_width = 16; \
        constexpr int block_height = 16; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 16 && var_block_height == 32) { \
        constexpr int block_width = 16; \
        constexpr int block_height = 32; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 32 && var_block_height == 16) { \
        constexpr int block_width = 32; \
        constexpr int block_height = 16; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 32 && var_block_height == 32) { \
        constexpr int block_width = 32; \
        constexpr int block_height = 32; \
        (void) block_width; \
        (void) block_height; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else { \
        LOG(FATAL) << "Error in CUDAAutoTuner: Invalid block size given."; \
      } \
      \
      if (tuner.tuning_active()) { \
        cudaEventRecord(tuning_post_event, stream); \
        cudaEventSynchronize(tuning_post_event); \
        \
        if (!block_size_works) { \
          cudaError_t last_error = cudaGetLastError(); \
          if (last_error == cudaErrorLaunchOutOfResources || last_error == cudaErrorInvalidConfiguration) { \
            -- tuning_iteration; \
            var_block_width = 0; \
            CHECK_GE(tuning_iteration, 0) << "CUDA auto tuner: Could not find any working configuration for this kernel."; \
            continue; \
          } \
          block_size_works = true; \
        } \
        \
        float elapsed_milliseconds; \
        cudaEventElapsedTime(&elapsed_milliseconds, tuning_pre_event, tuning_post_event); \
        tuner.AddTuningMeasurement(const_kernel_name, \
                                   var_block_width, var_block_height, \
                                   0.001 * elapsed_milliseconds); \
      } \
    } while (!block_size_works); \
    } while (false)

#define CUDA_AUTO_TUNE_2D_TEMPLATED( \
    kernel_name, \
    default_block_width, default_block_height, \
    domain_width, domain_height, \
    shared_memory_size, stream, \
    template_parameters, \
    ...) \
    CUDA_AUTO_TUNE_2D_BORDER_TEMPLATED( \
        kernel_name, \
        default_block_width, default_block_height, \
        0, 0, \
        domain_width, domain_height, \
        shared_memory_size, stream, \
        TEMPLATE_ARGUMENTS(template_parameters), \
        __VA_ARGS__)

// The kernel name must be given directly (not as a string). Template parameters
// may be included in the kernel name as long as they do not depend on the
// chosen block size (if they do, use CUDA_AUTO_TUNE_2D_TEMPLATED()).
// Tests the following block sizes:
// 32
// 64
// 128
// 256
// 512
// 1024
#define CUDA_AUTO_TUNE_1D( \
    kernel_name, \
    default_block_width, \
    domain_width, \
    shared_memory_size, stream, \
    ...) \
    do { \
    vis::CUDAAutoTuner& tuner = vis::CUDAAutoTuner::Instance(); \
    static bool block_size_works = !tuner.tuning_active(); \
    do { \
      static const char* const_kernel_name = #kernel_name; \
      static int tuning_iteration = tuner.tuning_iteration(); \
      static cudaEvent_t tuning_pre_event; \
      static cudaEvent_t tuning_post_event; \
      static int var_block_width = 0; \
      if (var_block_width == 0) { \
        tuner.GetParametersForKernel(const_kernel_name, 1, \
                                     tuning_iteration, \
                                     default_block_width, 1, \
                                     &var_block_width, nullptr); \
      } \
      dim3 grid_dim(GetBlockCount(domain_width, var_block_width)); \
      dim3 block_dim(var_block_width); \
      \
      if (tuner.tuning_active()) { \
        if (tuning_pre_event == 0) { \
          cudaEventCreate(&tuning_pre_event); \
          cudaEventCreate(&tuning_post_event); \
          cudaGetLastError(); \
        } \
        cudaEventRecord(tuning_pre_event, stream); \
      } \
      \
      kernel_name<<<grid_dim, block_dim, shared_memory_size, stream>>>( \
          __VA_ARGS__); \
      \
      if (tuner.tuning_active()) { \
        cudaEventRecord(tuning_post_event, stream); \
        cudaEventSynchronize(tuning_post_event); \
        \
        if (!block_size_works) { \
          cudaError_t last_error = cudaGetLastError(); \
          if (last_error == cudaErrorLaunchOutOfResources || last_error == cudaErrorInvalidConfiguration) { \
            -- tuning_iteration; \
            var_block_width = 0; \
            CHECK_GE(tuning_iteration, 0) << "CUDA auto tuner: Could not find any working configuration for this kernel."; \
            continue; \
          } \
          block_size_works = true; \
        } \
        \
        float elapsed_milliseconds; \
        cudaEventElapsedTime(&elapsed_milliseconds, tuning_pre_event, tuning_post_event); \
        tuner.AddTuningMeasurement(const_kernel_name, \
                                   var_block_width, 1, \
                                   0.001 * elapsed_milliseconds); \
      } \
    } while (!block_size_works); \
    } while (false)

// The kernel name must be given directly (not as a string).
// The block width and height can be used in the template parameters as
// block_width and block_height.
// Tests the following block sizes:
// 32
// 64
// 128
// 256
// 512
// 1024
#define CUDA_AUTO_TUNE_1D_TEMPLATED( \
    kernel_name, \
    default_block_width, \
    domain_width, \
    shared_memory_size, stream, \
    template_parameters, \
    ...) \
    do { \
    vis::CUDAAutoTuner& tuner = vis::CUDAAutoTuner::Instance(); \
    static bool block_size_works = !tuner.tuning_active(); \
    do { \
      static const char* const_kernel_name = #kernel_name; \
      static int tuning_iteration = tuner.tuning_iteration(); \
      static cudaEvent_t tuning_pre_event; \
      static cudaEvent_t tuning_post_event; \
      static int var_block_width = 0; \
      if (var_block_width == 0) { \
        tuner.GetParametersForKernel(const_kernel_name, 1, \
                                     tuning_iteration, \
                                     default_block_width, 1, \
                                     &var_block_width, nullptr); \
      } \
      dim3 grid_dim(GetBlockCount(domain_width, var_block_width)); \
      dim3 block_dim(var_block_width); \
      \
      if (tuner.tuning_active()) { \
        if (tuning_pre_event == 0) { \
          cudaEventCreate(&tuning_pre_event); \
          cudaEventCreate(&tuning_post_event); \
          cudaGetLastError(); \
        } \
        cudaEventRecord(tuning_pre_event, stream); \
      } \
      \
      if (var_block_width == 32) { \
        constexpr int block_width = 32; \
        (void) block_width; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 64) { \
        constexpr int block_width = 64; \
        (void) block_width; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 128) { \
        constexpr int block_width = 128; \
        (void) block_width; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 256) { \
        constexpr int block_width = 256; \
        (void) block_width; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 512) { \
        constexpr int block_width = 512; \
        (void) block_width; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else if (var_block_width == 1024) { \
        constexpr int block_width = 1024; \
        (void) block_width; \
        kernel_name<template_parameters> \
        <<<grid_dim, block_dim, shared_memory_size, stream>>>(__VA_ARGS__); \
      } else { \
        LOG(FATAL) << "Error in CUDAAutoTuner: Invalid block size given."; \
      } \
      \
      if (tuner.tuning_active()) { \
        cudaEventRecord(tuning_post_event, stream); \
        cudaEventSynchronize(tuning_post_event); \
        \
        if (!block_size_works) { \
          cudaError_t last_error = cudaGetLastError(); \
          if (last_error == cudaErrorLaunchOutOfResources || last_error == cudaErrorInvalidConfiguration) { \
            -- tuning_iteration; \
            var_block_width = 0; \
            CHECK_GE(tuning_iteration, 0) << "CUDA auto tuner: Could not find any working configuration for this kernel."; \
            continue; \
          } \
          block_size_works = true; \
        } \
        \
        float elapsed_milliseconds; \
        cudaEventElapsedTime(&elapsed_milliseconds, tuning_pre_event, tuning_post_event); \
        tuner.AddTuningMeasurement(const_kernel_name, \
                                   var_block_width, 1, \
                                   0.001 * elapsed_milliseconds); \
      } \
    } while (!block_size_works); \
    } while (false)

// Helper to pass template arguments as a single macro parameter
#define TEMPLATE_ARGUMENTS(...) __VA_ARGS__

}

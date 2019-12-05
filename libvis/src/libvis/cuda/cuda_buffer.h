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

#include <memory>

#include <cuda_runtime.h>

#include "libvis/cuda/cuda_buffer.cuh"
#include "libvis/image.h"
#include "libvis/image_display.h"

namespace vis {

// Utility class which encapsulates a block of pitched 2D CUDA device memory.
// The part of a CUDABuffer object which is transferred to the GPU on CUDA
// kernel invocations is represented by the contained CUDABuffer_ object.
template <typename T>
class CUDABuffer {
 public:
  // Make the type accessible from outside.
  typedef T Type;

  // Allocates a new CUDA buffer with the given size and undefined content.
  CUDABuffer(int height, int width);
  
  // Disallow copying.
  CUDABuffer(const CUDABuffer<T>&) = delete;

  // Frees the device buffer.
  ~CUDABuffer();

  // Uploads the (non-pitched) data to the device buffer.
  // For use in debug code only, otherwise use an async version.
  void DebugUpload(const T* data);
  // Uploads the data to the device buffer.
  // For use in debug code only, otherwise use an async version.
  void DebugUploadPitched(size_t pitch, const T* data);
  // Uploads the (non-pitched) data asynchronously to the device buffer.
  void UploadAsync(cudaStream_t stream, const T* data);
  // Uploads the image asynchronously to the device buffer.
  void UploadAsync(cudaStream_t stream, const Image<T>& data);
  // Uploads the data asynchronously to the device buffer.
  void UploadPitchedAsync(cudaStream_t stream, size_t pitch, const T* data);
  // Uploads data to a part of the device buffer asynchronously.
  // Start and length are given in bytes. Intended for 1D arrays.
  void UploadPartAsync(size_t start, size_t length, cudaStream_t stream,
                       const T* data);

  // Downloads the device buffer data to a (non-pitched) CPU buffer.
  // For use in debug code only, otherwise use an async version.
  void DebugDownload(T* data) const;
  // Downloads the device buffer data to a CPU buffer.
  // For use in debug code only, otherwise use an async version.
  void DebugDownloadPitched(size_t pitch, T* data) const;
  // Downloads the device buffer data asynchronously to the CPU.
  void DownloadAsync(cudaStream_t stream, T* data) const;
  // Downloads the device buffer data asynchronously to the CPU.
  void DownloadAsync(cudaStream_t stream, Image<T>* data) const;
  // Downloads the data asynchronously from the device buffer.
  void DownloadPitchedAsync(cudaStream_t stream, size_t pitch, T* data);
  // Downloads a part of the device buffer data asynchronously to the CPU.
  // Start and length are given in bytes. Intended for 1D arrays.
  void DownloadPartAsync(size_t start, size_t length, cudaStream_t stream,
                         T* data) const;

  // Fills the whole buffer with the given value.
  void Clear(T value, cudaStream_t stream);
  // Sets the buffer to the content of a texture of the same type
  // (non-normalized coordinate access, non-normalized read mode).
  void SetTo(cudaTextureObject_t texture, cudaStream_t stream);
  // Sets the buffer to the content of a texture of the same type
  // (non-normalized coordinate access, normalized read mode).
  void SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);
  // Sets the buffer to the content of another buffer of the same type.
  void SetTo(const CUDABuffer<T>& other, cudaStream_t stream);

  // Creates a texture object for accessing the buffer.
  // You must destroy it after use with cudaDestroyTextureObject().
  void CreateTextureObject(cudaTextureAddressMode address_mode_x,
                           cudaTextureAddressMode address_mode_y,
                           cudaTextureFilterMode filter_mode,
                           cudaTextureReadMode read_mode,
                           bool use_normalized_coordinates,
                           cudaTextureObject_t* texture_object) const;
  
  void DebugDisplay(cudaStream_t stream, const string& window_title, float min_value, float max_value) const;
  void DebugDisplay(cudaStream_t stream, ImageDisplay* display, const string& window_title, float min_value, float max_value) const;
  
  inline int width() const { return data_.width_; }
  inline int height() const { return data_.height_; }
  // Returns the entire buffer size in bytes.
  inline int Size() const { return data_.pitch_ * data_.height_; }

  // Returns the object that can be passed to CUDA code.
  __host__ __device__ inline const CUDABuffer_<T>& ToCUDA() const { return data_; }
  __host__ __device__ inline CUDABuffer_<T>& ToCUDA() { return data_; }

 private:
  // Data that will be passed to CUDA code.
  CUDABuffer_<T> data_;
};

template <typename T>
using CUDABufferPtr = std::shared_ptr<CUDABuffer<T>>;

template <typename T>
using CUDABufferConstPtr = std::shared_ptr<const CUDABuffer<T>>;

}  // namespace vis

#include "libvis/cuda/cuda_buffer_inl.h"

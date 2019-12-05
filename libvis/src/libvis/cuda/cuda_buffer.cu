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


#include "libvis/cuda/cuda_buffer.cuh"

#include <limits>

#include <curand_kernel.h>

#include "libvis/cuda/cuda_util.h"

namespace vis {

template<typename T>
__global__ void CUDABufferClearKernel(CUDABuffer_<T> buffer, T value) {
  // NOTE: according to fedeDev, it may be faster to ensure writing to at least
  // 2 to 4 words in every thread if sizeof(T) is small.
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < buffer.width() && y < buffer.height()) {
    buffer(y, x) = value;
  }
}

template<typename T>
void CUDABuffer_<T>::Clear(T value, cudaStream_t stream) {
  constexpr int kTileWidth = 32;
  constexpr int kTileHeight = 8;
  dim3 grid_dim(GetBlockCount(width(), kTileWidth),
                GetBlockCount(height(), kTileHeight));
  dim3 block_dim(kTileWidth, kTileHeight);
  CUDABufferClearKernel<<<grid_dim, block_dim, 0, stream>>>(*this, value);
}

template<typename T>
__global__ void CUDABufferSetToKernel(CUDABuffer_<T> buffer,
                                      cudaTextureObject_t texture) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < buffer.width() && y < buffer.height()) {
    buffer(y, x) = tex2D<T>(texture, x + 0.5f, y + 0.5f);
  }
}

template<typename T>
void CUDABuffer_<T>::SetTo(cudaTextureObject_t texture, cudaStream_t stream) {
  constexpr int kTileWidth = 32;
  constexpr int kTileHeight = 8;
  dim3 grid_dim(GetBlockCount(width(), kTileWidth),
                GetBlockCount(height(), kTileHeight));
  dim3 block_dim(kTileWidth, kTileHeight);
  CUDABufferSetToKernel<<<grid_dim, block_dim, 0, stream>>>(*this, texture);
}

template<typename T>
__global__ void CUDABufferSetToReadModeNormalizedKernel(
    CUDABuffer_<T> buffer,
    cudaTextureObject_t texture,
    T factor) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < buffer.width() && y < buffer.height()) {
    buffer(y, x) = factor * tex2D<float>(texture, x + 0.5f, y + 0.5f);
  }
}

template<typename T>
void CUDABuffer_<T>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream) {
  constexpr int kTileWidth = 32;
  constexpr int kTileHeight = 8;
  dim3 grid_dim(GetBlockCount(width(), kTileWidth),
                GetBlockCount(height(), kTileHeight));
  dim3 block_dim(kTileWidth, kTileHeight);
  CUDABufferSetToReadModeNormalizedKernel<<<grid_dim, block_dim, 0, stream>>>(
      *this, texture, numeric_limits<T>::max());
}

template<typename T>
__global__ void CUDABufferSetToKernel(CUDABuffer_<T> buffer,
                                      CUDABuffer_<T> other) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < buffer.width() && y < buffer.height()) {
    buffer(y, x) = other(y, x);
  }
}

template<typename T>
void CUDABuffer_<T>::SetTo(const CUDABuffer_<T>& other, cudaStream_t stream) {
  constexpr int kTileWidth = 32;
  constexpr int kTileHeight = 8;
  dim3 grid_dim(GetBlockCount(width(), kTileWidth),
                GetBlockCount(height(), kTileHeight));
  dim3 block_dim(kTileWidth, kTileHeight);
  CUDABufferSetToKernel<<<grid_dim, block_dim, 0, stream>>>(*this, other);
}

// Avoid compilation of some functions for some types by declaring but not
// defining a specialization for it.
template<> void CUDABuffer_<curandState>::SetTo(cudaTextureObject_t texture, cudaStream_t stream);
template<> void CUDABuffer_<curandState>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);

template<> void CUDABuffer_<float2>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);
template<> void CUDABuffer_<char2>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);
template<> void CUDABuffer_<char4>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);
template<> void CUDABuffer_<uchar4>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);

// TODO: This would need a specialization which uses 1 for the scaling factor, not numeric_limits<float>::max()!
template<> void CUDABuffer_<float>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);

// Same as above for double.
template<> void CUDABuffer_<double>::SetTo(cudaTextureObject_t texture, cudaStream_t stream);
template<> void CUDABuffer_<double>::SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);

// Precompile all variants of CUDABuffer_ that are used.
// The alternative would be to move above functions to an inl header,
// but then all files including cuda_buffer.h (and thus this inl header)
// would need to be compiled by nvcc.
template class CUDABuffer_<float>;
template class CUDABuffer_<float2>;
template class CUDABuffer_<double>;
template class CUDABuffer_<int>;
template class CUDABuffer_<unsigned int>;
template class CUDABuffer_<uint8_t>;
template class CUDABuffer_<uint16_t>;
template class CUDABuffer_<char2>;
template class CUDABuffer_<char4>;
template class CUDABuffer_<uchar4>;
template class CUDABuffer_<curandState>;

}  // namespace vis

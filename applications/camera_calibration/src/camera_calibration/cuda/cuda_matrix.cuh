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

#include <libvis/libvis.h>

namespace vis {

template <typename Scalar, int Rows, int Cols>
class CUDAMatrix;

template <typename Scalar, int Rows, int Cols, int Range>
class CUDAMatrixCols {
 public:
  enum {
    VisibleRows = Rows,
    VisibleCols = Range,
  };
  typedef Scalar _Scalar;
  
  __forceinline__ __device__ CUDAMatrixCols(CUDAMatrix<Scalar, Rows, Cols>* matrix, int start)
      : data_(matrix->data() + start) {}
  
  __forceinline__ __device__ Scalar& operator() (int row, int col) const {
    return data_[row * Cols + col];
  }
  
  __forceinline__ __device__ Scalar& operator() (int position) const {
    static_assert(Rows == 1 || Range == 1, "Single-dimension access is only for vectors.");
    if (Rows == 1) {
      return data_[position];
    } else {  // if (Range == 1)
      return data_[position * Cols];
    }
  }
  
  __forceinline__ __device__ Scalar* row(int row) const {
    return data_ + row * Cols;
  }
  
 private:
  Scalar* data_;
};

// Fixed-size matrix class compatible with CUDA code (unlike Eigen with Visual Studio), using row-wise storage.
template <typename Scalar, int Rows, int Cols>
class CUDAMatrix {
 public:
  enum {
    VisibleRows = Rows,
    VisibleCols = Cols,
  };
  typedef Scalar _Scalar;
  
  __forceinline__ __device__ const Scalar& operator() (int row, int col) const {
    return data_[row * Cols + col];
  }
  __forceinline__ __device__ Scalar& operator() (int row, int col) {
    return data_[row * Cols + col];
  }
  
  __forceinline__ __device__ const Scalar& operator() (int position) const {
    static_assert(Rows == 1 || Cols == 1, "Single-dimension access is only for vectors.");
    return data_[position];
  }
  __forceinline__ __device__ Scalar& operator() (int position) {
    static_assert(Rows == 1 || Cols == 1, "Single-dimension access is only for vectors.");
    return data_[position];
  }
  
  __forceinline__ __device__ const Scalar* row(int row) const {
    return data_ + row * Cols;
  }
  __forceinline__ __device__ Scalar* row(int row) {
    return data_ + row * Cols;
  }
  
  __forceinline__ __device__ const Scalar* data() const {
    return data_;
  }
  __forceinline__ __device__ Scalar* data() {
    return data_;
  }
  
  template <int Range>
  __forceinline__ __device__ CUDAMatrixCols<Scalar, Rows, Cols, Range> cols(int start) {
    return CUDAMatrixCols<Scalar, Rows, Cols, Range>(this, start);
  }
  
 private:
  Scalar data_[Rows * Cols];
};

// TODO: De-duplicate the implementations below (const / non-const ResultT for CUDAMatrixCols / CUDAMatrix)

template <typename ResultT, typename AT, typename BT>
__forceinline__ __device__ void MatrixMultiply(
    const ResultT& result,
    const AT& a,
    const BT& b) {
  static_assert(ResultT::VisibleRows == AT::VisibleRows, "MatrixMultiply: The matrix dimensions do not match.");
  static_assert(ResultT::VisibleCols == BT::VisibleCols, "MatrixMultiply: The matrix dimensions do not match.");
  static_assert(AT::VisibleCols == BT::VisibleRows, "MatrixMultiply: The matrix dimensions do not match.");
  
  for (int r = 0; r < ResultT::VisibleRows; ++ r) {
    for (int c = 0; c < ResultT::VisibleCols; ++ c) {
      typename ResultT::_Scalar sum = 0;
      for (int i = 0; i < AT::VisibleCols; ++ i) {
        sum += a(r, i) * b(i, c);
      }
      result(r, c) = sum;
    }
  }
}

template <typename ResultT, typename AT, typename BT>
__forceinline__ __device__ void MatrixMultiply(
    ResultT& result,
    const AT& a,
    const BT& b) {
  static_assert(ResultT::VisibleRows == AT::VisibleRows, "MatrixMultiply: The matrix dimensions do not match.");
  static_assert(ResultT::VisibleCols == BT::VisibleCols, "MatrixMultiply: The matrix dimensions do not match.");
  static_assert(AT::VisibleCols == BT::VisibleRows, "MatrixMultiply: The matrix dimensions do not match.");
  
  for (int r = 0; r < ResultT::VisibleRows; ++ r) {
    for (int c = 0; c < ResultT::VisibleCols; ++ c) {
      typename ResultT::_Scalar sum = 0;
      for (int i = 0; i < AT::VisibleCols; ++ i) {
        sum += a(r, i) * b(i, c);
      }
      result(r, c) = sum;
    }
  }
}

}

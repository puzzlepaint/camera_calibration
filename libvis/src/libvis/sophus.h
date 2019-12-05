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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/StdVector>
#include <sophus/rxso3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <sophus/so2.hpp>
#include <sophus/so3.hpp>


#include "libvis/libvis.h"

#ifdef __CUDA_ARCH__
#error "Compiling Eigen code with nvcc is not supported properly, therefore Eigen headers should not be included from .cu / .cuh files."
#endif

namespace vis {

// Using the full namespace was discarded as it conflicted with importing the
// Eigen namespace: Matrix was defined in both.
using Sophus::RxSO3f;
using Sophus::RxSO3d;

using Sophus::SE2f;
using Sophus::SE2d;

using Sophus::SE3f;
using Sophus::SE3d;

using Sophus::Sim3f;
using Sophus::Sim3d;

using Sophus::SO2f;
using Sophus::SO2d;

using Sophus::SO3f;
using Sophus::SO3d;

/// Averages the given Sophus::SE3 transformations.
template <typename Scalar>
Sophus::SE3<Scalar> AverageSE3(int count, Sophus::SE3<Scalar>* transformations) {
  Eigen::Matrix<Scalar, 3, 3> accumulated_rotations;
  accumulated_rotations.setZero();
  Eigen::Matrix<Scalar, 3, 1> accumulated_translations;
  accumulated_translations.setZero();
  
  for (int i = 0; i < count; ++ i) {
    accumulated_rotations += transformations[i].so3().matrix();
    accumulated_translations += transformations[i].translation();
  }
  
  Sophus::SE3<Scalar> result;
  Eigen::JacobiSVD<Eigen::Matrix<Scalar, 3, 3>> svd(accumulated_rotations, Eigen::ComputeFullU | Eigen::ComputeFullV);
  result.setRotationMatrix((svd.matrixU() * svd.matrixV().transpose()));
  result.translation() = (accumulated_translations / (static_cast<Scalar>(1) * count));
  return result;
}

}

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::RxSO3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::RxSO3d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE2f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE2d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SE3d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::Sim3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::Sim3d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SO2f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SO2d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SO3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Sophus::SO3d)

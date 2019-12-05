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


// Eigen should always be included via this file. This imports it into the vis
// namespace and defines suitable typedefs to use Eigen types in STL containers.
#pragma once

#include <vector>


#ifdef __CUDA_ARCH__
#error "Compiling Eigen code with nvcc is not supported properly, therefore Eigen headers should not be included from .cu / .cuh files."
#endif


#pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #if __GNUC__ >= 7
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    #pragma GCC diagnostic ignored "-Wmisleading-indentation"
    #pragma GCC diagnostic ignored "-Wint-in-bool-context"
  #endif
  #include <Eigen/Core>
  #include <Eigen/Dense>
  #include <Eigen/StdVector>
#pragma GCC diagnostic pop

#include "libvis/libvis.h"


// Specialize std::vector for common types for correct alignment without having to use aligned_vector.
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4d);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4d);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Affine3d);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Affine3f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Quaterniond);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Quaternionf);


namespace vis {

// Import the Eigen namespace into the vis namespace.
using namespace Eigen;

// Define aligned STL types which can be used with Eigen objects. Note that this
// is only required for fixed-size, vectorizable types. For example,
// Eigen::Vector3f is not vectorizable. See:
// http://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
template<class T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

// Rename the built-in types like Vector3f and Matrix3f to shorter names like
// Vec3f and Mat3f.
#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)         \
/** \ingroup matrixtypedefs */                                          \
typedef Matrix##SizeSuffix##TypeSuffix    Mat##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                          \
typedef Vector##SizeSuffix##TypeSuffix    Vec##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                          \
typedef RowVector##SizeSuffix##TypeSuffix RowVec##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)      \
/** \ingroup matrixtypedefs */                                 \
typedef Matrix##Size##X##TypeSuffix Mat##Size##X##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                 \
typedef Matrix##X##Size##TypeSuffix Mat##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int,                  i)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(float,                f)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(double,               d)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<float>,  cf)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<double>, cd)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

// Define VecXu8 types. The Vec3u8 type is commonly used as the pixel
// type for color images.
#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Size> Mat##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, 1>    Vec##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, 1, Size>    RowVec##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Dynamic> Mat##Size##X##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Dynamic, Size> Mat##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(u8, u8)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

}

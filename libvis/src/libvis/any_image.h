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

#include "libvis/eigen.h"
#include "libvis/image.h"
#include "libvis/libvis.h"

namespace vis {

// Supported types for the AnyImage class.
enum class ImageType {
  U8 = 0,
  U16,
  U32,
  U64,
  
  I8,
  I16,
  I32,
  I64,
  
  Vec2u8,
  Vec3u8,
  Vec4u8,
  
  Float,
  Double,
  
  Invalid
};

// Traits type allowing to get the ImageType enum value for a given type as:
// GetImageType<Type>::value. For example, GetImageType<u32>::value resoves to
// ImageType::U32.
template <typename T> struct GetImageType {
  // Default value for unknown type.
  static const ImageType value = ImageType::Invalid;
};

template<> struct GetImageType<u8> {static const ImageType value = ImageType::U8;};
template<> struct GetImageType<u16> {static const ImageType value = ImageType::U16;};
template<> struct GetImageType<u32> {static const ImageType value = ImageType::U32;};
template<> struct GetImageType<u64> {static const ImageType value = ImageType::U64;};

template<> struct GetImageType<i8> {static const ImageType value = ImageType::I8;};
template<> struct GetImageType<i16> {static const ImageType value = ImageType::I16;};
template<> struct GetImageType<i32> {static const ImageType value = ImageType::I32;};
template<> struct GetImageType<i64> {static const ImageType value = ImageType::I64;};

template<> struct GetImageType<Vec2u8> {static const ImageType value = ImageType::Vec2u8;};
template<> struct GetImageType<Vec3u8> {static const ImageType value = ImageType::Vec3u8;};
template<> struct GetImageType<Vec4u8> {static const ImageType value = ImageType::Vec4u8;};

template<> struct GetImageType<float> {static const ImageType value = ImageType::Float;};
template<> struct GetImageType<double> {static const ImageType value = ImageType::Double;};

// Traits type allowing to determine whether T is an Eigen::Matrix type by
// accessing IsMatrixType<T>::value for the type T in question.
template<typename T>
struct IsMatrixType {
  // Default value for unknown type.
  static const bool value = false;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct IsMatrixType<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  static const bool value = true;
};


// Can be used on an ImageType that is only known at runtime to execute code
// which knows the corresponding type at compile time.
// 
// Compiles the statement(s) given as variadic parameters with a typedef
// resembling object_type, which must be the name of a variable of type
// ImageType. If "name" is the variable name, the typedef will be named
// "_name".
//
// Example:
//   template <typename T>
//   void SomeFunction() {
//     // Use the image type T in some way ...
//   }
// 
//   ImageType type = image.type();
//   IDENTIFY_IMAGE_TYPE(type, SomeFunction<_image>());
#define IDENTIFY_IMAGE_TYPE(object_type, ...)                                \
  do {                                                                       \
    if ((object_type) == ImageType::U8) {                                    \
      typedef u8 _##object_type;                                             \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::U16) {                            \
      typedef u16 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::U32) {                            \
      typedef u32 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::U64) {                            \
      typedef u64 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I8) {                             \
      typedef i8 _##object_type;                                             \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I16) {                            \
      typedef i16 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I32) {                            \
      typedef i32 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I64) {                            \
      typedef i64 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Vec2u8) {                         \
      typedef Vec2u8 _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Vec3u8) {                         \
      typedef Vec3u8 _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Vec4u8) {                         \
      typedef Vec4u8 _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Float) {                          \
      typedef float _##object_type;                                          \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Double) {                         \
      typedef double _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else {                                                                 \
      LOG(FATAL) << "IDENTIFY_IMAGE_TYPE() encountered an invalid type: " << static_cast<int>(object_type); \
    }                                                                        \
  } while (false)

// Given an AnyImage, this macro can be used to call code statements which know
// the type of the image at compile time.
// 
// Compiles the statement(s) given as variadic parameters for all possible image
// types. The object parameter must be the name of an object of type AnyImage or
// (const) AnyImage&. If "name" is the object name, the given statement(s) can
// use the type of this object as a typdef "_name_type", and a pointer
// Image<_name_type>* named "_name".
// 
// If your code does not compile for all types DoSomething() calls it with,
// one solution is to make it call an overloaded function which does nothing or
// prints an error message for types which are not supported. Another solution
// is to call IDENTIFY_SCALAR_IMAGE() or IDENTIFY_MATRIX_IMAGE() if your code
// compiles for scalar or matrix types only, respectively.
// 
// Example use of IDENTIFY_IMAGE():
// 
// template <typename T>
// void DoSomething(Image<T>* image) {
//   // Use the typed image ...
// }
// 
// AnyImage image;
// // Initialize image ...
// IDENTIFY_IMAGE(image, DoSomething(_image));
#define IDENTIFY_IMAGE(object, ...)                                            \
  do {                                                                         \
    vis::ImageType object##_type = (object).type();                            \
    IDENTIFY_IMAGE_TYPE(object##_type,                                         \
        Image<_##object##_type>* _##object = (object).get<_##object##_type>(); \
        (void)_##object;                                                       \
        __VA_ARGS__;                                                           \
    );                                                                         \
  } while (false)


// Variant of IDENTIFY_IMAGE_TYPE. Only calls the given statement(s) if the
// given type is a scalar type, does nothing otherwise.
#define IDENTIFY_SCALAR_IMAGE_TYPE(object_type, ...)                         \
  do {                                                                       \
    if ((object_type) == ImageType::U8) {                                    \
      typedef u8 _##object_type;                                             \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::U16) {                            \
      typedef u16 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::U32) {                            \
      typedef u32 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::U64) {                            \
      typedef u64 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I8) {                             \
      typedef i8 _##object_type;                                             \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I16) {                            \
      typedef i16 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I32) {                            \
      typedef i32 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::I64) {                            \
      typedef i64 _##object_type;                                            \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Float) {                          \
      typedef float _##object_type;                                          \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Double) {                         \
      typedef double _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Invalid) {                        \
      LOG(FATAL) << "IDENTIFY_IMAGE_TYPE() encountered an invalid type: " << static_cast<int>(object_type); \
    }                                                                        \
  } while (false)

// Variant of IDENTIFY_IMAGE.
// Compiles the statement(s) given as variadic parameters for scalar image types.
// For matrix-valued images, does not execute the statements.
#define IDENTIFY_SCALAR_IMAGE(object, ...)                                     \
  do {                                                                         \
    vis::ImageType object##_type = (object).type();                            \
    IDENTIFY_SCALAR_IMAGE_TYPE(object##_type,                                  \
        Image<_##object##_type>* _##object = (object).get<_##object##_type>(); \
        (void)_##object;                                                       \
        __VA_ARGS__;                                                           \
    );                                                                         \
  } while (false)


// Variant of IDENTIFY_IMAGE_TYPE. Only calls the given statement(s) if the
// given type is a matrix resp. vector type, does nothing otherwise.
#define IDENTIFY_MATRIX_IMAGE_TYPE(object_type, ...)                         \
  do {                                                                       \
    if ((object_type) == ImageType::Vec2u8) {                                \
      typedef Vec2u8 _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Vec3u8) {                         \
      typedef Vec3u8 _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Vec4u8) {                         \
      typedef Vec4u8 _##object_type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((object_type) == ImageType::Invalid) {                        \
      LOG(FATAL) << "IDENTIFY_IMAGE_TYPE() encountered an invalid type: " << static_cast<int>(object_type); \
    }                                                                        \
  } while (false)

// Variant of IDENTIFY_IMAGE.
// Compiles the statement(s) given as variadic parameters for matrix image types.
// For scalar-valued images, does not execute the statements.
#define IDENTIFY_MATRIX_IMAGE(object, ...)                                     \
  do {                                                                         \
    vis::ImageType object##_type = (object).type();                            \
    IDENTIFY_MATRIX_IMAGE_TYPE(object##_type,                                  \
        Image<_##object##_type>* _##object = (object).get<_##object##_type>(); \
        (void)_##object;                                                       \
        __VA_ARGS__;                                                           \
    );                                                                         \
  } while (false)


// Holds image data of any type. This is in contrast to the Image class, which
// holds image data of a type that is known at compile time. If performance
// matters, Image should be preferred when possible.
// 
// For working with AnyImages without assuming a specific type, the
// IDENTIFY_IMAGE and IDENTIFY_IMAGE_TYPE macros can be used to call a templated
// function (or directly run a small templated code fragment) which will be
// called with the correct type. Notice that IDENTIFY_IMAGE will compile the
// templated code for any possible image type, which includes scalar and vector
// or matrix types. This can be an issue if the code only compiles for one of
// these (for example, static_cast<float>(value) only works for scalar types).
// Therefore, IDENTIFY_SCALAR_IMAGE and IDENTIFY_MATRIX_IMAGE can be used to
// test for scalar or matrix / vector types only and only compile the templated
// code for the given kind of types.
class AnyImage {
 public:
  // Creates an empty / invalid image.
  inline AnyImage()
      : type_(ImageType::Invalid), data_(nullptr) {}
  
  // Constructs a new image of the given type and size.
  inline AnyImage(ImageType type, int width, int height) {
    type_ = type;
    if (type == ImageType::Invalid) {
      data_ = nullptr;
    } else {
      IDENTIFY_IMAGE_TYPE(type, data_ = new Image<_type>(width, height););
    }
  }
  
  // Creates a deep copy of the other image.
  template <typename T>
  inline AnyImage(const Image<T>& other) {
    type_ = GetImageType<T>::value;
    if (type_ != ImageType::Invalid) {
      data_ = new Image<T>(other);
    } else {
      data_ = nullptr;
    }
  }
  
  ~AnyImage() {
    DeleteData();
  }
  
  
  // Creates a deep copy of the other image.
  template <typename T>
  AnyImage& operator= (const Image<T>& other) {
    DeleteData();
    type_ = GetImageType<T>::value;
    if (type_ != ImageType::Invalid) {
      data_ = new Image<T>(other);
    }
    return *this;
  }
  
  
  // Returns the image type (which may also be ImageType::Invalid).
  inline ImageType type() const { return type_; }
  
  // Returns true if this object does not contain image data.
  inline bool empty() const { return data_ == nullptr; }
  
  // Returns whether the contained type is a matrix (or vector) type (as opposed
  // to a scalar type). This function must not be called for images with invalid
  // type.
  inline bool is_matrix_type() const {
    IDENTIFY_IMAGE_TYPE(type_, return IsMatrixType<_type_>::value;);
    return false;
  }
  
  // Retrieves the underlying Image, assuming that it has the given type T.
  // The caller is responsible for ensuring that T matches the actual type.
  // Calling this with a wrong type can result in undefined behavior!
  template <typename T>
  inline Image<T>* get() { return reinterpret_cast<Image<T>*>(data_); }
  
  // Retrieves the underlying Image, assuming that it has the given type T.
  // The caller is responsible for ensuring that T matches the actual type.
  // Calling this with a wrong type can result in undefined behavior!
  template <typename T>
  inline const Image<T>* get() const { return reinterpret_cast<Image<T>*>(data_); }
  
 private:
  // Sets the image to empty.
  void DeleteData() {
    if (data_) {
      AnyImage& image = *this;
      IDENTIFY_IMAGE(image, delete _image;);
      data_ = nullptr;
    }
  }
  
  ImageType type_;
  
  // Pointer to an Image<Type>, where Type is only known at runtime.
  void* data_;
};

}


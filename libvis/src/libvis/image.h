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

#ifdef LIBVIS_HAVE_OPENCV
  #include <opencv2/core/core.hpp>
#endif

#include "libvis/logging.h"

#include "libvis/eigen.h"
// #include "libvis/image_display_qt_window.h"
#include "libvis/image_io_libpng.h"
#include "libvis/image_io_netpbm.h"
#include "libvis/image_io_qt.h"
#include "libvis/libvis.h"
#include "libvis/qt_thread.h"

namespace vis {

// Vector type for image sizes.
typedef Matrix<u32, 2, 1> ImageSize;

// Implementation helper which makes the value false appear to be dependent on
// the type T to the compiler. This way, it is only evaluated when the function
// containing it is directly instantiated (as opposed to the instantiation of
// the whole template class). This enables to prevent calling a non-specialized
// individual function in a template class using a static_assert.
template<typename T>
struct always_false {
  static const bool value = false;
};

// Implementation helper which is used to check whether some type is a
// Eigen::Matrix type and then return its element count.
template<typename T>
struct channel_count_helper {
  static constexpr inline u32 channel_count() {
    return 1;
  }
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct channel_count_helper<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  static constexpr inline u32 channel_count() {
    static_assert(_Rows >= 0 && _Cols >= 0, "Matrices of dynamic size are not supported.");
    return _Rows * _Cols;
  }
};

// Implementation helper which suggests a default interpolation result type for image interpolation.
template<typename T>
struct float_type_helper {
  typedef float Type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct float_type_helper<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  typedef Matrix<float, _Rows, _Cols, _Options, _MaxRows, _MaxCols> Type;
};

// Implementation helpers for Image::SetTo(), which requires a different
// implementation for Eigen::Matrix types.
template <typename T>
inline void SetImageTo(Image<T>* image, const T value) {
  // Use (hopefully fast) memset() implementation if possible. The checks are
  // not exhaustive but should cover the most important cases.
  if (sizeof(T) == 1) {
    memset(image->data(), value, image->height() * image->stride());
  } else if (value == static_cast<T>(0)) {
    memset(image->data(), 0, image->height() * image->stride());
  } else {
    for (u32 y = 0; y < image->height(); ++ y) {
      T* write_ptr = image->row(y);
      T* end = write_ptr + image->width();
      while (write_ptr < end) {
        *write_ptr = value;
        ++ write_ptr;
      }
    }
  }
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void SetImageTo(Image<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>* image,
                       const Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> value) {
  typedef Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> T;
  for (u32 y = 0; y < image->height(); ++ y) {
    T* write_ptr = image->row(y);
    T* end = write_ptr + image->width();
    while (write_ptr < end) {
      *write_ptr = value;
      ++ write_ptr;
    }
  }
}

// Implementation helpers for Image::InterpolateBilinear(), which requires a different
// implementation for Eigen::Matrix types and scalar types.
template <typename InterpolatedT, typename T, typename Derived>
inline InterpolatedT InterpolateImageBilinear(
  const Image<T>* image,
  const MatrixBase<Derived>& position) {
  int ix = static_cast<int>(position.coeff(0));
  int iy = static_cast<int>(position.coeff(1));
  
  float fx = position.coeff(0) - ix;
  float fy = position.coeff(1) - iy;
  float fx_inv = 1.f - fx;
  float fy_inv = 1.f - fy;
  
  const T* ptr = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(image->data()) + iy * image->stride()) + ix;
  const T* ptr2 = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(ptr) + image->stride());
  
  return fx_inv * fy_inv * static_cast<InterpolatedT>(*ptr) +
         fx * fy_inv * static_cast<InterpolatedT>(*(ptr + 1)) +
         fx_inv * fy * static_cast<InterpolatedT>(*ptr2) +
         fx * fy * static_cast<InterpolatedT>(*(ptr2 + 1));
}

template <typename InterpolatedT, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename Derived>
inline InterpolatedT InterpolateImageBilinear(
    const Image<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>* image,
    const MatrixBase<Derived>& position) {
  typedef Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> T;
  
  int ix = static_cast<int>(position.coeff(0));
  int iy = static_cast<int>(position.coeff(1));
  
  float fx = position.coeff(0) - ix;
  float fy = position.coeff(1) - iy;
  float fx_inv = 1.f - fx;
  float fy_inv = 1.f - fy;
  
  const T* ptr = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(image->data()) + iy * image->stride()) + ix;
  const T* ptr2 = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(ptr) + image->stride());
  
  return fx_inv * fy_inv * ptr->template cast<typename InterpolatedT::Scalar>() +
         fx * fy_inv * (ptr + 1)->template cast<typename InterpolatedT::Scalar>() +
         fx_inv * fy * ptr2->template cast<typename InterpolatedT::Scalar>() +
         fx * fy * (ptr2 + 1)->template cast<typename InterpolatedT::Scalar>();
}


// Implementation helpers for Image::InterpolateBilinearWithJacobian(), which requires a different
// implementation for Eigen::Matrix types and scalar types.
template <typename T, typename InterpolatedT, typename JacobianScalarT, typename Derived>
inline void InterpolateImageBilinearWithJacobian(
    const Image<T>* image,
    const MatrixBase<Derived>& position,
    InterpolatedT* value,
    Matrix<JacobianScalarT, 1, 2>* jacobian) {
  int ix = static_cast<int>(position.coeff(0));
  int iy = static_cast<int>(position.coeff(1));
  
  float fx = position.coeff(0) - ix;
  float fy = position.coeff(1) - iy;
  float fx_inv = 1.f - fx;
  float fy_inv = 1.f - fy;
  
  const T* ptr = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(image->data()) + iy * image->stride()) + ix;
  const T* ptr2 = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(ptr) + image->stride());
  const T top_left = *ptr;
  const T top_right = *(ptr + 1);
  const T bottom_left = *ptr2;
  const T bottom_right = *(ptr2 + 1);
  
  InterpolatedT top = fx_inv * top_left + fx * top_right;
  InterpolatedT bottom = fx_inv * bottom_left + fx * bottom_right;
  *value = fy_inv * top + fy * bottom;
  jacobian->coeffRef(0) = fy * (bottom_right - bottom_left) +
                          fy_inv * (top_right - top_left);
  jacobian->coeffRef(1) = bottom - top;
}

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename InterpolatedT, typename JacobianScalarT, typename Derived>
inline void InterpolateImageBilinearWithJacobian(
    const Image<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>* image,
    const MatrixBase<Derived>& position,
    InterpolatedT* value,
    Matrix<JacobianScalarT, _Rows, 2>* jacobian) {  // _Rows == channel_count_helper<Image<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>>().channel_count()
  typedef Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> T;
  
  int ix = static_cast<int>(position.coeff(0));
  int iy = static_cast<int>(position.coeff(1));
  
  float fx = position.coeff(0) - ix;
  float fy = position.coeff(1) - iy;
  float fx_inv = 1.f - fx;
  float fy_inv = 1.f - fy;
  
  const T* ptr = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(image->data()) + iy * image->stride()) + ix;
  const T* ptr2 = reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(ptr) + image->stride());
  const T top_left = *ptr;
  const T top_right = *(ptr + 1);
  const T bottom_left = *ptr2;
  const T bottom_right = *(ptr2 + 1);
  
  InterpolatedT top = fx_inv * top_left + fx * top_right;
  InterpolatedT bottom = fx_inv * bottom_left + fx * bottom_right;
  *value = fy_inv * top + fy * bottom;
  jacobian->col(0) = (fy * (bottom_right - bottom_left) +
                      fy_inv * (top_right - top_left)).template cast<JacobianScalarT>();
  jacobian->col(1) = (bottom - top).template cast<JacobianScalarT>();
}


// Implementation helper which is used to check whether some type is a
// Eigen::Matrix type and return the underlying scalar type.
// TODO: Is this useful somewhere? If not, remove.
// template<typename T>
// struct get_scalar_type {
//   typedef T Type;
// };
// 
// template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
// struct get_scalar_type<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
//   typedef _Scalar Type;
// };

// Registration helpers for I/O handlers. They need to be referenced somehow
// for static initialization to be carried out if this code is in a static
// library. Otherwise they will be left out. Thus, they have been placed here.
struct ImageIOLibPngRegistrator {
  ImageIOLibPngRegistrator() {
    shared_ptr<ImageIOLibPng> io(new ImageIOLibPng());
    ImageIORegistry::Instance()->Register(io);
  }
};
extern ImageIOLibPngRegistrator image_io_libpng_registrator_;

struct ImageIONetPBMRegistrator {
  ImageIONetPBMRegistrator() {
    shared_ptr<ImageIONetPBM> io(new ImageIONetPBM());
    ImageIORegistry::Instance()->Register(io);
  }
};
extern ImageIONetPBMRegistrator image_io_netpbm_registrator_;

#ifdef LIBVIS_HAVE_QT
struct ImageIOQtRegistrator {
  ImageIOQtRegistrator() {
    shared_ptr<ImageIOQt> io(new ImageIOQt());
    ImageIORegistry::Instance()->Register(io);
  }
};
extern ImageIOQtRegistrator image_io_qt_registrator_;
#endif


// Macro for iteration over all pixels in an image. Guarantees no performance
// loss compared to a manual implementation, while being convenient.
// 
// Use as follows:
// FOREACH_PIXEL_XY(x, y, image) {
//   // Can access x and y here. The variable names are determined by the
//   // parameters above.
// }
#define FOREACH_PIXEL_XY_IN_BORDER(x, y, border, image) \
    for (u32 y = (border), y##_end = (image).height() - (border); y < y##_end; ++ y) \
      for (u32 x = (border), x##_end = (image).width() - (border); x < x##_end; ++ x)

#define FOREACH_PIXEL_XY(x, y, image) \
    FOREACH_PIXEL_XY_IN_BORDER(x, y, 0, image)


// Holds image data of a specific type.
// 
// Range-based for iteration over all pixels is possible in two different ways:
// 
//   // Iterate over pixels, not knowing their coordinates:
//   for (T& pixel : image.pixels()) { ... }
// 
//   // Iterate over pixels, knowing their coordinates:
//   // TODO: Profile this.
//   for (Image<T>::Pixel& pixel : image.c_pixels()) {
//     // Access: T& pixel.value, u32 pixel.x, u32 pixel.y
//     ...
//   }
// 
// If no stride is specified in the functions to allocate the image, it is
// chosen such that the data is stored in memory continuously.
template <typename T>
class Image {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Helper struct for range-based for iteration over all image pixels (without
  // coordinates).
  struct ImageConstPixels {
    struct Iterator {
      inline Iterator(const T* pointer, const T* line_end, u32 line_jump,
                      u32 stride)
          : pointer(pointer), line_end(line_end), line_jump(line_jump),
            stride(stride) {}
      
      inline Iterator& operator++() {
        ++ pointer;
        if (pointer == line_end) {
          pointer = reinterpret_cast<const T*>(
              reinterpret_cast<const uint8_t*>(pointer) + line_jump);
          line_end = reinterpret_cast<const T*>(
              reinterpret_cast<const uint8_t*>(line_end) + stride);
        }
        return *this;
      }
      
      inline bool operator!=(const Iterator& other) const {
        return pointer != other.pointer;
      }
      
      inline const T& operator*() const {
        return *pointer;
      }
    
     private:
      const T* pointer;
      const T* line_end;
      u32 line_jump;
      u32 stride;
    };
    
    inline ImageConstPixels(const Image<T>* image)
        : image(image) {}
    
    inline Iterator begin() const {
      return Iterator(image->data(), image->data() + image->width(),
                      image->stride() - image->width() * sizeof(T),
                      image->stride());
    }
    inline Iterator end() const {
      return Iterator(image->row(image->height()), 0, 0, 0);
    };
    
   private:
    // Not owned.
    const Image<T>* image;
  };
  
  // Helper struct for range-based for iteration over all image pixels (without
  // coordinates).
  struct ImagePixels {
    struct Iterator {
      inline Iterator(T* pointer, const T* line_end, u32 line_jump,
                      u32 stride)
          : pointer(pointer), line_end(line_end), line_jump(line_jump),
            stride(stride) {}
      
      inline Iterator& operator++() {
        ++ pointer;
        if (pointer == line_end) {
          pointer = reinterpret_cast<T*>(
              reinterpret_cast<uint8_t*>(pointer) + line_jump);
          line_end = reinterpret_cast<const T*>(
              reinterpret_cast<const uint8_t*>(line_end) + stride);
        }
        return *this;
      }
      
      inline bool operator!=(const Iterator& other) const {
        return pointer != other.pointer;
      }
      
      inline T& operator*() const {
        return *pointer;
      }
    
     private:
      T* pointer;
      const T* line_end;
      u32 line_jump;
      u32 stride;
    };
    
    inline ImagePixels(Image<T>* image)
        : image(image) {}
    
    inline Iterator begin() const {
      return Iterator(image->data(), image->data() + image->width(),
                         image->stride() - image->width() * sizeof(T),
                         image->stride());
    }
    inline Iterator end() const {
      return Iterator(image->row(image->height()), 0, 0, 0);
    };
    
   private:
    // Not owned.
    Image<T>* image;
  };
  

  // Creates an empty image. Call SetSize() later to initialize the image
  // buffer.
  Image()
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {}
  
  // Creates a deep copy of the other image.
  Image(const Image<T>& other)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSizeToMatch(other);
    SetTo(other);
  }
  
  // Creates a deep copy of the other image.
  Image& operator= (const Image<T>& other) {
    FreeData();
    size_ = ImageSize(0, 0);
    data_ = nullptr;
    stride_ = 0;
    alignment_ = 0;
    SetSizeToMatch(other);
    SetTo(other);
    return *this;
  }
  
  // Attempts to read the given image file. Use empty() to check whether image
  // loading was successful.
  Image(const string& image_file_path)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    Read(image_file_path);
  }
  
  // Creates an image with the given dimensions. Does not initialize the image
  // memory. The memory alignment is chosen automatically.
  Image(u32 width, u32 height)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(width, height);
  }
  
  // Creates an image with the given dimensions and alignment (in bytes, must be
  // a power of two and multiple of sizeof(void*), or 1 for no alignment). The
  // stride is chosen automatically to align each row to the alignment
  // specification while minimizing the amount of excess memory use.
  Image(u32 width, u32 height, usize alignment)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(width, height, alignment);
  }
  
  // Creates an image with the given dimensions, stride (in bytes), and
  // alignment (in bytes, must be a power of two and multiple of sizeof(void*),
  // or 1 for no alignment). Does not initialize the image memory.
  Image(u32 width, u32 height, u32 stride, usize alignment)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(width, height, stride, alignment);
  }
  
  // Creates an image with the given dimensions and copies existing data to the
  // image buffer. The memory alignment is chosen automatically.
  Image(u32 width, u32 height, const T* data)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(width, height);
    if (!empty()) {
      SetTo(data);
    }
  }
  
  // Creates an image with the given dimensions, stride (in bytes), and
  // alignment (in bytes, must be a power of two and multiple of sizeof(void*),
  // or 1 for no alignment). Copies existing data to the image buffer, which is
  // assumed to be densely laid out (no stride).
  Image(u32 width, u32 height, u32 stride, usize alignment,
        const T* data)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(width, height, stride, alignment);
    if (!empty()) {
      SetTo(data);
    }
  }
  
  // Creates an image with the given dimensions. Does not initialize the image
  // memory. The memory alignment is chosen automatically.
  Image(const ImageSize& size)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(size(0), size(1));
  }
  
  // Creates an image with the given dimensions and alignment (in bytes, must be
  // a power of two and multiple of sizeof(void*), or 1 for no alignment). The
  // stride is chosen automatically to align each row to the alignment
  // specification while minimizing the amount of excess memory use.
  Image(const ImageSize& size, usize alignment)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(size(0), size(1), alignment);
  }
  
  // Creates an image with the given dimensions, stride (in bytes), and
  // alignment (in bytes, must be a power of two and multiple of sizeof(void*),
  // or 1 for no alignment). Does not initialize the image memory.
  Image(const ImageSize& size, u32 stride, usize alignment)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(size(0), size(1), stride, alignment);
  }
  
  // Creates an image with the given dimensions and copies existing data to the
  // image buffer. The memory alignment is chosen automatically.
  Image(const ImageSize& size, const T* data)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(size(0), size(1));
    if (!empty()) {
      SetTo(data);
    }
  }
  
  // Creates an image with the given dimensions, stride (in bytes), and
  // alignment (in bytes, must be a power of two and multiple of sizeof(void*),
  // or 1 for no alignment). Copies existing data to the image buffer, which is
  // assumed to be densely laid out (no stride).
  Image(const ImageSize& size, u32 stride, usize alignment, const T* data)
      : size_(0, 0), data_(nullptr), stride_(0), alignment_(0) {
    SetSize(size(0), size(1), stride, alignment);
    if (!empty()) {
      SetTo(data);
    }
  }
  
  // Destructor,
  ~Image() {
    FreeData();
  }
  
  
  // Changes the image size. The memory alignment is chosen automatically.
  // Re-allocates the image buffer if the new size is different from the current
  // size. Does not preserve the image data.
  void SetSize(u32 width, u32 height) {
    if (data_ && this->width() == width && this->height() == height) {
      return;
    }
    // Since there currently are no optimized implementations making use of
    // instructions that require a specific alignment, the stride is chosen
    // to match the image width.
    SetSize(width, height, width * sizeof(T), 1);
  }
  
  // Changes the image size, stride (in bytes), and alignment (in bytes, must be
  // a power of two and multiple of sizeof(void*), or 1 for no alignment).
  // Re-allocates the image buffer if the new settings are different from the
  // current ones. Does not preserve the image data. The stride is chosen
  // automatically to align each row to the alignment specification while
  // minimizing the amount of excess memory use.
  void SetSize(u32 width, u32 height, usize alignment) {
    if (data_ && this->width() == width && this->height() == height &&
        this->alignment() == alignment) {
      return;
    }
    u32 stride = alignment * ((width * sizeof(T) - 1) / alignment + 1);
    SetSize(width, height, stride, alignment);
  }
  
  // Changes the image size, stride (in bytes), and alignment (in bytes, must be
  // a power of two and multiple of sizeof(void*), or 1 for no alignment).
  // Re-allocates the image buffer if the new settings are different from the
  // current ones. Does not preserve the image data.
  void SetSize(u32 width, u32 height, u32 stride, usize alignment) {
    if (data_ && this->width() == width && this->height() == height &&
        this->stride() == stride && this->alignment() == alignment) {
      return;
    }
    
    FreeData();
    data_ = nullptr;
    
    if (width > 0 || height > 0) {
      int return_value;
      if (alignment == 1) {
        if (stride % sizeof(T) == 0) {
          data_ = new T[height * stride / sizeof(T)];
        } else {
          data_ = reinterpret_cast<T*>(malloc(height * stride));
        }
        return_value = (data_ == nullptr) ? (-1) : 0;
      } else {
#ifdef WIN32
        data_ = reinterpret_cast<T*>(_aligned_malloc(height * stride, alignment));
      }
      if (data_ == nullptr) {
        // An error ocurred.
        if (errno == EINVAL) {
          // The alignment argument was not a power of two, or was not a multiple of
          // sizeof(void*).
          // TODO
          LOG(FATAL) << "return_value == EINVAL";
        } else if (errno == ENOMEM) {
          // There was insufficient memory to fulfill the allocation request.
          // TODO
          LOG(FATAL) << "return_value == ENOMEM";
        } else {
          // Unknown error.
          // TODO
          LOG(FATAL) << "Unknown error";
        }
      }
#else
        return_value = posix_memalign(reinterpret_cast<void**>(&data_), alignment, height * stride);
      }
      if (return_value != 0) {
        // An error ocurred.
        if (return_value == EINVAL) {
          // The alignment argument was not a power of two, or was not a multiple of
          // sizeof(void*).
          // TODO
          LOG(FATAL) << "return_value == EINVAL";
        } else if (return_value == ENOMEM) {
          // There was insufficient memory to fulfill the allocation request.
          // TODO
          LOG(FATAL) << "return_value == ENOMEM";
        } else {
          // Unknown error.
          // TODO
          LOG(FATAL) << "Unknown error";
        }
      }
#endif // !WIN32
    }
    
    size_ = ImageSize(width, height);
    stride_ = stride;
    alignment_ = alignment;
  }
  
  // Changes the image size. The memory alignment is chosen automatically.
  // Re-allocates the image buffer if the new size is different from the current
  // size. Does not preserve the image data.
  void SetSize(const ImageSize& size) {
    SetSize(size(0), size(1));
  }
  
  // Changes the image size, stride (in bytes), and alignment (in bytes, must be
  // a power of two and multiple of sizeof(void*), or 1 for no alignment).
  // Re-allocates the image buffer if the new settings are different from the
  // current ones. Does not preserve the image data. The stride is chosen
  // automatically to align each row to the alignment specification while
  // minimizing the amount of excess memory use.
  void SetSize(const ImageSize& size, usize alignment) {
    SetSize(size(0), size(1), alignment);
  }
  
  // Changes the image size, stride (in bytes), and alignment (in bytes, must be
  // a power of two and multiple of sizeof(void*), or 1 for no alignment).
  // Re-allocates the image buffer if the new settings are different from the
  // current ones. Does not preserve the image data.
  void SetSize(const ImageSize& size, u32 stride, usize alignment) {
    SetSize(size(0), size(1), stride, alignment);
  }
  
  // Changes the image size, stride, and alignment to match the settings of the
  // given image. Re-allocates the image buffer if the new settings are
  // different from the current ones.
  void SetSizeToMatch(const Image<T>& other) {
    SetSize(other.width(), other.height(), other.stride(), other.alignment());
  }
  
  
  // Sets all image pixels to the given value.
  void SetTo(const T value) {
    SetImageTo(this, value);
  }
  
  // Copies the data from the given pointer to the image. The data is assumed to
  // be densely laid out (no stride).
  void SetTo(const T* pointer) {
    // Use (hopefully fast) memcpy() implementation if possible.
    if (stride() == width() * sizeof(T)) {
      memcpy(data_, pointer, height() * stride());
    } else {
      T* dest_ptr = data_;
      const T* src_ptr = pointer;
      for (u32 y = 0; y < height(); ++ y) {
        memcpy(dest_ptr, src_ptr, width() * sizeof(T));
        src_ptr += width() * sizeof(T);
        dest_ptr = reinterpret_cast<T*>(
            reinterpret_cast<uint8_t*>(dest_ptr) + stride());
      }
    }
  }
  
  // Copies the data from the given pointer with the given stride to the image.
  void SetTo(const T* pointer, u32 stride) {
    // Use (hopefully fast) memcpy() implementation if possible.
    if (this->stride() == stride) {
      memcpy(data_, pointer, height() * stride);
    } else {
      T* dest_ptr = data_;
      const T* src_ptr = pointer;
      for (u32 y = 0; y < height(); ++ y) {
        memcpy(dest_ptr, src_ptr, width() * sizeof(T));
        src_ptr = reinterpret_cast<const T*>(
            reinterpret_cast<const uint8_t*>(src_ptr) + stride);
        dest_ptr = reinterpret_cast<T*>(
            reinterpret_cast<uint8_t*>(dest_ptr) + this->stride());
      }
    }
  }
  
  // Sets the image content to the content of another image. The images must
  // have the same size.
  void SetTo(const Image<T>& other) {
    // Use (hopefully fast) memcpy() implementation if possible.
    if (stride() == other.stride()) {
      memcpy(static_cast<void*>(data_), other.data_, height() * stride());
    } else {
      T* dest_ptr = data_;
      const T* src_ptr = other.data_;
      for (u32 y = 0; y < height(); ++ y) {
        memcpy(static_cast<void*>(dest_ptr), src_ptr, width() * sizeof(T));
        src_ptr = reinterpret_cast<const T*>(
            reinterpret_cast<const uint8_t*>(src_ptr) + other.stride());
        dest_ptr = reinterpret_cast<T*>(
            reinterpret_cast<uint8_t*>(dest_ptr) + stride());
      }
    }
  }
  
  // Sets a rectangular region of the image to the content of another image.
  void SetRectTo(Vector2i dest_offset, ImageSize size, Vector2i src_offset,
                 const Image<T>& src) {
    usize row_size = size(0) * sizeof(T);
    for (u32 y = 0; y < size(1); ++ y) {
      T* dest_ptr = row(dest_offset(1) + y) + dest_offset(0);
      const T* src_ptr = src.row(src_offset(1) + y) + src_offset(0);
      memcpy(dest_ptr, src_ptr, row_size);
    }
  }
  
  // Draws a line from pixel (x0, y0) to pixel (x1, y1) with the given color.
  // Uses the Bresenham algorithm. The implementation was taken from:
  // https://gist.github.com/bert/1085538#file-plot_line-c
  void DrawLine(int x0, int y0, int x1, int y1, T color) {
    int dx =  abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1; 
    int err = dx + dy, e2; /* error value e_xy */
    
    while (true) {
      (*this)(x0, y0) = color;
      if (x0 == x1 && y0 == y1) {
        break;
      }
      
      e2 = 2 * err;
      
      if (e2 >= dy) { err += dy; x0 += sx; } /* e_xy+e_x > 0 */
      if (e2 <= dx) { err += dx; y0 += sy; } /* e_xy+e_y < 0 */
    }
  }
  
  // Returns true if the images have the same content. Note that the alignment
  // and stride are allowed to differ.
  bool operator==(const Image<T>& other) const {
    if (width() != other.width() || height() != other.height()) {
      return false;
    }
    for (u32 y = 0; y < height(); ++ y) {
      const T* this_ptr = row(y);
      const T* other_ptr = other.row(y);
      const T* end = this_ptr + width();
      while (this_ptr < end) {
        if (*this_ptr != *other_ptr) {
          return false;
        }
        ++ this_ptr;
        ++ other_ptr;
      }
    }
    return true;
  }
  
  template<typename TargetT>
  void ConvertToGrayscale(Image<TargetT>* target) const;
  
  // Returns true if the given pixel coordinate is within the image,
  // using the "pixel center" origin convention (i.e., the coordinate
  // (0, 0) refers to the center of the top left pixel). Coordinates
  // which are exactly on the right or bottom borders are treated as
  // outside the image.
  template <typename Derived>
  bool ContainsPixelCenterConv(const MatrixBase<Derived>& position) const {
    return position.x() >= 0 &&
           position.y() >= 0 &&
           position.x() < size_.coeff(0) - 1 &&
           position.y() < size_.coeff(1) - 1;
  }
  
  // Returns the bilinearly interpolated image value at the given position. The
  // coordinate (0, 0) refers to the center of the top left pixel. The largest
  // permitted position is the largest position smaller than (width - 1,
  // height - 1) in both axes. Out-of-bounds accesses are not allowed.
  // The validity of a coordinate can be checked with ContainsPixelCenterConv().
  template<typename InterpolatedT = typename float_type_helper<T>::Type, typename Derived>
  inline InterpolatedT InterpolateBilinear(const MatrixBase<Derived>& position) const {
    return InterpolateImageBilinear<InterpolatedT>(this, position);
  }
  
  // Returns the derivatives of InterpolateBilinear() at the given position with
  // respect to the x and y component of the lookup position.
  // TODO: As with the other bilinear interpolation functions, make this work for scalar and vector types
  template<typename InterpolatedT = float, typename Derived>
  inline Matrix<InterpolatedT, 2, 1> InterpolateBilinearJacobian(
      const MatrixBase<Derived>& position) const {
    int ix = static_cast<int>(position.coeff(0));
    int iy = static_cast<int>(position.coeff(1));
    
    float fx = position.coeff(0) - ix;
    float fy = position.coeff(1) - iy;
    float fx_inv = 1.f - fx;
    float fy_inv = 1.f - fy;
    
    const T* ptr = reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(data_) + iy * stride()) + ix;
    const T* ptr2 = reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(ptr) + stride());
    const T top_left = *ptr;
    const T top_right = *(ptr + 1);
    const T bottom_left = *ptr2;
    const T bottom_right = *(ptr2 + 1);
    
    Matrix<InterpolatedT, 2, 1> jacobian;
    jacobian.coeffRef(0) = fy * (bottom_right - bottom_left) +
                           fy_inv * (top_right - top_left);
    jacobian.coeffRef(1) = fx * (bottom_right - top_right) +
                           fx_inv * (bottom_left - top_left);
    return jacobian;
  }
  
  // Fast combined variant of InterpolateBilinear() and
  // InterpolateBilinearJacobian().
  template<typename InterpolatedT, typename JacobianScalarT, typename Derived>
  inline void InterpolateBilinearWithJacobian(
      const MatrixBase<Derived>& position,
      InterpolatedT* value,
      Matrix<JacobianScalarT, channel_count_helper<T>::channel_count(), 2>* jacobian) const {
    InterpolateImageBilinearWithJacobian(this, position, value, jacobian);
  }
  
  
  // TODO: Document. Warn about invalid accesses: valid parameter range for one axis is [1, image_length - 1[ !
  // TODO: Can this be merged with a scalar version?
  template<typename InterpolatedT = float, typename Derived>
  inline Matrix<InterpolatedT, channel_count_helper<T>::channel_count(), 1> InterpolateBicubicVector(const MatrixBase<Derived>& position) const {
    typedef Matrix<InterpolatedT, channel_count_helper<T>::channel_count(), 1> ResultT;
    
    int ix = std::floor(position.coeff(0));
    int iy = std::floor(position.coeff(1));
    
    double fx = position.coeff(0) - ix;
    double fy = position.coeff(1) - iy;
    
    const T* row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(data_) + (iy - 1) * stride()) + (ix - 1);
    ResultT row0_value = CubicHermiteSplineVector<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row_ptr) + stride());
    ResultT row1_value = CubicHermiteSplineVector<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row_ptr) + stride());
    ResultT row2_value = CubicHermiteSplineVector<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row_ptr) + stride());
    ResultT row3_value = CubicHermiteSplineVector<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    return CubicHermiteSplineVector<ResultT, InterpolatedT>(row0_value, row1_value, row2_value, row3_value, fy);
  }
  
  template<typename InterpolatedT = float, typename Derived>
  inline InterpolatedT InterpolateBicubic(const MatrixBase<Derived>& position) const {
    typedef InterpolatedT ResultT;
    
    int ix = std::floor(position.coeff(0));
    int iy = std::floor(position.coeff(1));
    
    double fx = position.coeff(0) - ix;
    double fy = position.coeff(1) - iy;
    
    const T* row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(data_) + (iy - 1) * stride()) + (ix - 1);
    ResultT row0_value = CubicHermiteSpline<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row_ptr) + stride());
    ResultT row1_value = CubicHermiteSpline<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row_ptr) + stride());
    ResultT row2_value = CubicHermiteSpline<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    row_ptr = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row_ptr) + stride());
    ResultT row3_value = CubicHermiteSpline<ResultT, InterpolatedT>(*row_ptr, *(row_ptr + 1), *(row_ptr + 2), *(row_ptr + 3), fx);
    
    return CubicHermiteSpline<ResultT, InterpolatedT>(row0_value, row1_value, row2_value, row3_value, fy);
  }
  
  
  // Downscales the image to half of its size and writes the result to output.
  // The image width and height must be divisible by two. Each output pixel is
  // assigned the average of its corresponding four input pixels.
  void DownscaleToHalfSize(Image<T>* output) const {
    CHECK_EQ(width() % 2, 0);
    CHECK_EQ(height() % 2, 0);
    
    output->SetSize(width() / 2, height() / 2);
    for (u32 y = 0; y < output->height(); ++ y) {
      T* write_ptr = output->row(y);
      T* write_ptr_end = write_ptr + output->width();
      const T* upper_read_ptr = row(2 * y + 0);
      const T* lower_read_ptr = row(2 * y + 1);
      while (write_ptr != write_ptr_end) {
        // TODO: This formulation was required to hande Vec3u8 (but does not round the results correctly).
        //       Consider a template specialization for these, and for correct rounding in general.
        *write_ptr = upper_read_ptr[0] / 4 + upper_read_ptr[1] / 4 + lower_read_ptr[0] / 4 + lower_read_ptr[1] / 4;
        ++ write_ptr;
        upper_read_ptr += 2;
        lower_read_ptr += 2;
      }
    }
  }
  
  // Downscales the image to the given size. Each result pixel's intensity is
  // computed as the median of its corresponding pixels in the original image.
  // Requires the image to have a scalar type.
  // TODO: Slow implementation. Average is not needed if the number of original
  //       pixels per output pixel is uneven.
  void DownscaleUsingMedian(int output_width, int output_height, Image<T>* output) const {
    output->SetSize(output_width, output_height);
    std::vector<T> values;
    for (u32 y = 0; y < output->height(); ++ y) {
      T* write_ptr = output->row(y);
      T* write_ptr_end = write_ptr + output->width();
      u32 x = 0;
      while (write_ptr != write_ptr_end) {
        u32 start_x = (width() * x) / output->width();
        u32 end_x = (width() * (x + 1)) / output->width();
        u32 start_y = (height() * y) / output->height();
        u32 end_y = (height() * (y + 1)) / output->height();
        
        values.clear();
        float value_sum = 0;
        for (u32 original_y = start_y; original_y < end_y; ++ original_y) {
          for (u32 original_x = start_x; original_x < end_x; ++ original_x) {
            values.push_back(operator()(original_x, original_y));
            value_sum += values.back();
          }
        }
        int value_count = ((end_x - start_x) * (end_y - start_y));
        float average = value_sum / value_count;
        
        // TODO: Only the median is required, not sorting the whole vector.
        std::sort(values.begin(), values.end());
        if (value_count % 2 == 1) {
          *write_ptr = values[values.size() / 2];
        } else {
          const T& low_value = values[values.size() / 2 - 1];
          const T& high_value = values[values.size() / 2];
          if (fabs(average - low_value) < fabs(average - high_value)) {
            *write_ptr = low_value;
          } else {
            *write_ptr = high_value;
          }
        }
        ++ write_ptr;
        ++ x;
      }
    }
  }
  
  // Downscales the image to the given size. Each result pixel's intensity is
  // computed as the median of its corresponding pixels in the original image.
  // Requires the image to have a scalar type.
  // TODO: Slow implementation. Average is not needed if the number of original
  //       pixels per output pixel is uneven.
  void DownscaleUsingMedianWhileExcluding(const T value_to_ignore, int output_width, int output_height, Image<T>* output) const {
    output->SetSize(output_width, output_height);
    std::vector<T> values;
    for (u32 y = 0; y < output->height(); ++ y) {
      T* write_ptr = output->row(y);
      T* write_ptr_end = write_ptr + output->width();
      u32 x = 0;
      while (write_ptr != write_ptr_end) {
        u32 start_x = (width() * x) / output->width();
        u32 end_x = (width() * (x + 1)) / output->width();
        u32 start_y = (height() * y) / output->height();
        u32 end_y = (height() * (y + 1)) / output->height();
        
        values.clear();
        float value_sum = 0;
        u32 value_count = 0;
        for (u32 original_y = start_y; original_y < end_y; ++ original_y) {
          for (u32 original_x = start_x; original_x < end_x; ++ original_x) {
            T value = operator()(original_x, original_y);
            if (value != value_to_ignore) {
              values.push_back(value);
              value_sum += values.back();
              ++ value_count;
            }
          }
        }
        
        if (value_count == 0) {
          *write_ptr = value_to_ignore;
        } else {
          float average = value_sum / value_count;
          
          // TODO: Only the median is required, not sorting the whole vector.
          std::sort(values.begin(), values.end());
          if (value_count % 2 == 1) {
            *write_ptr = values[values.size() / 2];
          } else {
            const T& low_value = values[values.size() / 2 - 1];
            const T& high_value = values[values.size() / 2];
            if (fabs(average - low_value) < fabs(average - high_value)) {
              *write_ptr = low_value;
            } else {
              *write_ptr = high_value;
            }
          }
        }
        ++ write_ptr;
        ++ x;
      }
    }
  }
  
  
  // Calculates and returns the average image value.
  template <typename ResultT>
  ResultT CalcAverage() const {
    T sum = static_cast<T>(0);
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        sum += *ptr;
        ++ ptr;
      }
    }
    
    return sum / static_cast<ResultT>(width() * height());
  }
  
  // Calculates and returns the average image value, excluding a value.
  template <typename ResultT>
  ResultT CalcAverageWhileExcluding(const T value_to_ignore) const {
    T sum = static_cast<T>(0);
    u32 count = 0;
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        if (*ptr != value_to_ignore) {
          sum += *ptr;
          ++ count;
        }
        ++ ptr;
      }
    }
    
    return sum / static_cast<ResultT>(count);
  }
  
  // Calculates and returns the median image value. If the number of
  // pixels is even, the larger value of the middle two is returned.
  T CalcMedian() const {
    // NOTE: Slow implementation.
    std::vector<T> values;
    values.reserve(width() * height());
    for (T value : pixels()) {
      values.push_back(value);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  }
  
  // Calculates and returns the median image value, excluding a value. If
  // the number of non-excluded pixels is even, the larger value of the middle
  // two is returned. If the image does not contain any other values than
  // value_to_ignore, the return value is value_to_ignore.
  T CalcMedianWhileExcluding(const T value_to_ignore) const {
    // NOTE: Slow implementation.
    std::vector<T> values;
    values.reserve(width() * height());
    for (T value : pixels()) {
      if (value != value_to_ignore) {
        values.push_back(value);
      }
    }
    if (values.size() == 0) {
      return value_to_ignore;
    } else {
      std::sort(values.begin(), values.end());
      return values[values.size() / 2];
    }
  }
  
  // Calculates and returns the minimum image value.
  T CalcMin() const {
    T result = std::numeric_limits<T>::max();
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        if (*ptr < result) {
          result = *ptr;
        }
        ++ ptr;
      }
    }
    
    return result;
  }
  
  // Calculates and returns the minimum image value, excluding a value. If no
  // pixel with an included value exists, returns std::numeric_limits<T>::max().
  T CalcMinWhileExcluding(const T value_to_ignore) const {
    T result = std::numeric_limits<T>::max();
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        if (*ptr < result && *ptr != value_to_ignore) {
          result = *ptr;
        }
        ++ ptr;
      }
    }
    
    return result;
  }
  
  // Calculates and returns the maximum image value.
  T CalcMax() const {
    T result = -1 * std::numeric_limits<T>::max();
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        if (*ptr > result) {
          result = *ptr;
        }
        ++ ptr;
      }
    }
    
    return result;
  }
  
  // Calculates and returns the maximum image value, excluding a value. If no
  // pixel with an included value exists, returns
  // -1 * std::numeric_limits<T>::max().
  T CalcMaxWhileExcluding(const T value_to_ignore) const {
    T result = -1 * std::numeric_limits<T>::max();
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        if (*ptr > result && *ptr != value_to_ignore) {
          result = *ptr;
        }
        ++ ptr;
      }
    }
    
    return result;
  }
  
  // Counts the number of pixels in the image which are not equal to
  // value_to_ignore. Can for example be used to count the number of valid
  // pixels in a depth image by setting value_to_ignore to the invalid depth
  // value.
  usize CountNotEqualTo(const T value_to_ignore) const {
    usize result = 0;
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* ptr = row(y);
      const T* end = ptr + width();
      while (ptr < end) {
        if (*ptr != value_to_ignore) {
          ++ result;
        }
        ++ ptr;
      }
    }
    
    return result;
  }
  
  
  // Replaces all pixels with values larger than max_value with
  // replacement_value.
  void MaxCutoff(const T max_value, const T replacement_value, Image<T>* result) const {
    result->SetSizeToMatch(*this);
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* read_ptr = row(y);
      const T* read_end = read_ptr + width();
      T* write_ptr = result->row(y);
      while (read_ptr < read_end) {
        *write_ptr = (*read_ptr <= max_value) ? *read_ptr : replacement_value;
        ++ read_ptr;
        ++ write_ptr;
      }
    }
  }
  
  
  // Flips the image left / right.
  void FlipX() {
    for (u32 y = 0; y < height(); ++ y) {
      T* left_ptr = row(y);
      T* right_ptr = left_ptr + width() - 1;
      while (left_ptr < right_ptr) {
        std::swap(*left_ptr, *right_ptr);
        ++ left_ptr;
        -- right_ptr;
      }
    }
  }
  
  // Flips the image up / down.
  void FlipY() {
    for (u32 y = 0; y < height() / 2; ++ y) {
      T* row0_ptr = row(y);
      T* row0_end = row0_ptr + width();
      T* row1_ptr = row(height() - 1 - y);
      while (row0_ptr < row0_end) {
        std::swap(*row0_ptr, *row1_ptr);
        ++ row0_ptr;
        ++ row1_ptr;
      }
    }
  }
  
  
  // Applies an affine function to the pixel values. Does not perform any
  // clamping on the result.
  void ApplyAffineFunction(const T scale, const T bias) {
    for (u32 y = 0; y < height(); ++ y) {
      T* write_ptr = row(y);
      T* end = write_ptr + width();
      while (write_ptr < end) {
        *write_ptr = scale * (*write_ptr) + bias;
        ++ write_ptr;
      }
    }
  }
  
  // Pixels with value_to_ignore will keep this value, and in addition they are
  // not considered when computing the filtered value for other pixels. This is
  // intended to be used for invalid depths in depth maps.
  // 3 is a safe value for radius_factor, smaller values can be used to improve
  // performance while neglecting far-away pixels for filtering.
  void BilateralFilter(float sigma_xy, const T sigma_value,
                       const T value_to_ignore, float radius_factor,
                       Image<T>* result) const {
    result->SetSizeToMatch(*this);
    
    int radius = radius_factor * sigma_xy + 0.5f;
    int radius_squared = radius * radius;
    
    float denom_xy = 2.0f * sigma_xy * sigma_xy;
    T denom_value = 2.0f * sigma_value * sigma_value;
    
    for (u32 y = 0; y < height(); ++ y) {
      const T* read_ptr = row(y);
      const T* read_end = read_ptr + width();
      T* write_ptr = result->row(y);
      int x = 0;
      while (read_ptr < read_end) {
        T center_value = *read_ptr;
        if (center_value == value_to_ignore) {
          *write_ptr = value_to_ignore;
          ++ read_ptr;
          ++ write_ptr;
          ++ x;
          continue;
        }
        
        float sum = 0;
        float weight = 0;
        
        u32 min_y = max<int>(0, y - radius);
        u32 max_y = min<int>(height() - 1, y + radius);
        for (u32 sample_y = min_y; sample_y <= max_y; ++ sample_y) {
          int dy = sample_y - y;
          
          u32 min_x = max<int>(0, x - radius);
          u32 max_x = min<int>(width() - 1, x + radius);
          for (u32 sample_x = min_x; sample_x <= max_x; ++ sample_x) {
            int dx = sample_x - x;
            
            int grid_distance_squared = dx * dx + dy * dy;
            if (grid_distance_squared > radius_squared) {
              continue;
            }
            
            T sample = operator()(sample_x, sample_y);
            if (sample == value_to_ignore) {
              continue;
            }
            
            float value_distance_squared = center_value - sample;  // NOTE: Cannot use T here as it might be an unsigned type.
            value_distance_squared *= value_distance_squared;
            float w = exp(-grid_distance_squared / denom_xy) *
                      exp(-value_distance_squared / denom_value);
            sum += w * sample;
            weight += w;
          }
        }
        
        if (weight == 0) {
          *write_ptr = value_to_ignore;
        } else {
          *write_ptr = sum / weight;
        }
        
        ++ read_ptr;
        ++ write_ptr;
        ++ x;
      }
    }
  }
  
  // TODO: Maybe separate this out into its own header connected_components.h?
  struct ConnectedComponent {
    ConnectedComponent(int _parent, int _pixel_count)
        : parent(_parent), pixel_count(_pixel_count) {}

    int parent;
    int pixel_count;
    bool should_be_removed;
  };
  
  void RemoveSmallConnectedComponents(
      T separator_value,
      int min_component_size,
      int min_x,
      int min_y,
      int max_x,
      int max_y) {
    // Find connected components and calculate number of certain pixels.
    vector<ConnectedComponent> components;
    constexpr size_t kPreallocationSize = 4096u;
    components.reserve(kPreallocationSize);
    Image<int> component_image(width(), height());
    
    for (int y = min_y; y <= max_y; ++ y) {
      for (int x = min_x; x <= max_x; ++ x) {
        // Mark pixel as invalid if it has the separator value.
        if (operator()(x, y) == separator_value) {
          component_image(x, y) = -1;
          continue;
        }
        
        if (x > min_x && component_image(x - 1, y) != -1) {
          // Merge into left component.
          component_image(x, y) = component_image(x - 1, y);
          ConnectedComponent* const component = &components[component_image(x - 1, y)];
          ConnectedComponent* parent = component;
          while (&components[parent->parent] != parent) {
            ConnectedComponent* const higher_parent = &components[parent->parent];
            parent->parent = higher_parent->parent;
            parent = higher_parent;
          }
          parent->pixel_count += 1;
          
          if (y > min_y && component_image(x, y - 1) != -1) {
            // Merge left into top component.
            // Notice: leaf, parent and target components may be the same.
            ConnectedComponent* const left_component = &components[component_image(x - 1, y)];
            ConnectedComponent* parent = left_component;
            while (&components[parent->parent] != parent) {
              ConnectedComponent* const higher_parent = &components[parent->parent];
              parent->parent = higher_parent->parent;
              parent = higher_parent;
            }
            int certain_pixels = left_component->pixel_count;
            left_component->pixel_count = 0;
            certain_pixels += parent->pixel_count;
            parent->pixel_count = 0;
            
            ConnectedComponent* const top_component = &components[component_image(x, y - 1)];
            parent->parent = top_component->parent;
            while (&components[parent->parent] != parent) {
              ConnectedComponent* const higher_parent = &components[parent->parent];
              parent->parent = higher_parent->parent;
              parent = higher_parent;
            }
            components[parent->parent].pixel_count += certain_pixels;
          }
        } else if (y > min_y && component_image(x, y - 1) != -1) {
          // Merge into top component.
          component_image(x, y) = component_image(x, y - 1);
          ConnectedComponent* const component = &components[component_image(x, y - 1)];
          ConnectedComponent* parent = component;
          while (&components[parent->parent] != parent) {
            ConnectedComponent* const higher_parent = &components[parent->parent];
            parent->parent = higher_parent->parent;
            parent = higher_parent;
          }
          parent->pixel_count += 1;
        } else {
          // Create a new component.
          components.emplace_back(components.size(), 1);
          component_image(x, y) = components.size() - 1;
        }
      }
    }
    
    // Resolve parents until the root and decide on which components to remove.
    for (size_t i = 0u, end = components.size(); i < end; ++i) {
      ConnectedComponent* const component = &components[i];
      ConnectedComponent* parent = &components[component->parent];
      while (&components[parent->parent] != parent) {
        ConnectedComponent* const higher_parent = &components[parent->parent];
        parent->parent = higher_parent->parent;
        parent = higher_parent;
      }
      component->should_be_removed = parent->pixel_count < min_component_size;
    }
    
    // Remove bad connected components from image.
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        if (component_image(x, y) >= 0 &&
            components[component_image(x, y)].should_be_removed) {
          operator()(x, y) = 0.f;
        }
      }
    }
  }
  
  template <typename OutputT>
  void ComputeIntegralImage(Image<OutputT>* output) const {
    output->SetSize(width(), height());
    
    OutputT row_sum = 0;
    
    // First row
    const T* read_ptr = row(0);
    const T* read_end = read_ptr + width();
    OutputT* write_ptr = output->row(0);
    while (read_ptr != read_end) {
      row_sum += *read_ptr;
      
      *write_ptr = row_sum;
      
      ++ read_ptr;
      ++ write_ptr;
    }
    
    // Remaining rows
    for (u32 y = 1; y < height(); ++ y) {
      row_sum = 0;
      
      read_ptr = row(y);
      read_end = read_ptr + width();
      const OutputT* prev_write_ptr = output->row(y - 1);
      write_ptr = output->row(y);
      
      while (read_ptr != read_end) {
        row_sum += *read_ptr;
        
        *write_ptr = row_sum + *prev_write_ptr;
        
        ++ read_ptr;
        ++ write_ptr;
        ++ prev_write_ptr;
      }
    }
  }
  
  // Returns the sum of the values in the rectangle which spans the given
  // values, including the boundaries (i.e., with left == 2 and right == 3,
  // the two columns with x == 2 and x == 3 would be included).
  // The boundary values may be outside of the image. In this case, the outside
  // parts are simply ignored (i.e., they contribute zero to the sum).
  T AccessIntegralImage(int left, int top, int right, int bottom) const {
    int left_m1 = left - 1;
    int top_m1 = top - 1;
    int clamped_right = std::min<int>(right, width() - 1);
    int clamped_bottom = std::min<int>(bottom, height() - 1);
    
    u64 top_left_value = (left_m1 < 0 || top_m1 < 0) ? 0 : at(left_m1, top_m1);
    u64 top_right_value = (top_m1 < 0) ? 0 : at(clamped_right, top_m1);
    u64 bottom_left_value = (left_m1 < 0) ? 0 : at(left_m1, clamped_bottom);
    u64 bottom_right_value = at(clamped_right, clamped_bottom);
    
    return bottom_right_value - top_right_value - bottom_left_value + top_left_value;
  }
  
  
  // Writes the image to a file. Returns true if successful.
  bool Write(const string& /*image_file_name*/) const {
    static_assert(always_false<T>::value, "Write() is not supported for this image type. u8 and Vec3u8 and Vec4u8 are supported.");
    return false;
  }
  
  // Loads the image from a file. Returns true if successful. This resizes the
  // image to the size of the image file. If a specific alignment or stride is
  // desired, the image currently must be set to the correct size beforehand,
  // otherwise these settings will be overwritten.
  // TODO: It would probably be better if the Image simply kept its current
  // stride and alignment settings on SetSize(width, height)? But this would
  // still require the user to allocate some arbitrary-sized buffer before
  // calling Read(). Maybe offer a Read() variant with stride and alignment
  // settings?
  bool Read(const string& /*image_file_name*/) {
    static_assert(always_false<T>::value, "Read() is not supported for this image type. u8 and Vec3u8 and Vec4u8 are supported.");
    return false;
  }
  
  
#ifdef LIBVIS_HAVE_QT
  // Creates a QImage around the image data. It uses the existing memory, which
  // is only valid as long as the Image is alive. This variant uses a default
  // QImage format, the mapping from Image type to the format is as follows:
  //   
  //   u8     -> QImage::Format_Grayscale8
  //   Vec3u8 -> QImage::Format_RGB888
  //   Vec4u8 -> QImage::Format_RGBA8888 (not premultiplied)
  QImage WrapInQImage() const {
    static_assert(always_false<T>::value, "This WrapInQImage() overload is not supported for this image type. u8 and Vec3u8 and Vec4u8 are supported.");
    return QImage();
  }
  
  // Overload of WrapInQImage() which allows to specify the QImage format to
  // be used.
  QImage WrapInQImage(QImage::Format format) const {
    return QImage(reinterpret_cast<const u8*>(data()), width(), height(),
                  stride(), format);
  }
#endif


#ifdef LIBVIS_HAVE_OPENCV
  // TODO: Version of WrapInCVMat() with automatic choice of type
  
  // Overload of WrapInCVMat() which allows to specify the OpenCV type to
  // be used.
  cv::Mat WrapInCVMat(int type) {
    return cv::Mat(height(), width(), type, reinterpret_cast<void*>(data()), stride());
  }
#endif
  
  
//   // Displays the image in a debug window.
//   shared_ptr<ImageDisplay> DebugDisplay(const string& title) const {
//     (void) title;
//     static_assert(always_false<T>::value, "DebugDisplay() is not supported for this image type. u8 and Vec3u8 are supported.");
//     return shared_ptr<ImageDisplay>();
//   }
//   
//   // Displays the image in a debug window. Intended for images with scalar
//   // values. Scales the content with an affine brightness transformation such
//   // that white_value is displayed as white and black_value is displayed as
//   // black.
//   shared_ptr<ImageDisplay> DebugDisplay(const string& title, const T& black_value, const T& white_value) const;
  
  
  // Returns true if the image buffer is invalid.
  inline bool empty() const { return data_ == nullptr; }
  
  // Returns the image size.
  inline const ImageSize& size() const { return size_; }
  
  // Returns the image width.
  inline u32 width() const { return size_(0); }
  
  // Returns the image height.
  inline u32 height() const { return size_(1); }
  
  // Returns the number of pixels in the image.
  inline u32 pixel_count() const { return width() * height(); }
  
  // Returns the bytes per pixel.
  inline constexpr u32 bytes_per_pixel() const { return sizeof(T); }
  
  // Returns the number of channels. If the pixel type is an Eigen::Matrix, this
  // is equal to the number of elements in the matrix. Otherwise, it is one.
  // Does not work for matrices of dynamic size. Gives wrong results for other
  // matrix or vector types than Eigen's ones (for which a different channel
  // count from one would be expected) because it is only specialized for these
  // ones.
  // NOTE: C++ does not allow partial template specialization (which could be
  // used to implement this in a cleaner way), so we work around this by using
  // the partial class specialization of channel_count_helper instead.
  inline constexpr u32 channel_count() const {
    return channel_count_helper<T>::channel_count();
  }
  
  // Returns the stride in bytes.
  inline u32 stride() const { return stride_; }
  
  // Returns the alignment in bytes.
  inline usize alignment() const { return alignment_; }
  
  // Returns whether the data is laid out continuously in memory.
  inline bool is_data_continuous() const { return stride() == width() * bytes_per_pixel(); }
  
  // Returns the image buffer (const).
  inline const T* data() const { return data_; }
  
  // Returns the image buffer (const).
  inline T* data() { return data_; }
  
  // Returns a pointer to the start of an image row (const).
  inline const T* row(u32 y) const {
    return reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(data_) + y * stride_);
  }
  
  // Returns a pointer to the start of an image row.
  inline T* row(u32 y) {
    return reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(data_) + y * stride_);
  }
  
  // Access to a given pixel (slow!).
  inline const T& at(u32 x, u32 y) const {
    return *(reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(data_) + y * stride_) + x);
  }
  inline const T& operator()(u32 x, u32 y) const {
    return at(x, y);
  }
  
  // Access to a given pixel (slow!).
  inline T& at(u32 x, u32 y) {
    return *(reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(data_) + y * stride_) + x);
  }
  inline T& operator()(u32 x, u32 y) {
    return at(x, y);
  }
  
  // Access to a given pixel using a vector coordinate (slow!).
  // The coordinate should be integer-valued. Floating-point values are converted
  // to integers. This means that e.g. (-0.5, 0.9) would access pixel (0, 0).
  template <typename Derived>
  inline const T& at(const Eigen::MatrixBase<Derived>& coordinate) const {
    return *(reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(data_) + coordinate.coeff(1) * stride_) + coordinate.coeff(0));
  }
  template <typename Derived>
  inline const T& operator()(const Eigen::MatrixBase<Derived>& coordinate) const {
    return at(coordinate);
  }
  
  // Access to a given pixel using a vector coordinate (slow!).
  // The coordinate should be integer-valued. Floating-point values are converted
  // to integers. This means that e.g. (-0.5, 0.9) would access pixel (0, 0).
  template <typename Derived>
  inline T& at(const Eigen::MatrixBase<Derived>& coordinate) {
    return *(reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(data_) + coordinate.coeff(1) * stride_) + coordinate.coeff(0));
  }
  template <typename Derived>
  inline T& operator()(const Eigen::MatrixBase<Derived>& coordinate) {
    return at(coordinate);
  }
  
  // Access to a given pixel by sequential index (NOTE: pay attention to the
  // stride when using this function! It is not applicable if the stride is not
  // a multiple of the element size).
  inline const T& operator[](u32 index) const {
    return *(data_ + index);
  }
  
  // Access to a given pixel by sequential index (NOTE: pay attention to the
  // stride when using this function! It is not applicable if the stride is not
  // a multiple of the element size).
  inline T& operator[](u32 index) {
    return *(data_ + index);
  }
  
  // Returns an object which can be used for iteration over pixels in the
  // following way:
  // for (const T& value : image.pixels()) {
  //   ...
  // }
  inline ImageConstPixels pixels() const { return ImageConstPixels(this); }
  
  // Returns an object which can be used for iteration over pixels in the
  // following way:
  // for (T& value : image.pixels()) {
  //   ...
  // }
  inline ImagePixels pixels() { return ImagePixels(this); }
  
 private:
  void FreeData() {
    // Does nothing if data_ is nullptr.
    if (stride_ % sizeof(T) == 0) {
      delete[] data_;
    } else {
      free(data_);
    }
  }
  
  // Helper function for bicubic interpolation.
  // Computes a Catmull–Rom spline (TODO: change the name to reflect this?)
  // TODO: Can factor out the 0.5f's
  template <typename ResultT, typename InterpolatedT>
  inline ResultT CubicHermiteSplineVector(const T& v0, const T& v1, const T& v2, const T& v3, double frac) const {
    const ResultT a = 0.5f * (-v0.template cast<InterpolatedT>() + 3.0f * v1.template cast<InterpolatedT>() - 3.0f * v2.template cast<InterpolatedT>() + v3.template cast<InterpolatedT>());
    const ResultT b = 0.5f * (2.0f * v0.template cast<InterpolatedT>() - 5.0f * v1.template cast<InterpolatedT>() + 4.0f * v2.template cast<InterpolatedT>() - v3.template cast<InterpolatedT>());
    const ResultT c = 0.5f * (-v0.template cast<InterpolatedT>() + v2.template cast<InterpolatedT>());
    return v1 + frac * (c + frac * (b + frac * a));
  }
  
  template <typename ResultT, typename InterpolatedT>
  inline ResultT CubicHermiteSpline(const T& v0, const T& v1, const T& v2, const T& v3, double frac) const {
    const ResultT a = 0.5f * (-static_cast<InterpolatedT>(v0) + 3.0f * static_cast<InterpolatedT>(v1) - 3.0f * static_cast<InterpolatedT>(v2) + static_cast<InterpolatedT>(v3));
    const ResultT b = 0.5f * (2.0f * static_cast<InterpolatedT>(v0) - 5.0f * static_cast<InterpolatedT>(v1) + 4.0f * static_cast<InterpolatedT>(v2) - static_cast<InterpolatedT>(v3));
    const ResultT c = 0.5f * (-static_cast<InterpolatedT>(v0) + static_cast<InterpolatedT>(v2));
    return v1 + frac * (c + frac * (b + frac * a));
  }
  
  ImageSize size_;
  T* data_;
  u32 stride_;
  usize alignment_;
};

// Writing / reading template specializations for the supported types.
template<>
bool Image<u8>::Write(const string& image_file_name) const;
template<>
bool Image<u16>::Write(const string& image_file_name) const;
template<>
bool Image<Vec3u8>::Write(const string& image_file_name) const;
template<>
bool Image<Vec4u8>::Write(const string& image_file_name) const;
template<>
bool Image<u8>::Read(const string& image_file_name);
template<>
bool Image<u16>::Read(const string& image_file_name);
template<>
bool Image<Vec3u8>::Read(const string& image_file_name);
template<>
bool Image<Vec4u8>::Read(const string& image_file_name);

// WrapInQImage() template specializations for the supported types.
#ifdef LIBVIS_HAVE_QT
template<>
QImage Image<u8>::WrapInQImage() const;
template<>
QImage Image<Vec3u8>::WrapInQImage() const;
template<>
QImage Image<Vec4u8>::WrapInQImage() const;
#endif

// // DebugDisplay() template specializations for the supported types.
// template<>
// shared_ptr<ImageDisplay> Image<u8>::DebugDisplay(const string& title) const;
// template<>
// shared_ptr<ImageDisplay> Image<Vec3u8>::DebugDisplay(const string& title) const;
// 
// // Defined outside of the Image class since NVCC complained about DebugDisplay()
// // being used before it is specialized.
// template <typename T>
// shared_ptr<ImageDisplay> Image<T>::DebugDisplay(const string& title, const T& black_value, const T& white_value) const {
//   // Convert the image.
//   Image<u8> display_image(width(), height());
//   double scale = 255.999 / (white_value - black_value);
//   double bias = (-255.999 * black_value) / (white_value - black_value);
//   for (u32 y = 0; y < height(); ++ y) {
//     const T* read_ptr = row(y);
//     u8* write_ptr = display_image.row(y);
//     u8* write_end = write_ptr + width();
//     while (write_ptr < write_end) {
//       *write_ptr = std::max<double>(0, std::min<double>(255, scale * (*read_ptr) + bias));
//       ++ write_ptr;
//       ++ read_ptr;
//     }
//   }
//   
//   // Display the converted image.
//   // TODO: Make the "value-under-cursor" display show the original image's values.
//   return display_image.DebugDisplay(title);
// }

template<> template<typename TargetT> void Image<Vec3u8>::ConvertToGrayscale(Image<TargetT>* target) const {
  target->SetSize(this->width(), this->height());
  
  for (u32 y = 0; y < height(); ++ y) {
    const Vec3u8* this_ptr = row(y);
    TargetT* target_ptr = target->row(y);
    const Vec3u8* end = this_ptr + width();
    while (this_ptr < end) {
      *target_ptr = 0.299f * this_ptr->coeffRef(0) +
                    0.587f * this_ptr->coeffRef(1) +
                    0.114f * this_ptr->coeffRef(2);
      
      ++ this_ptr;
      ++ target_ptr;
    }
  }
}

#ifdef WIN32
#include "libvis/image_template_specializations.h"
#endif

}

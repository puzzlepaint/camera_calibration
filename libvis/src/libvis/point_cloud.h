// Copyright 2018, 2019 ETH Zürich, Thomas Schöps
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

#include <boost/algorithm/string/predicate.hpp>

#include "libvis/logging.h"

#include "libvis/eigen.h"
#include "libvis/image.h"
#include "libvis/libvis.h"
#include "libvis/sophus.h"

namespace vis {

/// Point type storing only a position attribute.
template <typename PositionT>
struct Point {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline Point(const PositionT& position)
      : position_(position) {}
  
  inline PositionT& position() { return position_; }
  inline const PositionT& position() const { return position_; }
  
 private:
  PositionT position_;
};


/// Point type storing position and normal.
// TODO: Implement this


/// Point type storing position and color.
template <typename PositionT, typename ColorT>
struct PointC {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline PointC(const PositionT& position, const ColorT& color)
      : position_(position), color_(color) {}
  
  inline PositionT& position() { return position_; }
  inline const PositionT& position() const { return position_; }
  
  inline ColorT& color() { return color_; }
  inline const ColorT& color() const { return color_; }
  
 private:
  PositionT position_;
  ColorT color_;
};


/// Point type storing position, color, and normal.
template <typename PositionT, typename ColorT, typename NormalT>
struct PointCN {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline PointCN(const PositionT& position, const ColorT& color, const NormalT& normal)
      : position_(position), color_(color), normal_(normal) {}
  
  inline PositionT& position() { return position_; }
  inline const PositionT& position() const { return position_; }
  
  inline ColorT& color() { return color_; }
  inline const ColorT& color() const { return color_; }
  
  inline NormalT& normal() { return normal_; }
  inline const NormalT& normal() const { return normal_; }
  
 private:
  PositionT position_;
  ColorT color_;
  NormalT normal_;
};


/// Traits type for querying the presence of point attributes.
/// Must have a specialization for each point type that is used.
template <typename PointT>
struct PointTraits {};

template<typename _PositionT>
struct PointTraits<Point<_PositionT>> {
  typedef _PositionT PositionT;
  
  static const bool has_color = false;
  static const bool has_normal = false;
};

template<typename _PositionT, typename _ColorT>
struct PointTraits<PointC<_PositionT, _ColorT>> {
  typedef _PositionT PositionT;
  typedef _ColorT ColorT;
  
  static const bool has_color = true;
  static const bool has_normal = false;
};

template<typename _PositionT, typename _ColorT, typename _NormalT>
struct PointTraits<PointCN<_PositionT, _ColorT, _NormalT>> {
  typedef _PositionT PositionT;
  typedef _ColorT ColorT;
  typedef _NormalT NormalT;
  
  static const bool has_color = true;
  static const bool has_normal = true;
};


/// Traits type for applying a cast to Eigen Matrix types or to scalar types,
/// either using static_cast<>() or using Eigen::Matrix::cast<>(),
/// by passing for example unsigned char or Eigen::Matrix<float, 1, 1> as T.
template <typename InputT>
struct CastEigenOrScalar {
  template <typename T>
  inline T Cast(const InputT& input) {
    return static_cast<T>(input);
  }
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct CastEigenOrScalar<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  template <typename T>
  inline T Cast(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& input) {
    return input.template cast<typename T::Scalar>();
  }
};


/// Generic point cloud type, templated with the point type PointT, storing the
/// points in CPU memory.
/// 
/// The point types should have some attributes with common names such that they
/// can be used by PointCloud. At the moment, this is position() which is
/// expected to return a writable reference to an Eigen::Vector compatible type,
/// and if the point types has colors, color() to return the color in the same
/// way.
template <typename PointT>
class PointCloud {
 public:
  using PositionT = typename PointTraits<PointT>::PositionT;
  
  /// Creates an empty point cloud.
  PointCloud()
      : data_(nullptr), size_(0), alignment_(0) {}
  
  /// Creates a deep copy of the other point cloud.
  PointCloud(const PointCloud<PointT>& other)
      : data_(nullptr), size_(0), alignment_(0) {
    Resize(other.size_, other.alignment_);
    memcpy(data_, other.data_, SizeInBytes());
  }
  
  /// Creates a point cloud with the given point count. Does not initialize the
  /// point memory. The memory alignment is chosen automatically.
  PointCloud(usize size)
      : data_(nullptr), size_(0), alignment_(0) {
    Resize(size);
  }
  
  /// Creates a point cloud with the given point count and alignment (in bytes,
  /// must be a power of two and multiple of sizeof(void*), or 1 for no
  /// alignment).
  PointCloud(usize size, usize alignment)
      : data_(nullptr), size_(0), alignment_(0) {
    Resize(size, alignment);
  }
  
  ~PointCloud() {
    /// Does nothing if data_ is nullptr.
    free(data_);
  }
  
  /// Appends a point to the cloud. This is slow (since the point buffer is
  /// reallocated each time) and should only be used if performance does not
  /// matter. A faster alternative is to count the final required point count
  /// beforehand and allocate the point buffer with this size only once (on point
  /// cloud construction, or using Resize()).
  void AppendSlow(const PointT& point) {
    PointT* old_data = data_;
    usize old_size = size_;
    
    data_ = nullptr;
    size_ = 0;
    
    Resize(old_size + 1, alignment_);
    
    memcpy(data_, old_data, old_size * sizeof(PointT));
    data_[size_ - 1] = point;
    
    free(old_data);
  }
  
  /// Changes the size. Re-allocates the point buffer if the new size is
  /// different from the current one. Does not preserve the point data.
  void Resize(usize size) {
    if (data_ && size_ == size) {
      return;
    }
    Resize(size, 1);
  }
  
  /// Changes the size and alignment (in bytes, must be a power of two and
  /// multiple of sizeof(void*), or 1 for no alignment). Re-allocates the point
  /// buffer if the new settings are different from the current ones. Does not
  /// preserve the point data.
  void Resize(usize size, usize alignment) {
    if (data_ && size_ == size && alignment_ == alignment) {
      return;
    }
    
    // Does nothing if data_ is nullptr.
    free(data_);
    data_ = nullptr;
    
    int return_value;
    if (alignment == 1) {
      data_ = reinterpret_cast<PointT*>(malloc(size * sizeof(PointT)));
      return_value = (data_ == nullptr) ? (-1) : 0;
    } else {
#ifdef WIN32
      data_ = reinterpret_cast<PointT*>(_aligned_malloc(size * sizeof(PointT), alignment));
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
      return_value = posix_memalign(reinterpret_cast<void**>(&data_), alignment, size * sizeof(PointT));
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
#endif
    
    size_ = size;
    alignment_ = alignment;
  }
  
  /// Computes the axis-aligned bounding box extents for the point cloud. Does
  /// not do anything for empty point clouds.
  /// TODO: Compare performance between the current method and the one which
  ///       is commented out, and additionally, to using _min and _max directly
  ///       instead of using local variables on the stack.
  void ComputeMinMax(PositionT* _min, PositionT* _max) const {
    if (empty()) {
      return;
    }
    
    PositionT min = data_[0].position();
    PositionT max = data_[0].position();
    
    for (usize i = 1; i < size_; ++ i) {
      for (int c = 0; c < PositionT::RowsAtCompileTime; ++ c) {
        if (data_[i].position().coeffRef(c) <
            min.coeffRef(c)) {
          min.coeffRef(c) = data_[i].position().coeffRef(c);
        }
        if (data_[i].position().coeffRef(c) >
            max.coeffRef(c)) {
          max.coeffRef(c) = data_[i].position().coeffRef(c);
        }
      }
      
//       *min = min->cwiseMin(data_[i].position());
//       *max = max->cwiseMax(data_[i].position());
    }
    
    *_min = min;
    *_max = max;
  }
  
  /// Un-projects the depth map pixels into 3D space, creates a point for each
  /// valid pixel. Only the depth attribute of the points is set, possible other
  /// attributes remain uninitialized.
  ///
  /// \sa SetFromRGBDImage
  template <typename DepthT, typename CameraT>
  void SetFromDepthImage(const Image<DepthT> depth_image, bool depth_is_inverse, DepthT invalid_depth_value, const CameraT& camera) {
    // Count the valid pixels in the depth image.
    usize point_count = 0;
    for (u32 y = 0; y < depth_image.height(); ++ y) {
      const DepthT* ptr = depth_image.row(y);
      const DepthT* end = ptr + depth_image.width();
      while (ptr < end) {
        if (*ptr != invalid_depth_value) {
          ++ point_count;
        }
        ++ ptr;
      }
    }
    
    Resize(point_count);
    
    // Create the points.
    point_count = 0;
    for (u32 y = 0; y < depth_image.height(); ++ y) {
      const DepthT* ptr = depth_image.row(y);
      for (u32 x = 0; x < depth_image.width(); ++ x) {
        if (*ptr != invalid_depth_value) {
          data_[point_count].position() =
              (depth_is_inverse ? (1.f / *ptr) : *ptr) *
              camera.UnprojectFromPixelCenterConv(Vec2f(x, y)).template cast<typename PositionT::Scalar>();
          ++ point_count;
        }
        ++ ptr;
      }
    }
  }
  
  /// Un-projects the depth map pixels into 3D space, creates a point for each
  /// valid pixel. This function can only be called on point clouds with point
  /// types that contain a color attribute.
  ///
  /// \sa SetFromDepthImage
  template <typename DepthT, typename ColorT, typename CameraT>
  void SetFromRGBDImage(
      const Image<DepthT> depth_image,
      bool depth_is_inverse,
      DepthT invalid_depth_value,
      const Image<ColorT> color_image,
      const CameraT& camera) {
    // Count the valid pixels in the depth image.
    usize point_count = 0;
    for (u32 y = 0; y < depth_image.height(); ++ y) {
      const DepthT* ptr = depth_image.row(y);
      const DepthT* end = ptr + depth_image.width();
      while (ptr < end) {
        if (*ptr != invalid_depth_value) {
          ++ point_count;
        }
        ++ ptr;
      }
    }
    
    Resize(point_count);
    
    // Create the points.
    point_count = 0;
    for (u32 y = 0; y < depth_image.height(); ++ y) {
      const DepthT* d_ptr = depth_image.row(y);
      const ColorT* rgb_ptr = color_image.row(y);
      for (u32 x = 0; x < depth_image.width(); ++ x) {
        if (*d_ptr != invalid_depth_value) {
          data_[point_count].position() =
              (depth_is_inverse ? (1.f / *d_ptr) : *d_ptr) *
              camera.UnprojectFromPixelCenterConv(Vec2f(x, y)).template cast<typename PositionT::Scalar>();
          data_[point_count].color() =
              CastEigenOrScalar<ColorT>().template Cast<typename PointTraits<PointT>::ColorT>(*rgb_ptr);
          ++ point_count;
        }
        ++ d_ptr;
        ++ rgb_ptr;
      }
    }
  }
  
  /// Transforms all points in the cloud by left-multiplication with the given
  /// Sophus SE3 transformation. Also rotates point normals for clouds with
  /// normals.
  /// 
  /// There is a small overhead in computing the rotation matrix from the SE3's
  /// quaternion. If transforming a large number of point clouds, it would be
  /// faster to pre-compute the rotation matrix only once.
  /// 
  /// If renormalize_normals is true, the normal vectors will be normalized
  /// after rotation to avoid the build-up of small errors over time. If the
  /// point cloud does not contain normals, the value of renormalize_normals is
  /// ignored.
  template <typename Derived>
  void Transform(
      const Sophus::SE3Base<Derived>& transform,
      bool renormalize_normals = false) {
    // Convert the rotation quaternion to a matrix for faster point
    // multiplication.
    Matrix<typename PositionT::Scalar,
           PositionT::RowsAtCompileTime,
           PositionT::RowsAtCompileTime> rotation =
        transform.rotationMatrix().template cast<typename PositionT::Scalar>();
    Matrix<typename PositionT::Scalar,
           PositionT::RowsAtCompileTime, 1> translation =
        transform.translation().template cast<typename PositionT::Scalar>();
    
    if (renormalize_normals) {
      TransformHelper<true>(rotation, translation, this);
    } else {
      TransformHelper<false>(rotation, translation, this);
    }
  }
  
  /// Version of Transform() that directly takes a 3x3 rotation matrix and a
  /// 3x1 translation vector, avoiding their computation from an SE3 object
  /// (that contains a quaternion for the rotation).
  template <typename DerivedA, typename DerivedB>
  void Transform(
      const MatrixBase<DerivedA>& rotation,
      const MatrixBase<DerivedB>& translation,
      bool renormalize_normals = false) {
    if (renormalize_normals) {
      TransformHelper<true>(rotation, translation, this);
    } else {
      TransformHelper<false>(rotation, translation, this);
    }
  }
  
  /// Saves the point cloud in .obj format. This is a human-readable ASCII-based
  /// format; use WriteAsPLY() instead for a more efficient binary format.
  bool WriteAsOBJ(const char* path) {
    ofstream file_stream(path, std::ios::out);
    if (!file_stream) {
      return false;
    }
    
    bool result = WriteAsOBJ(&file_stream);
    
    file_stream.close();
    return result;
  }
  
  /// Saves the point cloud in .obj format. This is a human-readable ASCII-based
  /// format; use WriteAsPLY() instead for a more efficient binary format.
  template<typename _CharT, typename _Traits>
  bool WriteAsOBJ(basic_ostream<_CharT,_Traits>* stream) {
    WriteAsOBJHelper(stream, *this);
    return true;
  }
  
  /// Saves the point cloud in .ply format (in its binary variant). Returns true
  /// if successful.
  bool WriteAsPLY(const string& path) {
    return WriteAsPLY(path.c_str());
  }
  
  /// Saves the point cloud in .ply format (in its binary variant). Returns true
  /// if successful.
  bool WriteAsPLY(const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) {
      return false;
    }
    
    // Write header
    std::ostringstream header;  // TODO: Use something faster than ostringstream?
    header <<  "ply\n"
               "format binary_little_endian 1.0\n"
               "element vertex " << size() << "\n";
    
    // TODO: Make also work with other types than float
    header << "property float x\n"
              "property float y\n"
              "property float z\n";
    
    if (PointTraits<PointT>::has_color) {
      // TODO: Make also work with other types than uchar
      header << "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n";
    }
    
    if (PointTraits<PointT>::has_normal) {
      // TODO: Make also work with other types than float
      header << "property float nx\n"
                "property float ny\n"
                "property float nz\n";
    }
    
    header << "end_header\n";
    string header_string = header.str();
    fwrite(header_string.data(), 1, header_string.size(), file);
    
    // Write points
    WriteAsPLYHelper(file, *this);
    
    fclose(file);
    return true;
  }
  
  /// Loads the point cloud from .ply format (binary variant). Returns true if
  /// successful.
  /// TODO: This parser is likely not robust at all and should be improved to
  ///       handle the file format more generically instead of only understanding
  ///       files written by WriteAsPLY().
  bool Read(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
      return false;
    }
    
    // File format identifier
    u8 header[3];
    if (fread(header, 1, 3, file) != 3 ||
        header[0] != 'p' ||
        header[1] != 'l' ||
        header[2] != 'y') {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Invalid file header.";
      fclose(file);
      return false;
    }
    
    // Parse header
    vector<string> words;
    string current_word;
    
    vector<int> properties;
    
    while (true) {
      char c;
      if (fread(&c, 1, 1, file) != 1) {
        LOG(ERROR) << "Cannot parse file: " << path;
        LOG(ERROR) << "Unexpected end-of-file in header.";
        fclose(file);
        return false;
      }
      
      if (c == '\r') {
        continue;
      } else if (c == '\n') {
        if (!current_word.empty()) {
          words.push_back(current_word);
          current_word = "";
        }
        
        // Parse the words read in this line.
        if (!words.empty()) {
          if (words[0] == string("format")) {
            if (words.size() != 3) {
              LOG(ERROR) << "Cannot parse file: " << path;
              LOG(ERROR) << "Unexpected number of words in 'format' line.";
              fclose(file);
              return false;
            }
            
            if (words[1] != string("binary_little_endian")) {
              LOG(ERROR) << "Cannot parse file: " << path;
              LOG(ERROR) << "Unsupported format: " << words[1] << ". Only 'binary_little_endian' is supported currently.";
              fclose(file);
              return false;
            }
            
            if (words[2] != string("1.0")) {
              LOG(1) << "Warning: Unexpected format number of PLY file: " << words[2];
            }
          } else if (words[0] == string("element")) {
            if (words.size() != 3) {
              LOG(ERROR) << "Cannot parse file: " << path;
              LOG(ERROR) << "Unexpected number of words in 'element' line.";
              fclose(file);
              return false;
            }
            
            if (words[1] != string("vertex")) {
              LOG(ERROR) << "Cannot parse file: " << path;
              LOG(ERROR) << "Unsupported element: " << words[1] << ". Only 'vertex' is supported currently.";
              fclose(file);
              return false;
            }
            
            Resize(atoi(words[2].c_str()));
          } else if (words[0] == string("property")) {
            if (words.size() != 3) {
              LOG(ERROR) << "Cannot parse file: " << path;
              LOG(ERROR) << "Unexpected number of words in 'property' line.";
              fclose(file);
              return false;
            }
            
            // Heuristics for assigning names to point elements.
            // TODO: This must be much more relaxed to work well with files from other sources.
            if (boost::iequals(words[2], string("x"))) {
              properties.push_back(0);
            } else if (boost::iequals(words[2], string("y"))) {
              properties.push_back(1);
            } else if (boost::iequals(words[2], string("z"))) {
              properties.push_back(2);
            } else if (boost::iequals(words[2], string("red")) || boost::iequals(words[2], string("r"))) {
              properties.push_back(3);
            } else if (boost::iequals(words[2], string("green")) || boost::iequals(words[2], string("g"))) {
              properties.push_back(4);
            } else if (boost::iequals(words[2], string("blue")) || boost::iequals(words[2], string("b"))) {
              properties.push_back(5);
            } else if (boost::iequals(words[2], string("normalx")) || boost::iequals(words[2], string("normal_x")) || boost::iequals(words[2], string("nx"))) {
              properties.push_back(6);
            } else if (boost::iequals(words[2], string("normaly")) || boost::iequals(words[2], string("normal_y")) || boost::iequals(words[2], string("ny"))) {
              properties.push_back(7);
            } else if (boost::iequals(words[2], string("normalz")) || boost::iequals(words[2], string("normal_z")) || boost::iequals(words[2], string("nz"))) {
              properties.push_back(8);
            } else {
              LOG(WARNING) << "Unhandled PLY property: " << words[0] << " " << words[1] << " " << words[2];
            }
          } else if (words[0] == string("end_header")) {
            break;
          } else {
            LOG(ERROR) << "Cannot parse file: " << path;
            LOG(ERROR) << "Unexpected word at start of line: " << words[0];
            fclose(file);
            return false;
          }
          
          words.clear();
        }
      } else if (c == ' ') {
        if (!current_word.empty()) {
          words.push_back(current_word);
          current_word = "";
        }
      } else {
        current_word += c;
      }
    }
    
    // Read the vertices.
    // TODO: Could it be helpful for performance to use special case(s) to load common vertex configurations?
    if (!ReadFromPLYHelper(file, properties, this)) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Unexpected end-of-file in content.";
      fclose(file);
      return false;
    }
    
    fclose(file);
    return true;
  }
  
  /// Returns the i-th point in the cloud.
  inline PointT& operator[](int i) { return data_[i]; }
  inline const PointT& operator[](int i) const { return data_[i]; }
  
  inline PointT& at(int i) { return data_[i]; }
  inline const PointT& at(int i) const { return data_[i]; }
  
  /// Returns a pointer to the data.
  inline const PointT* data() const { return data_; }
  inline PointT* data_mutable() { return data_; }
  
  /// Returns whether the size of the point cloud is zero.
  inline bool empty() const { return size_ == 0; }
  
  /// Returns the number of points in the cloud.
  inline usize size() const { return size_; }
  
  /// Returns the size of the points in the cloud in bytes.
  inline usize SizeInBytes() const { return size_ * sizeof(PointT); }
  
 private:
  template<typename _CharT, typename _Traits, typename PositionT>
  static void WriteAsOBJHelper(basic_ostream<_CharT,_Traits>* stream, const PointCloud<Point<PositionT>>& cloud) {
    for (usize i = 0; i < cloud.size_; ++ i) {
      const PointT& point = cloud.data_[i];
      *stream << "v " << point.position().x()
              << " " << point.position().y()
              << " " << point.position().z()
              << std::endl;
    }
  }
  
  template<typename _CharT, typename _Traits, typename PositionT, typename ColorT>
  static void WriteAsOBJHelper(basic_ostream<_CharT,_Traits>* stream, const PointCloud<PointC<PositionT, ColorT>>& cloud) {
    // TODO: This will not work if using scalars for colors
    constexpr float kNormalizationFactor =
        1.0f / numeric_limits<typename ColorT::Scalar>::max();
    for (usize i = 0; i < cloud.size_; ++ i) {
      const PointT& point = cloud.data_[i];
      *stream << "v " << point.position().x()
              << " " << point.position().y()
              << " " << point.position().z()
              << " " << (kNormalizationFactor * point.color().x())
              << " " << (kNormalizationFactor * point.color().y())
              << " " << (kNormalizationFactor * point.color().z())
              << std::endl;
    }
  }
  
  template<typename _CharT, typename _Traits, typename PositionT, typename ColorT, typename NormalT>
  static void WriteAsOBJHelper(basic_ostream<_CharT,_Traits>* stream, const PointCloud<PointCN<PositionT, ColorT, NormalT>>& cloud) {
    // It seems that in .obj format, normals are specified separately from
    // vertices, and they are only merged in face definitions. So we might not
    // be able to assign normals to vertices directly.
    // TODO: It might be a reasonable use-case to save the positions and colors
    //       of a point cloud that also has normals as .obj. In this case, there
    //       should be a way to deactivate this warning.
    LOG(WARNING) << "Saving normals in .obj format is not supported. The normals will be missing.";
    
    // TODO: This will not work if using scalars for colors
    constexpr float kNormalizationFactor =
        1.0f / numeric_limits<typename ColorT::Scalar>::max();
    for (usize i = 0; i < cloud.size_; ++ i) {
      const PointT& point = cloud.data_[i];
      *stream << "v " << point.position().x()
              << " " << point.position().y()
              << " " << point.position().z()
              << " " << (kNormalizationFactor * point.color().x())
              << " " << (kNormalizationFactor * point.color().y())
              << " " << (kNormalizationFactor * point.color().z())
              << std::endl;
    }
  }
  
  
  template<typename PositionT>
  static void WriteAsPLYHelper(FILE* file, const PointCloud<Point<PositionT>>& cloud) {
    for (usize i = 0; i < cloud.size(); ++ i) {
      const PointT& point = cloud.data_[i];
      
      fwrite(&point.position().x(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.position().y(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.position().z(), sizeof(typename PositionT::Scalar), 1, file);
    }
  }
  
  template<typename PositionT, typename ColorT>
  static void WriteAsPLYHelper(FILE* file, const PointCloud<PointC<PositionT, ColorT>>& cloud) {
    for (usize i = 0; i < cloud.size(); ++ i) {
      const PointT& point = cloud.data_[i];
      
      fwrite(&point.position().x(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.position().y(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.position().z(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.color().x(), sizeof(typename ColorT::Scalar), 1, file);
      fwrite(&point.color().y(), sizeof(typename ColorT::Scalar), 1, file);
      fwrite(&point.color().z(), sizeof(typename ColorT::Scalar), 1, file);
    }
  }
  
  template<typename PositionT, typename ColorT, typename NormalT>
  static void WriteAsPLYHelper(FILE* file, const PointCloud<PointCN<PositionT, ColorT, NormalT>>& cloud) {
    for (usize i = 0; i < cloud.size(); ++ i) {
      const PointT& point = cloud.data_[i];
      
      fwrite(&point.position().x(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.position().y(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.position().z(), sizeof(typename PositionT::Scalar), 1, file);
      fwrite(&point.color().x(), sizeof(typename ColorT::Scalar), 1, file);
      fwrite(&point.color().y(), sizeof(typename ColorT::Scalar), 1, file);
      fwrite(&point.color().z(), sizeof(typename ColorT::Scalar), 1, file);
      fwrite(&point.normal().x(), sizeof(typename NormalT::Scalar), 1, file);
      fwrite(&point.normal().y(), sizeof(typename NormalT::Scalar), 1, file);
      fwrite(&point.normal().z(), sizeof(typename NormalT::Scalar), 1, file);
    }
  }
  
  
  template<typename PositionT>
  static bool ReadFromPLYHelper(FILE* file, const vector<int>& properties, PointCloud<Point<PositionT>>* cloud) {
    for (usize i = 0; i < cloud->size(); ++ i) {
      PointT& point = cloud->data_[i];
      
      for (usize p = 0; p < properties.size(); ++ p) {
        if ((p >= 0 && p <= 2) || (p >= 6 && p <= 8)) {
          float v;
          if (fread(&v, sizeof(float), 1, file) != 1) {
            return false;
          }
          if (p >= 0 && p <= 2) {
            point.position()(p) = v;
          } else {  // p >= 6 && p <= 8
            // Drop normal information.
          }
        } else {  // p >= 3 && p <= 5
          char c;
          if (fread(&c, 1, 1, file) != 1) {
            return false;
          }
          // Drop color information.
        }
      }
    }
    return true;
  }
  
  template<typename PositionT, typename ColorT>
  static bool ReadFromPLYHelper(FILE* file, const vector<int>& properties, PointCloud<PointC<PositionT, ColorT>>* cloud) {
    for (usize i = 0; i < cloud->size(); ++ i) {
      PointT& point = cloud->data_[i];
      point.color() = Vec3u8(0, 0, 0);  // initialize for the case of no color properties in file
      
      for (usize p = 0; p < properties.size(); ++ p) {
        if ((p >= 0 && p <= 2) || (p >= 6 && p <= 8)) {
          float v;
          if (fread(&v, sizeof(float), 1, file) != 1) {
            return false;
          }
          if (p >= 0 && p <= 2) {
            point.position()(p) = v;
          } else {  // p >= 6 && p <= 8
            // Drop normal information.
          }
        } else {  // p >= 3 && p <= 5
          char c;
          if (fread(&c, 1, 1, file) != 1) {
            return false;
          }
          point.color()(p - 3) = c;
        }
      }
    }
    return true;
  }
  
  template<typename PositionT, typename ColorT, typename NormalT>
  static bool ReadFromPLYHelper(FILE* file, const vector<int>& properties, PointCloud<PointCN<PositionT, ColorT, NormalT>>* cloud) {
    for (usize i = 0; i < cloud->size(); ++ i) {
      PointT& point = cloud->data_[i];
      point.color() = Vec3u8(0, 0, 0);  // initialize for the case of no color properties in file
      point.normal() = Vec3f(0, 0, 0);  // initialize for the case of no normal properties in file
      
      for (usize p = 0; p < properties.size(); ++ p) {
        if ((p >= 0 && p <= 2) || (p >= 6 && p <= 8)) {
          float v;
          if (fread(&v, sizeof(float), 1, file) != 1) {
            return false;
          }
          if (p >= 0 && p <= 2) {
            point.position()(p) = v;
          } else {  // p >= 6 && p <= 8
            point.normal()(p - 6) = v;
          }
        } else {  // p >= 3 && p <= 5
          char c;
          if (fread(&c, 1, 1, file) != 1) {
            return false;
          }
          point.color()(p - 3) = c;
        }
      }
    }
    return true;
  }
  
  
  template<bool renormalize_normals, typename DerivedA, typename DerivedB, typename PositionT>
  static void TransformHelper(
      const MatrixBase<DerivedA>& rotation,
      const MatrixBase<DerivedB>& translation,
      PointCloud<Point<PositionT>>* cloud) {
    for (usize i = 0; i < cloud->size_; ++ i) {
      auto& point = cloud->data_[i];
      point.position() = rotation * point.position() + translation;
    }
  }
  
  template<bool renormalize_normals, typename DerivedA, typename DerivedB, typename PositionT, typename ColorT>
  static void TransformHelper(
      const MatrixBase<DerivedA>& rotation,
      const MatrixBase<DerivedB>& translation,
      PointCloud<PointC<PositionT, ColorT>>* cloud) {
    for (usize i = 0; i < cloud->size_; ++ i) {
      auto& point = cloud->data_[i];
      point.position() = rotation * point.position() + translation;
    }
  }
  
  template<bool renormalize_normals, typename DerivedA, typename DerivedB, typename PositionT, typename ColorT, typename NormalT>
  static void TransformHelper(
      const MatrixBase<DerivedA>& rotation,
      const MatrixBase<DerivedB>& translation,
      PointCloud<PointCN<PositionT, ColorT, NormalT>>* cloud) {
    for (usize i = 0; i < cloud->size_; ++ i) {
      auto& point = cloud->data_[i];
      point.position() = rotation * point.position() + translation;
      if (renormalize_normals) {
        point.normal() = (rotation * point.normal()).normalized();
      } else {
        point.normal() = rotation * point.normal();
      }
    }
  }
  
  
  PointT* data_;
  usize size_;
  usize alignment_;
};


typedef Point<Vec3f> Point3f;
typedef PointC<Vec3f, u8> Point3fCu8;
typedef PointC<Vec3f, Vec3u8> Point3fC3u8;
typedef PointCN<Vec3f, Vec3u8, Vec3f> Point3fC3u8Nf;

typedef PointCloud<Point3f> Point3fCloud;
typedef PointCloud<Point3fCu8> Point3fCu8Cloud;
typedef PointCloud<Point3fC3u8> Point3fC3u8Cloud;
typedef PointCloud<Point3fC3u8Nf> Point3fC3u8NfCloud;

}

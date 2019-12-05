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
// 
// This file is in small parts based on Kalibr, which has the following license:
// 
// Copyright (c) 2014, Paul Furgale, Jérôme Maye and Jörn Rehder, Autonomous Systems Lab, 
//                     ETH Zurich, Switzerland
// Copyright (c) 2014, Thomas Schneider, Skybotix AG, Switzerland
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
//     Redistributions of source code must retain the above copyright notice, this 
//     list of conditions and the following disclaimer.
// 
//     Redistributions in binary form must reproduce the above copyright notice, 
//     this list of conditions and the following disclaimer in the documentation 
//     and/or other materials provided with the distribution.
// 
//     All advertising materials mentioning features or use of this software must 
//     display the following acknowledgement: This product includes software developed 
//     by the Autonomous Systems Lab and Skybotix AG.
// 
//     Neither the name of the Autonomous Systems Lab and Skybotix AG nor the names 
//     of its contributors may be used to endorse or promote products derived from 
//     this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTONOMOUS SYSTEMS LAB AND SKYBOTIX AG ''AS IS'' 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL the AUTONOMOUS SYSTEMS LAB OR SKYBOTIX AG BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
// OF SUCH DAMAGE.


#pragma once

#include <memory>

#include "libvis/logging.h"

#include "libvis/eigen.h"
#include "libvis/libvis.h"

namespace vis {

// Different image coordinate system conventions, which describe how coordinates
// in an image relate to its pixels.
enum class ImageCoordinateConvention {
  // The origin of the image (pixel) coordinates is at the top left image
  // corner, i.e., at the top left corner of the top left pixel in the image.
  // 
  // This convention is convenient for rounding a pixel value to integer
  // coordinates, since the result will be the pixel square which the original
  // floating point coordinate is in.
  // 
  // Coordinate of the top left corner: (0, 0)
  // Coordinate of the center of the top left pixel: (0.5, 0.5)
  // Coordinate of the center of the bottom right pixel: (width - 0.5, height - 0.5)
  // Coordinate of the bottom right corner: (width, height)
  kPixelCorner = 0,
  
  // The origin of the image (pixel) coordinates is in the center of the top
  // left pixel in the image.
  // 
  // This convention is convenient for bilinear filtering, since the offsets
  // from the pixel centers can be computed easily.
  // 
  // Coordinate of the top left corner: (-0.5, -0.5)
  // Coordinate of the center of the top left pixel: (0, 0)
  // Coordinate of the center of the bottom right pixel: (width - 1, height - 1)
  // Coordinate of the bottom right corner: (width - 0.5, height - 0.5)
  kPixelCenter = 1,
  
  // The origin of the image (pixel) coordinates is at the top left image
  // corner, i.e., at the top left corner of the top left pixel in the image.
  // The image coordinates are scaled such that (1, 1) corresponds to the bottom
  // right corner of the image (regardless of the image size).
  // 
  // This convention is convenient when working with textures on the GPU.
  // 
  // Coordinate of the top left corner: (0, 0)
  // Coordinate of the center of the top left pixel: (0.5 / width, 0.5 / height)
  // Coordinate of the center of the bottom right pixel: ((width - 0.5) / width, (height - 0.5) / height)
  // Coordinate of the bottom right corner: (1, 1)
  kRatio = 2
};


// NOTE: These helper functions could be moved to a helper header.
template <typename CameraT, typename Derived>
inline bool ProjectToPixelCornerConvIfVisibleHelper(
    const CameraT& camera, const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) {
  Matrix<typename CameraT::ScalarT, 2, 1> temp;
  bool result = camera.ProjectToPixelCornerConvIfVisible(camera_space_point.template cast<typename CameraT::ScalarT>(), pixel_border, &temp);
  if (result) {
    *pixel_coordinates = temp.template cast<double>();
  }
  return result;
}

template <typename CameraT, typename Derived>
inline bool ProjectToPixelCenterConvIfVisibleHelper(
    const CameraT& camera, const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) {
  Matrix<typename CameraT::ScalarT, 2, 1> temp;
  bool result = camera.ProjectToPixelCenterConvIfVisible(camera_space_point.template cast<typename CameraT::ScalarT>(), pixel_border, &temp);
  if (result) {
    *pixel_coordinates = temp.template cast<double>();
  }
  return result;
}

template <typename CameraT, typename Derived>
inline bool ProjectToRatioConvIfVisibleHelper(
    const CameraT& camera, const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) {
  Matrix<typename CameraT::ScalarT, 2, 1> temp;
  bool result = camera.ProjectToRatioConvIfVisible(camera_space_point.template cast<typename CameraT::ScalarT>(), pixel_border, &temp);
  if (result) {
    *pixel_coordinates = temp.template cast<double>();
  }
  return result;
}


// Base class for camera classes.
// 
// These camera classes allow to convert between 3D points in camera space (or
// rather, directions) and image points by modeling the geometric part of the
// image aquisition. This can be implemented in different ways, with various
// lens distortion models. This is accounted for here by modeling it as a
// relatively generic sequence of operations (depending on the camera's
// intrinsic parameters) given as template parameters to the CameraImpl class.
// To allow for a variable number of operations, a variadic template is used.
// The class hierarchy then looks like this example with two algorithm steps:
// 
//   Camera:
//       The base class. Useful to store pointers to camera objects with
//       different models.
//   CameraImplVariadic<Scalar>:
//       Last step in the type recursion.
//   CameraImplVariadic<Scalar, Step1>:
//       Recursion step for algorithm step 1.
//   CameraImplVariadic<Scalar, Step0, Step1>:
//       Recursion step for algorithm step 0.
//   CameraImpl<TypeID, Scalar, Step0, Step1>:
//       Child class which is able to access all information of the algorithm.
// 
// Only the topmost and bottommost classes in this hierarchy are relevant for
// users of the library. The intermediate classes are helpers.
// The Scalar parameter should be float or double, determining the type of the
// intrinsic camera parameters.
// 
// While the Camera base class provides access to most operations via virtual
// functions, for improved performance it should be preferred to determine the
// derived CameraImpl type and call its functions directly. This can be done
// using the IdentifyCamera() function. It determines the type via the
// type ID. Note that this only works with types that are defined in the library
// already, not with user-defined types.
// Design consideration:
// * User-defined types could be supported by having the user define a macro
//   with a given signature before the first libvis include. However, this might
//   be cumbersome to include before each first libvis include.
// 
// The camera classes support the following image coordinate conventions:
// 
// * "PixelCorner":
//   Pixel coordinates with origin at the top-left image corner.
//   The range of coordinates on the image is [0, side_length].
//   This convention allows to easily obtain the pixel a given point lies within
//   by converting its coordinates to integers. Care has to be taken for
//   coordinates in the ]-1, 0[ range since they will be converted to the valid
//   pixel coordinate zero.
// 
// * "PixelCenter":
//   Pixel coordinates with origin at the center of the top left pixel.
//   The range of coordinates on the image is [-0.5, side_length - 0.5].
//   This convention is convenient for doing bilinear interpolation, since
//   converting the pixel coordinates to integers results in the coordinates of
//   the top-left pixel for the interpolation. As above, care has to be taken
//   for coordinates in the ]-1, 0[ range since they will be converted to the
//   valid pixel coordinate zero.
// 
// * "Ratio":
//   These coordinates are defined such that (0, 0) is the top-left image corner
//   and (1, 1) is the bottom-right image corner. This convention is useful for
//   texture coordinates, which are usually expressed in this way.
// 
// Design considerations:
// * Different image coordinate conventions could also be implemented by
//   exchanging the pixel mapping step in a camera. However, algorithms should
//   be able to rely on getting the convention they expect, regardless of the
//   camera object they operate with, which would not be possible with this
//   approach.
// * The different conventions are chosen by calling different functions. There
//   is no default convention. Internally, the conventions are handled with a
//   template parameter to reduce the number of functions. This was not chosen
//   for the outside-facing implementation to reduce the perceived complexity.
// 
// The scheme for classes which are used as algorithm steps in a camera
// implementation is as follows:
//
// template<typename Scalar>
// class AlgorithmStep {
//  public:
//   // Input type to this algorithm step (when doing projection).
//   // The first input type must be Matrix<Scalar, 3, 1> (camera space 3D points).
//   typedef <...> InputType;
//   // Output type of this algorithm step (when doing projection).
//   // The last output type must be Matrix<Scalar, 2, 1> (image coordinates).
//   typedef <...> OutputType;
//   
//   template<ImageCoordinateConvention convention>
//   inline bool ProjectIfVisible(const InputType& <...>,
//                                float pixel_border,
//                                u32 width,
//                                u32 height,
//                                const Scalar* parameters,
//                                OutputType* <...>) const {
//     <...>
//   }
//   
//   template<ImageCoordinateConvention convention>
//   inline OutputType Project(const InputType& <...>,
//                             const Scalar* parameters) const {
//     <...>
//   }
//   
//   template<ImageCoordinateConvention convention>
//   inline InputType Unproject(const OutputType& <...>,
//                              const Scalar* parameters) const {
//     <...>
//   }
//   
//   inline void ScaleParameters(Scalar factor, Scalar* parameters) const {
//     <...>
//   }
//   
//   inline void CropParameters(int left, int top, int right, int bottom, Scalar* parameters) const {
//     <...>
//   }
//   
//   inline void CacheDerivedParameters(u32 width, u32 height, const Scalar* parameters) {
//     <...>
//   }
//   
//   // The number of Scalars allocated for this step in the parameter vector.
//   // They can be directly accessed as elements 0 .. (kParameterCount - 1) in
//   // the parameter pointers passed to the functions of this class.
//   static const u32 kParameterCount = <...>;
// };
class Camera {
 public:
  // Type IDs for derived classes for serialization.
  // NOTE: The values are written and read by WriteAsText() and ReadFromText(),
  //       so they should not change! In addition, the range of numeric values
  //       is used for checking type validity, so it should be successive.
  enum class Type {
    kInvalid = 0,  // must be the lowest number
    kPinholeCamera4f = 1,
    kRadtanCamera8d = 2,
    kRadtanCamera9d = 5,
    kThinPrismFisheyeCamera12d = 3,
    kNonParametricBicubicProjectionCamerad = 4,
    kFovCamera5f = 6,
    kNumTypes = 7  // must be the highest number, does not need to stay constant
  };
  
  // Constructor, sets the type id and image dimensions.
  inline Camera(int type_int, u32 width, u32 height)
      : type_int_(type_int), width_(width), height_(height) {}
  
  // Destructor.
  virtual ~Camera() {}
  
  // Projects a 3D point in camera space to an image point. See the Camera class
  // description for the different image coordinate conventions. True is
  // returned if the 3D point is visible in the image (if it is unobstructed),
  // false otherwise.
  // 
  // NOTE 1: These convenience functions always return double-typed vectors
  // since the Scalar type of the derived class cannot be accessed here. An
  // alternative would be to make two versions Project...Float() and
  // Project...Double() (or template it).
  // 
  // NOTE 2: If performance matters, you should call the corresponding functions of the
  // derived class directly. Write a helper template function and call it with
  // e.g. IDENTIFY_CAMERA() to access the derived camera type directly in this
  // function.
  template <typename Derived>
  inline bool ProjectToPixelCornerConvIfVisible(
      const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) const;
  
  template <typename Derived>
  inline bool ProjectToPixelCenterConvIfVisible(
      const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) const;
  
  template <typename Derived>
  inline bool ProjectToRatioConvIfVisible(
      const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) const;
  
  // Projects a 3D point in camera space to an image point. See the Camera class
  // description for the different image coordinate conventions.
  // 
  // NOTE 1: These convenience functions always return double-typed vectors
  // since the Scalar type of the derived class cannot be accessed here. An
  // alternative would be to make two versions Project...Float() and
  // Project...Double() (or template it, requiring explicitly specifying the type).
  // 
  // NOTE 2: If performance matters, you should call the corresponding functions of the
  // derived class directly. Write a helper template function and call it with
  // e.g. IDENTIFY_CAMERA() to access the derived camera type directly in this
  // function.
  template <typename Derived>
  inline Matrix<double, 2, 1> ProjectToPixelCornerConv(const MatrixBase<Derived>& camera_space_point) const;
  
  template <typename Derived>
  inline Matrix<double, 2, 1> ProjectToPixelCenterConv(const MatrixBase<Derived>& camera_space_point) const;
  
  template <typename Derived>
  inline Matrix<double, 2, 1> ProjectToRatioConv(const MatrixBase<Derived>& camera_space_point) const;
  
  // Unprojects an image point to a direction. See the Camera class
  // description for the different image coordinate conventions.
  // 
  // NOTE 1: These convenience functions always return double-typed vectors
  // since the Scalar type of the derived class cannot be accessed here. An
  // alternative would be to make two versions Project...Float() and
  // Project...Double() (or template it, requiring explicitly specifying the type).
  // 
  // NOTE 2: If performance matters, you should call the corresponding functions of the
  // derived class directly. Write a helper template function and call it with
  // e.g. IDENTIFY_CAMERA() to access the derived camera type directly in this
  // function.
  template <typename Derived>
  inline Matrix<double, 3, 1> UnprojectFromPixelCornerConv(const MatrixBase<Derived>& pixel_coordinates) const;
  
  template <typename Derived>
  inline Matrix<double, 3, 1> UnprojectFromPixelCenterConv(const MatrixBase<Derived>& pixel_coordinates) const;
  
  template <typename Derived>
  inline Matrix<double, 3, 1> UnprojectFromRatioConv(const MatrixBase<Derived>& pixel_coordinates) const;
  
  // Creates a scaled version of this camera, suitable for an image size which
  // is scaled by the same amount. The returned object has to be deleted using
  // delete.
  virtual Camera* Scaled(double factor) const = 0;
  
  // Creates a cropped version of this camera, suitable for images for which
  // the given number of pixels at each side is removed. The returned object
  // has to be deleted using delete.
  virtual Camera* Cropped(int left, int top, int right, int bottom) const = 0;
  
  // Writes the camera attributes in textual form to the stream, in the format:
  // type_int width height num_parameters [parameters ...]
  // Make sure that the stream's precision setting is sufficient before calling this!
  // TODO: It would be much more robust to write the type as a string to not depend on the type_int values staying constant
  virtual void WriteAsText(std::ostream* stream) const = 0;
  
  // Loads the camera attributes from the text stream in the format written by WriteAsText()
  // and allocates a camera object with these attributes. If parsing fails, returns a null pointer.
  // The returned object must be deleted using delete.
  static inline Camera* ReadFromText(std::istream* stream);
  
  // Returns the number of parameters of this camera model.
  // Notice that this can alternatively be gotten as a constant expression if the
  // type of the camera object is known at compile time.
  inline u32 parameter_count() const;
  
  // Returns a pointer to the parameters of this camera model.
  // Notice that this can alternatively be gotten more directly if the
  // type of the camera object is known at compile time.
  // 
  // NOTE: Since the Scalar type of the derived class cannot be accessed here, this
  // function returns void*. An alternative would be to make two versions
  // parameters_float() and parameters_double().
  inline const void* parameters() const;
  
  // Returns the width of the camera images.
  inline u32 width() const { return width_; }
  
  // Returns the height of the camera images.
  inline u32 height() const { return height_; }
  
  // Returns the type, which is the same for each derived class and should be
  // unique. This function returns a Camera::Type. There is also type_int(), which
  // returns the same value as an int.
  inline Camera::Type type() const { return static_cast<Camera::Type>(type_int_); }
  
  // Returns the type ID, which is the same for each derived class and should be
  // unique. This function returns an int. There is also type(), which returns the
  // same value as a Camera::Type.
  inline int type_int() const { return type_int_; }
  
 private:
  template <typename CameraT>
  static Camera* ReadFromTextHelper(std::istream* stream) {
    int width, height, read_parameter_count;
    *stream >> width >> height >> read_parameter_count;
    
    vector<typename CameraT::ScalarT> parameters(read_parameter_count);
    for (int i = 0; i < read_parameter_count; ++ i) {
      *stream >> parameters[i];
    }
    
    return new CameraT(width, height, parameters.data());
  }
  
  int type_int_;
  u32 width_;
  u32 height_;
};

typedef shared_ptr<Camera> CameraPtr;
typedef shared_ptr<const Camera> CameraConstPtr;


// Algorithm step which can be used in camera implementations:
// Projection from camera space to the virtual image plane (normalized image
// coordinates) for pinhole cameras. In unprojection, assigns a z-depth of 1.
template<typename Scalar>
class PinholeProjection {
 public:
  // Camera-space 3D point.
  typedef Matrix<Scalar, 3, 1> InputType;
  // Normalized image coordinates (before distortion).
  typedef Matrix<Scalar, 2, 1> OutputType;
  
  template<ImageCoordinateConvention convention, typename DerivedA, typename DerivedB>
  inline bool ProjectIfVisible(const MatrixBase<DerivedA>& camera_point,
                               float /*pixel_border*/,
                               u32 /*width*/,
                               u32 /*height*/,
                               const Scalar* parameters,
                               MatrixBase<DerivedB>* normalized_image_coordinates) const {
    if (camera_point.coeff(2) <= static_cast<Scalar>(0)) {
      return false;
    }
    *normalized_image_coordinates = Project<convention>(camera_point, parameters);
    return true;
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType Project(const MatrixBase<Derived>& camera_point,
                            const Scalar* /*parameters*/) const {
    return OutputType(camera_point.coeff(0) / camera_point.coeff(2),
                      camera_point.coeff(1) / camera_point.coeff(2));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline InputType Unproject(const MatrixBase<Derived>& projected_point,
                             const Scalar* /*parameters*/) const {
    return projected_point.homogeneous();
  }
  
  inline void ScaleParameters(Scalar /*factor*/, Scalar* /*parameters*/) const {}
  
  inline void CropParameters(int /*left*/, int /*top*/, int /*right*/, int /*bottom*/, Scalar* /*parameters*/) const {}
  
  static inline int GetParameterCount(const Scalar* /*parameters*/) {
    return 0;
  }
  
  inline void CacheDerivedParameters(u32 /*width*/, u32 /*height*/, const Scalar* /*parameters*/) {}
};


// Algorithm step which can be used in camera implementations:
// Radial-tangential distortion with 4 parameters k1, k2, r1, r2.
template<typename Scalar>
class RadtanDistortion4 {
 public:
  // Normalized image coordinates (before distortion).
  typedef Matrix<Scalar, 2, 1> InputType;
  // Normalized image coordinates (after distortion).
  typedef Matrix<Scalar, 2, 1> OutputType;
  
  template<ImageCoordinateConvention convention, typename DerivedA, typename DerivedB>
  inline bool ProjectIfVisible(const MatrixBase<DerivedA>& camera_point,
                               float /*pixel_border*/,
                               u32 /*width*/,
                               u32 /*height*/,
                               const Scalar* parameters,
                               MatrixBase<DerivedB>* normalized_image_coordinates) const {
    *normalized_image_coordinates = Project<convention>(camera_point, parameters);
    return true;
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType Project(const MatrixBase<Derived>& undistorted_point,
                            const Scalar* parameters) const {
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& r1 = parameters[2];
    const Scalar& r2 = parameters[3];
    
    const Scalar mx2_u = undistorted_point.coeff(0) * undistorted_point.coeff(0);
    const Scalar my2_u = undistorted_point.coeff(1) * undistorted_point.coeff(1);
    const Scalar mxy_u = undistorted_point.coeff(0) * undistorted_point.coeff(1);
    const Scalar rho2_u = mx2_u + my2_u;
    const Scalar rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    return OutputType(
        undistorted_point.coeff(0) + undistorted_point.coeff(0) * rad_dist_u + static_cast<Scalar>(2) * r1 * mxy_u + r2 * (rho2_u + static_cast<Scalar>(2) * mx2_u),
        undistorted_point.coeff(1) + undistorted_point.coeff(1) * rad_dist_u + static_cast<Scalar>(2) * r2 * mxy_u + r1 * (rho2_u + static_cast<Scalar>(2) * my2_u));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType ProjectWithJacobian(const MatrixBase<Derived>& undistorted_point,
                                        const Scalar* parameters,
                                        Matrix<Scalar, 2, 2>* jacobian) const {
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& r1 = parameters[2];
    const Scalar& r2 = parameters[3];
    
    const Scalar mx2_u = undistorted_point.coeff(0) * undistorted_point.coeff(0);
    const Scalar my2_u = undistorted_point.coeff(1) * undistorted_point.coeff(1);
    const Scalar mxy_u = undistorted_point.coeff(0) * undistorted_point.coeff(1);
    const Scalar rho2_u = mx2_u + my2_u;
    const Scalar rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;

    (*jacobian)(0, 0) = 1 + rad_dist_u + k1 * static_cast<Scalar>(2) * mx2_u + k2 * rho2_u * static_cast<Scalar>(4) * mx2_u + static_cast<Scalar>(2) * r1 * undistorted_point.coeff(1) + static_cast<Scalar>(6) * r2 * undistorted_point.coeff(0);
    (*jacobian)(1, 0) = k1 * static_cast<Scalar>(2) * mxy_u + k2 * static_cast<Scalar>(4) * rho2_u * mxy_u + r1 * static_cast<Scalar>(2) * undistorted_point.coeff(0) + static_cast<Scalar>(2) * r2 * undistorted_point.coeff(1);
    (*jacobian)(0, 1) = (*jacobian)(1, 0);
    (*jacobian)(1, 1) = 1 + rad_dist_u + k1 * static_cast<Scalar>(2) * my2_u + k2 * rho2_u * static_cast<Scalar>(4) * my2_u + static_cast<Scalar>(6) * r1 * undistorted_point.coeff(1) + static_cast<Scalar>(2) * r2 * undistorted_point.coeff(0);
    
    return OutputType(undistorted_point.coeff(0) + undistorted_point.coeff(0) * rad_dist_u + static_cast<Scalar>(2) * r1 * mxy_u + r2 * (rho2_u + static_cast<Scalar>(2) * mx2_u),
                      undistorted_point.coeff(1) + undistorted_point.coeff(1) * rad_dist_u + static_cast<Scalar>(2) * r2 * mxy_u + r1 * (rho2_u + static_cast<Scalar>(2) * my2_u));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline InputType Unproject(const MatrixBase<Derived>& distorted_point,
                             const Scalar* parameters) const {
    Matrix<Scalar, 2, 1> undistorted_point = distorted_point.template cast<Scalar>();
    Matrix<Scalar, 2, 2> jacobian;
    Matrix<Scalar, 2, 1> cur_redistorted_point;
    
    constexpr int kMaxIterations = 5;
    for (int i = 0; i < kMaxIterations; i++) {
      cur_redistorted_point = ProjectWithJacobian<convention>(undistorted_point, parameters, &jacobian);
      
      Matrix<Scalar, 2, 1> error = distorted_point.template cast<Scalar>() - cur_redistorted_point;
      undistorted_point += (jacobian.transpose() * jacobian).inverse() * jacobian.transpose() * error;
      
      if (error.squaredNorm() < numeric_limits<Scalar>::epsilon()) {
        break;
      }
    }
    
    return undistorted_point;
  }
  
  inline void ScaleParameters(Scalar /*factor*/, Scalar* /*parameters*/) const {}
  
  inline void CropParameters(int /*left*/, int /*top*/, int /*right*/, int /*bottom*/, Scalar* /*parameters*/) const {}
  
  static inline int GetParameterCount(const Scalar* /*parameters*/) {
    return 4;
  }
  
  inline void CacheDerivedParameters(u32 /*width*/, u32 /*height*/, const Scalar* /*parameters*/) {}
};


// Algorithm step which can be used in camera implementations:
// Radial-tangential distortion with 5 parameters k1, k2, k3, r1, r2.
template<typename Scalar>
class RadtanDistortion5 {
 public:
  // Normalized image coordinates (before distortion).
  typedef Matrix<Scalar, 2, 1> InputType;
  // Normalized image coordinates (after distortion).
  typedef Matrix<Scalar, 2, 1> OutputType;
  
  template<ImageCoordinateConvention convention, typename DerivedA, typename DerivedB>
  inline bool ProjectIfVisible(const MatrixBase<DerivedA>& camera_point,
                               float /*pixel_border*/,
                               u32 /*width*/,
                               u32 /*height*/,
                               const Scalar* parameters,
                               MatrixBase<DerivedB>* normalized_image_coordinates) const {
    *normalized_image_coordinates = Project<convention>(camera_point, parameters);
    return true;
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType Project(const MatrixBase<Derived>& undistorted_point,
                            const Scalar* parameters) const {
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& k3 = parameters[2];
    const Scalar& r1 = parameters[3];
    const Scalar& r2 = parameters[4];
    
    const Scalar mx2_u = undistorted_point.coeff(0) * undistorted_point.coeff(0);
    const Scalar my2_u = undistorted_point.coeff(1) * undistorted_point.coeff(1);
    const Scalar mxy_u = undistorted_point.coeff(0) * undistorted_point.coeff(1);
    const Scalar rho2_u = mx2_u + my2_u;
    const Scalar rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u + k3 * rho2_u * rho2_u * rho2_u;
    return OutputType(
        undistorted_point.coeff(0) + undistorted_point.coeff(0) * rad_dist_u + static_cast<Scalar>(2) * r1 * mxy_u + r2 * (rho2_u + static_cast<Scalar>(2) * mx2_u),
        undistorted_point.coeff(1) + undistorted_point.coeff(1) * rad_dist_u + static_cast<Scalar>(2) * r2 * mxy_u + r1 * (rho2_u + static_cast<Scalar>(2) * my2_u));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType ProjectWithJacobian(const MatrixBase<Derived>& undistorted_point,
                                        const Scalar* parameters,
                                        Matrix<Scalar, 2, 2>* jacobian) const {
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& k3 = parameters[2];
    const Scalar& r1 = parameters[3];
    const Scalar& r2 = parameters[4];
    
    const Scalar mx2_u = undistorted_point.coeff(0) * undistorted_point.coeff(0);
    const Scalar my2_u = undistorted_point.coeff(1) * undistorted_point.coeff(1);
    const Scalar mxy_u = undistorted_point.coeff(0) * undistorted_point.coeff(1);
    const Scalar rho2_u = mx2_u + my2_u;
    const Scalar rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u + k3 * rho2_u * rho2_u * rho2_u;
    
    (*jacobian)(0, 0) = 1 + rad_dist_u +
                        k1 * static_cast<Scalar>(2) * mx2_u +
                        k2 * static_cast<Scalar>(4) * rho2_u * mx2_u +
                        k3 * static_cast<Scalar>(6) * rho2_u * rho2_u * mx2_u +
                        r1 * static_cast<Scalar>(2) * undistorted_point.coeff(1) +
                        r2 * static_cast<Scalar>(6) * undistorted_point.coeff(0);
    (*jacobian)(1, 0) = k1 * static_cast<Scalar>(2) * mxy_u +
                        k2 * static_cast<Scalar>(4) * rho2_u * mxy_u +
                        k3 * static_cast<Scalar>(6) * rho2_u * rho2_u * mxy_u +
                        r1 * static_cast<Scalar>(2) * undistorted_point.coeff(0) +
                        r2 * static_cast<Scalar>(2) * undistorted_point.coeff(1);
    (*jacobian)(0, 1) = (*jacobian)(1, 0);
    (*jacobian)(1, 1) = 1 + rad_dist_u +
                        k1 * static_cast<Scalar>(2) * my2_u +
                        k2 * static_cast<Scalar>(4) * rho2_u * my2_u +
                        k3 * static_cast<Scalar>(6) * rho2_u * rho2_u * my2_u +
                        r1 * static_cast<Scalar>(6) * undistorted_point.coeff(1) +
                        r2 * static_cast<Scalar>(2) * undistorted_point.coeff(0);
    
    return OutputType(undistorted_point.coeff(0) + undistorted_point.coeff(0) * rad_dist_u + static_cast<Scalar>(2) * r1 * mxy_u + r2 * (rho2_u + static_cast<Scalar>(2) * mx2_u),
                      undistorted_point.coeff(1) + undistorted_point.coeff(1) * rad_dist_u + static_cast<Scalar>(2) * r2 * mxy_u + r1 * (rho2_u + static_cast<Scalar>(2) * my2_u));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline InputType Unproject(const MatrixBase<Derived>& distorted_point,
                             const Scalar* parameters) const {
    Scalar cur_x = distorted_point.coeff(0);
    Scalar cur_y = distorted_point.coeff(1);
    
    // Gauss-Newton optimization algorithm.
    const float kUndistortionEpsilon = 1e-10f;
    const usize kMaxIterations = 100;
    
    for (usize i = 0; i < kMaxIterations; ++i) {
      Matrix<Scalar, 2, 2> ddxy_dxy;
      Matrix<Scalar, 2, 1> distorted = ProjectWithJacobian<convention>(Matrix<Scalar, 2, 1>(cur_x, cur_y), parameters, &ddxy_dxy);
      
      // (Non-squared) residuals.
      float dx = distorted.x() - distorted_point.x();
      float dy = distorted.y() - distorted_point.y();
      
      // Accumulate H and b.
      float H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0);
      float H_1_0_and_0_1 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1);
      float H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1);
      float b_0 = dx * ddxy_dxy(0, 0) + dy * ddxy_dxy(1, 0);
      float b_1 = dx * ddxy_dxy(0, 1) + dy * ddxy_dxy(1, 1);
      
      // Solve the system and update the parameters.
      float x_1 = (b_1 - H_1_0_and_0_1 / H_0_0 * b_0) /
                  (H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0);
      float x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0;
      cur_x -= x_0;
      cur_y -= x_1;
      
      if (dx * dx + dy * dy < kUndistortionEpsilon) {
        break;
      }
    }
    
    return InputType(cur_x, cur_y);
  }
  
  inline void ScaleParameters(Scalar /*factor*/, Scalar* /*parameters*/) const {}
  
  inline void CropParameters(int /*left*/, int /*top*/, int /*right*/, int /*bottom*/, Scalar* /*parameters*/) const {}
  
  static inline int GetParameterCount(const Scalar* /*parameters*/) {
    return 5;
  }
  
  inline void CacheDerivedParameters(u32 /*width*/, u32 /*height*/, const Scalar* /*parameters*/) {}
};


// TODO: Can we move the "fisheye" / equidistant part into a separate step?
template<typename Scalar>
class ThinPrismFisheyeDistortion8 {
 public:
  // Normalized image coordinates (before distortion).
  typedef Matrix<Scalar, 2, 1> InputType;
  // Normalized image coordinates (after distortion).
  typedef Matrix<Scalar, 2, 1> OutputType;
  
  template<ImageCoordinateConvention convention, typename DerivedA, typename DerivedB>
  inline bool ProjectIfVisible(const MatrixBase<DerivedA>& camera_point,
                               float /*pixel_border*/,
                               u32 /*width*/,
                               u32 /*height*/,
                               const Scalar* parameters,
                               MatrixBase<DerivedB>* normalized_image_coordinates) const {
    *normalized_image_coordinates = Project<convention>(camera_point, parameters);
    return true;
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType Project(const MatrixBase<Derived>& undistorted_point,
                            const Scalar* parameters) const {
    Scalar r = undistorted_point.norm();
    
    Scalar fisheye_x, fisheye_y;
    const Scalar kEpsilon = static_cast<Scalar>(1e-6);
    if (r > kEpsilon) {
      Scalar theta_by_r = std::atan(r) / r;
      fisheye_x = theta_by_r * undistorted_point.coeff(0);
      fisheye_y = theta_by_r * undistorted_point.coeff(1);
    } else {
      fisheye_x = undistorted_point.coeff(0);
      fisheye_y = undistorted_point.coeff(1);
    }
    
    return ProjectInnerPart<convention>(Matrix<Scalar, 2, 1>(fisheye_x, fisheye_y), parameters);
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType ProjectInnerPart(const MatrixBase<Derived>& undistorted_point,
                                     const Scalar* parameters) const {
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& k3 = parameters[2];
    const Scalar& k4 = parameters[3];
    const Scalar& p1 = parameters[4];
    const Scalar& p2 = parameters[5];
    const Scalar& sx1 = parameters[6];
    const Scalar& sy1 = parameters[7];
    
    const Scalar x2 = undistorted_point.coeff(0) * undistorted_point.coeff(0);
    const Scalar xy = undistorted_point.coeff(0) * undistorted_point.coeff(1);
    const Scalar y2 = undistorted_point.coeff(1) * undistorted_point.coeff(1);
    const Scalar r2 = x2 + y2;
    const Scalar r4 = r2 * r2;
    const Scalar r6 = r4 * r2;
    const Scalar r8 = r6 * r2;
    
    const Scalar radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    const Scalar dx = static_cast<Scalar>(2) * p1 * xy + p2 * (r2 + static_cast<Scalar>(2) * x2) + sx1 * r2;
    const Scalar dy = static_cast<Scalar>(2) * p2 * xy + p1 * (r2 + static_cast<Scalar>(2) * y2) + sy1 * r2;
    
    return OutputType(undistorted_point.coeff(0) + radial * undistorted_point.coeff(0) + dx,
                      undistorted_point.coeff(1) + radial * undistorted_point.coeff(1) + dy);
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType ProjectWithJacobian(const MatrixBase<Derived>& undistorted_point,
                                        const Scalar* parameters,
                                        Matrix<Scalar, 2, 2>* jacobian) const {
    const Scalar& nx = undistorted_point.coeff(0);
    const Scalar& ny = undistorted_point.coeff(1);
    const Scalar nx_ny = nx * ny;
    const Scalar nx2 = nx * nx;
    const Scalar ny2 = ny * ny;
    const Scalar nr2 = nx2 + ny2;
    const Scalar r = sqrtf(nr2);
    
    Scalar fisheye_x, fisheye_y;
    if (r > kEpsilon) {
      Scalar theta_by_r = std::atan(r) / r;
      fisheye_x = theta_by_r * undistorted_point.coeff(0);
      fisheye_y = theta_by_r * undistorted_point.coeff(1);
    } else {
      fisheye_x = undistorted_point.coeff(0);
      fisheye_y = undistorted_point.coeff(1);
    }
    
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& k3 = parameters[2];
    const Scalar& k4 = parameters[3];
    const Scalar& p1 = parameters[4];
    const Scalar& p2 = parameters[5];
    const Scalar& sx1 = parameters[6];
    const Scalar& sy1 = parameters[7];
    
    const Scalar x2 = fisheye_x * fisheye_x;
    const Scalar xy = fisheye_x * fisheye_y;
    const Scalar y2 = fisheye_y * fisheye_y;
    const Scalar fr2 = x2 + y2;
    const Scalar fr4 = fr2 * fr2;
    const Scalar fr6 = fr4 * fr2;
    const Scalar fr8 = fr6 * fr2;
    
    const Scalar radial = k1 * fr2 + k2 * fr4 + k3 * fr6 + k4 * fr8;
    const Scalar dx = static_cast<Scalar>(2) * p1 * xy + p2 * (fr2 + static_cast<Scalar>(2) * x2) + sx1 * fr2;
    const Scalar dy = static_cast<Scalar>(2) * p2 * xy + p1 * (fr2 + static_cast<Scalar>(2) * y2) + sy1 * fr2;
    
//     if (r > radius_cutoff_) {
//       return Eigen::Vector4f(0, 0, 0, 0);
//     }
    
    // TODO: Should de-duplicate more computations between value computation and Jacobian computation
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float r3 = nr2 * r;
      
      const float term1 = nr2 * (nr2 + 1);
      const float term2 = atan_r / r3;
      
      // Derivatives of fisheye x / y coordinates by nx / ny:
      const float dnxf_dnx = ny2 * term2 + nx2 / term1;
      const float dnxf_dny = nx_ny / term1 - nx_ny * term2;
      const float dnyf_dnx = dnxf_dny;
      const float dnyf_dny = nx2 * term2 + ny2 / term1;
      
      // Compute fisheye x / y.
      const float theta_by_r = atan2(r, 1.f) / r;
      const float x = theta_by_r * nx;
      const float y = theta_by_r * ny;
      
      // Derivatives of distorted coordinates by fisheye x / y:
      // (same computation as in non-fisheye polynomial-tangential)

      const float x_y = x * y;
      const float x2 = x * x;
      const float y2 = y * y;
      
      const float rf2 = x2 + y2;
      const float rf4 = rf2 * rf2;
      const float rf6 = rf4 * rf2;
      const float rf8 = rf6 * rf2;
      
      // NOTE: Could factor out more terms here which might improve performance.
      const float term1f = 2*p1*x + 2*p2*y + 2*k1*x_y + 6*k3*x_y*rf4 + 8*k4*x_y*rf6 + 4*k2*x_y*rf2;
      const float ddx_dnxf = 2*k1*x2 + 4*k2*x2*rf2 + 6*k3*x2*rf4 + 8*k4*x2*rf6 + k2*rf4 + k3*rf6 + k4*rf8 + 6*p2*x + 2*p1*y + 2*sx1*x + k1*rf2 + 1;
      const float ddx_dnyf = 2*sx1*y + term1f;
      const float ddy_dnxf = 2*sy1*x + term1f;
      const float ddy_dnyf = 2*k1*y2 + 4*k2*y2*rf2 + 6*k3*y2*rf4 + 8*k4*y2*rf6 + k2*rf4 + k3*rf6 + k4*rf8 + 2*p2*x + 6*p1*y + 2*sy1*y + k1*rf2 + 1;
      
      (*jacobian)(0, 0) = ddx_dnxf * dnxf_dnx + ddx_dnyf * dnyf_dnx;
      (*jacobian)(0, 1) = ddx_dnxf * dnxf_dny + ddx_dnyf * dnyf_dny;
      (*jacobian)(1, 0) = ddy_dnxf * dnxf_dnx + ddy_dnyf * dnyf_dnx;
      (*jacobian)(1, 1) = ddy_dnxf * dnxf_dny + ddy_dnyf * dnyf_dny;
    } else {
      // Non-fisheye variant is used in this case.
      const float r4 = nr2 * nr2;
      const float r6 = r4 * nr2;
      const float r8 = r6 * nr2;
      
      // NOTE: Could factor out more terms here which might improve performance.
      const float term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*nr2;
      const float ddx_dnx = 2*k1*nx2 + 4*k2*nx2*nr2 + 6*k3*nx2*r4 + 8*k4*nx2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*nr2 + 1;
      const float ddx_dny = 2*sx1*ny + term1;
      const float ddy_dnx = 2*sy1*nx + term1;
      const float ddy_dny = 2*k1*ny2 + 4*k2*ny2*nr2 + 6*k3*ny2*r4 + 8*k4*ny2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*nr2 + 1;
      
      (*jacobian)(0, 0) = ddx_dnx;
      (*jacobian)(0, 1) = ddx_dny;
      (*jacobian)(1, 0) = ddy_dnx;
      (*jacobian)(1, 1) = ddy_dny;
    }
    
    return OutputType(fisheye_x + radial * fisheye_x + dx,
                      fisheye_y + radial * fisheye_y + dy);
  }
  
  // TODO: De-duplicate between here and the function above.
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType ProjectInnerPartWithJacobian(const MatrixBase<Derived>& undistorted_point,
                                                 const Scalar* parameters,
                                                 Matrix<Scalar, 2, 2>* jacobian) const {
    const Scalar& k1 = parameters[0];
    const Scalar& k2 = parameters[1];
    const Scalar& k3 = parameters[2];
    const Scalar& k4 = parameters[3];
    const Scalar& p1 = parameters[4];
    const Scalar& p2 = parameters[5];
    const Scalar& sx1 = parameters[6];
    const Scalar& sy1 = parameters[7];
    
    const Scalar x2 = undistorted_point.coeff(0) * undistorted_point.coeff(0);
    const Scalar xy = undistorted_point.coeff(0) * undistorted_point.coeff(1);
    const Scalar y2 = undistorted_point.coeff(1) * undistorted_point.coeff(1);
    const Scalar r2 = x2 + y2;
    const Scalar r4 = r2 * r2;
    const Scalar r6 = r4 * r2;
    const Scalar r8 = r6 * r2;
    
    const Scalar radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    const Scalar dx = static_cast<Scalar>(2) * p1 * xy + p2 * (r2 + static_cast<Scalar>(2) * x2) + sx1 * r2;
    const Scalar dy = static_cast<Scalar>(2) * p2 * xy + p1 * (r2 + static_cast<Scalar>(2) * y2) + sy1 * r2;
    
    const Scalar nx = undistorted_point.coeff(0);
    const Scalar ny = undistorted_point.coeff(1);
    const Scalar nx_ny = nx * ny;
    
    // NOTE: Could factor out more terms here which might improve performance.
    const Scalar term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*r2;
    (*jacobian)(0, 0) = 2*k1*x2 + 4*k2*x2*r2 + 6*k3*x2*r4 + 8*k4*x2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*r2 + 1;
    (*jacobian)(0, 1)= 2*sx1*ny + term1;
    (*jacobian)(1, 0) = 2*sy1*nx + term1;
    (*jacobian)(1, 1) = 2*k1*y2 + 4*k2*y2*r2 + 6*k3*y2*r4 + 8*k4*y2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*r2 + 1;
    
    return OutputType(undistorted_point.coeff(0) + radial * undistorted_point.coeff(0) + dx,
                      undistorted_point.coeff(1) + radial * undistorted_point.coeff(1) + dy);
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline InputType Unproject(const MatrixBase<Derived>& distorted_point,
                             const Scalar* parameters) const {
    Scalar cur_x = distorted_point.coeff(0);
    Scalar cur_y = distorted_point.coeff(1);
    
    // Gauss-Newton optimization algorithm.
    const float kUndistortionEpsilon = 1e-10f;
    const usize kMaxIterations = 100;
    
    for (usize i = 0; i < kMaxIterations; ++i) {
      Matrix<Scalar, 2, 2> ddxy_dxy;
      Matrix<Scalar, 2, 1> distorted = ProjectInnerPartWithJacobian<convention>(Matrix<Scalar, 2, 1>(cur_x, cur_y), parameters, &ddxy_dxy);
      
      // (Non-squared) residuals.
      float dx = distorted.x() - distorted_point.x();
      float dy = distorted.y() - distorted_point.y();
      
      // Accumulate H and b.
      float H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0);
      float H_1_0_and_0_1 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1);
      float H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1);
      float b_0 = dx * ddxy_dxy(0, 0) + dy * ddxy_dxy(1, 0);
      float b_1 = dx * ddxy_dxy(0, 1) + dy * ddxy_dxy(1, 1);
      
      // Solve the system and update the parameters.
      float x_1 = (b_1 - H_1_0_and_0_1 / H_0_0 * b_0) /
                  (H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0);
      float x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0;
      cur_x -= x_0;
      cur_y -= x_1;
      
      if (dx * dx + dy * dy < kUndistortionEpsilon) {
        break;
      }
    }
    
    const float theta = sqrtf(cur_x * cur_x + cur_y * cur_y);
    const float theta_cos_theta = theta * cosf(theta);
    if (theta_cos_theta > kEpsilon) {
      const float scale = sinf(theta) / theta_cos_theta;
      cur_x *= scale;
      cur_y *= scale;
    }
    
    return InputType(cur_x, cur_y);
  }
  
  inline void ScaleParameters(Scalar /*factor*/, Scalar* /*parameters*/) const {}
  
  inline void CropParameters(int /*left*/, int /*top*/, int /*right*/, int /*bottom*/, Scalar* /*parameters*/) const {}
  
  static inline int GetParameterCount(const Scalar* /*parameters*/) {
    return 8;
  }
  
  inline void CacheDerivedParameters(u32 /*width*/, u32 /*height*/, const Scalar* /*parameters*/) {}
  
  static constexpr Scalar kEpsilon = static_cast<Scalar>(1e-6);
};


// Algorithm step which can be used in camera implementations:
// Pixel mapping with the 4 parameters fx, fy, cx, cy. The parameters have to
// be given in the PixelCorner convention.
template<typename Scalar>
class PixelMapping4 {
 public:
  // Normalized image coordinates (after distortion).
  typedef Matrix<Scalar, 2, 1> InputType;
  // Pixel coordinates.
  typedef Matrix<Scalar, 2, 1> OutputType;
  
  template<ImageCoordinateConvention convention, typename DerivedA, typename DerivedB>
  inline bool ProjectIfVisible(const MatrixBase<DerivedA>& normalized_image_coordinates,
                               float pixel_border,
                               u32 width,
                               u32 height,
                               const Scalar* parameters,
                               MatrixBase<DerivedB>* pixel_coordinates) const {
    if (convention == ImageCoordinateConvention::kPixelCorner) {
      *pixel_coordinates = Project<convention>(normalized_image_coordinates, parameters);
      return pixel_coordinates->coeff(0) >= pixel_border &&
             pixel_coordinates->coeff(1) >= pixel_border &&
             pixel_coordinates->coeff(0) < width - pixel_border &&
             pixel_coordinates->coeff(1) < height - pixel_border;
    } else if (convention == ImageCoordinateConvention::kPixelCenter) {
      *pixel_coordinates = Project<convention>(normalized_image_coordinates, parameters);
      return pixel_coordinates->coeff(0) >= -0.5f + pixel_border &&
             pixel_coordinates->coeff(1) >= -0.5f + pixel_border &&
             pixel_coordinates->coeff(0) < width - 0.5f - pixel_border &&
             pixel_coordinates->coeff(1) < height - 0.5f - pixel_border;
    } else if (convention == ImageCoordinateConvention::kRatio) {
      *pixel_coordinates = Project<convention>(normalized_image_coordinates, parameters);
      return pixel_coordinates->coeff(0) >= 0 &&
             pixel_coordinates->coeff(1) >= 0 &&
             pixel_coordinates->coeff(0) < 1 &&
             pixel_coordinates->coeff(1) < 1;
    }
    LOG(FATAL) << "convention not supported";
    return false;
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType Project(const MatrixBase<Derived>& normalized_image_coordinates,
                            const Scalar* parameters) const {
    if (convention == ImageCoordinateConvention::kPixelCorner) {
      const Scalar& fx = parameters[0];
      const Scalar& fy = parameters[1];
      const Scalar& cx = parameters[2];
      const Scalar& cy = parameters[3];
      return OutputType(fx * normalized_image_coordinates.coeff(0) + cx,
                        fy * normalized_image_coordinates.coeff(1) + cy);
    } else if (convention == ImageCoordinateConvention::kPixelCenter) {
      const Scalar& fx = parameters[0];
      const Scalar& fy = parameters[1];
      return OutputType(fx * normalized_image_coordinates.coeff(0) + cx_pixel_center_,
                        fy * normalized_image_coordinates.coeff(1) + cy_pixel_center_);
    } else if (convention == ImageCoordinateConvention::kRatio) {
      return OutputType(fx_ratio_ * normalized_image_coordinates.coeff(0) + cx_ratio_,
                        fy_ratio_ * normalized_image_coordinates.coeff(1) + cy_ratio_);
    }
    LOG(FATAL) << "convention not supported";
    return OutputType();
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline InputType Unproject(const MatrixBase<Derived>& projected_point, const Scalar* /*parameters*/) const {
    if (convention == ImageCoordinateConvention::kPixelCorner) {
      return InputType(fx_inv_ * projected_point.coeff(0) + cx_inv_,
                       fy_inv_ * projected_point.coeff(1) + cy_inv_);
    } else if (convention == ImageCoordinateConvention::kPixelCenter) {
      return InputType(fx_inv_ * projected_point.coeff(0) + cx_inv_pixel_center_,
                       fy_inv_ * projected_point.coeff(1) + cy_inv_pixel_center_);
    } else if (convention == ImageCoordinateConvention::kRatio) {
      return InputType(fx_inv_ratio_ * projected_point.coeff(0) + cx_inv_ratio_,
                       fy_inv_ratio_ * projected_point.coeff(1) + cy_inv_ratio_);
    }
    LOG(FATAL) << "convention not supported";
    return InputType();
  }
  
  inline void ScaleParameters(Scalar factor, Scalar* parameters) const {
    Scalar* fx = &parameters[0];
    Scalar* fy = &parameters[1];
    Scalar* cx = &parameters[2];
    Scalar* cy = &parameters[3];
    *fx *= factor;
    *fy *= factor;
    // Convention: Origin at image corner.
    *cx *= factor;
    *cy *= factor;
  }
  
  inline void CropParameters(int left, int top, int /*right*/, int /*bottom*/, Scalar* parameters) const {
    Scalar* cx = &parameters[2];
    Scalar* cy = &parameters[3];
    *cx -= left;
    *cy -= top;
  }
  
  static inline int GetParameterCount(const Scalar* /*parameters*/) {
    return 4;
  }
  
  inline void CacheDerivedParameters(u32 width, u32 height,
                                     const Scalar* parameters) {
    const Scalar& fx = parameters[0];
    const Scalar& fy = parameters[1];
    const Scalar& cx = parameters[2];
    const Scalar& cy = parameters[3];
    
    // Cache derived parameters for PixelCorner unprojection.
    fx_inv_ = static_cast<Scalar>(1.0) / fx;
    fy_inv_ = static_cast<Scalar>(1.0) / fy;
    cx_inv_ = -cx / fx;
    cy_inv_ = -cy / fy;
    
    // Cache derived parameters for PixelCenter projection and unprojection.
    cx_pixel_center_ = cx - static_cast<Scalar>(0.5);
    cy_pixel_center_ = cy - static_cast<Scalar>(0.5);
    cx_inv_pixel_center_ = -cx_pixel_center_ / fx;
    cy_inv_pixel_center_ = -cy_pixel_center_ / fy;
    
    // Cache derived parameters for Ratio projection and unprojection.
    fx_ratio_ = fx / width;
    fy_ratio_ = fy / height;
    cx_ratio_ = cx / width;
    cy_ratio_ = cy / height;
    fx_inv_ratio_ = static_cast<Scalar>(1.0) / fx_ratio_;
    fy_inv_ratio_ = static_cast<Scalar>(1.0) / fy_ratio_;
    cx_inv_ratio_ = -cx_ratio_ / fx_ratio_;
    cy_inv_ratio_ = -cy_ratio_ / fy_ratio_;
  }
  
 private:
  // Cached derived parameters for PixelCorner unprojection.
  Scalar fx_inv_;
  Scalar fy_inv_;
  Scalar cx_inv_;
  Scalar cy_inv_;
  
  // Cached derived parameters for PixelCenter projection and unprojection.
  Scalar cx_pixel_center_;
  Scalar cy_pixel_center_;
  Scalar cx_inv_pixel_center_;
  Scalar cy_inv_pixel_center_;
  
  // Cached derived parameters for Ratio projection and unprojection.
  Scalar fx_ratio_;
  Scalar fy_ratio_;
  Scalar cx_ratio_;
  Scalar cy_ratio_;
  Scalar fx_inv_ratio_;
  Scalar fy_inv_ratio_;
  Scalar cx_inv_ratio_;
  Scalar cy_inv_ratio_;
};


// Algorithm step which can be used in camera implementations:
// Non-parametric mapping from normalized image coordinates to pixels, using bicubic interpolation.
// The parameters for this camera model step are laid out as follows:
// 
// resolution_x resolution_y min_nx min_ny max_nx max_ny data_points[resolution_x * resolution_y; row-major]
// 
// Each data point consists of an x and y pixel coordinate which the corresponding
// normalized image coordinate is mapped to, in "pixel corner" origin convention.
template<typename Scalar>
class NonParametricBicubicProjection {
 public:
  // Normalized image coordinates (before distortion).
  typedef Matrix<Scalar, 2, 1> InputType;
  // Pixel coordinates.
  typedef Matrix<Scalar, 2, 1> OutputType;
  
  template<ImageCoordinateConvention convention, typename DerivedA, typename DerivedB>
  inline bool ProjectIfVisible(const MatrixBase<DerivedA>& normalized_image_coordinates,
                               float pixel_border,
                               u32 width,
                               u32 height,
                               const Scalar* parameters,
                               MatrixBase<DerivedB>* pixel_coordinates) const {
    if (convention == ImageCoordinateConvention::kPixelCorner) {
      *pixel_coordinates = Project<convention>(normalized_image_coordinates, parameters);
      return pixel_coordinates->coeff(0) >= pixel_border &&
             pixel_coordinates->coeff(1) >= pixel_border &&
             pixel_coordinates->coeff(0) < width - pixel_border &&
             pixel_coordinates->coeff(1) < height - pixel_border;
    } else if (convention == ImageCoordinateConvention::kPixelCenter) {
      *pixel_coordinates = Project<convention>(normalized_image_coordinates, parameters);
      return pixel_coordinates->coeff(0) >= -0.5f + pixel_border &&
             pixel_coordinates->coeff(1) >= -0.5f + pixel_border &&
             pixel_coordinates->coeff(0) < width - 0.5f - pixel_border &&
             pixel_coordinates->coeff(1) < height - 0.5f - pixel_border;
    } else if (convention == ImageCoordinateConvention::kRatio) {
      *pixel_coordinates = Project<convention>(normalized_image_coordinates, parameters);
      return pixel_coordinates->coeff(0) >= 0 &&
             pixel_coordinates->coeff(1) >= 0 &&
             pixel_coordinates->coeff(0) < 1 &&
             pixel_coordinates->coeff(1) < 1;
    }
    LOG(FATAL) << "convention not supported";
    return false;
  }
  
  inline OutputType CubicHermiteSpline(
      const OutputType& p0,
      const OutputType& p1,
      const OutputType& p2,
      const OutputType& p3,
      const Scalar x) const {
    const OutputType a = static_cast<Scalar>(0.5) * (-p0 + static_cast<Scalar>(3.0) * p1 - static_cast<Scalar>(3.0) * p2 + p3);
    const OutputType b = static_cast<Scalar>(0.5) * (static_cast<Scalar>(2.0) * p0 - static_cast<Scalar>(5.0) * p1 + static_cast<Scalar>(4.0) * p2 - p3);
    const OutputType c = static_cast<Scalar>(0.5) * (-p0 + p2);
    const OutputType d = p1;
    
    // Use Horner's rule to evaluate the function value and its
    // derivative.
    
    // f = ax^3 + bx^2 + cx + d
    return d + x * (c + x * (b + x * a));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType Project(const MatrixBase<Derived>& normalized_image_coordinates,
                            const Scalar* parameters) const {
    const Scalar& min_nx = parameters[2];
    const Scalar& min_ny = parameters[3];
    const Scalar& max_nx = parameters[4];
    const Scalar& max_ny = parameters[5];
    
    Scalar fc = (normalized_image_coordinates.coeff(0) - min_nx) * ((resolution_x_ - 1) / (max_nx - min_nx));
    Scalar fr = (normalized_image_coordinates.coeff(1) - min_ny) * ((resolution_y_ - 1) / (max_ny - min_ny));
    const int row = std::floor(fr);
    const int col = std::floor(fc);
    Scalar r_frac = fr - row;
    Scalar c_frac = fc - col;
    
    int c[4];
    int r[4];
    for (int i = 0; i < 4; ++ i) {
      c[i] = std::min(std::max(0, col - 1 + i), resolution_x_ - 1);
      r[i] = std::min(std::max(0, row - 1 + i), resolution_y_ - 1);
    }
    
    OutputType f[4];
    for (int wrow = 0; wrow < 4; ++ wrow) {
      int base_idx = 6 + 2 * (c[0] + r[wrow] * resolution_x_);
      OutputType p0(parameters[base_idx + 0], parameters[base_idx + 1]);
      base_idx = 6 + 2 * (c[1] + r[wrow] * resolution_x_);
      OutputType p1(parameters[base_idx + 0], parameters[base_idx + 1]);
      base_idx = 6 + 2 * (c[2] + r[wrow] * resolution_x_);
      OutputType p2(parameters[base_idx + 0], parameters[base_idx + 1]);
      base_idx = 6 + 2 * (c[3] + r[wrow] * resolution_x_);
      OutputType p3(parameters[base_idx + 0], parameters[base_idx + 1]);
      
      f[wrow] = CubicHermiteSpline(p0, p1, p2, p3, c_frac);
    }
    
    OutputType result_pixel_corner = CubicHermiteSpline(f[0], f[1], f[2], f[3], r_frac);
    
    if (convention == ImageCoordinateConvention::kPixelCorner) {
      return result_pixel_corner;
    } else if (convention == ImageCoordinateConvention::kPixelCenter) {
      return result_pixel_corner - OutputType::Constant(0.5);
    } else if (convention == ImageCoordinateConvention::kRatio) {
      return OutputType(result_pixel_corner.x() * inv_width_, result_pixel_corner.y() * inv_height_);
    }
    LOG(FATAL) << "convention not supported";
    return OutputType();
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline OutputType ProjectWithJacobian(const MatrixBase<Derived>& normalized_image_coordinates,
                                        const Scalar* parameters,
                                        Matrix<Scalar, 2, 2>* jacobian) const {
    const Scalar& min_nx = parameters[2];
    const Scalar& min_ny = parameters[3];
    const Scalar& max_nx = parameters[4];
    const Scalar& max_ny = parameters[5];
    
    Scalar fc = (normalized_image_coordinates.coeff(0) - min_nx) * ((resolution_x_ - 1) / (max_nx - min_nx));
    Scalar fr = (normalized_image_coordinates.coeff(1) - min_ny) * ((resolution_y_ - 1) / (max_ny - min_ny));
    const int row = std::floor(fr);
    const int col = std::floor(fc);
    
    int c[4];
    int r[4];
    for (int i = 0; i < 4; ++ i) {
      c[i] = std::min(std::max(0, col - 1 + i), resolution_x_ - 1);
      r[i] = std::min(std::max(0, row - 1 + i), resolution_y_ - 1);
    }
    
    OutputType p[4][4];
    for (int y = 0; y < 4; ++ y) {
      for (int x = 0; x < 4; ++ x) {
        int base_idx = 6 + 2 * (c[x] + r[y] * resolution_x_);
        p[y][x] = OutputType(parameters[base_idx + 0], parameters[base_idx + 1]);
      }
    }
    
    Scalar nx = normalized_image_coordinates.coeff(0);
    Scalar ny = normalized_image_coordinates.coeff(1);
    
    // Auto-generated part.
    const Scalar term0 = -min_nx;
    const Scalar term1 = 1.0/(max_nx + term0);
    const Scalar term2 = (resolution_x_ - 1)*term1;
    const Scalar term3_nonfrac = term2*(nx + term0);
    const Scalar term3 = term3_nonfrac - std::floor(term3_nonfrac);
    const Scalar term4 = -0.5*p[1][0].x();
    const Scalar term5 = -2.5*p[1][1].x();
    const Scalar term6 = 0.5*p[1][3].x();
    const Scalar term7 = 1.0*p[1][0].x() + 2.0*p[1][2].x() + term5 - term6;
    const Scalar term8 = 1.5*p[1][1].x();
    const Scalar term9 = term3*(-1.5*p[1][2].x() + term4 + term6 + term8);
    const Scalar term10 = 0.5*p[1][2].x() + term3*(term7 + term9) + term4;
    const Scalar term11 = term10*term3;
    const Scalar term12 = -min_ny;
    const Scalar term13 = (resolution_y_ - 1)/(max_ny + term12);
    const Scalar term14_nonfrac = term13*(ny + term12);
    const Scalar term14 = term14_nonfrac - std::floor(term14_nonfrac);
    const Scalar term15 = -0.5*p[0][0].x();
    const Scalar term16 = 0.5*p[0][3].x();
    const Scalar term17 = 1.0*p[0][0].x() - 2.5*p[0][1].x() + 2.0*p[0][2].x() - term16;
    const Scalar term18 = term3*(1.5*p[0][1].x() - 1.5*p[0][2].x() + term15 + term16);
    const Scalar term19 = 0.5*p[0][2].x() + term15 + term3*(term17 + term18);
    const Scalar term20 = term19*term3;
    const Scalar term21 = -0.5*p[0][1].x() - 0.5*term20;
    const Scalar term22 = -0.5*p[2][0].x();
    const Scalar term23 = 0.5*p[2][3].x();
    const Scalar term24 = 1.0*p[2][0].x() - 2.5*p[2][1].x() + 2.0*p[2][2].x() - term23;
    const Scalar term25 = 1.5*p[2][1].x();
    const Scalar term26 = term3*(-1.5*p[2][2].x() + term22 + term23 + term25);
    const Scalar term27 = 0.5*p[2][2].x() + term22 + term3*(term24 + term26);
    const Scalar term28 = term27*term3;
    const Scalar term29 = 0.5*p[3][1].x();
    const Scalar term30 = 0.5*term3;
    const Scalar term31 = -0.5*p[3][0].x();
    const Scalar term32 = 0.5*p[3][3].x();
    const Scalar term33 = 1.0*p[3][0].x() - 2.5*p[3][1].x() + 2.0*p[3][2].x() - term32;
    const Scalar term34 = term3*(1.5*p[3][1].x() - 1.5*p[3][2].x() + term31 + term32);
    const Scalar term35 = 0.5*p[3][2].x() + term3*(term33 + term34) + term31;
    const Scalar term36 = term30*term35;
    const Scalar term37 = term14*(1.5*term11 + term21 - term25 - 1.5*term28 + term29 + term36 + term8);
    const Scalar term38 = 1.0*p[0][1].x() + 2.0*p[2][1].x() - 2.5*term11 + 1.0*term20 + 2.0*term28 - term29 - term36 + term37 + term5;
    const Scalar term39 = 0.5*p[2][1].x() + term14*term38 + term21 + 0.5*term28;
    const Scalar term40 = -0.5*p[1][0].y();
    const Scalar term41 = -2.5*p[1][1].y();
    const Scalar term42 = 0.5*p[1][3].y();
    const Scalar term43 = 1.0*p[1][0].y() + 2.0*p[1][2].y() + term41 - term42;
    const Scalar term44 = 1.5*p[1][1].y();
    const Scalar term45 = term3*(-1.5*p[1][2].y() + term40 + term42 + term44);
    const Scalar term46 = 0.5*p[1][2].y() + term3*(term43 + term45) + term40;
    const Scalar term47 = term3*term46;
    const Scalar term48 = -0.5*p[0][0].y();
    const Scalar term49 = 0.5*p[0][3].y();
    const Scalar term50 = 1.0*p[0][0].y() - 2.5*p[0][1].y() + 2.0*p[0][2].y() - term49;
    const Scalar term51 = term3*(1.5*p[0][1].y() - 1.5*p[0][2].y() + term48 + term49);
    const Scalar term52 = 0.5*p[0][2].y() + term3*(term50 + term51) + term48;
    const Scalar term53 = term3*term52;
    const Scalar term54 = -0.5*p[0][1].y() - 0.5*term53;
    const Scalar term55 = -0.5*p[2][0].y();
    const Scalar term56 = 0.5*p[2][3].y();
    const Scalar term57 = 1.0*p[2][0].y() - 2.5*p[2][1].y() + 2.0*p[2][2].y() - term56;
    const Scalar term58 = 1.5*p[2][1].y();
    const Scalar term59 = term3*(-1.5*p[2][2].y() + term55 + term56 + term58);
    const Scalar term60 = 0.5*p[2][2].y() + term3*(term57 + term59) + term55;
    const Scalar term61 = term3*term60;
    const Scalar term62 = 0.5*p[3][1].y();
    const Scalar term63 = -0.5*p[3][0].y();
    const Scalar term64 = 0.5*p[3][3].y();
    const Scalar term65 = 1.0*p[3][0].y() - 2.5*p[3][1].y() + 2.0*p[3][2].y() - term64;
    const Scalar term66 = term3*(1.5*p[3][1].y() - 1.5*p[3][2].y() + term63 + term64);
    const Scalar term67 = 0.5*p[3][2].y() + term3*(term65 + term66) + term63;
    const Scalar term68 = term30*term67;
    const Scalar term69 = term14*(term44 + 1.5*term47 + term54 - term58 - 1.5*term61 + term62 + term68);
    const Scalar term70 = 1.0*p[0][1].y() + 2.0*p[2][1].y() + term41 - 2.5*term47 + 1.0*term53 + 2.0*term61 - term62 - term68 + term69;
    const Scalar term71 = 0.5*p[2][1].y() + term14*term70 + term54 + 0.5*term61;
    const Scalar term72 = term2*term3;
    const Scalar term73 = term72*(term7 + 2*term9);
    const Scalar term74 = term10*term2;
    const Scalar term75 = term72*(term17 + 2*term18);
    const Scalar term76 = term2*term19;
    const Scalar term77 = -0.5*term75 - 0.5*term76;
    const Scalar term78 = term72*(term24 + 2*term26);
    const Scalar term79 = term2*term27;
    const Scalar term80 = 0.5*term72;
    const Scalar term81 = term80*(term33 + 2*term34);
    const Scalar term82 = 0.5*term2;
    const Scalar term83 = term35*term82;
    const Scalar term84 = term72*(term43 + 2*term45);
    const Scalar term85 = term2*term46;
    const Scalar term86 = term72*(term50 + 2*term51);
    const Scalar term87 = term2*term52;
    const Scalar term88 = -0.5*term86 - 0.5*term87;
    const Scalar term89 = term72*(term57 + 2*term59);
    const Scalar term90 = term2*term60;
    const Scalar term91 = term80*(term65 + 2*term66);
    const Scalar term92 = term67*term82;
    
    (*jacobian)(0, 0) = term14*(term14*(term14*(1.5*term73 + 1.5*term74 + term77 - 1.5*term78 - 1.5*term79 + term81 + term83) - 2.5*term73 - 2.5*term74 + 1.0*term75 + 1.0*term76 + 2.0*term78 + 2.0*term79 - term81 - term83) + term77 + 0.5*term78 + 0.5*term79) + term73 + term74;
    (*jacobian)(0, 1) = term13*term39 + term14*(term13*term37 + term13*term38);
    (*jacobian)(1, 0) = term14*(term14*(term14*(1.5*term84 + 1.5*term85 + term88 - 1.5*term89 - 1.5*term90 + term91 + term92) - 2.5*term84 - 2.5*term85 + 1.0*term86 + 1.0*term87 + 2.0*term89 + 2.0*term90 - term91 - term92) + term88 + 0.5*term89 + 0.5*term90) + term84 + term85;
    (*jacobian)(1, 1) = term13*term71 + term14*(term13*term69 + term13*term70);
    
    OutputType result_pixel_corner(
        p[1][1].x() + term11 + term14*term39,
        p[1][1].y() + term14*term71 + term47);
    
    if (convention == ImageCoordinateConvention::kPixelCorner) {
      return result_pixel_corner;
    } else if (convention == ImageCoordinateConvention::kPixelCenter) {
      return result_pixel_corner - OutputType::Constant(0.5);
    } else if (convention == ImageCoordinateConvention::kRatio) {
      // Adjust Jacobian
      (*jacobian)(0, 0) *= inv_width_;
      (*jacobian)(0, 1) *= inv_width_;
      (*jacobian)(1, 0) *= inv_height_;
      (*jacobian)(1, 1) *= inv_height_;
      return OutputType(result_pixel_corner.x() * inv_width_, result_pixel_corner.y() * inv_height_);
    }
    LOG(FATAL) << "convention not supported";
    return OutputType();
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline InputType Unproject(const MatrixBase<Derived>& projected_point, const Scalar* parameters) const {
    // TODO: (0, 0) as an initial estimate here can be very bad and might not converge.
    //       One solution would be to cache a mapping from image coordinates to
    //       approximate normalized image coordinates which is used as initialization.
    Scalar cur_x = 0;
    Scalar cur_y = 0;
    
    // Gauss-Newton optimization algorithm.
    const float kUndistortionEpsilon = 1e-10f;
    const usize kMaxIterations = 100;
    
    for (usize i = 0; i < kMaxIterations; ++i) {
      Matrix<Scalar, 2, 2> ddxy_dxy;
      Matrix<Scalar, 2, 1> distorted = ProjectWithJacobian<convention>(Matrix<Scalar, 2, 1>(cur_x, cur_y), parameters, &ddxy_dxy);
      
      // (Non-squared) residuals.
      float dx = distorted.x() - projected_point.x();
      float dy = distorted.y() - projected_point.y();
      
      // Accumulate H and b.
      float H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0);
      float H_1_0_and_0_1 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1);
      float H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1);
      float b_0 = dx * ddxy_dxy(0, 0) + dy * ddxy_dxy(1, 0);
      float b_1 = dx * ddxy_dxy(0, 1) + dy * ddxy_dxy(1, 1);
      
      // Solve the system and update the parameters.
      float x_1 = (b_1 - H_1_0_and_0_1 / H_0_0 * b_0) /
                  (H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0);
      float x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0;
      cur_x -= x_0;
      cur_y -= x_1;
      
      if (dx * dx + dy * dy < kUndistortionEpsilon) {
        break;
      }
    }
    
    return InputType(cur_x, cur_y);
  }
  
  inline void ScaleParameters(Scalar factor, Scalar* parameters) const {
    int point_count = resolution_x_ * resolution_y_;
    for (int i = 0; i < point_count; ++ i) {
      // Convention: Origin at image corner.
      parameters[6 + 2 * i + 0] *= factor;
      parameters[6 + 2 * i + 1] *= factor;
    }
  }
  
  inline void CropParameters(int left, int top, int /*right*/, int /*bottom*/, Scalar* parameters) const {
    int point_count = resolution_x_ * resolution_y_;
    for (int i = 0; i < point_count; ++ i) {
      parameters[6 + 2 * i + 0] -= left;
      parameters[6 + 2 * i + 1] -= top;
    }
  }
  
  static inline int GetParameterCount(const Scalar* parameters) {
    int resolution_x = static_cast<int>(parameters[0]);
    int resolution_y = static_cast<int>(parameters[1]);
    return 6 + 2 * resolution_x * resolution_y;
  }
  
  inline void CacheDerivedParameters(u32 width, u32 height,
                                     const Scalar* parameters) {
    resolution_x_ = static_cast<int>(parameters[0]);
    resolution_y_ = static_cast<int>(parameters[1]);
    
    inv_width_ = static_cast<Scalar>(1) / width;
    inv_height_ = static_cast<Scalar>(1) / height;
  }
  
 private:
  // Cached derived parameters:
  int resolution_x_;
  int resolution_y_;
  
  Scalar inv_width_;
  Scalar inv_height_;
};


// Last CameraImplVariadic type in recursion.
template <typename Scalar, class... Steps>
class CameraImplVariadic : public Camera {
 protected:
  CameraImplVariadic(int type_int, u32 width, u32 height)
      : Camera(type_int, width, height) {}
  
  int GetParameterCount(const Scalar* /*parameters*/) const {
    return 0;
  }
  
  void CacheDerivedParameters(const Scalar* /*parameters*/) {}
  
  template<ImageCoordinateConvention convention>
  inline bool ProjectIfVisibleImpl(
      const Matrix<Scalar, 2, 1>& input,
      float /*pixel_border*/,
      const Scalar* /*parameters*/,
      Matrix<Scalar, 2, 1>* pixel_coordinates) const {
    *pixel_coordinates = input;
    return true;
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline Matrix<Scalar, 2, 1> ProjectImpl(const MatrixBase<Derived>& input, const Scalar* /*parameters*/) const {
    return input.template cast<Scalar>();
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline Matrix<Scalar, 2, 1> UnprojectImpl(const MatrixBase<Derived>& pixel_coordinates, const Scalar* /*parameters*/) const {
    return pixel_coordinates.template cast<Scalar>();
  }
  
  inline void ScaleParameters(Scalar /*factor*/, Scalar* /*parameters*/) const {}
  
  inline void CropParameters(int /*left*/, int /*top*/, int /*right*/, int /*bottom*/, Scalar* /*parameters*/) const {}
};

// Specialization of CameraImplVariadic for recursion, peeling off the next Step
// from the Steps list.
template <typename Scalar, class Step, class... Steps>
class CameraImplVariadic<Scalar, Step, Steps...>
    : public CameraImplVariadic<Scalar, Steps...> {
 protected:
  typedef CameraImplVariadic<Scalar, Steps...> Base;
  
  CameraImplVariadic(int type_int, u32 width, u32 height)
      : Base(type_int, width, height) {}
  
  int GetParameterCount(const Scalar* parameters) const {
    int c = step_.GetParameterCount(parameters);
    c += Base::GetParameterCount(parameters + c);
    return c;
  }
  
  void CacheDerivedParameters(const Scalar* parameters) {
    step_.CacheDerivedParameters(Base::width(), Base::height(), parameters);
    Base::CacheDerivedParameters(parameters + Step::GetParameterCount(parameters));
  }
  
  template<ImageCoordinateConvention convention>
  inline bool ProjectIfVisibleImpl(
      const typename Step::InputType& input,
      float pixel_border,
      const Scalar* parameters,
      Matrix<Scalar, 2, 1>* pixel_coordinates) const {
    typename Step::OutputType step_output;
    bool visible = step_.template ProjectIfVisible<convention>(input, pixel_border, Camera::width(), Camera::height(), parameters, &step_output);
    if (!visible) {
      return false;
    }
    return Base::template ProjectIfVisibleImpl<convention>(step_output, pixel_border, parameters + Step::GetParameterCount(parameters), pixel_coordinates);
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline typename Step::OutputType ProjectImpl(const MatrixBase<Derived>& input, const Scalar* parameters) const {
    return Base::template ProjectImpl<convention>(step_.template Project<convention>(input, parameters), parameters + Step::GetParameterCount(parameters));
  }
  
  template<ImageCoordinateConvention convention, typename Derived>
  inline typename Step::InputType UnprojectImpl(const MatrixBase<Derived>& pixel_coordinates, const Scalar* parameters) const {
    return step_.template Unproject<convention>(Base::template UnprojectImpl<convention>(pixel_coordinates, parameters + Step::GetParameterCount(parameters)), parameters);
  }
  
  inline void ScaleParameters(Scalar factor, Scalar* parameters) const {
    step_.ScaleParameters(factor, parameters);
    Base::ScaleParameters(factor, parameters + Step::GetParameterCount(parameters));
  }
  
  inline void CropParameters(int left, int top, int right, int bottom, Scalar* parameters) const {
    step_.CropParameters(left, top, right, bottom, parameters);
    Base::CropParameters(left, top, right, bottom, parameters + Step::GetParameterCount(parameters));
  }
  
  Step step_;
};


// This type is used if the template parameter list is empty, forming the end of
// the recursion. It "returns" the value 0.
template<class... Steps>
struct GetCameraParameterCount
    : integral_constant<usize, 0> {};

// This type specialization peels of the next type from the list, adding its
// parameter count and recursing. The result can be obtained as:
// constexpr usize result = GetCameraParameterCount<...>::value.
template<class Step, class... Steps>
struct GetCameraParameterCount<Step, Steps...>
    : integral_constant<usize, Step::kParameterCount + GetCameraParameterCount<Steps...>::value > {};


// Bottommost type for camera implementations, starting the recursion and having
// access to all parts of the algorithm.
template<int TypeID, typename Scalar, class... Steps>
class CameraImpl : public CameraImplVariadic<Scalar, Steps...> {
 public:
  typedef CameraImplVariadic<Scalar, Steps...> Base;
  
  typedef Scalar ScalarT;
  
  // Creates an invalid camera.
  CameraImpl()
      : Base(static_cast<int>(Camera::Type::kInvalid), 0, 0) {}
  
  // Creates a valid camera.
  CameraImpl(u32 width, u32 height, const Scalar* parameters)
      : Base(TypeID, width, height) {
    parameters_.resize(this->GetParameterCount(parameters));
    memcpy(parameters_.data(), parameters, parameters_.size() * sizeof(Scalar));
    Base::CacheDerivedParameters(parameters_.data());
  }
  
  template <typename Derived>
  inline bool ProjectToPixelCornerConvIfVisible(
      const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<Scalar, 2, 1>* pixel_coordinates) const {
    return Base::template ProjectIfVisibleImpl<ImageCoordinateConvention::kPixelCorner>(camera_space_point, pixel_border, parameters_.data(), pixel_coordinates);
  }
  
  template <typename Derived>
  inline bool ProjectToPixelCenterConvIfVisible(
      const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<Scalar, 2, 1>* pixel_coordinates) const {
    return Base::template ProjectIfVisibleImpl<ImageCoordinateConvention::kPixelCenter>(camera_space_point, pixel_border, parameters_.data(), pixel_coordinates);
  }
  
  template <typename Derived>
  inline bool ProjectToRatioConvIfVisible(
      const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<Scalar, 2, 1>* pixel_coordinates) const {
    return Base::template ProjectIfVisibleImpl<ImageCoordinateConvention::kRatio>(camera_space_point, pixel_border, parameters_.data(), pixel_coordinates);
  }
  
  template <typename Derived>
  inline Matrix<Scalar, 2, 1> ProjectToPixelCornerConv(const MatrixBase<Derived>& camera_space_point) const {
    return Base::template ProjectImpl<ImageCoordinateConvention::kPixelCorner>(camera_space_point, parameters_.data());
  }
  
  template <typename Derived>
  inline Matrix<Scalar, 2, 1> ProjectToPixelCenterConv(const MatrixBase<Derived>& camera_space_point) const {
    return Base::template ProjectImpl<ImageCoordinateConvention::kPixelCenter>(camera_space_point, parameters_.data());
  }
  
  template <typename Derived>
  inline Matrix<Scalar, 2, 1> ProjectToRatioConv(const MatrixBase<Derived>& camera_space_point) const {
    return Base::template ProjectImpl<ImageCoordinateConvention::kRatio>(camera_space_point, parameters_.data());
  }
  
  template <typename Derived>
  inline Matrix<Scalar, 3, 1> UnprojectFromPixelCornerConv(const MatrixBase<Derived>& pixel_coordinates) const {
    return Base::template UnprojectImpl<ImageCoordinateConvention::kPixelCorner>(pixel_coordinates, parameters_.data());
  }
  
  template <typename Derived>
  inline Matrix<Scalar, 3, 1> UnprojectFromPixelCenterConv(const MatrixBase<Derived>& pixel_coordinates) const {
    return Base::template UnprojectImpl<ImageCoordinateConvention::kPixelCenter>(pixel_coordinates, parameters_.data());
  }
  
  template <typename Derived>
  inline Matrix<Scalar, 3, 1> UnprojectFromRatioConv(const MatrixBase<Derived>& pixel_coordinates) const {
    return Base::template UnprojectImpl<ImageCoordinateConvention::kRatio>(pixel_coordinates, parameters_.data());
  }
  
  inline virtual CameraImpl<TypeID, Scalar, Steps...>* Scaled(double factor) const override {
    vector<Scalar> scaled_parameters_vec(parameters_.size());
    Scalar* scaled_parameters = scaled_parameters_vec.data();
    memcpy(scaled_parameters, parameters_.data(), parameters_.size() * sizeof(Scalar));
    Base::ScaleParameters(factor, scaled_parameters);
    return new CameraImpl<TypeID, Scalar, Steps...>(
        factor * Camera::width() + static_cast<Scalar>(0.5),
        factor * Camera::height() + static_cast<Scalar>(0.5),
        scaled_parameters);
  }
  
  inline virtual CameraImpl<TypeID, Scalar, Steps...>* Cropped(int left, int top, int right, int bottom) const override {
    vector<Scalar> cropped_parameters_vec(parameters_.size());
    Scalar* cropped_parameters = cropped_parameters_vec.data();
    memcpy(cropped_parameters, parameters_.data(), parameters_.size() * sizeof(Scalar));
    Base::CropParameters(left, top, right, bottom, cropped_parameters);
    return new CameraImpl<TypeID, Scalar, Steps...>(
        Camera::width() - left - right,
        Camera::height() - top - bottom,
        cropped_parameters);
  }
  
  virtual void WriteAsText(std::ostream* stream) const override {
    *stream << Camera::type_int() << " "
            << Camera::width() << " "
            << Camera::height() << " "
            << Camera::parameter_count();
    for (usize i = 0; i < parameters_.size(); ++ i) {
      *stream << " " << parameters_[i];
    }
  }
  
  // Returns a pointer to the parameters.
  inline const Scalar* parameters() const { return parameters_.data(); }
  
  inline usize parameter_count() const { return parameters_.size(); }
  
 private:
  vector<Scalar> parameters_;
};


// Float pinhole camera class with 4 parameters:
// fx, fy, cx, cy.
typedef CameraImpl<static_cast<int>(Camera::Type::kPinholeCamera4f),
                   float,
                   PinholeProjection<float>,
                   PixelMapping4<float>> PinholeCamera4f;

// Double pinhole-radtan camera class with 8 parameters:
// k1, k2, r1, r2, fx, fy, cx, cy.
typedef CameraImpl<static_cast<int>(Camera::Type::kRadtanCamera8d),
                   double,
                   PinholeProjection<double>,
                   RadtanDistortion4<double>,
                   PixelMapping4<double>> RadtanCamera8d;

// Double pinhole-radtan camera class with 9 parameters:
// k1, k2, k3, r1, r2, fx, fy, cx, cy.
typedef CameraImpl<static_cast<int>(Camera::Type::kRadtanCamera9d),
                   double,
                   PinholeProjection<double>,
                   RadtanDistortion5<double>,
                   PixelMapping4<double>> RadtanCamera9d;

// Double thin prism fisheye camera class with 12 parameters:
// k1, k2, k3, k4, p1, p2, sx1, sy1, fx, fy, cx, cy.
typedef CameraImpl<static_cast<int>(Camera::Type::kThinPrismFisheyeCamera12d),
                   double,
                   PinholeProjection<double>,
                   ThinPrismFisheyeDistortion8<double>,
                   PixelMapping4<double>> ThinPrismFisheyeCamera12d;

// Double non-parametric camera with mapping from unprojected to projected coordinates. Parameters:
// resolution_x resolution_y min_nx min_ny max_nx max_ny data_points[resolution_x * resolution_y; row-major]
// Each data point consists of an x and y pixel coordinate which the corresponding
// normalized image coordinate is mapped to, in "pixel corner" origin convention.
typedef CameraImpl<static_cast<int>(Camera::Type::kNonParametricBicubicProjectionCamerad),
                   double,
                   PinholeProjection<double>,
                   NonParametricBicubicProjection<double>> NonParametricBicubicProjectionCamerad;


// Extracts the derived type of a Camera object and calls the templated
// functor with this template type and the given arguments.
// Const Camera version with functor.
template<template<typename CameraType> class Functor, class... Args>
void IdentifyCamera(const Camera& camera, Args... args) {
  if (camera.type_int() == static_cast<int>(Camera::Type::kPinholeCamera4f)) {
    Functor<PinholeCamera4f> functor;
    functor(*reinterpret_cast<const PinholeCamera4f*>(&camera), args...);
  } else {
    LOG(FATAL) << "Unsupported camera type: " << camera.type_int();
  }
}

// Mutable Camera version with functor.
template<template<typename CameraType> class Functor, class... Args>
void IdentifyCamera(Camera* camera, Args... args) {
  if (camera->type_int() == static_cast<int>(Camera::Type::kPinholeCamera4f)) {
    Functor<PinholeCamera4f> functor;
    functor(reinterpret_cast<PinholeCamera4f*>(camera), args...);
  } else {
    LOG(FATAL) << "Unsupported camera type: " << camera->type_int();
  }
}

// Since the function-based IdentifyCamera() variants require a functor which may
// be annoying, and it seems to be impossible to pass a template function
// (without a functor) as a template parameter, there is also a macro version of
// IdentifyCamera() (for const Camera object).
// The object type is available as _object_type within the call (with the actual
// camera variable name inserted for object), and the object with its derived
// type is available as _object. The call is to be inserted for the
// variable-length argument "...". This is because a single macro argument
// would cause trouble as soon as commas are within the call: They would be
// interpreted as argument separators for the macro.
#define IDENTIFY_CAMERA(object, ...)                                  \
  {                                                                          \
    if ((object).type() == Camera::Type::kPinholeCamera4f) {                 \
      typedef PinholeCamera4f _##object##_type;                              \
      const _##object##_type& _##object =                                    \
          static_cast<const _##object##_type&>(object);                      \
      (void)_##object;                                                       \
      __VA_ARGS__;                                                           \
    } else if ((object).type() == Camera::Type::kRadtanCamera8d) {           \
      typedef RadtanCamera8d _##object##_type;                               \
      const _##object##_type& _##object =                                    \
          static_cast<const _##object##_type&>(object);                      \
      (void)_##object;                                                       \
      __VA_ARGS__;                                                           \
    } else if ((object).type() == Camera::Type::kRadtanCamera9d) {           \
      typedef RadtanCamera9d _##object##_type;                               \
      const _##object##_type& _##object =                                    \
          static_cast<const _##object##_type&>(object);                      \
      (void)_##object;                                                       \
      __VA_ARGS__;                                                           \
    } else if ((object).type() == Camera::Type::kThinPrismFisheyeCamera12d) { \
      typedef ThinPrismFisheyeCamera12d _##object##_type;                    \
      const _##object##_type& _##object =                                    \
          static_cast<const _##object##_type&>(object);                      \
      (void)_##object;                                                       \
      __VA_ARGS__;                                                           \
    } else if ((object).type() == Camera::Type::kNonParametricBicubicProjectionCamerad) { \
      typedef NonParametricBicubicProjectionCamerad _##object##_type;        \
      const _##object##_type& _##object =                                    \
          static_cast<const _##object##_type&>(object);                      \
      (void)_##object;                                                       \
      __VA_ARGS__;                                                           \
    } else {                                                                 \
      LOG(FATAL) << "IDENTIFY_CAMERA() encountered an invalid type: " << static_cast<int>((object).type()); \
    }                                                                        \
  }

#define IDENTIFY_CAMERA2(objectA, objectB, ...)                       \
  {                                                                          \
    IDENTIFY_CAMERA(objectA, IDENTIFY_CAMERA(objectB, __VA_ARGS__)); \
  }

#define IDENTIFY_CAMERA_TYPE(type, ...)                                        \
  {                                                                          \
    if ((type) == Camera::Type::kPinholeCamera4f) {                          \
      typedef PinholeCamera4f _type;                                         \
      __VA_ARGS__;                                                           \
    } else if ((type) == Camera::Type::kRadtanCamera8d) {                    \
      typedef RadtanCamera8d _type;                                          \
      __VA_ARGS__;                                                           \
    } else if ((type) == Camera::Type::kRadtanCamera9d) {                    \
      typedef RadtanCamera9d _type;                                          \
      __VA_ARGS__;                                                           \
    } else if ((type) == Camera::Type::kThinPrismFisheyeCamera12d) {         \
      typedef ThinPrismFisheyeCamera12d _type;                               \
      __VA_ARGS__;                                                           \
    } else if ((type) == Camera::Type::kNonParametricBicubicProjectionCamerad) { \
      typedef NonParametricBicubicProjectionCamerad _type;                   \
      __VA_ARGS__;                                                           \
    } else {                                                                 \
      LOG(FATAL) << "IDENTIFY_CAMERA_TYPE() encountered an invalid type: " << static_cast<int>(type); \
    }                                                                        \
  }


// The implementations of the following template functions of the Camera
// class must be down here to be able to use IDENTIFY_CAMERA(), which
// needs to know about all derived classes.
template <typename Derived>
inline bool Camera::ProjectToPixelCornerConvIfVisible(
    const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return ProjectToPixelCornerConvIfVisibleHelper(_this_camera, camera_space_point, pixel_border, pixel_coordinates));
  return false;
}

template <typename Derived>
inline bool Camera::ProjectToPixelCenterConvIfVisible(
    const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return ProjectToPixelCenterConvIfVisibleHelper(_this_camera, camera_space_point, pixel_border, pixel_coordinates));
  return false;
}

template <typename Derived>
inline bool Camera::ProjectToRatioConvIfVisible(
    const MatrixBase<Derived>& camera_space_point, float pixel_border, Matrix<double, 2, 1>* pixel_coordinates) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return ProjectToRatioConvIfVisibleHelper(_this_camera, camera_space_point, pixel_border, pixel_coordinates));
  return false;
}

template <typename Derived>
inline Matrix<double, 2, 1> Camera::ProjectToPixelCornerConv(const MatrixBase<Derived>& camera_space_point) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.ProjectToPixelCornerConv(camera_space_point).template cast<double>());
  return Matrix<double, 2, 1>();
}

template <typename Derived>
inline Matrix<double, 2, 1> Camera::ProjectToPixelCenterConv(const MatrixBase<Derived>& camera_space_point) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.ProjectToPixelCenterConv(camera_space_point).template cast<double>());
  return Matrix<double, 2, 1>();
}

template <typename Derived>
inline Matrix<double, 2, 1> Camera::ProjectToRatioConv(const MatrixBase<Derived>& camera_space_point) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.ProjectToRatioConv(camera_space_point).template cast<double>());
  return Matrix<double, 2, 1>();
}

template <typename Derived>
inline Matrix<double, 3, 1> Camera::UnprojectFromPixelCornerConv(const MatrixBase<Derived>& pixel_coordinates) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.UnprojectFromPixelCornerConv(pixel_coordinates).template cast<double>());
  return Matrix<double, 3, 1>();
}

template <typename Derived>
inline Matrix<double, 3, 1> Camera::UnprojectFromPixelCenterConv(const MatrixBase<Derived>& pixel_coordinates) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.UnprojectFromPixelCenterConv(pixel_coordinates).template cast<double>());
  return Matrix<double, 3, 1>();
}

template <typename Derived>
inline Matrix<double, 3, 1> Camera::UnprojectFromRatioConv(const MatrixBase<Derived>& pixel_coordinates) const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.UnprojectFromRatioConv(pixel_coordinates).template cast<double>());
  return Matrix<double, 3, 1>();
}

inline u32 Camera::parameter_count() const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.parameter_count());
  return 0;
}

inline const void* Camera::parameters() const {
  const Camera& this_camera = *this;
  IDENTIFY_CAMERA(this_camera, return _this_camera.parameters());
  return nullptr;
}

inline Camera* Camera::ReadFromText(std::istream* stream) {
  int type_int;
  *stream >> type_int;
  
  if (type_int <= static_cast<int>(Camera::Type::kInvalid) ||
      type_int >= static_cast<int>(Camera::Type::kNumTypes)) {
    return nullptr;
  }
  
  IDENTIFY_CAMERA_TYPE(static_cast<Camera::Type>(type_int), return ReadFromTextHelper<_type>(stream));
  return nullptr;
}


template<typename CameraA, typename CameraB>
bool AreCamerasEqualHelper(const CameraA& camera_a, const CameraB& camera_b) {
  if (camera_a.type_int() != camera_b.type_int() ||
      camera_a.width() != camera_b.width() ||
      camera_a.height() != camera_b.height() ||
      camera_a.parameter_count() != camera_b.parameter_count()) {
    return false;
  }
  
  for (usize i = 0; i < camera_a.parameter_count(); ++ i) {
    if (camera_a.parameters()[i] != camera_b.parameters()[i]) {
      return false;
    }
  }
  
  return true;
};

inline bool AreCamerasEqual(const Camera& camera_a, const Camera& camera_b) {
  bool result = false;
  IDENTIFY_CAMERA2(camera_a, camera_b, result = AreCamerasEqualHelper(_camera_a, _camera_b));
  return result;
}

}

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

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>

namespace vis {

class CUDACameraModel;

typedef ParametrizedLine<double, 3> Line3d;

// Base class for camera models used in the camera calibrator.
class CameraModel {
 public:
  enum class Type {
    CentralGeneric = 0,
    NoncentralGeneric = 1,
    
    CentralRadial = 4,
    
    CentralThinPrismFisheye = 2,
    CentralOpenCV = 3,
    
    InvalidType = 5
  };
  
  
  /// Creates a new camera model.
  /// 
  /// @param width Width of the images recorded by this camera.
  /// @param height Height of the images recorded by this camera.
  /// @param calibration_min_x Leftmost calibrated pixel.
  /// @param calibration_min_y Topmost calibrated pixel.
  /// @param calibration_max_x Rightmost calibrated pixel.
  /// @param calibration_max_y Bottommost calibrated pixel.
  /// @param type Type of the derived class used with this base class.
  /// 
  /// Note that in the "pixel-corner" coordinate convention, the calibrated
  /// area is assumed to be in the range [min, max + 1[ for both dimensions.
  /// For example, if both min and max were 0, the range would go over the
  /// area of the zero-th pixel (excluding the right and bottom borders).
  inline CameraModel(
      int width, int height,
      int calibration_min_x, int calibration_min_y,
      int calibration_max_x, int calibration_max_y,
      Type type)
      : m_width(width),
        m_height(height),
        m_calibration_min_x(calibration_min_x),
        m_calibration_min_y(calibration_min_y),
        m_calibration_max_x(calibration_max_x),
        m_calibration_max_y(calibration_max_y),
        m_type(type) {}
  
  /// Destructor.
  virtual ~CameraModel() {}
  
  /// Creates a duplicate of this camera model (that must be deleted with delete).
  virtual CameraModel* duplicate() = 0;
  
  
  /// Projects a local 3D point to a pixel, in "pixel-corner" coordinate convention.
  virtual bool Project(
      const Vec3d& local_point,
      Vec2d* result) const = 0;
  
  /// Projects a local 3D point to a pixel (with the initial value of @p result
  /// used as initial estimate, which may speed up the process and / or
  /// ensure convergence of the optimization to the desired optimum).
  /// The pixel is in "pixel-corner" convention, meaning that the origin of the
  /// image pixel coordinate system is at the top-left corner of the top-left pixel.
  virtual bool ProjectWithInitialEstimate(
      const Vec3d& local_point,
      Vec2d* result) const = 0;
  
  // Un-projects a pixel to a direction. Must only be called for central camera
  // models.
  virtual inline bool Unproject(double /*x*/, double /*y*/, Vec3d* /*result*/) const {
    LOG(ERROR) << "This camera does not support un-projection to a direction.";
    return false;
  }
  
  // Un-projects a pixel to a line. Works with both central and non-central camera
  // models.
  virtual bool Unproject(double /*x*/, double /*y*/, Line3d* /*result*/) const = 0;
  
  
  /// Choose a "nice" orientation for the camera. Nice means that the camera's
  /// local +z direction approximately points forward (in the center of the
  /// image), and the camera's local +x direction approximately points right (in
  /// the camera image). This function returns the computed rotation, which
  /// should be left-multiplied with the camera_tr_rig transformation to keep
  /// the camera constant.
  /// This operation is purely for convention and convenience, so it is safe to
  /// leave this at the default implementation for derived classes.
  virtual Mat3d ChooseNiceCameraOrientation() { return Mat3d::Identity(); }
  
  /// Scales the camera model with the given factor. This is only relevant for
  /// non-central camera models and does nothing for central cameras.
  virtual inline void Scale(double /*factor*/) {}
  
  /// Creates a CUDA-compatible camera model instance representing the same
  /// camera as this object. Note: This function is not const such that the
  /// CPU-side class can store GPU memory that the GPU-side class references.
  virtual CUDACameraModel* CreateCUDACameraModel() { return nullptr; }
  
  /// If this camera model uses a parameter grid, returns its resolution in
  /// resolution_x and resolution_y and returns true. If the model does not use
  /// a grid, returns false.
  virtual inline bool GetGridResolution(int* resolution_x, int* resolution_y) const {
    (void) resolution_x;
    (void) resolution_y;
    return false;
  }
  
  /// Returns the number of parameters in local updates to this camera model.
  virtual int update_parameter_count() const = 0;
  
  /// Required static method:
  /// If this camera model uses a parameter grid, returns the number of grid
  /// cells that should be placed outside of the calibrated image area on each
  /// side. This is for example used for B-Spline interpolation, which requires
  /// 4x4 grid excerpts to have sufficient context for interpolation.
  static inline int exterior_cells_per_side() {
    return 0;
  }
  
  
  /// Returns whether the given pixel is within the calibrated image area.
  inline bool IsInCalibratedArea(double x, double y) const {
    return x >= m_calibration_min_x && y >= m_calibration_min_y &&
           x < m_calibration_max_x + 1 && y < m_calibration_max_y + 1;
  }
  
  inline Vec2d CenterOfCalibratedArea() const {
    return Vec2d(0.5f * (m_calibration_min_x + m_calibration_max_x + 1),
                 0.5f * (m_calibration_min_y + m_calibration_max_y + 1));
  }
  
  inline int width() const { return m_width; }
  inline int height() const { return m_height; }
  inline int calibration_min_x() const { return m_calibration_min_x; }
  inline int calibration_min_y() const { return m_calibration_min_y; }
  inline int calibration_max_x() const { return m_calibration_max_x; }
  inline int calibration_max_y() const { return m_calibration_max_y; }
  inline Type type() const { return m_type; }
  
  
  static inline bool IsCentral(Type type) {
    switch (type) {
    case Type::CentralGeneric: return true;
    case Type::NoncentralGeneric: return false;
    case Type::CentralRadial: return true;
    case Type::CentralThinPrismFisheye: return true;
    case Type::CentralOpenCV: return true;
    case Type::InvalidType: return false;  // value does not matter
    }
    LOG(FATAL) << "IsCentral(): type not recognized: " << static_cast<int>(type);
    return false;
  }
  
 protected:
  /// Size of the camera images in pixels.
  int m_width;
  int m_height;
  
  /// Extents of the calibrated image area within the image bounds.
  int m_calibration_min_x;
  int m_calibration_min_y;
  int m_calibration_max_x;
  int m_calibration_max_y;
  
  /// Type of this camera model, indicating the subclass instance.
  Type m_type;
};

}

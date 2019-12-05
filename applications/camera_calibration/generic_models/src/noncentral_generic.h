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

#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "noncentral_generic_unprojection_jacobian.h"
#include "util.h"

/// Helper struct for representing two (tangent) vectors.
template <typename Scalar>
struct DirectionTangents {
  Eigen::Matrix<Scalar, 3, 1> t1;
  Eigen::Matrix<Scalar, 3, 1> t2;
};

/// Treating the given direction as a vector with unit length, pointing to a
/// spot on the unit sphere, determines two right-angled tangent vectors for
/// this point of the unit sphere.
template <typename Scalar, typename Derived>
inline void ComputeTangentsForDirection(
    const Eigen::MatrixBase<Derived>& direction,
    DirectionTangents<Scalar>* tangents) {
  tangents->t1 = direction.cross((fabs(direction.x()) > 0.9f) ? Eigen::Matrix<Scalar, 3, 1>(0, 1, 0) : Eigen::Matrix<Scalar, 3, 1>(1, 0, 0)).normalized();
  tangents->t2 = direction.cross(tangents->t1);  // is already normalized
}

/// Computes the Jacobian of ComputeTangentsForDirection() wrt. the given
/// direction.
template <typename Scalar, typename Derived>
inline void TangentsJacobianWrtLineDirection(
    const Eigen::MatrixBase<Derived>& direction,
    Eigen::Matrix<Scalar, 6, 3>* jacobian) {
  if (fabs(direction.x()) > 0.9f) {
    const Scalar term0 = direction.x() * direction.x();
    const Scalar term1 = direction.z() * direction.z();
    const Scalar term2 = term0 + term1;
    const Scalar term7 = 1. / sqrt(term2);
    const Scalar term3 = term7 * term7 * term7;
    const Scalar term4 = direction.x()*direction.z()*term3;
    const Scalar term5 = term0*term3;
    const Scalar term6 = term1*term3;
    const Scalar term8 = direction.x()*term7;
    const Scalar term9 = -direction.y()*term4;
    const Scalar term10 = direction.z()*term7;
    
    *jacobian << term4, 0, -term5,
                 0, 0, 0,
                 term6, 0, -term4,
                 direction.y()*term6, term8, term9,
                 -term8, 0, -term10,
                 term9, term10, direction.y()*term5;
  } else {
    const Scalar term0 = direction.y() * direction.y();
    const Scalar term1 = direction.z() * direction.z();
    const Scalar term2 = term0 + term1;
    const Scalar term7 = 1. / sqrt(term2);
    const Scalar term3 = term7 * term7 * term7;
    const Scalar term4 = direction.y()*direction.z()*term3;
    const Scalar term5 = term0*term3;
    const Scalar term6 = term1*term3;
    const Scalar term8 = direction.y()*term7;
    const Scalar term9 = direction.z()*term7;
    const Scalar term10 = -direction.x()*term4;
    
    *jacobian << 0, 0, 0,
                 0, -term4, term5,
                 0, -term6, term4,
                 0, -term8, -term9,
                 term8, direction.x()*term6, term10,
                 term9, term10, direction.x()*term5;
  }
}

/// Implementation of a non-central-generic camera model. Supports:
/// * Projecting 3D points from the local coordinate frame of the camera to a
///   pixel position in the image.
/// * Computing Jacobians for projection (via finite differences).
/// * Un-projecting image pixel coordinates to 3D lines in the local
///   coordinate frame of the camera.
/// * Computing Jacobians for un-projection.
/// 
/// The coordinate system origin for pixel coordinates is considered to be at
/// the top-left corner of the top-left pixel in the image.
template <typename Scalar>
class NoncentralGenericCamera {
 public:
  typedef Eigen::ParametrizedLine<Scalar, 3> LineT;
  typedef Eigen::Matrix<Scalar, 3, 1> PointT;
  typedef Eigen::Matrix<Scalar, 2, 1> PixelT;
  
  /// Creates an uninitialized camera model. Use Read() to initialize it.
  NoncentralGenericCamera() = default;
  
  /// Creates a partly initialized camera model. It remains to initialize the
  /// grid values with grid_value().
  NoncentralGenericCamera(
      int width,
      int height,
      int calibration_min_x,
      int calibration_min_y,
      int calibration_max_x,
      int calibration_max_y,
      int grid_width,
      int grid_height)
      : m_width(width),
        m_height(height),
        m_calibration_min_x(calibration_min_x),
        m_calibration_min_y(calibration_min_y),
        m_calibration_max_x(calibration_max_x),
        m_calibration_max_y(calibration_max_y),
        m_grid_width(grid_width),
        m_grid_height(grid_height),
        m_point_grid(m_grid_width * m_grid_height),
        m_direction_grid(m_grid_width * m_grid_height) {}
  
  /// Loads the camera model from the given YAML file. Returns true on success,
  /// false on failure. If false is returned and error_reason is non-null, the
  /// reason for the error will be returned there.
  bool Read(const char* yaml_path, std::string* error_reason = nullptr) {
    FILE* file = fopen(yaml_path, "rb");
    if (!file) {
      if (error_reason) {
        *error_reason = "Could not open the file.";
      }
      return false;
    }
    
    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    std::vector<char> text(file_size);
    if (fread(text.data(), 1, file_size, file) != file_size) {
      fclose(file);
      if (error_reason) {
        *error_reason = "Could not read the file content.";
      }
      return false;
    }
    
    // Simple parsing of the YAML subset that is needed to read the files
    // written by the camera_calibration program.
    auto parse_tag = [&](const std::string& tag, const std::string& value) {
      if (tag == "type") {
        if (value != "NoncentralGenericModel" && value != "NoncentralGenericBSplineModel") {
          fclose(file);
          if (error_reason) {
            *error_reason = "The camera model type is wrong: " + value;
          }
          return false;
        }
      } else if (tag == "width") {
        m_width = atoi(value.c_str());
      } else if (tag == "height") {
        m_height = atoi(value.c_str());
      } else if (tag == "calibration_min_x") {
        m_calibration_min_x = atoi(value.c_str());
      } else if (tag == "calibration_min_y") {
        m_calibration_min_y = atoi(value.c_str());
      } else if (tag == "calibration_max_x") {
        m_calibration_max_x = atoi(value.c_str());
      } else if (tag == "calibration_max_y") {
        m_calibration_max_y = atoi(value.c_str());
      } else if (tag == "grid_width") {
        m_grid_width = atoi(value.c_str());
      } else if (tag == "grid_height") {
        m_grid_height = atoi(value.c_str());
      } else if (tag == "point_grid" || tag == "direction_grid") {
        if (m_grid_width < 4 || m_grid_height < 4) {
          fclose(file);
          if (error_reason) {
            *error_reason = "Grid width or height not initialized or incorrect.";
          }
          return false;
        }
        
        std::vector<PointT>* grid = (tag == "point_grid") ? &m_point_grid : &m_direction_grid;
        bool renormalize = (tag == "direction_grid");
        
        grid->resize(m_grid_width * m_grid_height);
        
        // Parse the grid values.
        std::string number;
        int cur_grid_point = 0;
        int cur_dimension = 0;
        for (std::size_t i = 0, size = value.size(); i < size; ++ i) {
          char v = value[i];
          if (v == '[' || v == ' ') {
            continue;
          } else if (v == ',' || v == ']') {
            if (cur_grid_point == m_grid_width * m_grid_height) {
              fclose(file);
              if (error_reason) {
                *error_reason = "Too many grid points given.";
              }
              return false;
            }
            
            // Parse the number.
            Scalar n = atof(number.c_str());
            (*grid)[cur_grid_point][cur_dimension] = n;
            
            ++ cur_dimension;
            if (cur_dimension == 3) {
              // Re-normalize to avoid non-normalized vectors due to limited numerical precision in the file
              if (renormalize) {
                (*grid)[cur_grid_point].normalize();
              }
              
              cur_dimension = 0;
              ++ cur_grid_point;
            }
            
            number = "";
          } else {
            number += v;
          }
        }
        
        if (cur_grid_point != m_grid_width * m_grid_height) {
          fclose(file);
          if (error_reason) {
            *error_reason = "Too few grid points given.";
          }
          return false;
        }
      } else {
        // Unknown tag encountered.
        fclose(file);
        if (error_reason) {
          *error_reason = "Encountered unknown YAML tag: " + tag;
        }
        return false;
      }
      return true;
    };
    
    m_grid_width = -1;
    m_grid_height = -1;
    
    int mode = 0;  // 0: read tag, 1: read value, 2: read comment
    std::string tag;
    std::string value;
    for (int index = 0; index < file_size; ++ index) {
      char c = text[index];
      
      if (mode == 0) {
        // Reading the tag
        if (c == ' ') {
          continue;
        } else if (c == ':') {
          mode = 1;
        } else if (c == '#') {
          mode = 2;
        } else if (c == '\n') {
          // Reset for the next tag
          mode = 0;
          tag = "";
          value = "";
        } else {
          tag += c;
        }
      } else if (mode == 1) {
        // Reading the value
        if (c == ' ') {
          continue;
        } else if (c == '\n') {
          // Parse the tag and value.
          if (!parse_tag(tag, value)) {
            return false;
          }
          
          // Reset for the next tag
          mode = 0;
          tag = "";
          value = "";
        } else {
          value += c;
        }
      } else {  // mode == 2
        if (c == '\n') {
          // Reset for the next tag
          mode = 0;
          tag = "";
          value = "";
        }
      }
    }
    
    // Parse final tag and value
    if (!tag.empty() && !parse_tag(tag, value)) {
      return false;
    }
    
    fclose(file);
    return true;
  }
  
  /// Projects the given 3D point in the local coordinate frame of the camera
  /// to a pixel position in the image.
  /// 
  /// Returns true if the point projects to the image, false if it does not. If
  /// true is returned, the result is stored in the pixel output parameter.
  template <typename Derived>
  bool Project(
      const Eigen::MatrixBase<Derived>& point,
      PixelT* pixel) const {
    *pixel = CenterOfCalibratedArea();
    return ProjectWithInitialEstimate(point, pixel);
  }
  
  /// Version of Project() for which an initial estimate must be given with the
  /// pixel parameter already. This may be used to speed up the process and / or
  /// ensure convergence of the optimization to the desired optimum.
  template <typename Derived>
  bool ProjectWithInitialEstimate(
      const Eigen::MatrixBase<Derived>& point,
      PixelT* pixel) const {
    // Levenberg-Marquardt optimization algorithm.
    constexpr Scalar kEpsilon = 1e-12;
    const int kMaxIterations = 100;
    
    Scalar lambda = -1;
    for (int i = 0; i < kMaxIterations; ++i) {
      Eigen::Matrix<Scalar, 6, 2> dline_dxy;
      LineT line;
      UnprojectWithJacobian(*pixel, &line, &dline_dxy);  // should always return true
      
      DirectionTangents<Scalar> tangents;
      ComputeTangentsForDirection(line.direction(), &tangents);
      
      // (Non-squared) residuals.
      PointT point_to_origin = line.origin() - point;
      Scalar d1 = tangents.t1.dot(point_to_origin);
      Scalar d2 = tangents.t2.dot(point_to_origin);
      
      // Jacobian of residuals wrt. pixel x, y [2 x 2]
      Eigen::Matrix<Scalar, 6, 3> tangents_wrt_direction;
      TangentsJacobianWrtLineDirection(
          line.direction(),
          &tangents_wrt_direction);
      
      Eigen::Matrix<Scalar, 2, 9> d_wrt_t1_t2_origin;
      d_wrt_t1_t2_origin <<
          point_to_origin.x(), point_to_origin.y(), point_to_origin.z(), 0, 0, 0, tangents.t1.x(), tangents.t1.y(), tangents.t1.z(),
          0, 0, 0, point_to_origin.x(), point_to_origin.y(), point_to_origin.z(), tangents.t2.x(), tangents.t2.y(), tangents.t2.z();
      
      Eigen::Matrix<Scalar, 9, 2> t1_t2_origin_wrt_xy = Eigen::Matrix<Scalar, 9, 2>::Zero();
      t1_t2_origin_wrt_xy.template block<6, 2>(0, 0) =
          tangents_wrt_direction * dline_dxy.template block<3, 2>(0, 0);
      t1_t2_origin_wrt_xy.template block<3, 2>(6, 0) =
          dline_dxy.template block<3, 2>(3, 0);
      
      Eigen::Matrix<Scalar, 2, 2> residuals_wrt_xy =
          d_wrt_t1_t2_origin * t1_t2_origin_wrt_xy;
      
      Scalar cost = d1 * d1 + d2 * d2;
      
      // Accumulate H and b.
      Scalar H_0_0 = residuals_wrt_xy(0, 0) * residuals_wrt_xy(0, 0) + residuals_wrt_xy(1, 0) * residuals_wrt_xy(1, 0);
      Scalar H_1_0_and_0_1 = residuals_wrt_xy(0, 0) * residuals_wrt_xy(0, 1) + residuals_wrt_xy(1, 0) * residuals_wrt_xy(1, 1);
      Scalar H_1_1 = residuals_wrt_xy(0, 1) * residuals_wrt_xy(0, 1) + residuals_wrt_xy(1, 1) * residuals_wrt_xy(1, 1);
      Scalar b_0 = d1 * residuals_wrt_xy(0, 0) + d2 * residuals_wrt_xy(1, 0);
      Scalar b_1 = d1 * residuals_wrt_xy(0, 1) + d2 * residuals_wrt_xy(1, 1);
      
      if (lambda < 0) {
        constexpr Scalar kInitialLambdaFactor = 0.01;
        lambda = kInitialLambdaFactor * 0.5 * (H_0_0 + H_1_1);
      }
      
      bool update_accepted = false;
      for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
        Scalar H_0_0_LM = H_0_0 + lambda;
        Scalar H_1_1_LM = H_1_1 + lambda;
        
        // Solve the system.
        Scalar x_1 = (b_1 - H_1_0_and_0_1 / H_0_0_LM * b_0) /
                    (H_1_1_LM - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0_LM);
        Scalar x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0_LM;
        
        // Compute the test state (constrained to the calibrated image area).
        PixelT test_result(
            std::max<Scalar>(m_calibration_min_x, std::min(m_calibration_max_x + 0.999, pixel->x() - x_0)),
            std::max<Scalar>(m_calibration_min_y, std::min(m_calibration_max_y + 0.999, pixel->y() - x_1)));
        
        // Compute the test cost.
        Scalar test_cost = std::numeric_limits<Scalar>::infinity();
        LineT test_line;
        if (Unproject(test_result, &test_line)) {
          DirectionTangents<Scalar> test_tangents;
          ComputeTangentsForDirection(test_line.direction(), &test_tangents);
          
          // (Non-squared) residuals.
          PointT test_point_to_origin = test_line.origin() - point;
          Scalar test_d1 = test_tangents.t1.dot(test_point_to_origin);
          Scalar test_d2 = test_tangents.t2.dot(test_point_to_origin);
          
          test_cost = test_d1 * test_d1 + test_d2 * test_d2;
        }
        
        if (test_cost < cost) {
          lambda *= 0.5;
          *pixel = test_result;
          update_accepted = true;
          break;
        } else {
          lambda *= 2;
        }
      }
      
      if (!update_accepted) {
        return cost < kEpsilon;
      }
      
      if (cost < kEpsilon) {
        return true;
      }
    }
    
    return false;
  }
  
  /// Projects the given 3D point in the local coordinate frame of the camera
  /// to a pixel position in the image. Additionally returns the Jacobian of
  /// this operation, which says how the projection result changes when varying
  /// the input point's coordinates. Note that the Jacobian is computed with
  /// finite differences.
  /// 
  /// Returns true if the point projects to the image, false if it does not. If
  /// true is returned, the result is stored in the pixel and jacobian output
  /// parameters.
  template <typename Derived>
  bool ProjectWithJacobian(
      const Eigen::MatrixBase<Derived>& point,
      PixelT* pixel,
      Eigen::Matrix<Scalar, 2, 3>* jacobian,
      Scalar numerical_diff_delta = 1e-4) const {
    *pixel = CenterOfCalibratedArea();
    return ProjectWithJacobianAndInitialEstimate(point, pixel, jacobian, numerical_diff_delta);
  }
  
  /// Version of ProjectWithJacobian() for which an initial estimate must be
  /// given with the pixel parameter already. This may be used to speed up the
  /// process and / or ensure convergence of the optimization to the desired
  /// optimum.
  template <typename Derived>
  bool ProjectWithJacobianAndInitialEstimate(
      const Eigen::MatrixBase<Derived>& point,
      PixelT* pixel,
      Eigen::Matrix<Scalar, 2, 3>* jacobian,
      Scalar numerical_diff_delta = 1e-4) const {
    if (!ProjectWithInitialEstimate(point, pixel)) {
      return false;
    }
    
    bool ok = true;
    for (int dimension = 0; dimension < 3; ++ dimension) {
      PointT offset_point = point;
      offset_point(dimension) += numerical_diff_delta;
      PixelT offset_pixel = *pixel;
      if (!ProjectWithInitialEstimate(offset_point, &offset_pixel)) {
        ok = false;
        break;
      }
      
      (*jacobian)(0, dimension) = (offset_pixel.x() - pixel->x()) / numerical_diff_delta;
      (*jacobian)(1, dimension) = (offset_pixel.y() - pixel->y()) / numerical_diff_delta;
    }
    if (!ok) {
      return false;
    }
    
    return true;
  }
  
  /// Un-projects the given image pixel coordinate to a 3D line in the
  /// local coordinate frame of the camera.
  /// 
  /// Returns true if the point can be unprojected, false if it not. If
  /// true is returned, the result is stored in the line output parameter.
  template <typename Derived>
  bool Unproject(
      const Eigen::MatrixBase<Derived>& pixel,
      LineT* line) const {
    if (!IsInCalibratedArea(pixel.x(), pixel.y())) {
      return false;
    }
    
    PixelT grid_point = PixelCornerConvToGridPoint(pixel.x(), pixel.y());
    
    EvalTwoUniformCubicBSplineSurfaces(m_direction_grid, m_point_grid, m_grid_width, grid_point.x(), grid_point.y(), &line->direction(), &line->origin());
    line->direction().normalize();
    
    return true;
  }
  
  /// Un-projects the given image pixel coordinate to a 3D line in the
  /// local coordinate frame of the camera. Additionally returns the Jacobian of
  /// this operation, which says how the unprojection result changes when
  /// varying the input pixel's coordinates.
  /// 
  /// Returns true if the point can be unprojected, false if it not. If
  /// true is returned, the result is stored in the line and jacobian
  /// output parameters.
  template <typename Derived>
  bool UnprojectWithJacobian(
      const Eigen::MatrixBase<Derived>& pixel,
      LineT* line,
      Eigen::Matrix<Scalar, 6, 2>* jacobian) const {
    if (!IsInCalibratedArea(pixel.x(), pixel.y())) {
      return false;
    }
    
    PixelT grid_point = PixelCornerConvToGridPoint(pixel.x(), pixel.y()) + PixelT(2, 2);
    
    int ix = std::floor(grid_point.x());
    int iy = std::floor(grid_point.y());
    
    Scalar frac_x = grid_point.x() - (ix - 3);
    Scalar frac_y = grid_point.y() - (iy - 3);
    
    Eigen::Matrix<Scalar, 6, 1> p[4][4];
    for (int y = 0; y < 4; ++ y) {
      for (int x = 0; x < 4; ++ x) {
        p[y][x].template topRows<3>() = m_direction_grid[(ix - 3 + x) + (iy - 3 + y) * m_grid_width];
        p[y][x].template bottomRows<3>() = m_point_grid[(ix - 3 + x) + (iy - 3 + y) * m_grid_width];
      }
    }
    
    NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, p, line, jacobian);
    for (int i = 0; i < 6; ++ i) {
      (*jacobian)(i, 0) = PixelScaleToGridScaleX((*jacobian)(i, 0));
      (*jacobian)(i, 1) = PixelScaleToGridScaleY((*jacobian)(i, 1));
    }
    return true;
  }
  
  /// Returns the width of the camera images.
  inline int width() const { return m_width; }
  
  /// Returns the height of the camera images.
  inline int height() const { return m_height; }
  
  /// Returns the left x coordinate of the calibrated image area rectangle.
  inline int calibration_min_x() const { return m_calibration_min_x; }
  
  /// Returns the top y coordinate of the calibrated image area rectangle.
  inline int calibration_min_y() const { return m_calibration_min_y; }
  
  /// Returns the right x coordinate of the calibrated image area rectangle.
  inline int calibration_max_x() const { return m_calibration_max_x; }
  
  /// Returns the bottom y coordinate of the calibrated image area rectangle.
  inline int calibration_max_y() const { return m_calibration_max_y; }
  
  /// Returns whether the given pixel is within the calibrated image area.
  inline bool IsInCalibratedArea(Scalar x, Scalar y) const {
    return x >= m_calibration_min_x && y >= m_calibration_min_y &&
           x < m_calibration_max_x + 1 && y < m_calibration_max_y + 1;
  }
  
  /// Returns the center point of the calibrated image area.
  inline PixelT CenterOfCalibratedArea() const {
    return PixelT(0.5f * (m_calibration_min_x + m_calibration_max_x + 1),
                  0.5f * (m_calibration_min_y + m_calibration_max_y + 1));
  }
  
  /// Returns the internal grid width.
  inline int grid_width() const { return m_grid_width; }
  
  /// Returns the internal grid height.
  inline int grid_height() const { return m_grid_height; }
  
  /// Provides access to point grid values.
  inline PointT& point_grid_value(int x, int y) { return m_point_grid[y * m_grid_width + x]; }
  
  /// Provides const access to point grid values.
  inline const PointT& point_grid_value(int x, int y) const { return m_point_grid[y * m_grid_width + x]; }
  
  /// Provides access to direction grid values.
  inline PointT& direction_grid_value(int x, int y) { return m_direction_grid[y * m_grid_width + x]; }
  
  /// Provides const access to direction grid values.
  inline const PointT& direction_grid_value(int x, int y) const { return m_direction_grid[y * m_grid_width + x]; }
  
 private:
  inline PixelT PixelCornerConvToGridPoint(Scalar x, Scalar y) const {
    return PixelT(
        1.f + (m_grid_width - 3.f) * (x - m_calibration_min_x) / (m_calibration_max_x + 1 - m_calibration_min_x),
        1.f + (m_grid_height - 3.f) * (y - m_calibration_min_y) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  inline Scalar PixelScaleToGridScaleX(Scalar length) const {
    return length * ((m_grid_width - 3.f) / (m_calibration_max_x + 1 - m_calibration_min_x));
  }
  
  inline Scalar PixelScaleToGridScaleY(Scalar length) const {
    return length * ((m_grid_height - 3.f) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  int m_width;
  int m_height;
  int m_calibration_min_x;
  int m_calibration_min_y;
  int m_calibration_max_x;
  int m_calibration_max_y;
  int m_grid_width;
  int m_grid_height;
  // Note that Eigen matrices with 3 elements (such as PointT) are not
  // vectorizable. So, no special care to align these std::vectors should be
  // needed.
  std::vector<PointT> m_point_grid;
  std::vector<PointT> m_direction_grid;
};

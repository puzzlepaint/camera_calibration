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

#include <fstream>
#include <iomanip>
#include <memory>
#include <unordered_map>
#include <vector>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>
#include <QDir>
#include <QFileInfo>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/models/all_models.h"
#include "camera_calibration/util.h"

namespace vis {

class CentralThinPrismFisheyeModel;

/// Creates a report showing the differences between the base model and the
/// fitted model.
template <typename AccurateModelT, typename FittedModelT>
bool CreateFittingErrorReport(
    const char* base_path,
    const AccurateModelT& base_model,
    const FittedModelT& fitted_model,
    const Mat3d& parametric_r_dense,
    int border_x = 0,
    int border_y = 0,
    double max_visualization_extent = -1,
    double max_visualization_extent_pixels = -1) {
  CHECK_EQ(base_model.width() - 2 * border_x, fitted_model.width());
  CHECK_EQ(base_model.height() - 2 * border_y, fitted_model.height());
  
  QFileInfo(base_path).dir().mkpath(".");
  
  double max_error_component = 0;
  double max_error_norm = 0;
  
  double reprojection_error_sum = 0;
  usize reprojection_error_count = 0;
  double reprojection_error_max = 0;
  vector<double> reprojection_errors;
  reprojection_errors.reserve(100000);
  
  Image<Vec3d> generic_direction_image(fitted_model.width(), fitted_model.height());
  Image<Vec3d> fitted_direction_image(fitted_model.width(), fitted_model.height());
  Image<Vec3d> error_image(fitted_model.width(), fitted_model.height());
  Image<Vec2d> reprojection_error_image(fitted_model.width(), fitted_model.height());
  for (int y = 0; y < fitted_model.height(); ++ y) {
    for (int x = 0; x < fitted_model.width(); ++ x) {
      reprojection_error_image(x, y) = Vec2d::Zero();
      
      Vec3d generic_direction_before_rotation;
      bool ok = base_model.Unproject(border_x + x + 0.5f, border_y + y + 0.5f, &generic_direction_before_rotation);
      if (!ok) {
        error_image(x, y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
        continue;
      }
      
      Vec3d generic_direction = parametric_r_dense * generic_direction_before_rotation;
      Vec3d fitted_direction;
      Vec3d error;
      if (fitted_model.Unproject(x + 0.5f, y + 0.5f, &fitted_direction)) {
        error = fitted_direction - generic_direction;
      } else {
        error = Vec3d::Constant(numeric_limits<double>::infinity());
      }
      
      generic_direction_image(x, y) = generic_direction;
      fitted_direction_image(x, y) = fitted_direction;
      error_image(x, y) = error;
      
      if (!std::isinf(error.x()) && !std::isinf(error.y()) && !std::isinf(error.z())) {
        max_error_component = std::max(max_error_component, fabs(error.x()));
        max_error_component = std::max(max_error_component, fabs(error.y()));
        max_error_component = std::max(max_error_component, fabs(error.z()));
        max_error_norm = std::max(max_error_norm, error.norm());
      }
      
      Vec2d reprojected_pixel;
      if (fitted_model.Project(generic_direction, &reprojected_pixel)) {
        Vec2d reprojection_error = Vec2d(x + 0.5f, y + 0.5f) - reprojected_pixel;
        double reprojection_error_magnitude = reprojection_error.norm();
        reprojection_error_image(x, y) = reprojection_error;
        reprojection_error_sum += reprojection_error_magnitude;
        ++ reprojection_error_count;
        reprojection_error_max = std::max(reprojection_error_max, reprojection_error_magnitude);
        reprojection_errors.push_back(reprojection_error_magnitude);
      }
    }
  }
  
  double max_angle_component = 0.025;  // TODO: Make configurable
  if (max_visualization_extent >= 0) {
    max_error_component = max_visualization_extent;
  }
  if (max_visualization_extent_pixels >= 0) {
    reprojection_error_max = max_visualization_extent_pixels;
  }
  
  Image<Vec3u8> error_direction_angle_visualization(fitted_model.width(), fitted_model.height());
  Image<Vec3u8> error_direction_visualization(fitted_model.width(), fitted_model.height());
  Image<u8> error_magnitude_visualization(fitted_model.width(), fitted_model.height());
  Image<u8> reprojection_error_magnitude_visualization(fitted_model.width(), fitted_model.height());
  Image<Vec3u8> reprojection_error_visualization(fitted_model.width(), fitted_model.height());
  
  for (int y = 0; y < fitted_model.height(); ++ y) {
    for (int x = 0; x < fitted_model.width(); ++ x) {
      const Vec3d& error = error_image(x, y);
      if (error.hasNaN()) {
        error_direction_angle_visualization(x, y) = Vec3u8(0, 0, 0);
        error_direction_visualization(x, y) = Vec3u8(0, 0, 0);
        error_magnitude_visualization(x, y) = 0;
      } else {
        Vec3d relative_error = (error / max_error_component).cwiseMax(Vec3d(-1, -1, -1)).cwiseMin(Vec3d(1, 1, 1));
        
        const Vec3d& gen_dir = generic_direction_image(x, y);
        const Vec3d& fit_dir = fitted_direction_image(x, y);
        // TODO: Not accounting for possible wrap-around of the angles here
        error_direction_angle_visualization(x, y) = Vec3u8(
            std::min<int>(255, std::max<int>(0, 127 + 127 / (M_PI / 180.f * max_angle_component) * (atan2(gen_dir.z(), gen_dir.x()) - atan2(fit_dir.z(), fit_dir.x())) + 0.5)),
            std::min<int>(255, std::max<int>(0, 127 + 127 / (M_PI / 180.f * max_angle_component) * (atan2(gen_dir.y(), gen_dir.z()) - atan2(fit_dir.y(), fit_dir.z())) + 0.5)),
            127);
        
        error_direction_visualization(x, y) =
            ((255.99f / 2) * (relative_error + Vec3d::Constant(1.f))).cast<u8>();
        error_magnitude_visualization(x, y) = 255.99f * (error.norm() / max_error_norm);
      }
      
      const Vec2d& reprojection_error = reprojection_error_image(x, y);
      double reprojection_error_magnitude = reprojection_error.norm();
      reprojection_error_magnitude_visualization(x, y) = std::max<float>(0.f, std::min<float>(255.f, 255.99f * reprojection_error_magnitude / reprojection_error_max));
      
      double strength = std::max(0., std::min(1., (reprojection_error_magnitude / max_visualization_extent_pixels)));
      // NOTE: Inverting the reprojection error here. This way, the visualizations
      //       were consistent with the residual visualizations.
      double dir = atan2(-reprojection_error.y(), -reprojection_error.x());  // from -M_PI to M_PI
      Vec3f color = Vec3f(
          127 + strength * 127 * sin(dir),
          127 + strength * 127 * cos(dir),
          127);
      reprojection_error_visualization(x, y) = (color + Vec3f::Constant(0.5f)).cast<u8>();
    }
  }
  
  error_magnitude_visualization.Write(string(base_path) + "_fitting_error_magnitudes.png");
  error_direction_angle_visualization.Write(string(base_path) + "_fitting_error_direction_angles.png");
  error_direction_visualization.Write(string(base_path) + "_fitting_error_directions.png");
  reprojection_error_magnitude_visualization.Write(string(base_path) + "_fitting_error_reprojection_magnitudes.png");
  reprojection_error_visualization.Write(string(base_path) + "_fitting_error_reprojections.png");
  
  ofstream stream(string(base_path) + "_fitting_info.txt", std::ios::out);
  if (!stream) {
    return false;
  }
  stream << std::setprecision(14);
  
  if (!reprojection_errors.empty()) {
    std::sort(reprojection_errors.begin(), reprojection_errors.end());
    stream << "median_reprojection_error : " << reprojection_errors[reprojection_errors.size() / 2] << std::endl;
  }
  stream << "average_reprojection_error : " << (reprojection_error_sum / reprojection_error_count) << std::endl;
  stream << "maximum_reprojection_error : " << reprojection_error_max << std::endl;
  
  stream << "error_magnitude_visualization_max_error_norm : " << max_error_norm << std::endl;
  stream << "error_direction_visualization_max_error_component : " << max_error_component << std::endl;
  
  return true;
}

template <typename FittedModelT>
bool CreateFittingErrorReport(
    const char* /*base_path*/,
    const NoncentralGenericModel& /*base_model*/,
    const FittedModelT& /*fitted_model*/,
    const Mat3d& /*parametric_r_dense*/,
    int /*border_x*/ = 0,
    int /*border_y*/ = 0,
    double /*max_visualization_extent*/ = -1,
    double /*max_visualization_extent_pixels*/ = -1) {
  LOG(WARNING) << "CreateFittingErrorReport() is not implemented for NoncentralGenericModel";
  return false;
}

inline int CreateFittingVisualization(
    const BAState& state,
    const string& report_base_path,
    float max_visualization_extent,
    float max_visualization_extent_pixels) {
  constexpr int kBorderX = 0;
  constexpr int kBorderY = 0;
  
  for (int camera_index = 0; camera_index < state.num_cameras(); ++ camera_index) {
    CameraModel* model = state.intrinsics[camera_index].get();
    if (dynamic_cast<NoncentralGenericModel*>(model)) {
      LOG(INFO) << "Fitting visualization not supported for non-central camera with index " << camera_index;
      continue;
    }
    
    Image<Vec3d> dense_model;
    if (CreateObservationDirectionsImage(model, &dense_model)) {
      constexpr bool use_equidistant_projection = true;  // (i == 0);
      CentralThinPrismFisheyeModel parametric_model(dense_model.width(),
                                                    dense_model.height(),
                                                    use_equidistant_projection);
      Mat3d parametric_r_dense = Mat3d::Identity();
      parametric_model.FitToDenseModel(dense_model, &parametric_r_dense, /*subsample_step*/ 5, /*print_progress*/ true);  // TODO: subsample_step: make configurable?
      
      ostringstream camera_name;
      camera_name << "_cam" << camera_index;
      CameraModel& model_ref = *model;
      IDENTIFY_CAMERA_MODEL(
          model_ref,
          CreateFittingErrorReport(
            (report_base_path + camera_name.str() + (use_equidistant_projection ? string("_equidistant") : string("_pinhole"))).c_str(),
            _model_ref,
            parametric_model,
            parametric_r_dense,
            kBorderX, kBorderY,
            max_visualization_extent,
            max_visualization_extent_pixels);
      );
    }
  }
  
  return EXIT_SUCCESS;
}

}

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

#include "camera_calibration/calibration.h"

#include <thread>

#include <boost/filesystem.hpp>
#include <libvis/logging.h>
#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/image_display.h>
#include <libvis/lm_optimizer.h>
#include <libvis/util.h>
#include <QApplication>
#include <QDir>
#include <QFileInfo>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/cuda_joint_optimization.h"
#include "camera_calibration/calibration_report.h"
#include "camera_calibration/calibration_initialization/dense_initialization.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/fitting_report.h"
#include "camera_calibration/hash_vec2i.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/all_models.h"
#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"
#include "camera_calibration/ui/calibration_window.h"
#include "camera_calibration/util.h"

namespace vis {

// TODO: Make the number of spline parameters configurable?
constexpr int kNumSplineParametersForRadialModel = 250;

void DeleteOutlierFeatures(
    int camera_index,
    Dataset* dataset,
    BAState* state,
    float outlier_removal_factor,
    CalibrationWindow* calibration_window,
    bool step_by_step,
    const char* outlier_visualization_path) {
  // Get statistics of reprojection errors
  vector<double> reprojection_errors;
  
  for (int imageset_index = 0; imageset_index < dataset->ImagesetCount(); ++ imageset_index) {
    if (!state->image_used.at(imageset_index)) {
      continue;
    }
    
    const SE3d& this_image_tr_global = state->image_tr_global(camera_index, imageset_index);
    Mat3d image_r_global = this_image_tr_global.rotationMatrix();
    const Vec3d& image_t_global = this_image_tr_global.translation();
    
    shared_ptr<const Imageset> imageset = dataset->GetImageset(imageset_index);
    const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
    
    for (const PointFeature& feature : features) {
      Vec3d local_point = image_r_global * state->points[feature.index] + image_t_global;
      Vec2d pixel;
      if (state->intrinsics[camera_index]->Project(local_point, &pixel)) {
        Vec2d reprojection_error = pixel - feature.xy.cast<double>();
        double reprojection_error_magnitude = reprojection_error.norm();
        reprojection_errors.push_back(reprojection_error_magnitude);
      }
    }
  }
  
  if (reprojection_errors.size() < 8) {  // arbitrary threshold
    // Too few reprojection errors to reliably detect outliers.
    return;
  }
  
  std::sort(reprojection_errors.begin(), reprojection_errors.end());
  
  // Determine outlier threshold
  double first_quartile_error = reprojection_errors[0.25f * reprojection_errors.size() + 0.5f];
  double third_quartile_error = reprojection_errors[0.75f * reprojection_errors.size() + 0.5f];
  // 1.5 is what matplotlib uses by default for outliers in box plots
  double outlier_threshold = (third_quartile_error + outlier_removal_factor * (third_quartile_error - first_quartile_error))*0.5;
  
  Image<Vec3u8> outlier_visualization(state->intrinsics[camera_index]->width(), state->intrinsics[camera_index]->height());
  outlier_visualization.SetTo(Vec3u8::Zero());
  
  // Remove outliers
  usize num_removed_outliers = 0;
  for (int imageset_index = 0; imageset_index < dataset->ImagesetCount(); ++ imageset_index) {
    if (!state->image_used.at(imageset_index)) {
      continue;
    }
    
    const SE3d& this_image_tr_global = state->image_tr_global(camera_index, imageset_index);
    Mat3d image_r_global = this_image_tr_global.rotationMatrix();
    const Vec3d& image_t_global = this_image_tr_global.translation();
    
    shared_ptr<Imageset> imageset = dataset->GetImageset(imageset_index);
    vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
    
    usize num_features = features.size();
    erase_if(features, [&](const PointFeature& feature){
      Vec3d local_point = image_r_global * state->points[feature.index] + image_t_global;
      Vec2d pixel;
      if (!state->intrinsics[camera_index]->Project(local_point, &pixel)) {
        LOG(1) << "Removing outlier which does not project into the image.";
        outlier_visualization(feature.xy.x(), feature.xy.y()) = Vec3u8(127, 127, 127);
        return true;
      }
      Vec2d reprojection_error = pixel - feature.xy.cast<double>();
      double reprojection_error_magnitude = reprojection_error.norm();
      if (reprojection_error_magnitude > outlier_threshold) {
        LOG(1) << "Removing outlier with reprojection error higher than threshold (" << reprojection_error_magnitude << " > " << outlier_threshold << ")";
        
        if (reprojection_error_magnitude > 10) {
          outlier_visualization(feature.xy.x(), feature.xy.y()) = Vec3u8(255, 0, 0);
        } else if (reprojection_error_magnitude > 5) {
          outlier_visualization(feature.xy.x(), feature.xy.y()) = Vec3u8(255, 127, 0);
        } else if (reprojection_error_magnitude > 1) {
          outlier_visualization(feature.xy.x(), feature.xy.y()) = Vec3u8(255, 255, 0);
        } else {
          outlier_visualization(feature.xy.x(), feature.xy.y()) = Vec3u8(255, 255, 255);
        }
      }
      return reprojection_error_magnitude > outlier_threshold;
    });
    num_removed_outliers += num_features - features.size();
    
    if (features.size() < 3) {
      state->image_used.at(imageset_index) = false;
    }
  }
  
  LOG(INFO) << "Outlier detection removed " << num_removed_outliers << " outlier features.";
  
  if (calibration_window) {
    calibration_window->UpdateRemovedOutliers(camera_index, outlier_visualization);
    if (step_by_step) {
      GetKeyInput();
    }
  }
  
  if (outlier_visualization_path) {
    ostringstream path;
    path << outlier_visualization_path << "_camera" << camera_index << "_removed_outliers.png";
    
    // Create containing folder
    QFileInfo(path.str().c_str()).dir().mkpath(".");
    
    if (!outlier_visualization.Write(path.str())) {
      LOG(ERROR) << "Cannot write file: " << path.str();
    }
  }
  
  // TODO: It might theoretically happen that some points in the points vector
  //       are not referenced anymore at all after removing outliers. Those
  //       should ideally be removed here, otherwise they might have a
  //       detrimental effect on the optimization.
}


void RunBundleAdjustment(
    bool use_cuda,
    SchurMode schur_mode,
    int max_iteration_count,
    double cost_reduction_threshold,
    Dataset* dataset,
    BAState* state,
    double regularization_weight,
    bool localize_only,
    CalibrationWindow* calibration_window,
    bool step_by_step,
    const char* state_output_path) {
   // NOTE: So far I didn't see any improvement from using multiple deltas.
   //       But it does have an effect, since making it too large (1e-1) resulted in problems.
  vector<double> numerical_diff_delta_range = {/*1e-3,*/ 1e-4/*, 1e-5*/};
  
  double lambda = -1;
  
  for (usize numerical_diff_delta_index = 0; numerical_diff_delta_index < numerical_diff_delta_range.size(); ++ numerical_diff_delta_index) {
    double numerical_diff_delta = numerical_diff_delta_range[numerical_diff_delta_index];
    
    double last_cost = numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < max_iteration_count; ++ iteration) {
      double cost;
      
      if (use_cuda) {
        if (localize_only) {
          LOG(ERROR) << "localize_only is not supported for CUDA-based optimization currently";
        }
        OptimizationReport report = CudaOptimizeJointly(
            *dataset,
            state,
            /*max_iteration_count*/ 1,
            /*int max_inner_iterations*/ 50,  // TODO: tune
            lambda,
            numerical_diff_delta,
            regularization_weight,
            &lambda);
        cost = report.final_cost;
      } else {
        cost = OptimizeJointly(
            *dataset,
            state,
            /*max_iteration_count*/ 1,
            lambda,
            numerical_diff_delta,
            regularization_weight,
            localize_only,
            /*eliminate_points*/ false,
            schur_mode,
            &lambda);
      }
      
      LOG(INFO) << "[" << (iteration + 1) << " of " << numerical_diff_delta_range.size() << "] Cost: " << cost;
      
      // Save the optimization state
      if (state_output_path) {
        SaveBAState(state_output_path, *state);
      }
      
      // Beautify all camera orientations
      if (!localize_only) {
        for (int camera_index = 0; camera_index < state->num_cameras(); ++ camera_index) {
          Mat3d rotation = state->intrinsics[camera_index]->ChooseNiceCameraOrientation();
          SE3d rotation_transform(rotation, Vec3d::Zero());
          state->camera_tr_rig[camera_index] = rotation_transform * state->camera_tr_rig[camera_index];
        }
      }
      
      // Visualize current state
      if (calibration_window) {
        for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
          calibration_window->SetCurrentCameraIndex(camera_index);
          
          // Visualize model intrinsics
          Image<Vec3u8> visualization;
          VisualizeModelDirections(*state->intrinsics[camera_index], &visualization);
          calibration_window->UpdateObservationDirections(camera_index, visualization);
          
          // Visualize reprojection errors
          Image<u8> reprojection_histogram_image;
          CreateReprojectionErrorHistogram(
              camera_index,
              *dataset,
              *state,
              &reprojection_histogram_image);
          calibration_window->UpdateErrorHistogram(camera_index, reprojection_histogram_image);
          
          // Visualize reprojection errors with their magnitudes
          CreateReprojectionErrorMagnitudeVisualization(
              *dataset, camera_index, *state, 5.0f, &visualization);
          calibration_window->UpdateReprojectionErrors(camera_index, visualization, dataset, state);
          
          // Visualize reprojection error directions
          CreateReprojectionErrorDirectionVisualization(
              *dataset, camera_index, *state, &visualization);
          calibration_window->UpdateErrorDirections(camera_index, visualization);
        }
        
        if (step_by_step) {
          LOG(INFO) << "> Press Return to continue";
          GetKeyInput();
        }
      }
      
      // Stop if the q key was pressed
      if (PollKeyInput() == 'q') {
        break;
      }
      
      // Stopping criterion
      if (cost >= last_cost - cost_reduction_threshold) {
        break;
      }
      last_cost = cost;
    }
  }
}


void ScaleToMetric(
    Dataset* dataset,
    BAState* state) {
  // Average a scaling correction factor by comparing the distance of neighboring corners
  // in the ideal pattern and all optimized patterns. Comparing neighboring corners (as
  // opposed to comparing other corner pairs) makes sense since we expect non-rigid
  // deformations to the calibration pattern (e.g., if it is printed on paper).
  double scaling_log_sum = 0;
  int scaling_count = 0;
  
  for (usize k = 0; k < dataset->KnownGeometriesCount(); ++ k) {
    const KnownGeometry& geometry = dataset->GetKnownGeometry(k);
    
    // Build a map from corner positions to indices in state->points
    unordered_map<Vec2i, int> corner_position_to_index;
    for (const pair<int, Vec2i>& item : geometry.feature_id_to_position) {
      int feature_id = item.first;
      const Vec2i& position = item.second;
      
      auto it = state->feature_id_to_points_index.find(feature_id);
      if (it != state->feature_id_to_points_index.end()) {
        corner_position_to_index[position] = it->second;
      }
    }
    if (corner_position_to_index.empty()) {
      continue;
    }
    
    // Loop over all corners
    for (const pair<int, Vec2i>& item : geometry.feature_id_to_position) {
      const Vec2i& position = item.second;
      auto it = corner_position_to_index.find(position);
      if (it == corner_position_to_index.end()) {
        continue;
      }
      int index = it->second;
      
      // Loop over neighbors of this corner (looking in one direction is sufficient)
      constexpr int kNeighbors[2][2] = {{1, 0}, {0, 1}};
      
      for (int n = 0; n < 2; ++ n) {
        // Is the neighbor within the pattern?
        Vec2i neighbor_position = position + Vec2i(kNeighbors[n][0], kNeighbors[n][1]);
        
        auto cit = corner_position_to_index.find(neighbor_position);
        if (cit == corner_position_to_index.end()) {
          continue;
        }
        int neighbor_index = cit->second;
        
        double ideal_distance = geometry.cell_length_in_meters;
        double actual_distance =
            (state->points[index] -
            state->points[neighbor_index]).norm();
        
        scaling_log_sum += std::log(ideal_distance / actual_distance);
        ++ scaling_count;
      }
    }
  }
  
  double scaling_factor = std::exp(scaling_log_sum / scaling_count);
  state->ScaleState(scaling_factor);
}


bool ResampleModel(
    shared_ptr<CameraModel>& model_to_optimize,
    SE3d* camera_tr_rig,
    int calibration_min_x,
    int calibration_min_y,
    int calibration_max_x,
    int calibration_max_y,
    CameraModel::Type model_type,
    int target_resolution_x,
    int target_resolution_y) {
  LOG(INFO) << "Resampling model ...";
  
  // Special case for resampling NoncentralGeneric --> NoncentralGeneric
  if (model_to_optimize->type() == CameraModel::Type::NoncentralGeneric &&
      model_type == CameraModel::Type::NoncentralGeneric) {
    NoncentralGenericModel* ngb_model = dynamic_cast<NoncentralGenericModel*>(model_to_optimize.get());
    Image<Vec3d> new_point_grid(target_resolution_x, target_resolution_y);
    Image<Vec3d> new_direction_grid(target_resolution_x, target_resolution_y);
    for (u32 y = 0; y < new_point_grid.height(); ++ y) {
      for (u32 x = 0; x < new_point_grid.width(); ++ x) {
        // Convert the grid point (in the new grid) to pixel-corner convention coordinates
        Vec2d pixel = CentralGenericModel::GridPointToPixelCornerConv(
            x, y,
            calibration_min_x, calibration_min_y,
            calibration_max_x, calibration_max_y,
            new_direction_grid.width(), new_direction_grid.height());
        
        Vec2d old_grid_point = ngb_model->PixelCornerConvToGridPoint(pixel.x(), pixel.y());
        old_grid_point = old_grid_point.cwiseMax(Vec2d(0, 0)).cwiseMin(Vec2d(ngb_model->point_grid().width() - 1.001, ngb_model->point_grid().height() - 1.001));
        
        // The result of InterpolateBilinear() will not fit perfectly, but
        // should be fine as an initial state for optimization
        new_point_grid(x, y) = ngb_model->point_grid().InterpolateBilinear<Vec3d>(Vec2d(
            old_grid_point.x(),
            old_grid_point.y()));
        new_direction_grid(x, y) = ngb_model->direction_grid().InterpolateBilinear<Vec3d>(Vec2d(
            old_grid_point.x(),
            old_grid_point.y()));
      }
    }
    NoncentralGenericModel* new_ngb_model = new NoncentralGenericModel(
        target_resolution_x, target_resolution_y,
        calibration_min_x, calibration_min_y,
        calibration_max_x, calibration_max_y,
        model_to_optimize->width(), model_to_optimize->height());
    new_ngb_model->SetPointGrid(new_point_grid);
    new_ngb_model->SetDirectionGrid(new_direction_grid);
    model_to_optimize.reset(new_ngb_model);
    return true;
  }
  
  if (model_to_optimize->type() == CameraModel::Type::NoncentralGeneric) {
    LOG(ERROR) << "Camera model resampling with NoncentralGeneric as a source model type is not implemented for anything else than NoncentralGeneric as target type.";
    return false;
  }
  
  // Create dense direction model from old camera
  Image<Vec3d> dense_model(model_to_optimize->width(), model_to_optimize->height());
  for (u32 y = 0; y < dense_model.height(); ++ y) {
    for (u32 x = 0; x < dense_model.width(); ++ x) {
      if (!model_to_optimize->Unproject(x + 0.5, y + 0.5, &dense_model(x, y))) {
        dense_model(x, y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
      }
    }
  }
  
  // Fit new camera to the dense model
  const int calibration_area_width = calibration_max_x - calibration_min_x + 1;
  const int calibration_area_height = calibration_max_y - calibration_min_y + 1;
  
  // TODO: Try to avoid the need for a special case for each camera model here,
  //       probably use AllocateCameraModel().
  if (model_type == CameraModel::Type::CentralGeneric ||
      model_type == CameraModel::Type::NoncentralGeneric) {
    CentralGenericModel* new_cgbsp_model = new CentralGenericModel(
        target_resolution_x, target_resolution_y,
        calibration_min_x, calibration_min_y,
        calibration_max_x, calibration_max_y,
        model_to_optimize->width(), model_to_optimize->height());
    constexpr int kMaxXSamplesForFitting = 300;
    constexpr int kMaxYSamplesForFitting = 300;
    int subsample_step = std::max<int>(1, std::min(std::round(calibration_area_width / kMaxXSamplesForFitting),
                                                   std::round(calibration_area_height / kMaxYSamplesForFitting)));
    bool success = new_cgbsp_model->FitToDenseModel(dense_model, subsample_step, 3);
    if (success) {
      if (model_type == CameraModel::Type::NoncentralGeneric) {
        NoncentralGenericModel* new_ngbsp_model = new NoncentralGenericModel(
            target_resolution_x, target_resolution_y,
            calibration_min_x, calibration_min_y,
            calibration_max_x, calibration_max_y,
            model_to_optimize->width(), model_to_optimize->height());
        new_ngbsp_model->InitializeFromCentralGenericModel(*new_cgbsp_model);
        delete new_cgbsp_model;
        model_to_optimize.reset(new_ngbsp_model);
      } else {
        model_to_optimize.reset(new_cgbsp_model);
      }
    }
    return success;
  } else if (model_type == CameraModel::Type::CentralRadial) {
    CentralRadialModel* new_cr_model = new CentralRadialModel(
        model_to_optimize->width(), model_to_optimize->height(), kNumSplineParametersForRadialModel);
    constexpr int kMaxXSamplesForFitting = 300;
    constexpr int kMaxYSamplesForFitting = 300;
    int subsample_step = std::max<int>(1, std::min(std::round(calibration_area_width / kMaxXSamplesForFitting),
                                                   std::round(calibration_area_height / kMaxYSamplesForFitting)));
    bool success = new_cr_model->FitToDenseModel(dense_model, subsample_step, 15);
    if (success) {
      model_to_optimize.reset(new_cr_model);
    }
    return success;
  } else if (model_type == CameraModel::Type::CentralThinPrismFisheye) {
    CentralThinPrismFisheyeModel* new_ctpf_model = new CentralThinPrismFisheyeModel(
        model_to_optimize->width(), model_to_optimize->height(), true);
    constexpr int kMaxXSamplesForFitting = 300;
    constexpr int kMaxYSamplesForFitting = 300;
    int subsample_step = std::max<int>(1, std::min(std::round(calibration_area_width / kMaxXSamplesForFitting),
                                                   std::round(calibration_area_height / kMaxYSamplesForFitting)));
    Mat3d parametric_r_dense = Mat3d::Identity();
    bool success = new_ctpf_model->FitToDenseModel(dense_model, &parametric_r_dense, subsample_step, 3);
    if (success) {
      model_to_optimize.reset(new_ctpf_model);
      
      SE3d parametric_tr_dense;
      parametric_tr_dense.setRotationMatrix(parametric_r_dense);
      *camera_tr_rig = parametric_tr_dense * (*camera_tr_rig);
    }
    return success;
  } else if (model_type == CameraModel::Type::CentralOpenCV) {
    CentralOpenCVModel* new_cocv_model = new CentralOpenCVModel(
        model_to_optimize->width(), model_to_optimize->height());
    constexpr int kMaxXSamplesForFitting = 300;
    constexpr int kMaxYSamplesForFitting = 300;
    int subsample_step = std::max<int>(1, std::min(std::round(calibration_area_width / kMaxXSamplesForFitting),
                                                   std::round(calibration_area_height / kMaxYSamplesForFitting)));
    Mat3d parametric_r_dense = Mat3d::Identity();
    bool success = new_cocv_model->FitToDenseModel(dense_model, &parametric_r_dense, subsample_step, 3);
    if (success) {
      model_to_optimize.reset(new_cocv_model);
      
      SE3d parametric_tr_dense;
      parametric_tr_dense.setRotationMatrix(parametric_r_dense);
      *camera_tr_rig = parametric_tr_dense * (*camera_tr_rig);
    }
    return success;
  } else {
    LOG(ERROR) << "Resampling to this target model type is not implemented yet.";
    return false;
  }
}


/// Computes the parameter grid resolution for the model's calibrated image area,
/// depending on the desired approximate number of pixels per cell, and the
/// number of cells on each side that should be outside the image area. For
/// example, for B-Spline interpolation, there should be one more cell on each
/// side to provide a sufficient context size for interpolation (unless using
/// clamped access to the parameters).
void ComputeGridResolution(
    int calibration_area_width,
    int calibration_area_height,
    int exterior_cells_per_side,
    int approx_pixels_per_cell,
    int* resolution_x,
    int* resolution_y) {
  *resolution_x = calibration_area_width / approx_pixels_per_cell + 0.5f + 2 * exterior_cells_per_side;
  *resolution_y = calibration_area_height / approx_pixels_per_cell + 0.5f + 2 * exterior_cells_per_side;
}

void ComputeGridResolution(
    const CameraModel& model,
    int approx_pixels_per_cell,
    int* resolution_x,
    int* resolution_y) {
  int calibration_area_width = model.calibration_max_x() - model.calibration_min_x() + 1;
  int calibration_area_height = model.calibration_max_y() - model.calibration_min_y() + 1;
  int exterior_cells_per_side = 0;
  CameraModel::Type type = model.type();
  IDENTIFY_CAMERA_MODEL_TYPE(type, exterior_cells_per_side = _type::exterior_cells_per_side();)
  ComputeGridResolution(
      calibration_area_width,
      calibration_area_height,
      exterior_cells_per_side,
      approx_pixels_per_cell,
      resolution_x,
      resolution_y);
}


/// Given the grid resolution on the full-sized level 0, computes the grid
/// resolution for a smaller pyramid level.
// TODO: What is a good upsampling factor?
void CalcGridResolutionForLevel(int pyramid_level, int full_resolution_x, int full_resolution_y, int* x, int* y) {
  *x = static_cast<int>(full_resolution_x * pow(1.333, -pyramid_level) + 0.5f);
  *y = static_cast<int>(full_resolution_y * pow(1.333, -pyramid_level) + 0.5f);
};


void ResampleModelsIfNecessary(
    Dataset* dataset,
    BAState* state,
    CameraModel::Type model_type,
    int approx_pixels_per_cell,
    int pyramid_level) {
  for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
    shared_ptr<CameraModel>& model = state->intrinsics[camera_index];
    
    int loaded_grid_resolution_x, loaded_grid_resolution_y;
    bool have_grid = model->GetGridResolution(&loaded_grid_resolution_x, &loaded_grid_resolution_y);
    if (have_grid) {
      LOG(INFO) << "Grid resolution loaded from file: " << loaded_grid_resolution_x << " x " << loaded_grid_resolution_y;
    }
    
    int desired_full_grid_resolution_x, desired_full_grid_resolution_y;
    ComputeGridResolution(
        *model,
        approx_pixels_per_cell,
        &desired_full_grid_resolution_x,
        &desired_full_grid_resolution_y);
    LOG(INFO) << "Choosing grid resolution (for highest pyramid level): " << desired_full_grid_resolution_x << " x " << desired_full_grid_resolution_y;
    
    int desired_grid_resolution_x;
    int desired_grid_resolution_y;
    CalcGridResolutionForLevel(pyramid_level, desired_full_grid_resolution_x, desired_full_grid_resolution_y, &desired_grid_resolution_x, &desired_grid_resolution_y);
    
    // Re-sample in case the loaded grid resolution differs from the desired one,
    // or the loaded model type differs from the desired one.
    if ((have_grid && (desired_grid_resolution_x != loaded_grid_resolution_x ||
                       desired_grid_resolution_y != loaded_grid_resolution_y)) ||
        model->type() != model_type) {
      ResampleModel(
          model,
          &state->camera_tr_rig[camera_index],
          model->calibration_min_x(), model->calibration_min_y(),
          model->calibration_max_x(), model->calibration_max_y(),
          model_type,
          desired_grid_resolution_x, desired_grid_resolution_y);
    }
  }
}


void ComputeIntegerBoundingRectForFeatures(
    Dataset* dataset,
    int camera_index,
    const vector<bool>& image_used,
    int& calibration_min_x,
    int& calibration_min_y,
    int& calibration_max_x,
    int& calibration_max_y) {
  calibration_min_x = numeric_limits<int>::max();
  calibration_min_y = numeric_limits<int>::max();
  calibration_max_x = 0;
  calibration_max_y = 0;
  
  for (int i = 0; i < dataset->ImagesetCount(); ++ i) {
    if (!image_used[i]) {
      continue;
    }
    
    vector<PointFeature>& matches = dataset->GetImageset(i)->FeaturesOfCamera(camera_index);
    for (PointFeature& feature : matches) {
      calibration_min_x = std::min(calibration_min_x, static_cast<int>(feature.xy.x()));
      calibration_min_y = std::min(calibration_min_y, static_cast<int>(feature.xy.y()));
      calibration_max_x = std::max(calibration_max_x, static_cast<int>(feature.xy.x()));
      calibration_max_y = std::max(calibration_max_y, static_cast<int>(feature.xy.y()));
    }
  }
}


template <typename ModelT>
CameraModel* AllocateCameraModel(
    Dataset* dataset,
    int camera_index,
    const vector<bool>& image_used,
    int approx_pixels_per_cell,
    int num_pyramid_levels) {
  // Determine bounding rectangle of feature matches. The calibration will be
  // constrained to this rectangle to avoid mostly-unconstrained variables.
  int calibration_min_x, calibration_min_y;
  int calibration_max_x, calibration_max_y;
  ComputeIntegerBoundingRectForFeatures(
      dataset, camera_index, image_used,
      calibration_min_x, calibration_min_y,
      calibration_max_x, calibration_max_y);
  
  LOG(INFO) << "Area that will be calibrated for camera " << camera_index
            << ": (" << calibration_min_x << ", " << calibration_min_y
            << ") to (" << calibration_max_x << ", " << calibration_max_y << ")";
  
  int calibration_area_width = calibration_max_x - calibration_min_x + 1;
  int calibration_area_height = calibration_max_y - calibration_min_y + 1;
  
  // Fit the final model to the initialization.
  int camera_width = dataset->GetImageSize(camera_index).x();
  int camera_height = dataset->GetImageSize(camera_index).y();
  
  int grid_resolution_x, grid_resolution_y;
  ComputeGridResolution(
      calibration_area_width,
      calibration_area_height,
      ModelT::exterior_cells_per_side(),
      approx_pixels_per_cell,
      &grid_resolution_x,
      &grid_resolution_y);
  LOG(INFO) << "Choosing grid resolution (for highest pyramid level): " << grid_resolution_x << " x " << grid_resolution_y;
  
  
  int first_resolution_x;
  int first_resolution_y;
  CalcGridResolutionForLevel(num_pyramid_levels - 1, grid_resolution_x, grid_resolution_y, &first_resolution_x, &first_resolution_y);
  
  return new ModelT(
      first_resolution_x, first_resolution_y,
      calibration_min_x, calibration_min_y,
      calibration_max_x, calibration_max_y,
      camera_width, camera_height);
}

template <>
CameraModel* AllocateCameraModel<CentralRadialModel>(
    Dataset* dataset,
    int camera_index,
    const vector<bool>& /*image_used*/,
    int /*approx_pixels_per_cell*/,
    int /*num_pyramid_levels*/) {
  return new CentralRadialModel(
      dataset->GetImageSize(camera_index).x(),
      dataset->GetImageSize(camera_index).y(),
      kNumSplineParametersForRadialModel);
}

template <>
CameraModel* AllocateCameraModel<CentralThinPrismFisheyeModel>(
    Dataset* dataset,
    int camera_index,
    const vector<bool>& /*image_used*/,
    int /*approx_pixels_per_cell*/,
    int /*num_pyramid_levels*/) {
  // TODO: Make use_equidistant_projection configurable?
  return new CentralThinPrismFisheyeModel(
      dataset->GetImageSize(camera_index).x(),
      dataset->GetImageSize(camera_index).y(),
      /*use_equidistant_projection*/ true);
}

template <>
CameraModel* AllocateCameraModel<CentralOpenCVModel>(
    Dataset* dataset,
    int camera_index,
    const vector<bool>& /*image_used*/,
    int /*approx_pixels_per_cell*/,
    int /*num_pyramid_levels*/) {
  return new CentralOpenCVModel(
      dataset->GetImageSize(camera_index).x(),
      dataset->GetImageSize(camera_index).y());
}


template <typename ModelT>
bool FitCameraModelToDenseInitialization(
    int camera_index,
    const Image<Vec3d>& dense_model,
    ModelT* model,
    CalibrationWindow* calibration_window,
    bool step_by_step) {
  constexpr int kMaxXSamplesForFitting = 300;
  constexpr int kMaxYSamplesForFitting = 300;
  int calibration_area_width = model->calibration_max_x() - model->calibration_min_x() + 1;
  int calibration_area_height = model->calibration_max_y() - model->calibration_min_y() + 1;
  int subsample_step = std::max<int>(1, std::min(std::round(calibration_area_width / kMaxXSamplesForFitting),
                                                 std::round(calibration_area_height / kMaxYSamplesForFitting)));
  LOG(INFO) << "FitToDenseModel() ...";
  if (!model->FitToDenseModel(dense_model, subsample_step, 3)) {
    LOG(INFO) << "Calibration failed: Fitting the model to the dense calibration initialization failed.";
    return false;
  }
  
  if (calibration_window) {
    Image<Vec3u8> visualization;
    VisualizeModelDirections(*model, &visualization);
    calibration_window->SetCurrentCameraIndex(camera_index);
    calibration_window->UpdateObservationDirections(camera_index, visualization);
    
    if (step_by_step) {
      LOG(INFO) << "> Press Return to continue";
      GetKeyInput();
    }
  }
  
  return true;
}

template <>
bool FitCameraModelToDenseInitialization<NoncentralGenericModel>(
    int /*camera_index*/,
    const Image<Vec3d>& /*dense_model*/,
    NoncentralGenericModel* /*model*/,
    CalibrationWindow* /*calibration_window*/,
    bool /*step_by_step*/) {
  LOG(FATAL) << "FitCameraModelToDenseInitialization is not supported for camera model type NoncentralGenericModel.";
  return false;
}


bool InitializeBAStateFromDenseInitialization(
    Dataset* dataset,
    const DenseInitialization& dense,
    CameraModel::Type model_type,
    int approx_pixels_per_cell,
    int num_pyramid_levels,
    bool initialize_intrinsics,
    CalibrationWindow* calibration_window,
    bool step_by_step,
    BAState* state) {
  // Initialize state->image_used by using all imagesets that are used by at least
  // one camera in the dense initialization.
  state->image_used.clear();
  state->image_used.resize(dense.image_used[0].size(), false);
  for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
    for (int imageset_index = 0; imageset_index < dense.image_used[camera_index].size(); ++ imageset_index) {
      state->image_used[imageset_index] =
          state->image_used[imageset_index] ||
          dense.image_used[camera_index][imageset_index];
    }
  }
  
  // Initialize state->intrinsics by fitting to the dense observation directions.
  if (initialize_intrinsics) {
    state->intrinsics.resize(dataset->num_cameras());
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      CameraModel::Type final_type = model_type;
      if (model_type == CameraModel::Type::NoncentralGeneric) {
        final_type = model_type;
        model_type = CameraModel::Type::CentralGeneric;
      }
      
      IDENTIFY_CAMERA_MODEL_TYPE(model_type,
        state->intrinsics[camera_index].reset(AllocateCameraModel<_model_type>(
            dataset,
            camera_index,
            dense.image_used[camera_index],
            approx_pixels_per_cell,
            num_pyramid_levels));
      );
      
      CameraModel& model = *state->intrinsics[camera_index];
      IDENTIFY_CAMERA_MODEL(model,
        if (!FitCameraModelToDenseInitialization<_model_type>(
            camera_index,
            dense.observation_directions[camera_index],
            &_model,
            calibration_window,
            step_by_step)) {
          return false;
        }
      );
      
      if (final_type != model_type) {
        if (final_type == CameraModel::Type::NoncentralGeneric &&
            model_type == CameraModel::Type::CentralGeneric) {
          LOG(INFO) << "NoncentralGenericModel::InitializeFromCentralGenericModel() ...";
          NoncentralGenericModel* noncentral_model = new NoncentralGenericModel();
          noncentral_model->InitializeFromCentralGenericModel(*dynamic_cast<CentralGenericModel*>(state->intrinsics[camera_index].get()));
          state->intrinsics[camera_index].reset(noncentral_model);
          model_type = final_type;
        } else {
          LOG(FATAL) << "This combination of initial and final model is not supported.";
        }
      }
    }
  }
  
  // Initialize state->points to the known geometry.
  state->points.clear();
  state->feature_id_to_points_index.clear();
  for (usize k = 0; k < dense.known_geometry_localized.size(); ++ k) {
    if (!dense.known_geometry_localized[k]) {
      continue;
    }
    
    const auto& known_geometry = dataset->GetKnownGeometry(k);
    const auto& feature_id_to_position = known_geometry.feature_id_to_position;
    for (auto it = feature_id_to_position.cbegin(); it != feature_id_to_position.cend(); ++ it) {
      Vec3f pattern_point = Vec3f(known_geometry.cell_length_in_meters * it->second.x(), known_geometry.cell_length_in_meters * it->second.y(), 0);
      Vec3f global_point = dense.global_r_known_geometry[k] * pattern_point + dense.global_t_known_geometry[k];
      state->points.emplace_back(global_point.cast<double>());
      
      state->feature_id_to_points_index[it->first] = state->points.size() - 1;
    }
  }
  
  // Initialize state->camera_tr_rig by averaging the poses in the dense
  // initialization (which are determined individually, without rig constraints).
  // TODO: Would the use of a RANSAC process help here?
  constexpr int kRigCameraIndex = 0;
  
  vector<vector<SE3d>> camera_tr_rig(dataset->num_cameras());
  for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
    for (int imageset_index = 0; imageset_index < dense.num_imagesets(); ++ imageset_index) {
      if (dense.image_used[camera_index][imageset_index] && dense.image_used[kRigCameraIndex][imageset_index]) {
        camera_tr_rig[camera_index].push_back(
            dense.image_tr_global[camera_index][imageset_index] *
            dense.image_tr_global[kRigCameraIndex][imageset_index].inverse());
      }
    }
  }
  
  state->camera_tr_rig.resize(dataset->num_cameras());
  for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
    vector<SE3d>& this_camera_tr_rig = camera_tr_rig[camera_index];
    state->camera_tr_rig[camera_index] = AverageSE3(this_camera_tr_rig.size(), this_camera_tr_rig.data());
  }
  
  // Initialize state->rig_tr_global by using the previously estimated
  // camera_tr_rig transformations to let each image in an imageset vote for
  // where the rig pose should be, and average the result.
  // TODO: Would the use of a RANSAC process help here?
  vector<SE3d> rig_tr_camera(state->camera_tr_rig.size());
  for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
    rig_tr_camera[camera_index] = state->camera_tr_rig[camera_index].inverse();
  }
  
  vector<SE3d> rig_tr_global;
  state->rig_tr_global.resize(dense.num_imagesets());
  for (int imageset_index = 0; imageset_index < dense.num_imagesets(); ++ imageset_index) {
    rig_tr_global.clear();
    
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      if (dense.image_used[camera_index][imageset_index]) {
        rig_tr_global.push_back(
            rig_tr_camera[camera_index] *
            dense.image_tr_global[camera_index][imageset_index]
        );
      }
    }
    
    state->rig_tr_global[imageset_index] = AverageSE3(rig_tr_global.size(), rig_tr_global.data());
  }
  
  return true;
}


bool Calibrate(
    Dataset* dataset,
    const char* dense_initialization_path,
    const char* state_initialization_base_path,
    const char* outlier_visualization_path,
    bool use_cuda,
    SchurMode schur_mode,
    int num_pyramid_levels,
    CameraModel::Type model_type,
    int approx_pixels_per_cell,
    double regularization_weight,
    float outlier_removal_factor,
    bool localize_only,
    CalibrationWindow* calibration_window,
    BAState* state,
    const char* dataset_output_path,
    const char* state_output_path) {
  constexpr bool step_by_step = false;
  
  if (dataset->ImagesetCount() < 3) {
    LOG(INFO) << "Calibration failed: too few input images given (" << dataset->ImagesetCount()
              << "), calibration requires at least 3. (In practice, many more should be used.)";
    return false;
  }
  LOG(INFO) << "Calibrate() starting ...";
  
  
  // --- Load or estimate the dense initialization (if needed) ---
  DenseInitialization dense;
  
  if (!state_initialization_base_path) {  // skip if loading a state later
    if (dense_initialization_path && boost::filesystem::exists(dense_initialization_path)) {
      LOG(INFO) << "Loading dense initialization from: " << dense_initialization_path;
      if (!LoadDenseInitialization(dense_initialization_path, &dense)) {
        return false;
      }
    } else {
      for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
        LOG(INFO) << "Estimating the dense initialization for camera " << camera_index << " ...";
        if (calibration_window) {
          calibration_window->SetCurrentCameraIndex(camera_index);
        }
        if (!dense.InitializeCamera(dataset, camera_index, /*localize_only*/ false, calibration_window, step_by_step)) {
          return false;
        }
      }
      
      // Save the initialization such that the calibration process can be re-started
      // from here.
      if (dense_initialization_path) {
        LOG(INFO) << "Saving dense initialization to " << dense_initialization_path;
        if (!SaveDenseInitialization(dense_initialization_path, dense)) {
          LOG(ERROR) << "Could not save file: " << dense_initialization_path;
        }
      }
    }
  }
  
  
  // --- Load or estimate the initial optimization state ---
  if (state_initialization_base_path) {
    // Passing nullptr as dataset here since we are going to call ComputeFeatureIdToPointsIndex() later anyway.
    if (!LoadBAState(state_initialization_base_path, state, nullptr)) {
      return false;
    }
    if (localize_only) {
      // The model type does not need to be given as input for localize_only,
      // so get it from the loaded state.
      model_type = state->intrinsics[0]->type();
    }
    
    if (calibration_window) {
      Image<Vec3u8> visualization;
      for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
        VisualizeModelDirections(*state->intrinsics[camera_index], &visualization);
        calibration_window->UpdateObservationDirections(camera_index, visualization);
      }
    }
    
    // Do the loaded models need to be resampled?
    if (!localize_only) {
      ResampleModelsIfNecessary(dataset, state, model_type, approx_pixels_per_cell, num_pyramid_levels - 1);
    }
    
    if (localize_only) {
      // Convert the intrinsics to a dense model.
      dense.observation_directions.resize(dataset->num_cameras());
      for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
        if (!CreateObservationDirectionsImage(state->intrinsics[camera_index].get(), &dense.observation_directions[camera_index])) {
          LOG(FATAL) << "This camera model is not supported for localization.";
        }
        if (calibration_window) {
          calibration_window->SetCurrentCameraIndex(camera_index);
        }
        if (!dense.InitializeCamera(dataset, camera_index, /*localize_only*/ true, calibration_window, step_by_step)) {
          return false;
        }
      }
      
      vector<shared_ptr<CameraModel>> original_intrinsics = state->intrinsics;
      InitializeBAStateFromDenseInitialization(
          dataset, dense, model_type, approx_pixels_per_cell, num_pyramid_levels, /*initialize_intrinsics*/ false,
          calibration_window, step_by_step, state);
      state->intrinsics = original_intrinsics;
    }
  } else {
    InitializeBAStateFromDenseInitialization(
        dataset, dense, model_type, approx_pixels_per_cell, num_pyramid_levels, /*initialize_intrinsics*/ true,
        calibration_window, step_by_step, state);
  }
  
  
  // --- Use bundle adjustment (BA) to refine the state ---
  
  // Preparation for bundle adjustment: For each feature, store the index of the
  // observed point in the points vector for fast lookup.
  state->ComputeFeatureIdToPointsIndex(dataset);
  
  // Compute and store the grid resolutions on the final pyramid level.
  vector<pair<int, int>> full_grid_resolutions(state->num_cameras());
  for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
    int dummy, dummy2;
    if (state->intrinsics[camera_index]->GetGridResolution(&dummy, &dummy2)) {
      ComputeGridResolution(
          *state->intrinsics[camera_index],
          approx_pixels_per_cell,
          &full_grid_resolutions[camera_index].first,
          &full_grid_resolutions[camera_index].second);
    }
  }
  
  // Loop over bundle adjustment iterations for pyramid levels which are not full size
  for (int pyramid_level = num_pyramid_levels - 1; pyramid_level > 0 && !localize_only; -- pyramid_level) {
    LOG(INFO) << "Bundle adjustment with pyramid level: " << pyramid_level;
    
    // Print current grid resolutions and ensure that the camera models are
    // actually set to these resolutions.
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      if (full_grid_resolutions[camera_index].first >= 0) {
        int resolution_x, resolution_y;
        CalcGridResolutionForLevel(pyramid_level, full_grid_resolutions[camera_index].first, full_grid_resolutions[camera_index].second, &resolution_x, &resolution_y);
        int actual_resolution_x, actual_resolution_y;
        CHECK(state->intrinsics[camera_index]->GetGridResolution(&actual_resolution_x, &actual_resolution_y)) << "Camera model without GetGridResolution() used with pyramid scheme; set --num_pyramid_levels to 1";
        CHECK_EQ(resolution_x, actual_resolution_x);
        CHECK_EQ(resolution_y, actual_resolution_y);
        LOG(INFO) << "Grid resolution on pyramid level " << pyramid_level << " for camera " << camera_index << ": " << resolution_x << " x " << resolution_y;
      }
    }
    
    // Loosen the cost reduction threshold after a few iterations. Reasoning: We
    // want to make sure to do a few iterations, and want to continue doing them
    // as long as they make reasonable progress, but don't need to solve the
    // problem to an extreme precision as long as we are not on the final
    // pyramid level yet.
    RunBundleAdjustment(
        use_cuda, schur_mode, /*max_iteration_count*/ 10, /*cost_reduction_threshold*/ 0.0001, dataset, state, regularization_weight,
        localize_only, calibration_window, step_by_step, state_output_path);
    RunBundleAdjustment(
        use_cuda, schur_mode, /*max_iteration_count*/ 50, /*cost_reduction_threshold*/ 1, dataset, state, regularization_weight,
        localize_only, calibration_window, step_by_step, state_output_path);
    
    // Upsample the camera models to the next level
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      CameraModel& model = *state->intrinsics[camera_index];
      if (full_grid_resolutions[camera_index].first >= 0) {
        int target_resolution_x, target_resolution_y;
        CalcGridResolutionForLevel(pyramid_level - 1, full_grid_resolutions[camera_index].first, full_grid_resolutions[camera_index].second, &target_resolution_x, &target_resolution_y);
        ResampleModel(
            state->intrinsics[camera_index],
            &state->camera_tr_rig[camera_index],
            model.calibration_min_x(), model.calibration_min_y(),
            model.calibration_max_x(), model.calibration_max_y(),
            model_type,
            target_resolution_x, target_resolution_y);
      }
    }
  }
  
  // Print final grid resolutions
  if (!localize_only) {
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      if (full_grid_resolutions[camera_index].first >= 0) {
        LOG(INFO) << "Bundle adjustment with final grid resolution for camera " << camera_index << ": "
                  << full_grid_resolutions[camera_index].first << " x " << full_grid_resolutions[camera_index].second << " ...";
      }
    }
  }
  
  // Run some BA iterations and delete outliers?
  if (outlier_removal_factor > 0) {
    // Do a few initial BA iterations before doing outlier rejection.
    RunBundleAdjustment(
        use_cuda, schur_mode, /*max_iteration_count*/ (num_pyramid_levels == 1) ? 100 : 10, /*cost_reduction_threshold*/ 0.0001, dataset, state, regularization_weight,
        localize_only, calibration_window, step_by_step, state_output_path);
    
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      DeleteOutlierFeatures(camera_index, dataset, state, outlier_removal_factor, calibration_window, step_by_step, outlier_visualization_path);
    }
    
    if (dataset_output_path) {
      SaveDataset(dataset_output_path, *dataset);
    }
  }
  
  // Run main BA iterations.
  RunBundleAdjustment(
      use_cuda, schur_mode, /*max_iteration_count*/ 100, /*cost_reduction_threshold*/ 0.0001, dataset, state, regularization_weight,
      localize_only, calibration_window, step_by_step, state_output_path);
  
  // If we used CUDA, which has a PCG-based BA implementation that is less accurate
  // than the CPU implementation, finish up with some CPU iterations.
  if (use_cuda) {
    RunBundleAdjustment(
        /*use_cuda*/ false, schur_mode, /*max_iteration_count*/ 10, /*cost_reduction_threshold*/ 0.0001, dataset, state, regularization_weight,
        localize_only, calibration_window, step_by_step, state_output_path);
  }
  
  // Scale the result as good as possible.
  // If we only localize, we skip this. Otherwise, we would very likely *modify*
  // non-central cameras here, unless we get a perfect 1 as scaling factor.
  if (!localize_only) {
    ScaleToMetric(dataset, state);
  }
  
  return true;
}


void ExtractFeatures(
    const vector<string>& image_directories,
    FeatureDetectorTaggedPattern& detector,
    Dataset* dataset,
    CalibrationWindow* calibration_window) {
  // Find all images in the first folder and ensure that they are also in the other folders.
  // We assume that all folders have images with the same names in them since they should correspond.
  vector<string> filenames;
  unordered_set<string> filename_set;
  boost::filesystem::directory_iterator it(image_directories[0]), end;
  while (it != end) {
    filenames.push_back(it->path().filename().string());
    filename_set.insert(it->path().filename().string());
    ++ it;
  }
  sort(filenames.begin(), filenames.end());
  
  for (int camera_index = 1; camera_index < image_directories.size(); ++ camera_index) {
    int num_files = 0;
    boost::filesystem::directory_iterator it(image_directories[camera_index]);
    while (it != end) {
      if (filename_set.count(it->path().filename().string()) == 0) {
        LOG(ERROR) << "The file " << it->path().filename().string() << " is in the image folder with index " << camera_index << ", but not in the first image folder. The same number of files with matching filenames must be in each folder.";
        return;
      }
      ++ num_files;
      ++ it;
    }
    if (num_files != filenames.size()) {
      LOG(ERROR) << "The number of files in the first image folder and in the image folder with index " << camera_index << " differs. The same number of files with matching filenames must be in each folder.";
      return;
    }
  }
  
  LOG(INFO) << "Found " << filenames.size() << " images to extract features from.";
  
  for (int camera_index = 0; camera_index < image_directories.size(); ++ camera_index) {
    bool image_size_set = false;
    for (int imageset_index = 0; imageset_index < filenames.size(); ++ imageset_index) {
      const string& filename = filenames[imageset_index];
      
      string path = (boost::filesystem::path(image_directories[camera_index]) / filename).string();
      Image<Vec3u8> image(path);
      if (image.empty()) {
        LOG(ERROR) << "Cannot read image: " << path << ". Aborting.";
        return;
      }
      
      if (!image_size_set) {
        dataset->SetImageSize(camera_index, image.size());
      } else {
        if (image.size() != dataset->GetImageSize(camera_index).cast<u32>()) {
          LOG(ERROR) << "The size of image " << path << " (" << image.width() << " x " << image.height()
                    << ") does not match that of the previous images (" << dataset->GetImageSize(camera_index).x() << " x " << dataset->GetImageSize(camera_index).y() << "). Aborting.";
          return;
        }
      }
      
      shared_ptr<Imageset> imageset;
      if (camera_index == 0) {
        imageset = dataset->NewImageset();
      } else {
        imageset = dataset->GetImageset(imageset_index);
      }
      imageset->SetFilename(filename);
      vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
      Image<Vec3u8> detection_visualization;
      detector.DetectFeatures(image, &features, calibration_window ? &detection_visualization : nullptr);
      
      LOG(INFO) << path << ": " << features.size() << " features";
      
      if (calibration_window) {
        calibration_window->UpdateFeatureDetection(camera_index, detection_visualization);
      }
    }
  }
  
  // Delete empty imagesets.
  for (int imageset_index = static_cast<int>(filenames.size()) - 1; imageset_index >= 0; -- imageset_index) {
    bool is_empty = true;
    for (int camera_index = 0; camera_index < image_directories.size(); ++ camera_index) {
      if (!dataset->GetImageset(imageset_index)->FeaturesOfCamera(camera_index).empty()) {
        is_empty = false;
        break;
      }
    }
    
    if (is_empty) {
      dataset->DeleteImageset(imageset_index);
    }
  }
}


void CalibrateBatch(
    const vector<string>& image_directories,
    const vector<string>& dataset_files,
    const string& dense_initialization_base_path,
    const string& state_directory,
    const string& dataset_output_path,
    const string& state_output_directory,
    const string& pruned_dataset_output_path,
    const string& report_base_path,
    FeatureDetectorTaggedPattern* detector,
    int num_pyramid_levels,
    CameraModel::Type model_type,
    int cell_length_in_pixels,
    double regularization_weight,
    float outlier_removal_factor,
    bool localize_only,
    SchurMode schur_mode,
    CalibrationWindow* calibration_window) {
  Dataset dataset(image_directories.size());
  
  // Either create the dataset by extracting features from images, or by loading
  // it from a file.
  if (!image_directories.empty()) {
    if (calibration_window) {
      calibration_window->SetDataset(&dataset);
    }
    dataset.ExtractKnownGeometries(*detector);
    ExtractFeatures(image_directories, *detector, &dataset, calibration_window);
    
    if (!dataset_output_path.empty()) {
      SaveDataset(dataset_output_path.c_str(), dataset);
    }
  } else if (!dataset_files.empty()) {
    LOG(INFO) << "Dataset 0: " << dataset_files[0];
    if (!LoadDataset(dataset_files[0].c_str(), &dataset)) {
      return;
    }
    for (int i = 1; i < dataset_files.size(); ++ i) {
      Dataset additional_dataset(0);
      LOG(INFO) << "Dataset " << i << ": " << dataset_files[i];
      if (!LoadDataset(dataset_files[i].c_str(), &additional_dataset)) {
        return;
      }
      if (!dataset.Merge(additional_dataset)) {
        return;
      }
    }
    if (calibration_window) {
      calibration_window->SetDataset(&dataset);
    }
  } else {
    LOG(FATAL) << "Either image_directories or dataset_files must be non-empty.";
  }
  
  // If no calibration output was requested, do not perform calibration.
  if (state_output_directory.empty() &&
      report_base_path.empty()) {
    return;
  }
  
  constexpr bool use_cuda = false;  // TODO: make configurable once the CUDA version works well
  
  BAState calibration;
  if (!Calibrate(&dataset,
                 dense_initialization_base_path.empty() ? nullptr : dense_initialization_base_path.c_str(),
                 state_directory.empty() ? nullptr : state_directory.c_str(),
                 report_base_path.empty() ? nullptr : report_base_path.c_str(),
                 use_cuda,
                 schur_mode,
                 num_pyramid_levels,
                 model_type,
                 cell_length_in_pixels,
                 regularization_weight,
                 outlier_removal_factor,
                 localize_only,
                 calibration_window,
                 &calibration,
                 pruned_dataset_output_path.empty() ? nullptr : pruned_dataset_output_path.c_str(),
                 state_output_directory.empty() ? nullptr : state_output_directory.c_str())) {
    LOG(ERROR) << "Calibration failed.";
    return;
  }
  
  // Save the resulting calibration.
  if (!state_output_directory.empty()) {
    SaveBAState(state_output_directory.c_str(), calibration);
  }
  if (!pruned_dataset_output_path.empty()) {
    SaveDataset(pruned_dataset_output_path.c_str(), dataset);
  }
  
  // Create the calibration error report.
  if (!report_base_path.empty()) {
    CreateCalibrationReport(dataset, calibration, report_base_path);
  }
  
  // // Fit parametric models to see how well they fit.
  // CreateFittingVisualization(
  //     calibration,
  //     report_base_path,
  //     /*max_visualization_extent*/ -1,
  //     /*max_visualization_extent_pixels*/ -1);
}


int BatchCalibrationWithGUI(
    int argc,
    char** argv,
    const vector<string>& image_directories,
    const vector<string>& dataset_files,
    const string& dense_initialization_base_path,
    const string& state_directory,
    const string& dataset_output_path,
    const string& state_output_directory,
    const string& pruned_dataset_output_path,
    const string& report_base_path,
    FeatureDetectorTaggedPattern* detector,
    int num_pyramid_levels,
    CameraModel::Type model_type,
    int cell_length_in_pixels,
    double regularization_weight,
    float outlier_removal_factor,
    bool localize_only,
    SchurMode schur_mode,
    bool show_visualizations) {
  QApplication qapp(argc, argv);
  qapp.setQuitOnLastWindowClosed(false);
  
  // Create the main window.
  CalibrationWindow calibration_window(nullptr, Qt::WindowFlags());
  if (show_visualizations) {
    calibration_window.show();
    calibration_window.raise();
  }
  
  // Start the actual application in its own thread
  thread calibrate_thread([&]{
    CalibrateBatch(
        image_directories,
        dataset_files,
        dense_initialization_base_path,
        state_directory,
        dataset_output_path,
        state_output_directory,
        pruned_dataset_output_path,
        report_base_path,
        detector,
        num_pyramid_levels,
        model_type,
        cell_length_in_pixels,
        regularization_weight,
        outlier_removal_factor,
        localize_only,
        schur_mode,
        show_visualizations ? &calibration_window : nullptr);
    
    RunInQtThreadBlocking([&]() {
      if (calibration_window.isVisible()) {
        qapp.setQuitOnLastWindowClosed(true);
      } else {
        qapp.quit();
      }
    });
  });
  
  // Run the Qt event loop
  qapp.exec();
  
  calibrate_thread.join();
  return EXIT_SUCCESS;
}

}

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

#include <vector>

#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>

#include "camera_calibration/bundle_adjustment/joint_optimization.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"
#include "camera_calibration/models/camera_model.h"
#include "camera_calibration/models/central_opencv.h"
#include "camera_calibration/models/central_radial.h"
#include "camera_calibration/models/central_thin_prism_fisheye.h"
#include "camera_calibration/models/noncentral_generic.h"

namespace vis {

struct BAState;
class CalibrationWindow;
class FeatureDetectorTaggedPattern;

/// Main function for calibrating a camera rig.
/// Returns true if a valid calibration was obtained.
/// Attention: This function may modify the dataset by deleting outlier features!
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
    const char* dataset_output_path = nullptr,
    const char* state_output_path = nullptr);

/// Attempts to load a dataset, calls Calibrate() to perform the actual
/// calibration, and saves the results.
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
    CalibrationWindow* calibration_window);

/// Runs the calibration function (CalibrateBatch()) while providing the user
/// interface in a different thread.
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
    bool show_visualizations);

}

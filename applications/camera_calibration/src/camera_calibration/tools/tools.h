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

#include <string>
#include <vector>

#include <libvis/libvis.h>

namespace vis {

// These functions represent implementations of various small tools that are
// more or less related to the camera calibration, that can be called via the
// command line. See main.cc for some explanation on what they do.

int BundleAdjustment(const string& state_directory, const string& model_input_directory, const string& model_output_directory);

int CompareCalibrations(const string& calibration_a, const string& calibration_b, const string& report_base_path);

int ComparePointClouds(const string& stereo_directory_target, const string& stereo_directory_source, const string& output_directory);

int CompareReconstructions(const string& reconstruction_path_1, const string& reconstruction_path_2);
  
int ConvertDataset(const string& dataset_files, const string& output_path);

int CreateLegends();

int IntersectDatasets(const vector<string>& features_path, double intersection_threshold);

int LocalizationAccuracyTest(const char* gt_model_yaml_path, const char* compared_model_yaml_path);

int RenderSyntheticDataset(const char* binary_path, const string& path);

int StereoDepthEstimation(const string& state_base_path, const vector<string>& image_paths, const string& output_directory);

int VisualizeKalibrCalibration(const string& camchain_path);

int VisualizeColmapCalibration(const string& cameras_path);

}

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

#include <memory>
#include <unordered_map>
#include <vector>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>

namespace vis {

class CameraModel;
class Dataset;
struct BAState;

/// Creates an error report for the calibration on the given dataset.
int CreateCalibrationReport(
    const Dataset& dataset,
    const BAState& calibration,
    const string& report_base_path);

bool CreateCalibrationReportForCamera(
    const char* base_path,
    int camera_index,
    const Dataset& dataset,
    const BAState& calibration);

bool CreateCalibrationReportForData(
    const char* base_path,
    int camera_index,
    const int width,
    const int height,
    vector<Vec2d> const& reprojection_errors,
    vector<Vec2f> const& features);

void CreateReprojectionErrorHistogram(
    int camera_index,
    const Dataset& dataset,
    const BAState& state,
    Image<u8>* histogram_image);

void CreateReprojectionErrorDirectionVisualization(
    const Dataset& dataset,
    int camera_index,
    const BAState& calibration,
    Image<Vec3u8>* visualization);

void CreateReprojectionErrorMagnitudeVisualization(
    const Dataset& dataset,
    int camera_index,
    const BAState& calibration,
    double max_error,
    Image<Vec3u8>* visualization);

/// Visualizes the observation direction for each camera pixel.
/// For non-central cameras, ignores the line offset.
void VisualizeModelDirections(
    const CameraModel& model,
    Image<Vec3u8>* visualization);

}

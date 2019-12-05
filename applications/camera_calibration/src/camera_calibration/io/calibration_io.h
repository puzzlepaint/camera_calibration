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

#include <unordered_map>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>

#include "camera_calibration/models/camera_model.h"

namespace vis {

struct BAState;
class Dataset;
struct DatasetCalibration;
struct DenseInitialization;


/// Tries to save the dataset to the given path. Returns true if successful.
bool SaveDataset(
    const char* path,
    const Dataset& dataset);

/// Tries to load a dataset from the given path. Returns true if successful.
bool LoadDataset(
    const char* path,
    Dataset* dataset);


/// Tries to save the dense initialization to the given path. Returns true if successful.
bool SaveDenseInitialization(
    const char* path,
    const DenseInitialization& dense);

/// Tries to load a dense initialization from the given path. Returns true if successful.
bool LoadDenseInitialization(
    const char* path,
    DenseInitialization* dense);


/// Tries to save the BAState to the given path. Returns true if successful.
bool SaveBAState(
    const char* base_path,
    const BAState& state);

/// Tries to load a BAState from the given path. Returns true if successful.
/// If dataset is non-null, calls ComputeFeatureIdToPointsIndex() to set the index
/// attributes of the dataset's features according to the loaded point mapping.
bool LoadBAState(
    const char* base_path,
    BAState* state,
    Dataset* dataset);


/// Saves the camera model to a YAML file. Returns true if successful.
bool SaveCameraModel(
    const CameraModel& model,
    const char* path);

/// Loads a camera model from a YAML file. Returns true if successful.
shared_ptr<CameraModel> LoadCameraModel(
    const char* path);

/// Saves the poses to a YAML file. Returns true if successful.
bool SavePoses(
    const vector<bool>& image_used,
    const vector<SE3d>& image_tr_pattern,
    const char* path);

/// Loads the poses from a YAML file.. Returns true if successful.
bool LoadPoses(
    vector<bool>* image_used,
    vector<SE3d>* image_tr_pattern,
    const char* path);

/// Savest the pattern geometry to a YAML file. Returns true if successful.
bool SavePointsAndIndexMapping(
    const BAState& calibration,
    const char* path);

/// Loads a point and index mapping from a file. Returns true if successful.
bool LoadPointsAndIndexMapping(
    vector<Vec3d>* optimized_geometry,
    unordered_map<int, int>* feature_id_to_points_index,
    const char* path);

}

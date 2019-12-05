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
#include <libvis/libvis.h>
#include <libvis/sophus.h>

#include "camera_calibration/models/camera_model.h"

namespace vis {

class Dataset;

/// Represents an optimization state for bundle adjustment (BA).
struct BAState {
  /// Constructs an uninitialized state.
  BAState() = default;
  
  /// Constructs a deep copy of the other state.
  BAState(const BAState& other);
  BAState& operator= (const BAState& other);
  
  /// Scales the whole reconstruction represented by the state. The BA cost
  /// should remain constant when doing this.
  void ScaleState(double scaling_factor);
  
  /// Uses feature_id_to_points_index to assign the point index to
  /// each feature in the dataset (member PointFeature::index).
  void ComputeFeatureIdToPointsIndex(Dataset* dataset);
  
  inline int num_cameras() const { return intrinsics.size(); }
  inline int num_imagesets() const { return image_used.size(); }
  
  inline SE3d image_tr_global(int camera_index, int imageset_index) const {
    return camera_tr_rig[camera_index] * rig_tr_global[imageset_index];
  }
  
  
  // --- Constant members ---
  
  /// One entry for each imageset in the dataset.
  vector<bool> image_used;
  
  /// Indexing information for points.
  unordered_map<int, int> feature_id_to_points_index;
  
  
  // --- Optimized members ---
  
  /// Indexed by: [camera_index].
  vector<SE3d> camera_tr_rig;
  
  /// Camera rig poses (one for each imageset in the dataset). An entry i is
  /// only valid if image_used[i] is true.
  /// Indexed by: [imageset_index].
  vector<SE3d> rig_tr_global;
  
  /// Camera intrinsics.
  /// Indexed by: [camera_index].
  vector<shared_ptr<CameraModel>> intrinsics;
  
  /// Optimized pattern points.
  /// Indexed by PointFeature::index, respectively:
  /// feature_id_to_points_index[PointFeature::id].
  vector<Vec3d> points;
};

}

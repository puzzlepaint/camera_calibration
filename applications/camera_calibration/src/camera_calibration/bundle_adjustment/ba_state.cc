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

#include "camera_calibration/bundle_adjustment/ba_state.h"

#include "camera_calibration/dataset.h"

namespace vis {

BAState::BAState(const BAState& other)
    : image_used(other.image_used),
      feature_id_to_points_index(other.feature_id_to_points_index),
      camera_tr_rig(other.camera_tr_rig),
      rig_tr_global(other.rig_tr_global),
      points(other.points) {
  intrinsics.resize(other.intrinsics.size());
  for (usize i = 0; i < intrinsics.size(); ++ i) {
    intrinsics[i].reset(other.intrinsics[i]->duplicate());
  }
}

BAState& BAState::operator= (const BAState& other) {
  image_used = other.image_used;
  feature_id_to_points_index = other.feature_id_to_points_index;
  camera_tr_rig = other.camera_tr_rig;
  rig_tr_global = other.rig_tr_global;
  points = other.points;
  intrinsics.resize(other.intrinsics.size());
  for (usize i = 0; i < intrinsics.size(); ++ i) {
    intrinsics[i].reset(other.intrinsics[i]->duplicate());
  }
  return *this;
}

void BAState::ScaleState(double scaling_factor) {
  for (usize i = 0; i < camera_tr_rig.size(); ++ i) {
    camera_tr_rig[i].translation() *= scaling_factor;
  }
  
  for (usize i = 0; i < rig_tr_global.size(); ++ i) {
    rig_tr_global[i].translation() *= scaling_factor;
  }
  
  for (usize i = 0; i < points.size(); ++ i) {
    points[i] *= scaling_factor;
  }
  
  for (int camera_index = 0; camera_index < num_cameras(); ++ camera_index) {
    intrinsics[camera_index]->Scale(scaling_factor);
  }
}

void BAState::ComputeFeatureIdToPointsIndex(Dataset* dataset) {
  for (int camera_index = 0; camera_index < num_cameras(); ++ camera_index) {
    for (int i = 0; i < num_imagesets(); ++ i) {
      if (!image_used[i]) {
        continue;
      }
      
      vector<PointFeature>& matches = dataset->GetImageset(i)->FeaturesOfCamera(camera_index);
      for (PointFeature& feature : matches) {
        feature.index = feature_id_to_points_index[feature.id];
      }
    }
  }
}

}

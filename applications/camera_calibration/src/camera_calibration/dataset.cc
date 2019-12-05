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

#include "camera_calibration/dataset.h"

#include <libvis/logging.h>

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"

namespace vis {

Imageset::Imageset(int num_cameras) {
  m_features.resize(num_cameras);
}


Dataset::Dataset() {
  m_num_cameras = 0;
  image_sizes.resize(m_num_cameras, Vec2i::Constant(-1));
  first_imageset_indices_for_datasets = {0};
}

Dataset::Dataset(int num_cameras) {
  m_num_cameras = num_cameras;
  image_sizes.resize(m_num_cameras, Vec2i::Constant(-1));
  first_imageset_indices_for_datasets = {0};
}

Dataset::Dataset(const Dataset& other) {
  m_num_cameras = other.m_num_cameras;
  
  image_sizes = other.image_sizes;
  
  m_imagesets.resize(other.m_imagesets.size());
  for (usize i = 0; i < m_imagesets.size(); ++ i) {
    m_imagesets[i].reset(new Imageset(*other.m_imagesets[i]));
  }
  
  m_known_geometries = other.m_known_geometries;
  
  first_imageset_indices_for_datasets = other.first_imageset_indices_for_datasets;
}

void Dataset::Reset(int num_cameras) {
  m_num_cameras = num_cameras;
  image_sizes.clear();
  image_sizes.resize(num_cameras, Vec2i::Constant(-1));
  m_imagesets.clear();
  m_known_geometries.clear();
  first_imageset_indices_for_datasets = {0};
}

bool Dataset::Merge(const Dataset& other) {
  if (m_num_cameras != other.m_num_cameras) {
    return false;
  }
  for (int camera_index = 0; camera_index < m_num_cameras; ++ camera_index) {
    if (image_sizes[camera_index] != other.image_sizes[camera_index]) {
      return false;
    }
  }
  
  // Treating each known geometry from each dataset as different.
  // Find the number we have to add to new feature IDs to make them not overlap
  // with the existing feature IDs.
  int max_feature_id = 0;
  for (int k = 0; k < m_known_geometries.size(); ++ k) {
    for (const auto& feature_id_to_pos : m_known_geometries[k].feature_id_to_position) {
      max_feature_id = std::max(max_feature_id, feature_id_to_pos.first);
    }
  }
  int new_feature_id_offset = max_feature_id + 1;
  
  // Copy over known geometries while adding new_feature_id_offset.
  for (int k = 0; k < other.m_known_geometries.size(); ++ k) {
    const KnownGeometry& other_kg = other.m_known_geometries[k];
    
    m_known_geometries.emplace_back();
    KnownGeometry& new_kg = m_known_geometries.back();
    new_kg.cell_length_in_meters = other_kg.cell_length_in_meters;
    for (const auto& feature_id_to_pos : other_kg.feature_id_to_position) {
      new_kg.feature_id_to_position[feature_id_to_pos.first + new_feature_id_offset] = feature_id_to_pos.second;
    }
  }
  
  // Copy over imagesets while adding new_feature_id_offset.
  first_imageset_indices_for_datasets.push_back(m_imagesets.size());
  
  for (const shared_ptr<Imageset>& imageset : other.m_imagesets) {
    shared_ptr<Imageset> new_is = NewImageset();
    new_is->SetFilename(imageset->GetFilename());
    for (int camera_index = 0; camera_index < num_cameras(); ++ camera_index) {
      new_is->FeaturesOfCamera(camera_index) = imageset->FeaturesOfCamera(camera_index);
      for (PointFeature& feature : new_is->FeaturesOfCamera(camera_index)) {
        feature.id += new_feature_id_offset;
      }
    }
  }
  
  return true;
}

shared_ptr<Imageset> Dataset::NewImageset() {
  m_imagesets.emplace_back(new Imageset(m_num_cameras));
  return m_imagesets.back();
}

void Dataset::DeleteImageset(int index) {
  m_imagesets.erase(m_imagesets.begin() + index);
}

void Dataset::DeleteLastImageset() {
  m_imagesets.pop_back();
}

void Dataset::ExtractKnownGeometries(const FeatureDetectorTaggedPattern& detector) {
  SetKnownGeometriesCount(detector.GetPatternCount());
  for (int p = 0; p < detector.GetPatternCount(); ++ p) {
    KnownGeometry& pattern = GetKnownGeometry(p);
    pattern.cell_length_in_meters = detector.GetCellLengthInMeters();
    detector.GetCorners(p, &pattern.feature_id_to_position);
  }
}

}

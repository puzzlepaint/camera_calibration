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

#include <libvis/libvis.h>
#include <libvis/image.h>

#include "camera_calibration/dataset.h"
#include "camera_calibration/feature_detection/feature_refinement.h"

namespace vis {

/// Base class for feature detectors.
class FeatureDetector {
 public:
  virtual ~FeatureDetector() {}
  
  /// Sets the list of patterns that the detector should attempt to detect.
  /// Returns true if all given YAML files could be loaded successfully.
  virtual bool SetPatternYAMLPaths(
      const vector<string>& paths) = 0;
  
  /// Detects features in the given image. The resulting features are returned
  /// in "features". If detection_visualization is non-null, a visualization of
  /// the detected features will be returned in this image. window_extent
  /// corresponds to half the size of the search window for features. There is
  /// a tradeoff here: on the one hand, it should be as small as possible to be
  /// able to detect features close to the image borders, and to contain as
  /// little lens distortion as possible. On the other hand, it
  /// needs to be large enough to be able to detect features properly.
  /// Especially if corners are blurred, it is necessary to increase it.
  virtual void DetectFeatures(
      const Image<Vec3u8>& image,
      vector<PointFeature>* features,
      Image<Vec3u8>* detection_visualization) = 0;
  
  /// Returns the number of independent patterns that were detected or searched
  /// for. The geometry is known for each pattern, but the relative poses of
  /// different patterns to each other are not known.
  virtual int GetPatternCount() const = 0;
  
  /// Returns the corners within a given pattern. The valid range for
  /// pattern_index is: [0, GetPatternCount() - 1].
  virtual void GetCorners(
      int pattern_index,
      unordered_map<int, Vec2i>* feature_id_to_coord) const = 0;
  
  inline int GetFeatureWindowHalfExtent() const {
    return window_half_extent;
  }
  
  inline void SetFeatureWindowHalfExtent(int window_half_extent) {
    this->window_half_extent = window_half_extent;
  }
  
  inline float GetCellLengthInMeters() const {
    return cell_length_in_meters;
  }
  
  inline const vector<string>& GetPatternYAMLPaths() const {
    return pattern_yaml_paths;
  }
  
 protected:
  // These must be initialized by the derived class.
  int window_half_extent;
  FeatureRefinement refinement_type;
  float cell_length_in_meters;
  vector<string> pattern_yaml_paths;
};

}

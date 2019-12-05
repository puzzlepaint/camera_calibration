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

namespace vis {

class CameraModel;
class FeatureDetectorTaggedPattern;

/// Chunk of known geometry that is used for calibration.
/// For example, if printing three patterns on three sheets of paper, each
/// sheet would correspond to one instance of KnownGeometry since we know the
/// printed pattern (approximately). However, we do not know the relative poses
/// of the sheets.
struct KnownGeometry {
  /// Maps feature IDs to their known 2D integer position in the calibration target's checkerboard.
  unordered_map<int, Vec2i> feature_id_to_position;
  
  /// Side length of one checkerboard cell of the calibration target in meters.
  float cell_length_in_meters;
};

struct PointFeature {
  inline PointFeature() = default;
  
  template <typename T>
  inline PointFeature(const MatrixBase<T>& xy, int id)
      : xy(xy), id(id) {}
  
  /// Detected feature position in "pixel corner" image coordinate origin
  /// convention, i.e., the point (0, 0) is at the top-left corner of the
  /// top-left pixel in the image.
  Vec2f xy;
  
  /// Unique ID of this feature. This is used to associate features detected
  /// in different images with each other.
  int id;
  
  /// Sequential index of the estimated 3D point corresponding to this feature
  /// observation, which is used in bundle adjustment. This value is set by
  /// BAState::ComputeFeatureIdToPointsIndex(). It is just the cached result of
  /// looking up BAState::feature_id_to_points_index[PointFeature::id].
  int index = -1;
  
  /// The last projected position of the corresponding 3D point in bundle
  /// adjustment can be cached here. This can help speeding up the projection,
  /// since it can serve as a good initial estimate for the next time the
  /// projection is run.
  Vec2d last_projection = Vec2d::Zero();
};

/// A single image or set of images recorded by one or more cameras at the same
/// time.
class Imageset {
 public:
  Imageset(int num_cameras);
  
  Imageset(const Imageset& other) = default;
  
  inline const vector<PointFeature>& FeaturesOfCamera(int camera_index) const {
    return m_features[camera_index];
  }
  inline vector<PointFeature>& FeaturesOfCamera(int camera_index) {
    return m_features[camera_index];
  }
  
  inline bool CameraHasFeatures(int camera_index) const {
    return !m_features[camera_index].empty();
  }
  
  /// Returns the image filename. Note that the filename is equal for all
  /// cameras; the files are in separate folders.
  const string& GetFilename() const {
    return filename;
  }
  
  void SetFilename(const string& new_filename) {
    filename = new_filename;
  }
  
 private:
  /// Indexed by: [camera_index][feature_index] .
  vector<vector<PointFeature>> m_features;
  
  string filename;
};

/// Stores a collection of feature detections, extracted from images of one or
/// more cameras. Features are grouped into Imagesets. Each set contains
/// features of images that were recorded with consistent relative poses between
/// the different cameras (with one image per camera). I.e., for a video
/// taken with a camera rig, an Imageset would correspond to the images taken
/// at one point in time.
class Dataset {
 public:
  Dataset();
  Dataset(int num_cameras);
  
  Dataset(const Dataset& other);
  
  void Reset(int num_cameras);
  
  /// Merges the other dataset into this one. This assumes that both dataset
  /// contain the same cameras in the same order.
  bool Merge(const Dataset& other);
  
  template <typename T>
  inline void SetImageSize(int camera_index, const MatrixBase<T>& size) {
    image_sizes[camera_index] = size.template cast<int>();
  }
  
  const Vec2i& GetImageSize(int camera_index) const {
    return image_sizes[camera_index];
  }
  
  shared_ptr<Imageset> NewImageset();
  void DeleteImageset(int index);
  void DeleteLastImageset();
  
  inline shared_ptr<const Imageset> GetImageset(int index) const {
    return m_imagesets[index];
  }
  inline shared_ptr<Imageset> GetImageset(int index) {
    return m_imagesets[index];
  }
  
  inline int ImagesetCount() const {
    return m_imagesets.size();
  }
  
  inline int KnownGeometriesCount() const {
    return m_known_geometries.size();
  }
  
  inline void SetKnownGeometriesCount(int count) {
    m_known_geometries.resize(count);
  }
  
  inline KnownGeometry& GetKnownGeometry(int index) {
    return m_known_geometries[index];
  }
  
  inline const KnownGeometry& GetKnownGeometry(int index) const {
    return m_known_geometries[index];
  }
  
  /// Extracts the known calibration pattern geometries from the feature detector
  /// and puts them as known geometries into the dataset.
  void ExtractKnownGeometries(const FeatureDetectorTaggedPattern& detector);
  
  inline int SourceDatasetsCount() const {
    return first_imageset_indices_for_datasets.size();
  }
  
  inline int FirstImagesetForDataset(int dataset_index) const {
    return first_imageset_indices_for_datasets[dataset_index];
  }
  
  inline int num_cameras() const { return m_num_cameras; }
  
 private:
  int m_num_cameras;
  
  /// Indexed by: [camera_index] .
  vector<Vec2i> image_sizes;
  
  /// Indexed by: [imageset_index] . There is no particular ordering.
  vector<shared_ptr<Imageset>> m_imagesets;
  
  /// Indexed by: [known_geometry_index] . There is no particular ordering.
  vector<KnownGeometry> m_known_geometries;
  
  /// Metadata: If the dataset is created by merging multiple datasets from disk,
  /// this can be used to find which imagesets come from which dataset. The vector
  /// has the same number of entries as the number of datasets that were merged,
  /// and each entry is the index of the first imageset corresponding to the dataset.
  vector<int> first_imageset_indices_for_datasets;
};

}

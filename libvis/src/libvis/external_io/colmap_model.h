// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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
#include <unordered_set>
#include <vector>

#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>

namespace vis {

// Holds data of a COLMAP camera.
struct ColmapCamera {
  // Unique camera id.
  int camera_id;
  
  // Name of the distortion model. Determines the number of parameters.
  string model_name;
  
  // Image width in pixels.
  int width;
  
  // Image height in pixels.
  int height;
  
  // Distortion parameters. Their number and interpretation depends on the
  // distortion model. Colmap uses the "pixel-corner" coordinate convention.
  vector<double> parameters;
};

typedef shared_ptr<ColmapCamera> ColmapCameraPtr;
typedef shared_ptr<const ColmapCamera> ColmapCameraConstPtr;

typedef vector<ColmapCameraPtr> ColmapCameraPtrVector;
// Indexed by: [camera_id] .
typedef unordered_map<int, ColmapCameraPtr> ColmapCameraPtrMap;


struct ColmapFeatureObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Sub-pixel coordinates of the observation in its image, given in pixels.
  Eigen::Vector2f xy;
  
  // Id of the corresponding 3D point or -1 if no 3D point is associated to this
  // observation.
  int point3d_id;
};

// Holds data of a COLMAP image.
struct ColmapImage {
  // Unique image id.
  int image_id;
  
  // Id of the camera model for this image.
  int camera_id;
  
  // Path to the image file, may be a relative path.
  string file_path;
  
  // Global-to-image transformation.
  Sophus::SE3f image_tr_global;
  
  // Image-to-global transformation.
  Sophus::SE3f global_tr_image;
  
  // 2D feature observations in this image.
  vector<ColmapFeatureObservation, Eigen::aligned_allocator<ColmapFeatureObservation>> observations;
};

typedef shared_ptr<ColmapImage> ColmapImagePtr;
typedef shared_ptr<const ColmapImage> ColmapImageConstPtr;

typedef vector<ColmapImagePtr> ColmapImagePtrVector;
// Indexed by [colmap_image_id] .
typedef unordered_map<int, ColmapImagePtr> ColmapImagePtrMap;


// Holds data for a camera within a COLMAP rig.
struct ColmapRigCamera {
  // Camera ID.
  int camera_id;
  
  // Prefix to recognize images of this rig camera.
  string image_prefix;
};

// Holds data of a COLMAP rig.
struct ColmapRig {
  // Reference camera ID.
  int ref_camera_id;
  
  // List of cameras attached to this rig.
  vector<ColmapRigCamera> cameras;
};

typedef vector<ColmapRig> ColmapRigVector;


/// Holds data of a single point in a COLMAP model.
struct ColmapPoint3D {
  /// ID of this 3D point.
  u64 id;
  
  /// Position of this point.
  Vec3f position;
  
  /// Color of this point.
  Vec3u8 color;
  
  /// Error of this point as given in the file.
  double error;
  
  /// Array of (image id, observation index).
  std::vector<pair<int, u32>> track;
};

typedef unordered_map<u64, ColmapPoint3D> ColmapPoint3DMap;


// Loads ColmapCameraPtr from a COLMAP cameras.txt file and appends
// them to the cameras map (indexed by camera_id). Returns true if successful.
bool ReadColmapCameras(const string& cameras_txt_path,
                       ColmapCameraPtrMap* cameras);

bool WriteColmapCameras(const string& cameras_txt_path,
                        const ColmapCameraPtrMap& cameras);

// Loads ColmapImagePtr from a COLMAP images.txt file and appends them
// to the images map (indexed by image_id). Returns true if successful.
bool ReadColmapImages(const string& images_txt_path,
                      bool read_observations,
                      ColmapImagePtrMap* images);

bool WriteColmapImages(const string& images_txt_path,
                       const ColmapImagePtrMap& images);

// Loads ColmapRigVector from a COLMAP rigs JSON file and appends them to
// the given rigs list. Returns true if successful.
bool ReadColmapRigs(const string& rigs_json_path,
                    ColmapRigVector* rigs);

bool WriteColmapRigs(const string& rigs_json_path,
                     const ColmapRigVector& rigs);

// Loads ColmapPoints3D from a COLMAP points3D.txt file and appends them
// to the given point map. Returns true if successful.
// TODO: This currently does not load the point tracks!
bool ReadColmapPoints3D(const string& points3d_txt_path,
                        ColmapPoint3DMap* points);

bool WriteColmapPoints3D(const string& points3d_txt_path,
                         const ColmapPoint3DMap& points);

}

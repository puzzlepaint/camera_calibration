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

#include "camera_calibration/io/calibration_io.h"

#include <fstream>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <QDir>
#include <QFileInfo>
#include <yaml-cpp/yaml.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/calibration_initialization/dense_initialization.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/io/io_util.h"
#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/models/central_opencv.h"
#include "camera_calibration/models/central_radial.h"
#include "camera_calibration/models/central_thin_prism_fisheye.h"
#include "camera_calibration/models/noncentral_generic.h"
#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"

namespace vis {

bool SaveDataset(const char* path, const Dataset& dataset) {
  QFileInfo(path).dir().mkpath(".");
  
  FILE* file = fopen(path, "wb");
  if (!file) {
    return false;
  }
  
  // File format identifier
  u8 header[10];
  header[0] = 'c';
  header[1] = 'a';
  header[2] = 'l';
  header[3] = 'i';
  header[4] = 'b';
  header[5] = '_';
  header[6] = 'd';
  header[7] = 'a';
  header[8] = 't';
  header[9] = 'a';
  fwrite(header, 1, 10, file);
  
  // File format version
  const u32 version = 0;
  write_one(&version, file);
  
  // Cameras
  u32 num_cameras = dataset.num_cameras();
  write_one(&num_cameras, file);
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    const Vec2i& image_size = dataset.GetImageSize(camera_index);
    u32 width = image_size.x();
    write_one(&width, file);
    u32 height = image_size.y();
    write_one(&height, file);
  }
  
  // Imagesets
  u32 num_imagesets = dataset.ImagesetCount();
  write_one(&num_imagesets, file);
  for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
    shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
    
    const string& filename = imageset->GetFilename();
    u32 filename_len = filename.size();
    write_one(&filename_len, file);
    fwrite(filename.data(), 1, filename_len, file);
    
    for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
      const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
      
      u32 num_features = features.size();
      write_one(&num_features, file);
      for (const PointFeature& feature : features) {
        write_one(&feature.xy.x(), file);
        write_one(&feature.xy.y(), file);
        i32 id_32bit = feature.id;
        write_one(&id_32bit, file);
      }
    }
  }
  
  // Known geometries
  u32 num_known_geometries = dataset.KnownGeometriesCount();
  write_one(&num_known_geometries, file);
  for (int geometry_index = 0; geometry_index < dataset.KnownGeometriesCount(); ++ geometry_index) {
    const KnownGeometry& geometry = dataset.GetKnownGeometry(geometry_index);
    
    write_one(&geometry.cell_length_in_meters, file);
    
    u32 feature_id_to_position_size = geometry.feature_id_to_position.size();
    write_one(&feature_id_to_position_size, file);
    for (const std::pair<int, Vec2i>& item : geometry.feature_id_to_position) {
      i32 id_32bit = item.first;
      write_one(&id_32bit, file);
      i32 x_32bit = item.second.x();
      write_one(&x_32bit, file);
      i32 y_32bit = item.second.y();
      write_one(&y_32bit, file);
    }
  }
  
  fclose(file);
  return true;
}

std::string strip_slashes(std::string const& src) {
    std::string result;
    for (char c : src) {
        if (c == '/') {
            result += '-';
        }
        else {
            result += c;
        }
    }
    return result;
}

bool SaveDatasetAndState(const char* path, const Dataset& dataset, const BAState& state) {
  QFileInfo(path).dir().mkpath(".");

  FILE* file = fopen(path, "wb");
  if (!file) {
    return false;
  }

  // File format identifier
  u8 header[10];
  header[0] = 'c';
  header[1] = 'a';
  header[2] = 'l';
  header[3] = 'i';
  header[4] = 'b';
  header[5] = '_';
  header[6] = 'd';
  header[7] = 'a';
  header[8] = 't';
  header[9] = 'a';
  fwrite(header, 1, 10, file);

  // File format version
  const u32 version = 0;
  write_one(&version, file);

  // Cameras
  u32 num_cameras = dataset.num_cameras();
  write_one(&num_cameras, file);
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    const Vec2i& image_size = dataset.GetImageSize(camera_index);
    u32 width = image_size.x();
    write_one(&width, file);
    u32 height = image_size.y();
    write_one(&height, file);
  }

  // Imagesets
  u32 num_imagesets = dataset.ImagesetCount();
  write_one(&num_imagesets, file);
  for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
    shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);

    const string& filename = imageset->GetFilename();
    u32 filename_len = filename.size();
    write_one(&filename_len, file);
    fwrite(filename.data(), 1, filename_len, file);

    for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
      const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);

      const SE3d& image_tr_global = state.camera_tr_rig[camera_index] * state.rig_tr_global[imageset_index];
      Quaterniond image_q_global = image_tr_global.unit_quaternion();
      Mat3d image_r_global = image_tr_global.rotationMatrix().cast<double>();
      const Vec3d& image_t_global = image_tr_global.translation();

      std::string const points_filename = std::string(path) + "-" + strip_slashes(filename) + "-" + std::to_string(camera_index) + ".yaml";
      std::ofstream points_file(points_filename);
      points_file << "%YAML:1.0" << std::endl
                  << "---" << std::endl
                  << "correspondences:" << std::endl;

      LOG(INFO) << "Saving points at " << points_filename;

      u32 num_features = features.size();
      write_one(&num_features, file);
      for (const PointFeature& feature : features) {
        write_one(&feature.xy.x(), file);
        write_one(&feature.xy.y(), file);
        i32 id_32bit = feature.id;
        write_one(&id_32bit, file);

        const Vec3d& point = state.points[feature.index];
        Vec3d local_point = image_r_global * point + image_t_global;
        points_file << "   -" << std::endl
                    << "      id: " << feature.id << std::endl
                    << "      point: [" << local_point[0] << ", " << local_point[1] << ", " << local_point[1] << "]" << std::endl;
      }
    }
  }

  // Known geometries
  u32 num_known_geometries = dataset.KnownGeometriesCount();
  write_one(&num_known_geometries, file);
  for (int geometry_index = 0; geometry_index < dataset.KnownGeometriesCount(); ++ geometry_index) {
    const KnownGeometry& geometry = dataset.GetKnownGeometry(geometry_index);

    write_one(&geometry.cell_length_in_meters, file);

    u32 feature_id_to_position_size = geometry.feature_id_to_position.size();
    write_one(&feature_id_to_position_size, file);
    for (const std::pair<int, Vec2i>& item : geometry.feature_id_to_position) {
      i32 id_32bit = item.first;
      write_one(&id_32bit, file);
      i32 x_32bit = item.second.x();
      write_one(&x_32bit, file);
      i32 y_32bit = item.second.y();
      write_one(&y_32bit, file);
    }
  }

  fclose(file);
  return true;
}

bool LoadDataset(const char* path, Dataset* dataset) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    LOG(ERROR) << "Cannot read file: " << path;
    return false;
  }
  
  // File format identifier
  u8 header[10];
  if (fread(header, 1, 10, file) != 10 ||
      header[0] != 'c' ||
      header[1] != 'a' ||
      header[2] != 'l' ||
      header[3] != 'i' ||
      header[4] != 'b' ||
      header[5] != '_' ||
      header[6] != 'd' ||
      header[7] != 'a' ||
      header[8] != 't' ||
      header[9] != 'a') {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "Invalid file header.";
    fclose(file);
    return false;
  }
  
  // File format version
  u32 version;
  read_one(&version, file);
  if (version != 0) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "Unsupported file format version.";
    fclose(file);
    return false;
  }
  
  // Cameras
  u32 num_cameras;
  read_one(&num_cameras, file);
  LOG(INFO) << "Number of cameras: " << num_cameras;
  dataset->Reset(num_cameras);
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    u32 width;
    read_one(&width, file);
    u32 height;
    read_one(&height, file);
    dataset->SetImageSize(camera_index, Vec2i(width, height));
    LOG(INFO) << "Width and height of camera #: " << camera_index << " " << width << ", " << height;
  }
  
  // Imagesets
  u32 num_imagesets;
  u32 total_num_features = 0;
  read_one(&num_imagesets, file);
  LOG(INFO) << "Number of imagesets: " << num_imagesets;
  for (int imageset_index = 0; imageset_index < num_imagesets; ++ imageset_index) {
    shared_ptr<Imageset> new_imageset = dataset->NewImageset();
    
    u32 filename_len;
    read_one(&filename_len, file);
    string filename;
    filename.resize(filename_len);
    if (fread(&filename[0], 1, filename_len, file) != filename_len) {
      fclose(file);
      LOG(ERROR) << "Unexpected end of file: " << path;
      return false;
    }
    new_imageset->SetFilename(filename);
    
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      vector<PointFeature>& features = new_imageset->FeaturesOfCamera(camera_index);
      
      u32 num_features;
      read_one(&num_features, file);
      total_num_features += num_features;
      features.resize(num_features);
      for (PointFeature& feature : features) {
        read_one(&feature.xy.x(), file);
        read_one(&feature.xy.y(), file);
        i32 id_32bit;
        read_one(&id_32bit, file);
        feature.id = id_32bit;
      }
      LOG(INFO) << "Number of features in image: " << filename << ": " << num_features;
    }
  }
  
  // Known geometries
  u32 num_known_geometries;
  read_one(&num_known_geometries, file);
  dataset->SetKnownGeometriesCount(num_known_geometries);
  LOG(INFO) << "Number of known geometries: " << num_known_geometries;
  for (u32 geometry_index = 0; geometry_index < num_known_geometries; ++ geometry_index) {
    KnownGeometry& geometry = dataset->GetKnownGeometry(geometry_index);
    
    read_one(&geometry.cell_length_in_meters, file);
    
    u32 feature_id_to_position_size;
    read_one(&feature_id_to_position_size, file);
    for (u32 i = 0; i < feature_id_to_position_size; ++ i) {
      i32 id_32bit;
      read_one(&id_32bit, file);
      i32 x_32bit;
      read_one(&x_32bit, file);
      i32 y_32bit;
      read_one(&y_32bit, file);
      
      geometry.feature_id_to_position[id_32bit] = Vec2i(x_32bit, y_32bit);
    }
  }
  
  fclose(file);
  LOG(INFO) << "Loaded dataset with " << total_num_features << " feature observations.";
  return true;
}


bool SaveDenseInitialization(
    const char* path,
    const DenseInitialization& dense) {
  QFileInfo(path).dir().mkpath(".");
  
  FILE* file = fopen(path, "wb");
  if (!file) {
    LOG(ERROR) << "Cannot write file: " << path;
    return false;
  }
  
  // File format identifier
  u8 header[10];
  header[0] = 'c';
  header[1] = 'a';
  header[2] = 'l';
  header[3] = 'i';
  header[4] = 'b';
  header[5] = '_';
  header[6] = 'i';
  header[7] = 'n';
  header[8] = 'i';
  header[9] = 't';
  fwrite(header, 1, 10, file);
  
  // File format version
  const u32 version = 0;
  write_one(&version, file);
  
  // Geometry properties
  u32 num_known_geometries = dense.known_geometry_localized.size();
  write_one(&num_known_geometries, file);
  
  for (int g = 0; g < num_known_geometries; ++ g) {
    u8 localized = dense.known_geometry_localized[g];
    write_one(&localized, file);
    
    fwrite(dense.global_r_known_geometry[g].data(), sizeof(float), 9, file);
    fwrite(dense.global_t_known_geometry[g].data(), sizeof(float), 3, file);
  }
  
  // Image properties
  u32 num_cameras = dense.image_used.size();
  write_one(&num_cameras, file);
  
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    u32 num_images = dense.image_used[camera_index].size();
    write_one(&num_images, file);
    
    for (int image_index = 0; image_index < num_images; ++ image_index) {
      u8 image_used = dense.image_used[camera_index][image_index];
      write_one(&image_used, file);
      
      fwrite(dense.image_tr_global[camera_index][image_index].data(), sizeof(double), SE3d::num_parameters, file);
    }
  }
  
  // Dense models
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    u32 width = dense.observation_directions[camera_index].width();
    write_one(&width, file);
    u32 height = dense.observation_directions[camera_index].height();
    write_one(&height, file);
    
    fwrite(dense.observation_directions[camera_index].data(),
           sizeof(Vec3d),
           width * height,
           file);
  }
  
  fclose(file);
  return true;
}

bool LoadDenseInitialization(
    const char* path,
    DenseInitialization* dense) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    LOG(ERROR) << "Cannot read file: " << path;
    return false;
  }
  
  // File format identifier
  u8 header[10];
  if (fread(header, 1, 10, file) != 10 ||
      header[0] != 'c' ||
      header[1] != 'a' ||
      header[2] != 'l' ||
      header[3] != 'i' ||
      header[4] != 'b' ||
      header[5] != '_' ||
      header[6] != 'i' ||
      header[7] != 'n' ||
      header[8] != 'i' ||
      header[9] != 't') {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "Invalid file header.";
    fclose(file);
    return false;
  }
  
  // File format version
  u32 version;
  read_one(&version, file);
  if (version != 0) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "Unsupported file format version.";
    fclose(file);
    return false;
  }
  
  // Geometry properties
  u32 num_known_geometries;
  read_one(&num_known_geometries, file);
  dense->known_geometry_localized.resize(num_known_geometries);
  dense->global_r_known_geometry.resize(num_known_geometries);
  dense->global_t_known_geometry.resize(num_known_geometries);
  
  for (int g = 0; g < num_known_geometries; ++ g) {
    u8 localized;
    read_one(&localized, file);
    dense->known_geometry_localized[g] = localized;
    
    int n = fread(dense->global_r_known_geometry[g].data(), sizeof(float), 9, file);
    n += fread(dense->global_t_known_geometry[g].data(), sizeof(float), 3, file);
    if (n != 9 + 3) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Unexpected end of file.";
      fclose(file);
      return false;
    }
  }
  
  // Image properties
  u32 num_cameras;
  read_one(&num_cameras, file);
  dense->image_used.resize(num_cameras);
  dense->image_tr_global.resize(num_cameras);
  
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    u32 num_images;
    read_one(&num_images, file);
    dense->image_used[camera_index].resize(num_images);
    dense->image_tr_global[camera_index].resize(num_images);
    
    for (int image_index = 0; image_index < num_images; ++ image_index) {
      u8 image_used;
      read_one(&image_used, file);
      dense->image_used[camera_index][image_index] = image_used;
      
      if (fread(dense->image_tr_global[camera_index][image_index].data(), sizeof(double), SE3d::num_parameters, file) != SE3d::num_parameters) {
        LOG(ERROR) << "Cannot parse file: " << path;
        LOG(ERROR) << "Unexpected end of file.";
        fclose(file);
        return false;
      }
    }
  }
  
  // Dense models
  dense->observation_directions.resize(num_cameras);
  
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    u32 width;
    read_one(&width, file);
    u32 height;
    read_one(&height, file);
    dense->observation_directions[camera_index].SetSize(width, height);
    
    if (fread(dense->observation_directions[camera_index].data(), sizeof(Vec3d), width * height, file) != width * height) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Unexpected end of file.";
      fclose(file);
      return false;
    }
  }
  
  fclose(file);
  return true;
}


bool SaveBAState(
    const char* base_path,
    const BAState& state) {
  QDir(base_path).mkpath(".");
  
  string rig_tr_global_path = (boost::filesystem::path(base_path) / "rig_tr_global.yaml").string();
  if (!SavePoses(state.image_used, state.rig_tr_global, rig_tr_global_path.c_str())) {
    return false;
  }
  
  string camera_tr_rig_path = (boost::filesystem::path(base_path) / "camera_tr_rig.yaml").string();
  vector<bool> dummy(state.camera_tr_rig.size(), true);
  if (!SavePoses(dummy, state.camera_tr_rig, camera_tr_rig_path.c_str())) {
    return false;
  }
  
  for (int camera_index = 0; camera_index < state.num_cameras(); ++ camera_index) {
    ostringstream filename;
    filename << "intrinsics" << camera_index << ".yaml";
    string model_path = (boost::filesystem::path(base_path) / filename.str()).string();
    
    if (!SaveCameraModel(*state.intrinsics[camera_index], model_path.c_str())) {
      return false;
    }
  }
  
  string pattern_path = (boost::filesystem::path(base_path) / "points.yaml").string();
  if (!SavePointsAndIndexMapping(state, pattern_path.c_str())) {
    return false;
  }
  
  return true;
}

bool LoadBAState(
    const char* base_path,
    BAState* state,
    Dataset* dataset) {
  string rig_tr_global_path = (boost::filesystem::path(base_path) / "rig_tr_global.yaml").string();
  if (!LoadPoses(
      &state->image_used,
      &state->rig_tr_global,
      rig_tr_global_path.c_str())) {
    LOG(ERROR) << "Cannot load file: " << rig_tr_global_path;
    return false;
  }
  
  string camera_tr_rig_path = (boost::filesystem::path(base_path) / "camera_tr_rig.yaml").string();
  vector<bool> dummy;
  if (!LoadPoses(
      &dummy,
      &state->camera_tr_rig,
      camera_tr_rig_path.c_str())) {
    LOG(ERROR) << "Cannot load file: " << camera_tr_rig_path;
    return false;
  }
  
  int camera_index = 0;
  while (true) {
    ostringstream filename;
    filename << "intrinsics" << camera_index << ".yaml";
    string model_path = (boost::filesystem::path(base_path) / filename.str()).string();
    if (!boost::filesystem::exists(model_path)) {
      if (camera_index == 0) {
        LOG(ERROR) << "No intrinsics file found since " << model_path << " does not exist.";
        return false;
      }
      break;
    }
    shared_ptr<CameraModel> model(LoadCameraModel(model_path.c_str()));
    if (!model) {
      LOG(ERROR) << "Cannot load file: " << model_path;
      return false;
    }
    state->intrinsics.push_back(model);
    ++ camera_index;
  }
  
  string pattern_path = (boost::filesystem::path(base_path) / "points.yaml").string();
  if (!LoadPointsAndIndexMapping(
      &state->points,
      &state->feature_id_to_points_index,
      pattern_path.c_str())) {
    LOG(ERROR) << "Cannot load file: " << pattern_path;
    return false;
  }
  if (dataset) {
    state->ComputeFeatureIdToPointsIndex(dataset);
  }
  
  return true;
}


bool SaveCameraModel(const CameraModel& model, const char* path) {
  QFileInfo(path).dir().mkpath(".");
  
  auto save_grid = [&](const Image<Vec3d>& grid, std::ofstream& stream) {
    stream << "[";
    for (u32 y = 0; y < grid.height(); ++ y) {
      for (u32 x = 0; x < grid.width(); ++ x) {
        if (x != 0 || y != 0) {
          stream << ", ";
        }
        const Vec3d& element = grid(x, y);
        stream << element.x() << ", " << element.y() << ", " << element.z();
      }
    }
    stream << "]" << std::endl;
  };
  
  // TODO: Would it be possible to implement more of the following in a generic way?
  //       Or maybe implement a save function in each camera model to avoid having
  //       to create a special case for each model here.
  if (const CentralGenericModel* cgbsp_model = dynamic_cast<const CentralGenericModel*>(&model)) {
    std::ofstream stream(path, std::ios::out);
    if (!stream) {
      return false;
    }
    stream << std::setprecision(14);
    
    stream << "type : CentralGenericModel" << std::endl;
    stream << "width : " << cgbsp_model->width() << std::endl;
    stream << "height : " << cgbsp_model->height() << std::endl;
    stream << "calibration_min_x : " << cgbsp_model->calibration_min_x() << std::endl;
    stream << "calibration_min_y : " << cgbsp_model->calibration_min_y() << std::endl;
    stream << "calibration_max_x : " << cgbsp_model->calibration_max_x() << std::endl;
    stream << "calibration_max_y : " << cgbsp_model->calibration_max_y() << std::endl;
    stream << "grid_width : " << cgbsp_model->grid().width() << std::endl;
    stream << "grid_height : " << cgbsp_model->grid().height() << std::endl;
    stream << "# The grid is stored in row-major order, top to bottom. Each row is stored left to right. Each grid point is stored as x, y, z." << std::endl;
    stream << "grid : ";
    save_grid(cgbsp_model->grid(), stream);
  } else if (const CentralOpenCVModel* cocv_model = dynamic_cast<const CentralOpenCVModel*>(&model)) {
    std::ofstream stream(path, std::ios::out);
    if (!stream) {
      return false;
    }
    stream << std::setprecision(14);
    
    stream << "type : CentralOpenCVModel" << std::endl;
    stream << "width : " << cocv_model->width() << std::endl;
    stream << "height : " << cocv_model->height() << std::endl;
    stream << "parameters : [";
    for (int i = 0; i < 12; ++ i) {
      if (i > 0) {
        stream << ", ";
      }
      stream << cocv_model->parameters()[i];
    }
    stream << "]" << std::endl;
  } else if (const CentralThinPrismFisheyeModel* ctpf_model = dynamic_cast<const CentralThinPrismFisheyeModel*>(&model)) {
    std::ofstream stream(path, std::ios::out);
    if (!stream) {
      return false;
    }
    stream << std::setprecision(14);
    
    stream << "type : CentralThinPrismFisheyeModel" << std::endl;
    stream << "width : " << ctpf_model->width() << std::endl;
    stream << "height : " << ctpf_model->height() << std::endl;
    stream << "use_equidistant_projection : " << (ctpf_model->use_equidistant_projection() ? "true" : "false") << std::endl;
    stream << "parameters : [";
    for (int i = 0; i < 12; ++ i) {
      if (i > 0) {
        stream << ", ";
      }
      stream << ctpf_model->parameters()[i];
    }
    stream << "]" << std::endl;
  } else if (const CentralRadialModel* cr_model = dynamic_cast<const CentralRadialModel*>(&model)) {
    std::ofstream stream(path, std::ios::out);
    if (!stream) {
      return false;
    }
    stream << std::setprecision(14);
    
    stream << "type : CentralRadialModel" << std::endl;
    stream << "width : " << cr_model->width() << std::endl;
    stream << "height : " << cr_model->height() << std::endl;
    stream << "parameters : [";
    for (int i = 0; i < cr_model->parameters().size(); ++ i) {
      if (i > 0) {
        stream << ", ";
      }
      stream << cr_model->parameters()[i];
    }
    stream << "]" << std::endl;
  } else if (const NoncentralGenericModel* ngbsp_model = dynamic_cast<const NoncentralGenericModel*>(&model)) {
    std::ofstream stream(path, std::ios::out);
    if (!stream) {
      return false;
    }
    stream << std::setprecision(14);
    
    stream << "type : NoncentralGenericModel" << std::endl;
    stream << "width : " << ngbsp_model->width() << std::endl;
    stream << "height : " << ngbsp_model->height() << std::endl;
    stream << "calibration_min_x : " << ngbsp_model->calibration_min_x() << std::endl;
    stream << "calibration_min_y : " << ngbsp_model->calibration_min_y() << std::endl;
    stream << "calibration_max_x : " << ngbsp_model->calibration_max_x() << std::endl;
    stream << "calibration_max_y : " << ngbsp_model->calibration_max_y() << std::endl;
    stream << "grid_width : " << ngbsp_model->point_grid().width() << std::endl;
    stream << "grid_height : " << ngbsp_model->point_grid().height() << std::endl;
    stream << "# The grids are stored in row-major order, top to bottom. Each row is stored left to right. Each grid point is stored as x, y, z." << std::endl;
    stream << "point_grid : ";
    save_grid(ngbsp_model->point_grid(), stream);
    stream << "direction_grid : ";
    save_grid(ngbsp_model->direction_grid(), stream);
  } else {
    LOG(ERROR) << "SaveCameraModel() is not implemented for this camera model type yet.";
    return false;
  }
  
  return true;
}

shared_ptr<CameraModel> LoadCameraModel(const char* path) {
  auto load_grid = [&](const YAML::Node& grid_node, int grid_width, int grid_height,
                       bool normalized, Image<Vec3d>* grid_image) {
    if (!grid_node.IsSequence()) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "The '" << grid_node.Tag() << "' item is not a sequence.";
      return false;
    }
    if (grid_node.size() != 3 * grid_width * grid_height) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Expected " << (3 * grid_width * grid_height) << " entries in '" << grid_node.Tag() << "' list, got " << grid_node.size() << ".";
      return false;
    }
    
    grid_image->SetSize(grid_width, grid_height);
    for (u32 y = 0; y < grid_height; ++ y) {
      for (u32 x = 0; x < grid_width; ++ x) {
        int index = 3 * (x + grid_width * y);
        (*grid_image)(x, y) = Vec3d(
            grid_node[index + 0].as<double>(),
            grid_node[index + 1].as<double>(),
            grid_node[index + 2].as<double>());
        if (normalized) {
          // Re-normalize to avoid non-normalized vectors due to limited numerical precision
          (*grid_image)(x, y).normalize();
        }
      }
    }
    return true;
  };
  
  YAML::Node file_node = YAML::LoadFile(path);
  if (file_node.IsNull()) {
    LOG(ERROR) << "Cannot read file: " << path;
    return nullptr;
  }
  
  int width = file_node["width"].as<int>();
  int height = file_node["height"].as<int>();
  if (width < 1 || height < 1) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "Invalid image dimensions: " << width << " x " << height << ".";
    return nullptr;
  }
  
  std::string type = file_node["type"].as<string>();
  
  LOG(INFO) << "Loaded type: " << type;
  
  if (type == "CentralThinPrismFisheyeModel") {
    bool use_equidistant_projection = file_node["use_equidistant_projection"].as<bool>();
    CentralThinPrismFisheyeModel* model = new CentralThinPrismFisheyeModel(width, height, use_equidistant_projection);
    YAML::Node parameters_node = file_node["parameters"];
    CHECK_EQ(parameters_node.size(), model->parameters().size());
    for (int i = 0; i < parameters_node.size(); ++ i) {
      model->parameters()[i] = parameters_node[i].as<double>();
    }
    return shared_ptr<CameraModel>(model);
  } else if (type == "CentralOpenCVModel") {
    CentralOpenCVModel* model = new CentralOpenCVModel(width, height);
    YAML::Node parameters_node = file_node["parameters"];
    CHECK_EQ(parameters_node.size(), model->parameters().size());
    for (int i = 0; i < parameters_node.size(); ++ i) {
      model->parameters()[i] = parameters_node[i].as<double>();
    }
    return shared_ptr<CameraModel>(model);
  } else if (type == "CentralRadialModel") {
    YAML::Node parameters_node = file_node["parameters"];
    CentralRadialModel* model = new CentralRadialModel(width, height, parameters_node.size() - 8);
    for (int i = 0; i < parameters_node.size(); ++ i) {
      model->parameters()[i] = parameters_node[i].as<double>();
    }
    return shared_ptr<CameraModel>(model);
  }
  
  int calibration_min_x = file_node["calibration_min_x"].as<int>();
  int calibration_min_y = file_node["calibration_min_y"].as<int>();
  int calibration_max_x = file_node["calibration_max_x"].as<int>();
  int calibration_max_y = file_node["calibration_max_y"].as<int>();
  
  if (type == "CentralGenericModel" || type == "CentralGenericBSplineModel") {
    int grid_width = file_node["grid_width"].as<int>();
    int grid_height = file_node["grid_height"].as<int>();
    if (grid_width < 4 || grid_height < 4) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Invalid grid dimensions: " << grid_width << " x " << grid_height << ".";
      return nullptr;
    }
    
    Image<Vec3d> grid_image;
    if (!load_grid(file_node["grid"], grid_width, grid_height, /*normalized*/ true, &grid_image)) {
      return nullptr;
    }
    
    CentralGenericModel* model = new CentralGenericModel(
        grid_width, grid_height,
        calibration_min_x, calibration_min_y,
        calibration_max_x, calibration_max_y,
        width, height);
    model->SetGrid(grid_image);
    return shared_ptr<CameraModel>(model);
  } else if (type == "NoncentralGenericModel" || type == "NoncentralGenericBSplineModel") {
    int grid_width = file_node["grid_width"].as<int>();
    int grid_height = file_node["grid_height"].as<int>();
    if (grid_width < 4 || grid_height < 4) {
      LOG(ERROR) << "Cannot parse file: " << path;
      LOG(ERROR) << "Invalid grid dimensions: " << grid_width << " x " << grid_height << ".";
      return nullptr;
    }
    
    Image<Vec3d> point_grid_image;
    if (!load_grid(file_node["point_grid"], grid_width, grid_height, /*normalized*/ false, &point_grid_image)) {
      return nullptr;
    }
    
    Image<Vec3d> direction_grid_image;
    if (!load_grid(file_node["direction_grid"], grid_width, grid_height, /*normalized*/ true, &direction_grid_image)) {
      return nullptr;
    }
    
    NoncentralGenericModel* model = new NoncentralGenericModel(
        grid_width, grid_height,
        calibration_min_x, calibration_min_y,
        calibration_max_x, calibration_max_y,
        width, height);
    model->SetPointGrid(point_grid_image);
    model->SetDirectionGrid(direction_grid_image);
    return shared_ptr<CameraModel>(model);
  } else {
    LOG(ERROR) << "Cannot load camera model type: " << type;
    return nullptr;
  }
  
  return nullptr;
}

bool SavePoses(
    const vector<bool>& image_used,
    const vector<SE3d>& image_tr_pattern,
    const char* path) {
  QFileInfo(path).dir().mkpath(".");
  
  CHECK_EQ(image_used.size(), image_tr_pattern.size());
  
  std::ofstream stream(path, std::ios::out);
  if (!stream) {
    return false;
  }
  stream << std::setprecision(14);
  
  stream << "# Each pose gives the B_tr_A transformation (i.e., A to B with right-multiplication), where the spaces A and B are defined by the filename. Quaternions are written as used by the Eigen library." << std::endl;
  
  stream << "pose_count: " << image_used.size() << std::endl;
  stream << "poses:" << std::endl;
  
  for (usize i = 0; i < image_used.size(); ++ i) {
    if (image_used[i]) {
      const SE3d& pose = image_tr_pattern[i];
      
      stream << "  - index: " << i << std::endl;
      stream << "    tx: " << pose.translation().x() << std::endl;
      stream << "    ty: " << pose.translation().y() << std::endl;
      stream << "    tz: " << pose.translation().z() << std::endl;
      stream << "    qx: " << pose.unit_quaternion().x() << std::endl;
      stream << "    qy: " << pose.unit_quaternion().y() << std::endl;
      stream << "    qz: " << pose.unit_quaternion().z() << std::endl;
      stream << "    qw: " << pose.unit_quaternion().w() << std::endl;
    }
  }
  
  // TODO: Should be in a separate function.
  // For convenience, we always also save an .obj file that can be used to
  // visualize the pose positions. All vertices are colored red.
  std::ofstream obj_stream((string(path) + ".obj").c_str(), std::ios::out);
  if (!obj_stream) {
    return false;
  }
  obj_stream << std::setprecision(14);
  
  for (usize i = 0; i < image_used.size(); ++ i) {
    if (image_used[i]) {
      const SE3d& pose = image_tr_pattern[i];
      Vec3d camera_position = pose.inverse().translation();
      obj_stream << "v " << camera_position.x() << " " << camera_position.y() << " " << camera_position.z() << " 1 0 0" << std::endl;
    }
  }
  
  obj_stream.close();
  
  return true;
}

bool LoadPoses(
    vector<bool>* image_used,
    vector<SE3d>* image_tr_pattern,
    const char* path) {
  YAML::Node file_node;
  try {
    file_node = YAML::LoadFile(path);
  } catch (YAML::BadFile& ex) {
    LOG(ERROR) << "Cannot read file: " << path;
    return false;
  }
  
  int pose_count = file_node["pose_count"].as<int>();
  image_used->clear();
  image_used->resize(pose_count, false);
  image_tr_pattern->resize(pose_count);
  
  YAML::Node poses_node = file_node["poses"];
  
  if (!poses_node.IsSequence()) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "Root node is not a sequence";
    return false;
  }
  
  for (usize i = 0; i < poses_node.size(); ++ i) {
    YAML::Node node = poses_node[i];
    int index = node["index"].as<int>();
    if (index >= image_used->size()) {
      LOG(ERROR) << "Error while parsing file: " << path;
      LOG(ERROR) << "Size of image_used vector given for parsing is too small";
      return false;
    }
    image_used->at(index) = true;
    
    SE3d& pose = image_tr_pattern->at(index);
    pose.translation().x() = node["tx"].as<double>();
    pose.translation().y() = node["ty"].as<double>();
    pose.translation().z() = node["tz"].as<double>();
    pose.setQuaternion(Quaterniond(
        node["qw"].as<double>(),
        node["qx"].as<double>(),
        node["qy"].as<double>(),
        node["qz"].as<double>()));
  }
  
  return true;
}

bool SavePointsAndIndexMapping(
    const BAState& calibration,
    const char* path) {
  QFileInfo(path).dir().mkpath(".");
  
  std::ofstream stream(path, std::ios::out);
  if (!stream) {
    return false;
  }
  stream << std::setprecision(14);
  
  stream << "# Each point is stored as x, y, z." << std::endl;
  stream << "points : [";
  for (usize i = 0; i < calibration.points.size(); ++ i) {
    const Vec3d& point = calibration.points[i];
    stream << point.x() << ", " << point.y() << ", " << point.z();
    if (i < calibration.points.size() - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;
  
  stream << "feature_id_to_point_index:" << std::endl;
  for (const auto& item : calibration.feature_id_to_points_index) {
    stream << "  - feature_id: " << item.first << std::endl;
    stream << "    point_index: " << item.second << std::endl;
  }

  unordered_map<int, Vec2i>* feature_id_to_coord;
  /*
  FeatureDetectorTaggedPattern().GetCorners(
       0,
              & feature_id_to_coord);
    //*/

  for (const auto& item : calibration.feature_id_to_points_index) {
    stream << "  - feature_id: " << item.first << std::endl;
    stream << "    point_index: " << item.second << std::endl;
  }

  stream.close();
  
  // TODO: Should be in a separate function.
  // For convenience, we always also save an .obj file that can be used to
  // visualize the patterns. All vertices are colored blue.
  std::ofstream obj_stream((string(path) + ".obj").c_str(), std::ios::out);
  if (!obj_stream) {
    return false;
  }
  obj_stream << std::setprecision(14);
  
  for (usize i = 0; i < calibration.points.size(); ++ i) {
    const Vec3d& point = calibration.points[i];
    obj_stream << "v " << point.x() << " " << point.y() << " " << point.z() << " 0 0 1" << std::endl;
  }
  
  obj_stream.close();
  
  return true;
}

bool LoadPointsAndIndexMapping(
    vector<Vec3d>* optimized_geometry,
    unordered_map<int, int>* feature_id_to_points_index,
    const char* path) {
  YAML::Node file_node = YAML::LoadFile(path);
  if (file_node.IsNull()) {
    LOG(ERROR) << "Cannot read file: " << path;
    return false;
  }
  
  YAML::Node points_node = file_node["points"];
  if (!points_node.IsSequence()) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "points node is not a sequence";
    return false;
  }
  if (points_node.size() % 3 != 0) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "points node size is not an integer multiple of 3";
    return false;
  }
  
  int num_points = points_node.size() / 3;
  optimized_geometry->resize(num_points);
  for (int i = 0; i < num_points; ++ i) {
    optimized_geometry->at(i) = Vec3d(
        points_node[3 * i + 0].as<double>(),
        points_node[3 * i + 1].as<double>(),
        points_node[3 * i + 2].as<double>());
  }
  
  YAML::Node indexing_node = file_node["feature_id_to_point_index"];
  if (!indexing_node.IsSequence()) {
    LOG(ERROR) << "Cannot parse file: " << path;
    LOG(ERROR) << "feature_id_to_point_index node is not a sequence";
    return false;
  }
  
  feature_id_to_points_index->clear();
  for (int i = 0; i < indexing_node.size(); ++ i) {
    feature_id_to_points_index->insert(make_pair(
        indexing_node[i]["feature_id"].as<int>(),
        indexing_node[i]["point_index"].as<int>()));
  }
  
  return true;
}

}

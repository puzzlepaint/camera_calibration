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


#include "libvis/external_io/colmap_model.h"

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <fstream>
#include <iomanip>
#include <map>
#include <sys/stat.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

namespace vis {

bool ReadColmapCameras(const std::string& cameras_txt_path,
                       ColmapCameraPtrMap* cameras) {
  std::ifstream cameras_file_stream(cameras_txt_path, std::ios::in);
  if (!cameras_file_stream) {
    return false;
  }
  
  while (!cameras_file_stream.eof() && !cameras_file_stream.bad()) {
    std::string line;
    std::getline(cameras_file_stream, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    ColmapCamera* new_camera = new ColmapCamera();
    std::istringstream line_stream(line);
    line_stream >> new_camera->camera_id >> new_camera->model_name >> new_camera->width >> new_camera->height;
    while (!line_stream.eof() && !line_stream.bad()) {
      new_camera->parameters.emplace_back();
      line_stream >> new_camera->parameters.back();
    }
    
    cameras->insert(std::make_pair(new_camera->camera_id, ColmapCameraPtr(new_camera)));
  }
  return true;
}

bool WriteColmapCameras(const std::string& cameras_txt_path,
                        const ColmapCameraPtrMap& cameras) {
  std::ofstream cameras_file_stream(cameras_txt_path, std::ios::out);
  cameras_file_stream << "# Camera list with one line of data per camera:" << std::endl;
  cameras_file_stream << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
  cameras_file_stream << "# Number of cameras: " << cameras.size() << std::endl;
  for (ColmapCameraPtrMap::const_iterator it = cameras.begin(); it != cameras.end(); ++ it) {
    const ColmapCamera& colmap_camera = *it->second;
    cameras_file_stream << colmap_camera.camera_id << " "
                        << colmap_camera.model_name
                        << " " << colmap_camera.width
                        << " " << colmap_camera.height;
    for (std::size_t i = 0; i < colmap_camera.parameters.size(); ++ i) {
      cameras_file_stream << " " << colmap_camera.parameters[i];
    }
    cameras_file_stream << std::endl;
  }
  cameras_file_stream.close();
  
  return true;
}

bool ReadColmapImages(const std::string& images_txt_path,
                      bool read_observations,
                      ColmapImagePtrMap* images) {
  std::ifstream images_file_stream(images_txt_path, std::ios::in);
  if (!images_file_stream) {
    return false;
  }
  
  while (!images_file_stream.eof() && !images_file_stream.bad()) {
    std::string line;
    std::getline(images_file_stream, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    // Read image info line.
    ColmapImage* new_image = new ColmapImage();
    std::istringstream image_stream(line);
    image_stream >> new_image->image_id
                 >> new_image->image_tr_global.data()[3]
                 >> new_image->image_tr_global.data()[0]
                 >> new_image->image_tr_global.data()[1]
                 >> new_image->image_tr_global.data()[2]
                 >> new_image->image_tr_global.data()[4]
                 >> new_image->image_tr_global.data()[5]
                 >> new_image->image_tr_global.data()[6]
                 >> new_image->camera_id
                 >> new_image->file_path;
    new_image->global_tr_image = new_image->image_tr_global.inverse();
    
    // Read feature observations line.
    std::getline(images_file_stream, line);
    if (read_observations) {
      std::istringstream observations_stream(line);
      while (!observations_stream.eof() && !observations_stream.bad()) {
        new_image->observations.emplace_back();
        ColmapFeatureObservation* new_observation = &new_image->observations.back();
        observations_stream >> new_observation->xy.x()
                            >> new_observation->xy.y()
                            >> new_observation->point3d_id;
      }
    }
    
    images->insert(std::make_pair(new_image->image_id, ColmapImagePtr(new_image)));
  }
  return true;
}

bool WriteColmapImages(const std::string& images_txt_path, const ColmapImagePtrMap& images) {
  std::ofstream images_file_stream(images_txt_path, std::ios::out);
  images_file_stream << "# Image list with two lines of data per image:" << std::endl;
  images_file_stream << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
  images_file_stream << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
  images_file_stream << "# Number of images: " << images.size() << std::endl;
  
  for (ColmapImagePtrMap::const_iterator it = images.begin(); it != images.end(); ++ it) {
    const ColmapImage& colmap_image = *it->second;
    
    // Image with pose.
    images_file_stream << colmap_image.image_id
                       << " " << colmap_image.image_tr_global.data()[3]
                       << " " << colmap_image.image_tr_global.data()[0]
                       << " " << colmap_image.image_tr_global.data()[1]
                       << " " << colmap_image.image_tr_global.data()[2]
                       << " " << colmap_image.image_tr_global.data()[4]
                       << " " << colmap_image.image_tr_global.data()[5]
                       << " " << colmap_image.image_tr_global.data()[6]
                       << " " << colmap_image.camera_id
                       << " " << colmap_image.file_path << std::endl;
    
    // Point observations.
    for (const ColmapFeatureObservation& observation : colmap_image.observations) {
      images_file_stream << " " << observation.xy.x()
                         << " " << observation.xy.y()
                         << " " << observation.point3d_id;
    }
    
    images_file_stream << std::endl;
  }
  images_file_stream.close();
  
  return true;
}

bool ReadColmapRigs(const std::string& rigs_json_path,
                    ColmapRigVector* rigs) {
  rapidjson::Document document;
  
  FILE* json_file = fopen(rigs_json_path.c_str(), "rb");
  if (!json_file) {
    return false;
  }
  constexpr int kBufferSize = 256;
  char buffer[kBufferSize];
  rapidjson::FileReadStream json_stream(json_file, buffer, kBufferSize);
  if (document.ParseStream(json_stream).HasParseError()) {
    return false;
  }
  fclose(json_file);
  
  if (!document.IsArray()) {
    return false;
  }
  
  for (rapidjson::SizeType rig_index = 0; rig_index < document.Size(); ++ rig_index) {
    rigs->emplace_back();
    ColmapRig* new_rig = &rigs->back();
    
    const auto& json_rig = document[rig_index];
    if (!json_rig.IsObject()) {
      return false;
    }
    
    new_rig->ref_camera_id = json_rig["ref_camera_id"].GetInt();
    const auto& json_rig_cameras = json_rig["cameras"];
    for (rapidjson::SizeType rig_camera_index = 0; rig_camera_index < json_rig_cameras.Size(); ++ rig_camera_index) {
      new_rig->cameras.emplace_back();
      ColmapRigCamera* new_camera = &new_rig->cameras.back();
      
      const auto& json_rig_camera = json_rig_cameras[rig_camera_index];
      new_camera->camera_id = json_rig_camera["camera_id"].GetInt();
      new_camera->image_prefix = json_rig_camera["image_prefix"].GetString();
    }
  }
  
  return true;
}

bool WriteColmapRigs(const std::string& rigs_json_path,
                     const ColmapRigVector& rigs) {
  rapidjson::Document document;
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  
  document.SetArray();
  for (const ColmapRig& rig : rigs) {
    rapidjson::GenericValue<rapidjson::UTF8<> > rig_object;
    rig_object.SetObject();
    
    rig_object.AddMember("ref_camera_id", rig.ref_camera_id, allocator);
    
    rapidjson::GenericValue<rapidjson::UTF8<> > rig_cameras_array;
    rig_cameras_array.SetArray();
    for (const ColmapRigCamera& rig_camera : rig.cameras) {
      rapidjson::GenericValue<rapidjson::UTF8<> > rig_camera_object;
      rig_camera_object.SetObject();
      
      rig_camera_object.AddMember("camera_id", rig_camera.camera_id, allocator);
      rapidjson::Value image_prefix;
      image_prefix.SetString(rig_camera.image_prefix.c_str(), static_cast<rapidjson::SizeType>(rig_camera.image_prefix.size()), allocator);
      rig_camera_object.AddMember("image_prefix", image_prefix, allocator);
      rig_cameras_array.PushBack(rig_camera_object, allocator);
    }
    rig_object.AddMember("cameras", rig_cameras_array, allocator);
    
    document.PushBack(rig_object, allocator);
  }
  
  rapidjson::StringBuffer string_buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(string_buffer);
  document.Accept(writer);
  
  std::ofstream json_file(rigs_json_path);
  if (!json_file) {
    return false;
  }
  json_file << string_buffer.GetString();
  return true;
}

bool ReadColmapPoints3D(const string& points3d_txt_path,
                        ColmapPoint3DMap* points) {
  std::ifstream points_file_stream(points3d_txt_path, std::ios::in);
  if (!points_file_stream) {
    return false;
  }
  
  while (!points_file_stream.eof() && !points_file_stream.bad()) {
    std::string line;
    std::getline(points_file_stream, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    // Read image info line.
    ColmapPoint3D new_point;
    std::istringstream point_stream(line);
    point_stream >> new_point.id
                 >> new_point.position.x()
                 >> new_point.position.y()
                 >> new_point.position.z()
                 >> new_point.color.x()
                 >> new_point.color.y()
                 >> new_point.color.z()
                 >> new_point.error;
//     while (!point_stream.eof() && !point_stream.bad()) {
//       new_point.track.emplace_back();
//       point_stream >> new_point.track.back().first >> new_point.track.back().second;
//     }
    
    points->insert(std::make_pair(new_point.id, new_point));
  }
  
  return true;
}

bool WriteColmapPoints3D(const string& points3d_txt_path,
                         const ColmapPoint3DMap& points) {
  std::ofstream points_file_stream(points3d_txt_path, std::ios::out);
  points_file_stream << "# 3D point list with one line of data per point:" << std::endl;
  points_file_stream << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)" << std::endl;
  points_file_stream << "# Number of points: " << points.size() << std::endl;  // << ", mean track length: " << mean_track_length << std::endl;
  
  for (ColmapPoint3DMap::const_iterator it = points.begin(), end = points.end(); it != end; ++ it) {
    const ColmapPoint3D& point = it->second;
    
    points_file_stream << point.id
                       << " " << point.position.x()
                       << " " << point.position.y()
                       << " " << point.position.z()
                       << " " << point.color.x()
                       << " " << point.color.y()
                       << " " << point.color.z()
                       << " " << point.error;
    for (const auto& item : point.track) {
      points_file_stream << " " << item.first << " " << item.second;
    }
    points_file_stream << std::endl;
  }
  points_file_stream.close();
  
  return true;
}

}

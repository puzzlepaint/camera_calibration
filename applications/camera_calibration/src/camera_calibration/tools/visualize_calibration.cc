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

#include "camera_calibration/tools/tools.h"

#include <libvis/camera.h>
#include <libvis/external_io/colmap_model.h>
#include <libvis/image.h>
#include <libvis/logging.h>
#include <yaml-cpp/yaml.h>

namespace vis {

void VisualizeCameraModel(
    const RadtanCamera8d& camera,
    const string& path) {
  // Determine a rotation of the observation directions into a canonical orientation (ChooseNiceCameraOrientation())
  // NOTE: It is important that this function obtains the same result for
  //       different calibrations of the same camera (to make them more easily
  //       comparable). For example, one thing to avoid is to make the location
  //       of the pixel(s) used for estimating the forward direction dependent
  //       on the calibrated image area rectangle extents.
  
  Vec3d forward = camera.UnprojectFromPixelCornerConv(Vec2d(
      0.5f * camera.width(),
      0.5f * camera.height()));
  Mat3d forward_rotation = Quaterniond::FromTwoVectors(forward, Vec3d(0, 0, 1)).toRotationMatrix();
  
  const int right_min_x = std::min<int>(camera.width() - 1, camera.width() / 2 + 11);
  const int right_max_x = camera.width() - 1;
  const int right_min_y = std::max<int>(0, camera.height() / 2 - 10);
  const int right_max_y = std::min<int>(camera.height() - 1, camera.height() / 2 + 10);
  Vec3d right_sum = Vec3d::Zero();
  u32 right_count = 0;
  for (u32 y = right_min_y; y <= right_max_y; ++ y) {
    for (u32 x = right_min_x; x <= right_max_x; ++ x) {
      Vec3d direction = camera.UnprojectFromPixelCornerConv(Vec2d(x + 0.5f, y + 0.5f));
      
      right_sum += direction.cast<double>();
      right_count += 1;
    }
  }
  
  Mat3d right_rotation;
  if (right_count > 0) {
    Vec3d forward_rotated_right = forward_rotation.cast<double>() * (right_sum / right_count);
    
    // We want to rotate forward_rotated_right around the forward vector (0, 0, 1) such as to maximize its x value.
    double angle = atan2(-forward_rotated_right.y(), forward_rotated_right.x());  // TODO: Is the minus here correct?
    right_rotation = AngleAxisd(angle, Vec3d(0, 0, 1)).toRotationMatrix();
  } else {
    right_rotation = Mat3d::Identity();
  }
  
  Mat3d rotation = right_rotation * forward_rotation;
  
  // Create the visualization
  Image<Vec3u8> visualization(camera.width(), camera.height());
  
  for (u32 y = 0; y < visualization.height(); ++ y) {
    for (u32 x = 0; x < visualization.width(); ++ x) {
      Vec3d direction = rotation * camera.UnprojectFromPixelCornerConv(Vec2d(x + 0.5f, y + 0.5f)).normalized();
      visualization.at(x, y) = Vec3u8(
          70 * 255.99f / 2.f * (direction.x() + 1),
          70 * 255.99f / 2.f * (direction.y() + 1),
          270 * 255.99f / 2.f * (direction.z() + 1));
    }
  }
  
  visualization.Write(path);
}

int VisualizeKalibrCalibration(const string& camchain_path) {
  YAML::Node file_node;
  try {
    file_node = YAML::LoadFile(camchain_path);
  } catch (YAML::BadFile& ex) {
    LOG(ERROR) << "Cannot read file: " << camchain_path;
    return EXIT_FAILURE;
  }
  
  int camera_index = 0;
  while (true) {
    ostringstream camera_name;
    camera_name << "cam" << camera_index;
    YAML::Node cam_node = file_node[camera_name.str()];
    if (!cam_node.IsDefined()) {
      break;
    }
    ++ camera_index;
    
    // Check camera model
    string camera_model = cam_node["camera_model"].as<string>();
    if (camera_model != "pinhole") {
      LOG(ERROR) << "Camera model not handled: " << camera_model;
      continue;
    }
    
    // Check distortion model
    string distortion_model = cam_node["distortion_model"].as<string>();
    if (distortion_model != "radtan") {
      LOG(ERROR) << "Distortion model not handled: " << distortion_model;
      continue;
    }
    
    // Read camera resolution
    YAML::Node resolution_node = cam_node["resolution"];
    int width = resolution_node[0].as<int>();
    int height = resolution_node[1].as<int>();
    LOG(INFO) << "Resolution: " << width << " x " << height;
    
    // Read intrinsics (concatenate the "distortion_coeffs" and "intrinsics" parameter arrays)
    vector<double> intrinsics;
    
    YAML::Node distortion_coeffs_node = cam_node["distortion_coeffs"];
    int base_index = intrinsics.size();
    intrinsics.resize(intrinsics.size() + distortion_coeffs_node.size());
    for (int i = 0; i < distortion_coeffs_node.size(); ++ i) {
      intrinsics[base_index + i] = distortion_coeffs_node[i].as<double>();
    }
    
    YAML::Node intrinsics_node = cam_node["intrinsics"];
    base_index = intrinsics.size();
    intrinsics.resize(intrinsics.size() + intrinsics_node.size());
    for (int i = 0; i < intrinsics_node.size(); ++ i) {
      intrinsics[base_index + i] = intrinsics_node[i].as<double>();
    }
    
    for (int i = 0; i < intrinsics.size(); ++ i) {
      LOG(INFO) << "intrinsics[" << i << "]: " << intrinsics[i];
    }
    
    // Put the parameters into a camera model class
    RadtanCamera8d camera(width, height, intrinsics.data());
    
    VisualizeCameraModel(camera, camchain_path + "." + camera_name.str() + ".png");
  }
  
  return EXIT_SUCCESS;
}

int VisualizeColmapCalibration(const string& cameras_path) {
  ColmapCameraPtrMap cameras;
  if (!ReadColmapCameras(cameras_path, &cameras)) {
    LOG(ERROR) << "Cannot read file: " << cameras_path;
    return EXIT_FAILURE;
  }
  
  for (auto it : cameras) {
    const ColmapCamera& colmap_camera = *it.second;
    
    if (colmap_camera.model_name != "OPENCV") {
      LOG(ERROR) << "Camera model not handled: " << colmap_camera.model_name;
      continue;
    }
    
    // The parameter ordering for RadtanCamera8d is:
    //   k1, k2, r1, r2, fx, fy, cx, cy.
    // The parameter ordering for the "OPENCV" camera model in Colmap is:
    //   fx, fy, cx, cy, k1, k2, p1, p2
    // So, parameters 0 .. 3 must be swapped with parameters 4 .. 7.
    vector<double> swapped_parameters(8);
    
    swapped_parameters[0] = colmap_camera.parameters[4];
    swapped_parameters[1] = colmap_camera.parameters[5];
    swapped_parameters[2] = colmap_camera.parameters[6];
    swapped_parameters[3] = colmap_camera.parameters[7];
    
    swapped_parameters[4] = colmap_camera.parameters[0];
    swapped_parameters[5] = colmap_camera.parameters[1];
    swapped_parameters[6] = colmap_camera.parameters[2];
    swapped_parameters[7] = colmap_camera.parameters[3];
    
    RadtanCamera8d camera(colmap_camera.width, colmap_camera.height, swapped_parameters.data());
    
    ostringstream camera_name;
    camera_name << "cam" << colmap_camera.camera_id;
    VisualizeCameraModel(camera, cameras_path + "." + camera_name.str() + ".png");
  }
  
  return EXIT_SUCCESS;
}

}

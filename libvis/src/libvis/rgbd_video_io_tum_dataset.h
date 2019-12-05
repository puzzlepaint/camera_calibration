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

#include <fstream>
#include <iostream>
#include <string>

#include "libvis/logging.h"

#include "libvis/libvis.h"
#include "libvis/rgbd_video.h"

namespace vis {

template <typename PoseScalar>
bool InterpolatePose(
    double timestamp,
    const vector<double>& pose_timestamps,
    const vector<Sophus::SE3<PoseScalar>>& poses,
    Sophus::SE3<PoseScalar>* pose,
    double max_interpolation_time_extent = numeric_limits<double>::infinity()) {
  CHECK_EQ(pose_timestamps.size(), poses.size());
  CHECK_GE(pose_timestamps.size(), 2u);
  
  if (timestamp <= pose_timestamps[0]) {
    *pose = poses[0];
    return true;
  } else if (timestamp >= pose_timestamps.back()) {
    *pose = poses.back();
    return true;
  }
  
  // TODO: Binary search should be faster (or with given starting point if having monotonically increasing query points as is the case below).
  for (usize i = 0; i < pose_timestamps.size() - 1; ++ i) {
    if (timestamp >= pose_timestamps[i] && timestamp <= pose_timestamps[i + 1]) {
      if ((timestamp - pose_timestamps[i]) > max_interpolation_time_extent ||
          (pose_timestamps[i + 1] - timestamp) > max_interpolation_time_extent) {
        return false;
      }
      
      double factor = (timestamp - pose_timestamps[i]) / (pose_timestamps[i + 1] - pose_timestamps[i]);
      
      const Sophus::SE3<PoseScalar>& pose_a = poses[i];
      const Sophus::SE3<PoseScalar>& pose_b = poses[i + 1];
      
      *pose = Sophus::SE3<PoseScalar>(
          pose_a.unit_quaternion().slerp(factor, pose_b.unit_quaternion()),
          pose_a.translation() + factor * (pose_b.translation() - pose_a.translation()));
      return true;
    }
  }
  
  return false;
}

template <typename T>
bool ReadTUMRGBDTrajectory(
    const char* path,
    vector<double>* pose_timestamps,
    vector<Sophus::SE3<T>>* poses_global_T_frame) {
  ifstream trajectory_file(path);
  if (!trajectory_file) {
    LOG(ERROR) << "Could not open trajectory file: " << path;
    return false;
  }
  string line;
  getline(trajectory_file, line);
  while (! line.empty()) {
    char time_string[128];
    Vector3d cam_translation;
    Quaterniond cam_rotation;
    
    if (line[0] == '#') {
      getline(trajectory_file, line);
      continue;
    }
    if (sscanf(line.c_str(), "%s %lf %lf %lf %lf %lf %lf %lf",
        time_string,
        &cam_translation[0],
        &cam_translation[1],
        &cam_translation[2],
        &cam_rotation.x(),
        &cam_rotation.y(),
        &cam_rotation.z(),
        &cam_rotation.w()) != 8) {
      LOG(ERROR) << "Cannot read poses! Line:";
      LOG(ERROR) << line;
      return false;
    }
    
    Sophus::SE3<T> global_T_frame = Sophus::SE3<T>(cam_rotation.cast<T>(),
                                                   cam_translation.cast<T>());
    
    pose_timestamps->push_back(atof(time_string));
    poses_global_T_frame->push_back(global_T_frame);
    
    getline(trajectory_file, line);
  }
  return true;
}

// Reads a variant of the TUM RGB-D dataset format. Returns true if the data
// was successfully read. Compared to the raw RGB-D datasets (as tgz archives),
// the calibration has to be added in a file calibration.txt, given as
// fx fy cx cy in a single line, and the associate.py tool from the benchmark
// website must be run as follows:
// python associate.py rgb.txt depth.txt > associated.txt
// The trajectory filename can be left empty to not load a trajectory.
template<typename ColorT, typename DepthT>
bool ReadTUMRGBDDatasetAssociatedAndCalibrated(
    const char* dataset_folder_path,
    const char* trajectory_filename,
    RGBDVideo<ColorT, DepthT>* rgbd_video,
    double max_interpolation_time_extent = numeric_limits<double>::infinity()) {
  rgbd_video->color_frames_mutable()->clear();
  rgbd_video->depth_frames_mutable()->clear();
  
  string calibration_path = string(dataset_folder_path) + "/calibration.txt";
  ifstream calibration_file(calibration_path.c_str());
  if (!calibration_file) {
    LOG(ERROR) << "Could not open calibration file: " << calibration_path;
    return false;
  }
  string line;
  getline(calibration_file, line);
  double fx, fy, cx, cy;
  if (sscanf(line.c_str(), "%lf %lf %lf %lf",
      &fx, &fy, &cx, & cy) != 4) {
    LOG(ERROR) << "Cannot read calibration!";
    return false;
  }
  
  vector<double> pose_timestamps;
  vector<SE3f> poses_global_T_frame;
  
  if (trajectory_filename != nullptr) {
    string trajectory_path = string(dataset_folder_path) + "/" + trajectory_filename;
    if (!ReadTUMRGBDTrajectory(trajectory_path.c_str(), &pose_timestamps, &poses_global_T_frame)) {
      return false;
    }
  }
  
  u32 width = 0;
  u32 height = 0;
  
  string associated_filename = string(dataset_folder_path) + "/associated.txt";
  ifstream associated_file(associated_filename.c_str());
  if (!associated_file) {
    LOG(ERROR) << "Could not open associated file: " << associated_filename;
    return false;
  }
  
  while (!associated_file.eof() && !associated_file.bad()) {
    std::getline(associated_file, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    char rgb_time_string[128];
    char rgb_filename[128];
    char depth_time_string[128];
    char depth_filename[128];
    
    if (sscanf(line.c_str(), "%s %s %s %s",
        rgb_time_string, rgb_filename, depth_time_string, depth_filename) != 4) {
      LOG(ERROR) << "Cannot read association line!";
      return false;
    }
    
    SE3f rgb_global_T_frame;
    double rgb_timestamp = atof(rgb_time_string);
    if (!poses_global_T_frame.empty()) {
      if (!InterpolatePose(rgb_timestamp, pose_timestamps, poses_global_T_frame, &rgb_global_T_frame, max_interpolation_time_extent)) {
        continue;
      }
    }
    
    SE3f depth_global_T_frame;
    double depth_timestamp = atof(depth_time_string);
    if (!poses_global_T_frame.empty()) {
      if (!InterpolatePose(depth_timestamp, pose_timestamps, poses_global_T_frame, &depth_global_T_frame, max_interpolation_time_extent)) {
        continue;
      }
    }
    
    string color_filepath =
        string(dataset_folder_path) + "/" + rgb_filename;
    ImageFramePtr<ColorT, SE3f> image_frame(new ImageFrame<ColorT, SE3f>(color_filepath, rgb_timestamp, rgb_time_string));
    image_frame->SetGlobalTFrame(rgb_global_T_frame);
    rgbd_video->color_frames_mutable()->push_back(image_frame);
    
    string depth_filepath =
        string(dataset_folder_path) + "/" + depth_filename;
    ImageFramePtr<DepthT, SE3f> depth_frame(new ImageFrame<DepthT, SE3f>(depth_filepath, depth_timestamp, depth_time_string));
    depth_frame->SetGlobalTFrame(depth_global_T_frame);
    rgbd_video->depth_frames_mutable()->push_back(depth_frame);
    
    if (width == 0) {
      // Get width and height by loading one image file.
      shared_ptr<Image<ColorT>> image_ptr =
          image_frame->GetImage();
      if (!image_ptr) {
        LOG(ERROR) << "Cannot load image to determine image dimensions.";
        return false;
      }
      width = image_ptr->width();
      height = image_ptr->height();
      image_frame->ClearImageAndDerivedData();
    }
  }
  
  float camera_parameters[4];
  camera_parameters[0] = fx;
  camera_parameters[1] = fy;
  camera_parameters[2] = cx + 0.5;
  camera_parameters[3] = cy + 0.5;
  rgbd_video->color_camera_mutable()->reset(
      new PinholeCamera4f(width, height, camera_parameters));
  rgbd_video->depth_camera_mutable()->reset(
      new PinholeCamera4f(width, height, camera_parameters));
  
  return true;
}

}

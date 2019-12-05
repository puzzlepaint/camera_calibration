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

#include <boost/filesystem.hpp>
#include <libvis/image.h>
#include <libvis/image_display.h>
#include <libvis/point_cloud.h>
#include <yaml-cpp/yaml.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/io/calibration_io.h"

namespace vis {

bool LoadDepthImage(const char* path, int width, int height, Image<float>* output) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    LOG(ERROR) << "Cannot open file for reading: " << path;
    return false;
  }
  
  output->SetSize(width, height);
  if (fread(output->data(), sizeof(float), width * height, file) != width * height) {
    LOG(ERROR) << "File is too small: " << path;
    fclose(file);
    return false;
  }
  
  fclose(file);
  return true;
}

bool LoadMetadata(const char* path, int* image_width, int* image_height, int* context_radius) {
  YAML::Node file_node = YAML::LoadFile(path);
  if (file_node.IsNull()) {
    LOG(ERROR) << "Cannot read file: " << path;
    return false;
  }
  
  *image_width = file_node["image_width"].as<int>();
  *image_height = file_node["image_height"].as<int>();
  *context_radius = file_node["context_radius"].as<int>();
  return true;
}

int ComparePointClouds(const string& stereo_directory_target, const string& stereo_directory_source, const string& output_directory) {
  // Load metadata
  int image_widths[2];
  int image_heights[2];
  int context_radii[2];
  if (!LoadMetadata((boost::filesystem::path(stereo_directory_target) / "metadata.yaml").string().c_str(), &image_widths[0], &image_heights[0], &context_radii[0])) {
    return EXIT_FAILURE;
  }
  if (!LoadMetadata((boost::filesystem::path(stereo_directory_source) / "metadata.yaml").string().c_str(), &image_widths[1], &image_heights[1], &context_radii[1])) {
    return EXIT_FAILURE;
  }
  if (image_widths[0] != image_widths[1] || image_heights[0] != image_heights[1] || context_radii[0] != context_radii[1]) {
    LOG(ERROR) << "Metadata differs between the two depth images, this is not supported.";
    return EXIT_FAILURE;
  }
  int width = image_widths[0];
  int height = image_heights[0];
  
  // Load images
  vector<Image<float>> images(2);
  if (!LoadDepthImage((boost::filesystem::path(stereo_directory_target) / "depth_image.bin").string().c_str(), width, height, &images[0])) {
    return EXIT_FAILURE;
  }
  if (!LoadDepthImage((boost::filesystem::path(stereo_directory_source) / "depth_image.bin").string().c_str(), width, height, &images[1])) {
    return EXIT_FAILURE;
  }
  
  // Load point clouds
  vector<shared_ptr<Point3fC3u8Cloud>> clouds(2);
  clouds[0].reset(new Point3fC3u8Cloud());
  if (!clouds[0]->Read((boost::filesystem::path(stereo_directory_target) / "point_cloud.ply").string().c_str())) {
    return EXIT_FAILURE;
  }
  clouds[1].reset(new Point3fC3u8Cloud());
  if (!clouds[1]->Read((boost::filesystem::path(stereo_directory_source) / "point_cloud.ply").string().c_str())) {
    return EXIT_FAILURE;
  }
  
  // Intersect the point clouds.
  CHECK_EQ(images[0].size(), images[1].size());
  
  usize cloud_indices[2] = {0, 0};
  vector<Point3fC3u8> outputs[2];
  
  const int context_radius = context_radii[0];
  for (int y = context_radius; y < images[0].height() - context_radius; ++ y) {
    for (int x = context_radius; x < images[0].width() - context_radius; ++ x) {
      if (images[0](x, y) != 0 && images[1](x, y) != 0) {
        outputs[0].push_back((*clouds[0])[cloud_indices[0]]);
        outputs[1].push_back((*clouds[1])[cloud_indices[1]]);
      }
      
      if (images[0](x, y) != 0) {
        ++ cloud_indices[0];
      }
      if (images[1](x, y) != 0) {
        ++ cloud_indices[1];
      }
    }
  }
  
  for (int i = 0; i < 2; ++ i) {
    CHECK_EQ(cloud_indices[i], clouds[i]->size()) << "i == " << i;
    clouds[i]->Resize(outputs[i].size());
    memcpy(clouds[i]->data_mutable(), outputs[i].data(), outputs[i].size() * sizeof(Point3fC3u8));
  }
  
  // Align the intersected point clouds (whose points at the same indices correspond) with a similarity transform.
  Eigen::Matrix<float, 3, Eigen::Dynamic> x;
  x.resize(Eigen::NoChange, clouds[1]->size());
  for (usize i = 0; i < clouds[1]->size(); ++ i) {
    x.col(i) = (*clouds[1])[i].position().cast<float>();
  }
  
  Eigen::Matrix<float, 3, Eigen::Dynamic> y;
  y.resize(Eigen::NoChange, clouds[0]->size());
  for (usize i = 0; i < clouds[0]->size(); ++ i) {
    y.col(i) = (*clouds[0])[i].position().cast<float>();
  }
  
  Mat4f y_tr_x = umeyama(x, y, /*with_scaling*/ true);
  clouds[1]->Transform(y_tr_x.topLeftCorner<3, 3>(), y_tr_x.topRightCorner<3, 1>(), /*renormalize_normals*/ true);
  
  // Show a difference image which for each pixel shows the relative distance between the corresponding points after alignment.
  Image<float> point_distance_image(images[0].size());
  point_distance_image.SetTo(0.f);
  int cloud_index = 0;
  float max_distance = 0;
  for (int y = context_radius; y < images[0].height() - context_radius; ++ y) {
    for (int x = context_radius; x < images[0].width() - context_radius; ++ x) {
      if (images[0](x, y) != 0 && images[1](x, y) != 0) {
        // NOTE: Relative distance was discarded since it wrongly suggests that the error grows proportionally with depth
        float distance = ((*clouds[0])[cloud_index].position() - (*clouds[1])[cloud_index].position()).norm();
        
        max_distance = std::max(distance, max_distance);
        point_distance_image(x, y) = distance;
        ++ cloud_index;
      } else {
        point_distance_image(x, y) = 0;
      }
    }
  }
  CHECK_EQ(cloud_index, clouds[0]->size());
  ImageDisplay point_distance_display;
  point_distance_display.Update(point_distance_image, "Point distances", 0.f, max_distance);
  
  // Save the results.
  boost::filesystem::create_directories(output_directory);
  // TODO: Save the point difference image? It is in float format though, which cannot be written directly.
  //       We could either write a binary blob that cannot be viewed easily,
  //       or write an image that can be viewed but loses precision (or both).
  // point_distance_image.Write((boost::filesystem::path(output_directory) / "point_distance_image.png").string().c_str());
  clouds[0]->WriteAsPLY((boost::filesystem::path(output_directory) / (boost::filesystem::path(stereo_directory_target).filename().string() + ".ply")).string().c_str());
  clouds[1]->WriteAsPLY((boost::filesystem::path(output_directory) / (boost::filesystem::path(stereo_directory_source).filename().string() + ".ply")).string().c_str());
  
  std::getchar();
  
  return EXIT_SUCCESS;
}

}

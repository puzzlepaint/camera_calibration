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

#include <iomanip>
#include <set>
#include <sstream>

#include <boost/filesystem.hpp>
#include <libvis/external_io/colmap_model.h>
#include <libvis/external_io/meshlab_project.h>
#include <libvis/logging.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/joint_optimization.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/camera_model.h"
#include "camera_calibration/models/central_thin_prism_fisheye.h"
#include "camera_calibration/util.h"

namespace vis {

int BundleAdjustment(const string& state_directory, const string& model_input_directory, const string& model_output_directory) {
  // Load intrinsics
  constexpr int camera_index = 0;
  ostringstream intrinsics_filename;
  intrinsics_filename << "intrinsics" << camera_index << ".yaml";
  string model_path = (boost::filesystem::path(state_directory) / intrinsics_filename.str()).string();
  if (!boost::filesystem::exists(model_path)) {
    LOG(ERROR) << "Cannot open intrinsics file: " << model_path;
    return EXIT_FAILURE;
  }
  shared_ptr<CameraModel> model(LoadCameraModel(model_path.c_str()));
  if (!model) {
    LOG(ERROR) << "Cannot load file: " << model_path;
    return EXIT_FAILURE;
  }
  
  
  // Load the input model
  Dataset dataset(/*num_cameras*/ 1);
  BAState state;
  
  // Set camera model
  state.intrinsics.resize(1);
  state.intrinsics[0] = model;
  dataset.SetImageSize(0, Vec2i(model->width(), model->height()));
  
  // cameras.txt (not used since we replace the intrinsics with our own):
//   ColmapCameraPtrMap cameras;
//   string cameras_path = (boost::filesystem::path(model_input_directory) / "cameras.txt").string();
//   if (!ReadColmapCameras(cameras_path, &cameras)) {
//     LOG(ERROR) << "Cannot read file: " << cameras_path;
//     return EXIT_FAILURE;
//   }
//   for (auto it : cameras) {
//     const ColmapCamera& colmap_camera = *it.second;
//     
//     if (colmap_camera.model_name != "THIN_PRISM_FISHEYE") {
//       LOG(ERROR) << "Camera model not handled: " << colmap_camera.model_name;
//       return EXIT_FAILURE;
//     }
//     
//     // Change the parameter ordering to match ours.
//     // Parameter order expected by Colmap:
//     // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
//     // Parameter order used by CameraCalibration:
//     // fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1
//     // --> Swap entries 6/8 and 7/9
//     bool use_equidistant_projection = true;
//     CentralThinPrismFisheyeModel* model = new CentralThinPrismFisheyeModel(colmap_camera.width, colmap_camera.height, use_equidistant_projection);
//     CHECK_EQ(colmap_camera.parameters.size(), model->parameters().size());
//     for (int i = 0; i < colmap_camera.parameters.size(); ++ i) {
//       model->parameters()[i] = colmap_camera.parameters[i];
//     }
//     std::swap(model->parameters()[6], model->parameters()[8]);
//     std::swap(model->parameters()[7], model->parameters()[9]);
//     shared_ptr<CameraModel> model_ptr = shared_ptr<CameraModel>(model);
//     
//     // ...
//   }
  
  // images.txt:
  string images_path = (boost::filesystem::path(model_input_directory) / "images.txt").string();
  ColmapImagePtrMap images;
  if (!ReadColmapImages(images_path, /*read_observations*/ true, &images)) {
    LOG(ERROR) << "Cannot read file: " << images_path;
    return EXIT_FAILURE;
  }
  
  // Determine a defined order for the images (increasing ID) such that different
  // runs of this tool will produce the same image indexings
  map<int, ColmapImagePtr> ordered_images;
  for (const auto& it : images) {
    ordered_images[it.first] = it.second;
  }
  
  // We don't support camera rigs in this tool, so set the single rig pose to identity
  state.camera_tr_rig.resize(1);
  state.camera_tr_rig[0] = SE3d();
  
  state.image_used.resize(ordered_images.size(), true);
  state.rig_tr_global.resize(ordered_images.size());
  int image_index = 0;
  for (const auto& it : ordered_images) {
    const ColmapImage& image = *it.second;
    
    state.rig_tr_global[image_index] = image.image_tr_global.cast<double>();
    
    shared_ptr<Imageset> imageset = dataset.NewImageset();
    imageset->SetFilename(image.file_path);
    
    vector<PointFeature>& features = imageset->FeaturesOfCamera(0);
    features.resize(image.observations.size());
    int output_index = 0;
    for (int i = 0; i < image.observations.size(); ++ i) {
      if (image.observations[i].point3d_id < 0) {
        continue;
      }
      features[output_index].xy = image.observations[i].xy;
      features[output_index].id = image.observations[i].point3d_id;
      ++ output_index;
    }
    features.resize(output_index);
    
    ++ image_index;
  }
  
  // points3D.txt:
  string points3d_path = (boost::filesystem::path(model_input_directory) / "points3D.txt").string();
  ColmapPoint3DMap points;
  if (!ReadColmapPoints3D(points3d_path, &points)) {
    LOG(ERROR) << "Cannot read file: " << points3d_path;
    return EXIT_FAILURE;
  }
  
  // Determine a defined order for the points (increasing ID) such that different
  // runs of this tool will produce the same image indexings
  set<int> point3d_ordering;
  for (const auto& it : points) {
    point3d_ordering.insert(it.first);
  }
  
  state.points.resize(points.size());
  state.feature_id_to_points_index.reserve(points.size());
  int point_index = 0;
  for (const auto& point_id : point3d_ordering) {
    const ColmapPoint3D& point = points.at(point_id);
    
    state.points[point_index] = point.position.cast<double>();
    state.feature_id_to_points_index[point_id] = point_index;
    ++ point_index;
  }
  
  
  // Run bundle adjustment
  state.ComputeFeatureIdToPointsIndex(&dataset);
  
  constexpr int max_iteration_count = 30;
  double lambda = -1;
  double numerical_diff_delta = 1e-4;
  for (int iteration = 0; iteration < max_iteration_count; ++ iteration) {
    double cost = OptimizeJointly(
        dataset,
        &state,
        /*max_iteration_count*/ 1,
        lambda,
        numerical_diff_delta,
        /*regularization_weight*/ 0,
        /*localize_only*/ true,
        /*eliminate_points*/ true,
        SchurMode::Dense,
        &lambda);
    
    LOG(INFO) << "[" << (iteration + 1) << "] Cost: " << cost;
    
    // Save the optimization state after each iteration
    SaveBAState(model_output_directory.c_str(), state);
    SaveDatasetAndState(model_output_directory.c_str(), dataset, state);

    // Save the final cost
    std::ofstream cost_stream((boost::filesystem::path(model_output_directory) / "cost.txt").string().c_str(), std::ios::out);
    if (cost_stream) {
      cost_stream << std::setprecision(14);
      cost_stream << cost << std::endl;
    }
    
    if (PollKeyInput() == 'q') {
      break;
    }
  }
  
  return EXIT_SUCCESS;
}


int CompareReconstructions(const string& reconstruction_path_1, const string& reconstruction_path_2) {
  // Load reconstructions
  BAState state1;
  if (!LoadBAState(reconstruction_path_1.c_str(), &state1, nullptr)) {
    return EXIT_FAILURE;
  }
  
  BAState state2;
  if (!LoadBAState(reconstruction_path_2.c_str(), &state2, nullptr)) {
    return EXIT_FAILURE;
  }
  
  // The reconstructions must contain the same images; as a weak verification,
  // check that the image count is equal
  CHECK_EQ(state1.rig_tr_global.size(), state2.rig_tr_global.size());
  
  // Align the camera centers of the two reconstructions with Umeyama.
  // This provides a scaling estimate.
  Eigen::Matrix<float, 3, Eigen::Dynamic> centers1;
  vector<SE3d> global_tr_images1(state1.rig_tr_global.size());
  centers1.resize(Eigen::NoChange, state1.rig_tr_global.size());
  for (usize i = 0; i < state1.rig_tr_global.size(); ++ i) {
    global_tr_images1[i] = (state1.camera_tr_rig[0] * state1.rig_tr_global[i]).inverse();
    centers1.col(i) = global_tr_images1[i].translation().cast<float>();
  }
  
  Eigen::Matrix<float, 3, Eigen::Dynamic> centers2;
  vector<SE3d> global_tr_images2(state2.rig_tr_global.size());
  centers2.resize(Eigen::NoChange, state2.rig_tr_global.size());
  for (usize i = 0; i < state2.rig_tr_global.size(); ++ i) {
    global_tr_images2[i] = (state2.camera_tr_rig[0] * state2.rig_tr_global[i]).inverse();
    centers2.col(i) = global_tr_images2[i].translation().cast<float>();
  }
  
  Mat4f centers2_tr_centers1 = umeyama(centers1, centers2, /*with_scaling*/ true);
  // Eigen::Matrix<float, 3, Eigen::Dynamic> aligned_centers1 = centers2_tr_centers1 * centers1;
  
  // Extract scaling
  double centers2_s_centers1 = (1 / 3.) * (
      centers2_tr_centers1.block<3, 1>(0, 0).norm() +
      centers2_tr_centers1.block<3, 1>(0, 1).norm() +
      centers2_tr_centers1.block<3, 1>(0, 2).norm());
  
  // Apply scaling to the camera poses of reconstruction 1
  centers1 *= centers2_s_centers1;
  for (usize i = 0; i < state1.rig_tr_global.size(); ++ i) {
    global_tr_images1[i].translation() *= centers2_s_centers1;
  }
  
  // Determine the intrinsics rotation between the two calibrations
  CHECK_EQ(state1.intrinsics.size(), 1);
  CHECK_EQ(state2.intrinsics.size(), 1);
  CameraModel* model1 = state1.intrinsics[0].get();
  CameraModel* model2 = state2.intrinsics[0].get();
  CHECK_EQ(model1->width(), model2->width());
  CHECK_EQ(model1->height(), model2->height());
  
  constexpr int kPixelStep = 10;
  vector<Vec3d> intrinsics1Points;
  vector<Vec3d> intrinsics2Points;
  
  // NOTE: We match line directions here. Depending on the non-central camera geometry, this might make more or less sense.
  for (int y = 0; y < model1->height(); y += kPixelStep) {
    for (int x = 0; x < model1->width(); x += kPixelStep) {
      Line3d line1, line2;
      if (model1->Unproject(x + 0.5, y + 0.5, &line1) &&
          model2->Unproject(x + 0.5, y + 0.5, &line2)) {
        intrinsics1Points.push_back(line1.direction().normalized());
        intrinsics2Points.push_back(line2.direction().normalized());
      }
    }
  }
  
  // Returns a_r_b, such that ideally: a[i] = a_r_b * b[i], for all i.
  Mat3d intrinsics1_r_intrinsics2 = DeterminePointCloudRotation(intrinsics1Points, intrinsics2Points);
  Mat4d intrinsics1_r_intrinsics2_4x4 = Mat4d::Identity();
  intrinsics1_r_intrinsics2_4x4.topLeftCorner<3, 3>() = intrinsics1_r_intrinsics2;
  
  LOG(INFO) << "intrinsics1_r_intrinsics2_4x4:\n" << intrinsics1_r_intrinsics2_4x4;
  
  // Set the first camera poses of the two reconstructions to be the same,
  // and measure the translation difference of the last one, relative to its
  // average distance to the first camera.
  // TODO: The image indices to compare should be configurable.
  //       Right now, we always use the first and last image.
  // X * global_tr_images2[0] == global_tr_images1[0] * intrinsics1_r_intrinsics2
  // --> X == global_tr_images1[0] * intrinsics1_r_intrinsics2 * global_tr_images2[0].inverse()
  SE3d firstimage1_tr_firstimage2 = SE3d(global_tr_images1.front().matrix() * intrinsics1_r_intrinsics2_4x4 * global_tr_images2.front().inverse().matrix());
  
  SE3d global_tr_images2_back_aligned_to1 = firstimage1_tr_firstimage2 * global_tr_images2.back();
  
  double endpoint_translation_difference = (global_tr_images2_back_aligned_to1.translation() - global_tr_images1.back().translation()).norm();
  
  double trajectory_length1 = 0;
  double trajectory_length2 = 0;
  for (int i = 0; i < global_tr_images1.size() - 1; ++ i) {
    trajectory_length1 += (global_tr_images1[i].translation() - global_tr_images1[i + 1].translation()).norm();
    trajectory_length2 += (global_tr_images2[i].translation() - global_tr_images2[i + 1].translation()).norm();
  }
  
  double relative_endpoint_difference = endpoint_translation_difference / (0.5 * (trajectory_length1 + trajectory_length2));
  LOG(INFO) << "relative endpoint difference: " << (100 * relative_endpoint_difference) << "%";
  
  // Write a MeshLab project that loads the reconstruction point clouds with aligned transformation
  string meshlab_project_path;
  string remaining_path_1;
  string remaining_path_2;
  for (int i = 0; i < std::min(reconstruction_path_1.size(), reconstruction_path_2.size()); ++ i) {
    if (reconstruction_path_1[i] == reconstruction_path_2[i]) {
      meshlab_project_path += reconstruction_path_1[i];
    } else {
      remaining_path_1 = reconstruction_path_1.substr(i);
      remaining_path_2 = reconstruction_path_2.substr(i);
      break;
    }
  }
  while (!meshlab_project_path.empty() && meshlab_project_path.back() != '/') {
    meshlab_project_path.pop_back();
  }
  
  MeshLabMeshInfoVector meshlab_project(4);
  
  Mat4f global_tr_1 = Mat4f::Identity();
  global_tr_1(0, 0) = centers2_s_centers1;
  global_tr_1(1, 1) = centers2_s_centers1;
  global_tr_1(2, 2) = centers2_s_centers1;
  
  
  MeshLabMeshInfo& cloud_1 = meshlab_project[0];
  cloud_1.label = "SfM cloud 1: " + remaining_path_1;
  cloud_1.filename = (boost::filesystem::absolute(reconstruction_path_1) / "points.yaml.obj").string();
  cloud_1.global_tr_mesh = global_tr_1;
  
  
  MeshLabMeshInfo& images_1 = meshlab_project[1];
  images_1.label = "SfM camera poses 1: " + remaining_path_1;
  images_1.filename = (boost::filesystem::absolute(reconstruction_path_1) / "rig_tr_global.yaml.obj").string();
  images_1.global_tr_mesh = global_tr_1;
  
  Mat4f global_tr_2 = firstimage1_tr_firstimage2.matrix().cast<float>();
  
  MeshLabMeshInfo& cloud_2 = meshlab_project[2];
  cloud_2.label = "SfM cloud 2: " + remaining_path_2;
  cloud_2.filename = (boost::filesystem::absolute(reconstruction_path_2) / "points.yaml.obj").string();
  cloud_2.global_tr_mesh = global_tr_2;
  
  MeshLabMeshInfo& images_2 = meshlab_project[3];
  images_2.label = "SfM camera poses 2: " + remaining_path_2;
  images_2.filename = (boost::filesystem::absolute(reconstruction_path_2) / "rig_tr_global.yaml.obj").string();
  images_2.global_tr_mesh = global_tr_2;
  
  string meshlab_project_aligned_at_start_path = meshlab_project_path + "reconstructions_aligned_at_start.mlp";
  if (!WriteMeshLabProject(meshlab_project_aligned_at_start_path, meshlab_project)) {
    LOG(ERROR) << "Failed to save MeshLab project to: " << meshlab_project_aligned_at_start_path;
  }
  
  // NOTE: This does only account for the camera centers, not for the point clouds, so it might
  //       introduce large deviations and might thus be badly suited for comparison.
  // cloud_1.global_tr_mesh = centers2_tr_centers1;
  // images_1.global_tr_mesh = centers2_tr_centers1;
  // cloud_2.global_tr_mesh = Mat4f::Identity();
  // images_2.global_tr_mesh = Mat4f::Identity();
  // 
  // string meshlab_project_aligned_umeyama_path = meshlab_project_path + "reconstructions_aligned_umeyama.mlp";
  // if (!WriteMeshLabProject(meshlab_project_aligned_umeyama_path, meshlab_project)) {
  //   LOG(ERROR) << "Failed to save MeshLab project to: " << meshlab_project_aligned_umeyama_path;
  // }
  
  return EXIT_SUCCESS;
}

}

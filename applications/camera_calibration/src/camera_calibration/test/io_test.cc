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


#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>
#include <libvis/sophus.h>
#include <gtest/gtest.h>
#include <QTemporaryFile>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/calibration_initialization/dense_initialization.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/models/noncentral_generic.h"

using namespace vis;

namespace vis {
void CompareDatasets(const Dataset& original, const Dataset& other) {
  ASSERT_EQ(original.num_cameras(), other.num_cameras());
  for (int camera_index = 0; camera_index < original.num_cameras(); ++ camera_index) {
    EXPECT_EQ(original.GetImageSize(camera_index), other.GetImageSize(camera_index));
  }
  
  ASSERT_EQ(original.KnownGeometriesCount(), other.KnownGeometriesCount());
  for (int known_geometry_index = 0; known_geometry_index < original.KnownGeometriesCount(); ++ known_geometry_index) {
    const KnownGeometry& kg0 = original.GetKnownGeometry(known_geometry_index);
    const KnownGeometry& kg1 = other.GetKnownGeometry(known_geometry_index);
    
    EXPECT_NEAR(kg0.cell_length_in_meters, kg1.cell_length_in_meters, 1e-6);
    
    ASSERT_EQ(kg0.feature_id_to_position.size(), kg1.feature_id_to_position.size());
    for (const auto& item : kg0.feature_id_to_position) {
      EXPECT_EQ(item.second, kg1.feature_id_to_position.at(item.first));
    }
  }
  
  ASSERT_EQ(original.ImagesetCount(), other.ImagesetCount());
  for (int imageset_index = 0; imageset_index < original.ImagesetCount(); ++ imageset_index) {
    shared_ptr<const Imageset> im0 = original.GetImageset(imageset_index);
    shared_ptr<const Imageset> im1 = other.GetImageset(imageset_index);
    
    EXPECT_EQ(im0->GetFilename(), im1->GetFilename());
    for (int camera_index = 0; camera_index < original.num_cameras(); ++ camera_index) {
      const vector<PointFeature>& f0 = im0->FeaturesOfCamera(camera_index);
      const vector<PointFeature>& f1 = im1->FeaturesOfCamera(camera_index);
      
      ASSERT_EQ(f0.size(), f1.size());
      for (usize i = 0; i < f0.size(); ++ i) {
        EXPECT_EQ(f0[i].id, f1[i].id);
        EXPECT_NEAR(f0[i].xy.x(), f1[i].xy.x(), 1e-6);
        EXPECT_NEAR(f0[i].xy.y(), f1[i].xy.y(), 1e-6);
      }
    }
  }
}
}

TEST(IO, DenseInitializationSaveAndLoad) {
  DenseInitialization dense;
  
  constexpr int kNumKGs = 6;
  dense.known_geometry_localized.resize(kNumKGs);
  dense.global_r_known_geometry.resize(kNumKGs);
  dense.global_t_known_geometry.resize(kNumKGs);
  
  for (int i = 0; i < kNumKGs; ++ i) {
    dense.known_geometry_localized[i] = i % 2;
    SE3f pose = SE3f::exp(SE3f::Tangent::Random());
    dense.global_r_known_geometry[i] = pose.rotationMatrix();
    dense.global_t_known_geometry[i] = pose.translation();
  }
  
  constexpr int kNumCameras = 2;
  constexpr int kNumImages = 11;
  dense.image_used.resize(kNumCameras);
  dense.image_tr_global.resize(kNumCameras);
  
  for (int c = 0; c < kNumCameras; ++ c) {
    dense.image_used[c].resize(kNumImages);
    dense.image_tr_global[c].resize(kNumImages);
    for (int i = 0; i < kNumImages; ++ i) {
      dense.image_used[c][i] = (rand() % 2 == 0);
      dense.image_tr_global[c][i] = SE3d::exp(SE3d::Tangent::Random());
    }
  }
  
  dense.observation_directions.resize(kNumCameras);
  for (int c = 0; c < kNumCameras; ++ c) {
    dense.observation_directions[c].SetSize(10 + rand() % 10, 10 + rand() % 10);
    for (int y = 0; y < dense.observation_directions[c].height(); ++ y) {
      for (int x = 0; x < dense.observation_directions[c].width(); ++ x) {
        dense.observation_directions[c](x, y) = Vec3d::Random();
      }
    }
  }
  
  // Save and load
  QTemporaryFile temp_file;
  CHECK(temp_file.open());
  ASSERT_TRUE(SaveDenseInitialization(temp_file.fileName().toStdString().c_str(), dense));
  DenseInitialization loaded;
  ASSERT_TRUE(LoadDenseInitialization(temp_file.fileName().toStdString().c_str(), &loaded));
  
  // Compare
  EXPECT_EQ(dense.known_geometry_localized.size(), loaded.known_geometry_localized.size());
  for (int i = 0; i < std::min(dense.known_geometry_localized.size(), loaded.known_geometry_localized.size()); ++ i) {
    EXPECT_EQ(dense.known_geometry_localized[i], loaded.known_geometry_localized[i]);
  }
  
  EXPECT_EQ(dense.global_r_known_geometry.size(), loaded.global_r_known_geometry.size());
  for (int i = 0; i < std::min(dense.global_r_known_geometry.size(), loaded.global_r_known_geometry.size()); ++ i) {
    for (int r = 0; r < 3; ++ r) {
      for (int c = 0; c < 3; ++ c) {
        EXPECT_EQ(dense.global_r_known_geometry[i](r, c), loaded.global_r_known_geometry[i](r, c));
      }
    }
  }
  
  EXPECT_EQ(dense.global_t_known_geometry.size(), loaded.global_t_known_geometry.size());
  for (int i = 0; i < std::min(dense.global_t_known_geometry.size(), loaded.global_t_known_geometry.size()); ++ i) {
    EXPECT_EQ(dense.global_t_known_geometry[i], loaded.global_t_known_geometry[i]);
  }
  
  EXPECT_EQ(dense.image_used.size(), loaded.image_used.size());
  for (int i = 0; i < std::min(dense.image_used.size(), loaded.image_used.size()); ++ i) {
    EXPECT_EQ(dense.image_used[i].size(), loaded.image_used[i].size());
    for (int k = 0; k < std::min(dense.image_used[i].size(), loaded.image_used[i].size()); ++ k) {
      EXPECT_EQ(dense.image_used[i][k], loaded.image_used[i][k]);
    }
  }
  
  EXPECT_EQ(dense.image_tr_global.size(), loaded.image_tr_global.size());
  for (int i = 0; i < std::min(dense.image_tr_global.size(), loaded.image_tr_global.size()); ++ i) {
    EXPECT_EQ(dense.image_tr_global[i].size(), loaded.image_tr_global[i].size());
    for (int k = 0; k < std::min(dense.image_tr_global[i].size(), loaded.image_tr_global[i].size()); ++ k) {
      for (int v = 0; v < 7; ++ v) {
        EXPECT_FLOAT_EQ(dense.image_tr_global[i][k].data()[v], loaded.image_tr_global[i][k].data()[v]);
      }
    }
  }
  
  EXPECT_EQ(dense.observation_directions.size(), loaded.observation_directions.size());
  for (int i = 0; i < std::min(dense.observation_directions.size(), loaded.observation_directions.size()); ++ i) {
    EXPECT_EQ(dense.observation_directions[i].size(), loaded.observation_directions[i].size());
    if (dense.observation_directions[i].size() == loaded.observation_directions[i].size()) {
      for (int y = 0; y < dense.observation_directions[i].height(); ++ y) {
        for (int x = 0; x < dense.observation_directions[i].width(); ++ x) {
          EXPECT_FLOAT_EQ(dense.observation_directions[i](x, y).x(), loaded.observation_directions[i](x, y).x());
          EXPECT_FLOAT_EQ(dense.observation_directions[i](x, y).y(), loaded.observation_directions[i](x, y).y());
          EXPECT_FLOAT_EQ(dense.observation_directions[i](x, y).z(), loaded.observation_directions[i](x, y).z());
        }
      }
    }
  }
}

TEST(IO, DatasetSaveAndLoad) {
  Dataset dataset(2);
  
  dataset.SetImageSize(0, Vec2i(123, 456));
  dataset.SetImageSize(1, Vec2i(789, 135));
  
  dataset.SetKnownGeometriesCount(2);
  
  KnownGeometry& kg0 = dataset.GetKnownGeometry(0);
  kg0.cell_length_in_meters = 1.23;
  kg0.feature_id_to_position[0] = Vec2i(12, 34);
  kg0.feature_id_to_position[16] = Vec2i(9, 8);
  
  KnownGeometry& kg1 = dataset.GetKnownGeometry(1);
  kg1.cell_length_in_meters = 2.22;
  kg1.feature_id_to_position[3] = Vec2i(44, 33);
  kg1.feature_id_to_position[5] = Vec2i(1, 2);
  kg1.feature_id_to_position[6] = Vec2i(3, 4);
  
  shared_ptr<Imageset> is0 = dataset.NewImageset();
  is0->SetFilename("imageset0.png");
  vector<PointFeature>& is0_f0 = is0->FeaturesOfCamera(0);
  is0_f0.emplace_back(Vec2f(0.f, 2.f), 123);
  is0_f0.emplace_back(Vec2f(1.f, 1.5f), 665);
  
  shared_ptr<Imageset> is1 = dataset.NewImageset();
  is1->SetFilename("imageset0.png");
  vector<PointFeature>& is1_f0 = is1->FeaturesOfCamera(0);
  is1_f0.emplace_back(Vec2f(4.f, 6.f), 123);
  is1_f0.emplace_back(Vec2f(2.f, 2.5f), 660);
  vector<PointFeature>& is1_f1 = is1->FeaturesOfCamera(1);
  is1_f1.emplace_back(Vec2f(6.f, 7.f), 123);
  is1_f1.emplace_back(Vec2f(4.f, 4.5f), 665);
  
  // Save and load
  QTemporaryFile temp_file;
  CHECK(temp_file.open());
  ASSERT_TRUE(SaveDataset(temp_file.fileName().toStdString().c_str(), dataset));
  Dataset loaded_dataset_binary(2);
  ASSERT_TRUE(LoadDataset(temp_file.fileName().toStdString().c_str(), &loaded_dataset_binary));
  CompareDatasets(dataset, loaded_dataset_binary);
}

// TODO: Also create tests for saving and loading the other camera models
TEST(IO, CentralGenericModelSaveAndLoad) {
  // Define model
  constexpr int kCameraWidth = 640;
  constexpr int kCameraHeight = 480;
  
  CentralGenericModel model(
      /*grid_resolution_x*/ 8,
      /*grid_resolution_y*/ 6,
      10, 20,
      kCameraWidth - 5,
      kCameraHeight - 8,
      kCameraWidth,
      kCameraHeight);
  
  Image<Vec3d> grid(model.grid().size());
  for (u32 y = 0; y < grid.height(); ++ y) {
    for (u32 x = 0; x < grid.width(); ++ x) {
      grid(x, y) = Vec3d(x, y, 1).normalized();
    }
  }
  model.SetGrid(grid);
  
  // Save and load model
  QTemporaryFile temp_file;
  CHECK(temp_file.open());
  ASSERT_TRUE(SaveCameraModel(model, temp_file.fileName().toStdString().c_str()));
  shared_ptr<CameraModel> loaded_generic_model = LoadCameraModel(temp_file.fileName().toStdString().c_str());
  CentralGenericModel* loaded_model = dynamic_cast<CentralGenericModel*>(loaded_generic_model.get());
  
  ASSERT_TRUE(loaded_model != nullptr);
  
  for (u32 y = 0; y < grid.height(); ++ y) {
    for (u32 x = 0; x < grid.width(); ++ x) {
      EXPECT_LE((grid(x, y) - loaded_model->grid()(x, y)).norm(), 1e-6);
    }
  }
  
  EXPECT_NEAR(model.calibration_min_x(), loaded_model->calibration_min_x(), 1e-6);
  EXPECT_NEAR(model.calibration_min_y(), loaded_model->calibration_min_y(), 1e-6);
  EXPECT_NEAR(model.calibration_max_x(), loaded_model->calibration_max_x(), 1e-6);
  EXPECT_NEAR(model.calibration_max_y(), loaded_model->calibration_max_y(), 1e-6);
  
  EXPECT_EQ(model.width(), loaded_model->width());
  EXPECT_EQ(model.height(), loaded_model->height());
  EXPECT_EQ(model.type(), loaded_model->type());
}

TEST(IO, NoncentralGenericModelSaveAndLoad) {
  // Define model
  constexpr int kCameraWidth = 640;
  constexpr int kCameraHeight = 480;
  
  NoncentralGenericModel model(
      /*grid_resolution_x*/ 8,
      /*grid_resolution_y*/ 6,
      10, 20,
      kCameraWidth - 5,
      kCameraHeight - 8,
      kCameraWidth,
      kCameraHeight);
  
  Image<Vec3d> direction_grid(model.direction_grid().size());
  for (u32 y = 0; y < direction_grid.height(); ++ y) {
    for (u32 x = 0; x < direction_grid.width(); ++ x) {
      direction_grid(x, y) = Vec3d(x, y, 1).normalized();
    }
  }
  model.SetDirectionGrid(direction_grid);
  
  Image<Vec3d> point_grid(model.point_grid().size());
  for (u32 y = 0; y < point_grid.height(); ++ y) {
    for (u32 x = 0; x < point_grid.width(); ++ x) {
      point_grid(x, y) = Vec3d(x, y, 1);
    }
  }
  model.SetPointGrid(point_grid);
  
  // Save and load model
  QTemporaryFile temp_file;
  CHECK(temp_file.open());
  ASSERT_TRUE(SaveCameraModel(model, temp_file.fileName().toStdString().c_str()));
  shared_ptr<CameraModel> loaded_generic_model = LoadCameraModel(temp_file.fileName().toStdString().c_str());
  NoncentralGenericModel* loaded_model = dynamic_cast<NoncentralGenericModel*>(loaded_generic_model.get());
  
  ASSERT_TRUE(loaded_model != nullptr);
  
  for (u32 y = 0; y < direction_grid.height(); ++ y) {
    for (u32 x = 0; x < direction_grid.width(); ++ x) {
      EXPECT_LE((direction_grid(x, y) - loaded_model->direction_grid()(x, y)).norm(), 1e-6);
      EXPECT_LE((point_grid(x, y) - loaded_model->point_grid()(x, y)).norm(), 1e-6);
    }
  }
  
  EXPECT_NEAR(model.calibration_min_x(), loaded_model->calibration_min_x(), 1e-6);
  EXPECT_NEAR(model.calibration_min_y(), loaded_model->calibration_min_y(), 1e-6);
  EXPECT_NEAR(model.calibration_max_x(), loaded_model->calibration_max_x(), 1e-6);
  EXPECT_NEAR(model.calibration_max_y(), loaded_model->calibration_max_y(), 1e-6);
  
  EXPECT_EQ(model.width(), loaded_model->width());
  EXPECT_EQ(model.height(), loaded_model->height());
  EXPECT_EQ(model.type(), loaded_model->type());
}

TEST(IO, PosesSaveAndLoad) {
  constexpr int kNumImages = 50;
  
  vector<bool> image_used(kNumImages);
  vector<SE3d> image_tr_pattern(kNumImages);
  
  for (int i = 0; i < kNumImages; ++ i) {
    image_used[i] = (i % 2) == 1;
    image_tr_pattern[i] = SE3d::exp(SE3d::Tangent::Random());
  }
  
  // Save and load poses
  QTemporaryFile temp_file;
  CHECK(temp_file.open());
  ASSERT_TRUE(SavePoses(image_used, image_tr_pattern, temp_file.fileName().toStdString().c_str()));
  
  vector<bool> loaded_image_used;
  vector<SE3d> loaded_image_tr_pattern;
  ASSERT_TRUE(LoadPoses(&loaded_image_used, &loaded_image_tr_pattern, temp_file.fileName().toStdString().c_str()));
  
  for (int i = 0; i < kNumImages; ++ i) {
    EXPECT_EQ(image_used[i], loaded_image_used[i]);
    if (loaded_image_used[i]) {
      EXPECT_LE((image_tr_pattern[i].inverse() * loaded_image_tr_pattern[i]).log().norm(), 1e-6);
    }
  }
}

TEST(IO, PatternSaveAndLoad) {
  BAState calibration;
  
  calibration.points.resize(50);
  for (u32 i = 0; i < calibration.points.size(); ++ i) {
    calibration.points[i] = Vec3d::Random();
    calibration.feature_id_to_points_index.insert(make_pair(rand(), i));
  }
  
  // Save and load pattern
  QTemporaryFile temp_file;
  CHECK(temp_file.open());
  ASSERT_TRUE(SavePointsAndIndexMapping(calibration, temp_file.fileName().toStdString().c_str()));
  
  vector<Vec3d> loaded_optimized_geometry;
  unordered_map<int, int> loaded_feature_id_to_points_index;
  
  ASSERT_TRUE(LoadPointsAndIndexMapping(
      &loaded_optimized_geometry,
      &loaded_feature_id_to_points_index,
      temp_file.fileName().toStdString().c_str()));
  
  for (u32 i = 0; i < calibration.points.size(); ++ i) {
    EXPECT_NEAR(calibration.points[i].x(), loaded_optimized_geometry[i].x(), 1e-6);
    EXPECT_NEAR(calibration.points[i].y(), loaded_optimized_geometry[i].y(), 1e-6);
    EXPECT_NEAR(calibration.points[i].z(), loaded_optimized_geometry[i].z(), 1e-6);
  }
  
  ASSERT_EQ(calibration.feature_id_to_points_index.size(), loaded_feature_id_to_points_index.size());
  for (auto& item : loaded_feature_id_to_points_index) {
    EXPECT_EQ(calibration.feature_id_to_points_index[item.first], item.second);
  }
}

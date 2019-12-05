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

#include <unordered_map>

#include <libvis/image_display.h>
#include <libvis/logging.h>

#include "camera_calibration/dataset.h"
#include "camera_calibration/io/calibration_io.h"

namespace vis {

int IntersectDatasets(const vector<string>& features_path, double intersection_threshold) {
  bool match_by_filename = true;  // If false, matches by imageset index. // TODO: Make parameter?
  
  double threshold_squared = intersection_threshold * intersection_threshold;
  vector<Dataset> datasets(features_path.size());
  
  // Load all datasets
  for (int i = 0; i < datasets.size(); ++ i) {
    LOG(INFO) << "Dataset " << i << ": " << features_path[i];
    if (!LoadDataset(features_path[i].c_str(), &datasets[i])) {
      return EXIT_FAILURE;
    }
    if (i > 0 && datasets[i].num_cameras() != datasets[0].num_cameras()) {
      LOG(ERROR) << "Number of cameras in dataset " << features_path[i] << " does not match the number of cameras in dataset " << features_path[0];
      return EXIT_FAILURE;
    }
  }
  
  // Print number of original features
  for (int i = 0; i < datasets.size(); ++ i) {
    int feature_count = 0;
    
    for (int imageset_index = 0; imageset_index < datasets[i].ImagesetCount(); ++ imageset_index) {
      const shared_ptr<Imageset>& imageset = datasets[i].GetImageset(imageset_index);
      for (int camera_index = 0; camera_index < datasets[i].num_cameras(); ++ camera_index) {
        feature_count += imageset->FeaturesOfCamera(camera_index).size();
      }
    }
    
    LOG(INFO) << "Input features in dataset " << i << ": " << feature_count << " (#imagesets: " << datasets[i].ImagesetCount() << ")";
  }
  
  if (!match_by_filename) {
    // Assign each imageset a fake filename based on its imageset index.
    for (int i = 0; i < datasets.size(); ++ i) {
      for (int imageset_index = 0; imageset_index < datasets[i].ImagesetCount(); ++ imageset_index) {
        ostringstream number;
        number << imageset_index;
        datasets[i].GetImageset(imageset_index)->SetFilename(number.str());
      }
    }
  }
  
  // Build filenames_to_imageset_indices
  vector<unordered_map<string, int>> filenames_to_imageset_indices(datasets.size());  // maps [dataset_index][filename] --> imageset_index
  for (int i = 0; i < datasets.size(); ++ i) {
    for (int imageset_index = 0; imageset_index < datasets[i].ImagesetCount(); ++ imageset_index) {
      filenames_to_imageset_indices[i].insert(make_pair(datasets[i].GetImageset(imageset_index)->GetFilename(), imageset_index));
    }
  }
  
  // Iterate over all imagesets in dataset 0 and intersect them with those from the other datasets
  for (int imageset_index = 0; imageset_index < datasets[0].ImagesetCount(); ++ imageset_index) {
    vector<shared_ptr<Imageset>> imagesets(datasets.size());
    imagesets[0] = datasets[0].GetImageset(imageset_index);
    
    // Check whether all other datasets also contain an imageset with this filename.
    // If not, delete all the imagesets with this filename.
    bool is_in_all_datasets = true;
    for (int i = 1; i < datasets.size(); ++ i) {
      auto it = filenames_to_imageset_indices[i].find(imagesets[0]->GetFilename());
      if (it == filenames_to_imageset_indices[i].end()) {
        is_in_all_datasets = false;
        break;
      }
      imagesets[i] = datasets[i].GetImageset(it->second);
    }
    if (!is_in_all_datasets) {
      // Delete all imagesets with this filename.
      string filename = imagesets[0]->GetFilename();
      for (int i = 0; i < datasets.size(); ++ i) {
        auto it = filenames_to_imageset_indices[i].find(filename);
        if (it == filenames_to_imageset_indices[i].end()) {
          continue;
        }
        int index_to_delete = it->second;
        datasets[i].DeleteImageset(index_to_delete);
        for (auto& item : filenames_to_imageset_indices[i]) {
          if (item.second > index_to_delete) {
            -- item.second;
          }
        }
        filenames_to_imageset_indices[i].erase(filename);
      }
      
      -- imageset_index;
      continue;
    }
    
    // Intersect the features in these imagesets.
    // Greedily pick features from imagesets[0] successively and find others to intersect them with.
    for (int camera_index = 0; camera_index < datasets[0].num_cameras(); ++ camera_index) {
      vector<vector<PointFeature>*> features(datasets.size());
      for (int i = 0; i < datasets.size(); ++ i) {
        features[i] = &imagesets[i]->FeaturesOfCamera(camera_index);
      }
      
      vector<Vec2f> intersection_centers;
      for (int f = 0; f < features[0]->size(); ++ f) {
        Vec2f intersection_center = features[0]->at(f).xy;
        
        // Repeat until the set of "covered" features does not change anymore
        vector<int> covered_feature_indices(datasets.size(), -1);
        vector<int> old_covered_feature_indices;
        while (true) {
          // Find all features that are closer than the threshold, and set intersection_center to the average of the covered features
          Vec2f covered_features_sum = Vec2f::Zero();
          int covered_features_count = 0;
          
          for (int i = 0; i < datasets.size(); ++ i) {
            // Find closest feature
            int closest_feature_index = -1;
            double closest_feature_distance_squared = threshold_squared;
            
            for (int o = 0; o < features[i]->size(); ++ o) {
              double distance_squared = (features[i]->at(o).xy - intersection_center).squaredNorm();
              if (distance_squared <= closest_feature_distance_squared) {
                closest_feature_distance_squared = distance_squared;
                closest_feature_index = o;
              }
            }
            
            if (closest_feature_index >= 0) {
              covered_features_sum += features[i]->at(closest_feature_index).xy;
              ++ covered_features_count;
              
              covered_feature_indices[i] = closest_feature_index;
            } else {
              covered_feature_indices[i] = -1;
            }
          }
          
          intersection_center = covered_features_sum / covered_features_count;
          
          if (covered_feature_indices == old_covered_feature_indices) {
            break;
          }
          old_covered_feature_indices = covered_feature_indices;
        }
        
        // If we have found a feature for each dataset, accept the features.
        // If not, delete them.
        bool accept = true;
        for (int i = 0; i < datasets.size(); ++ i) {
          if (covered_feature_indices[i] < 0) {
            accept = false;
            break;
          }
        }
        
        if (accept) {
          intersection_centers.push_back(intersection_center);
        } else {
          for (int i = 0; i < datasets.size(); ++ i) {
            if (covered_feature_indices[i] >= 0) {
              features[i]->erase(features[i]->begin() + covered_feature_indices[i]);
            }
          }
          if (covered_feature_indices[0] <= f) {
            -- f;
          }
        }
      }
      
      // Delete all features from the current set of imagesets that haven't been included in intersections
      // (i.e., are too far away from all intersection_centers).
      for (int i = 0; i < datasets.size(); ++ i) {
        for (int f = 0; f < features[i]->size(); ++ f) {
          bool accept = false;
          for (int c = 0; c < intersection_centers.size(); ++ c) {
            float distance_squared = (intersection_centers[c] - features[i]->at(f).xy).squaredNorm();
            if (distance_squared <= threshold_squared) {
              accept = true;
              break;
            }
          }
          
          // Delete the feature if no close-enough intersection center was found
          if (!accept) {
            features[i]->erase(features[i]->begin() + f);
            -- f;
          }
        }
      }
    }  // loop over all cameras [camera_index]
  }  // loop over all imagesets in dataset 0 [imageset_index]
  
  // Delete all imagesets from datasets 1 .. N whose filename we did not see in dataset 0.
  for (int i = 1; i < datasets.size(); ++ i) {
    for (int imageset_index = 0; imageset_index < datasets[i].ImagesetCount(); ++ imageset_index) {
      if (filenames_to_imageset_indices[0].count(datasets[i].GetImageset(imageset_index)->GetFilename()) == 0) {
        datasets[i].DeleteImageset(imageset_index);
        -- imageset_index;
      }
    }
  }
  
  // Save intersected datasets
  for (int i = 0; i < datasets.size(); ++ i) {
    string intersected_dataset_path = features_path[i] + ".intersected.bin";
    if (!SaveDataset(intersected_dataset_path.c_str(), datasets[i])) {
      return EXIT_FAILURE;
    }
  }
  
  // Print number of accepted features (for debugging: per dataset)
  for (int i = 0; i < datasets.size(); ++ i) {
    int feature_count = 0;
    
    for (int imageset_index = 0; imageset_index < datasets[i].ImagesetCount(); ++ imageset_index) {
      const shared_ptr<Imageset>& imageset = datasets[i].GetImageset(imageset_index);
      for (int camera_index = 0; camera_index < datasets[i].num_cameras(); ++ camera_index) {
        feature_count += imageset->FeaturesOfCamera(camera_index).size();
      }
    }
    
    LOG(INFO) << "Remaining features in dataset " << i << ": " << feature_count;
  }
  
  return EXIT_SUCCESS;
}

}

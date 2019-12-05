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

#include <atomic>
#include <mutex>
#include <vector>

#include <libvis/libvis.h>
#include <QDir>

#include "camera_calibration/image_input/image_input.h"
#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"

namespace vis {

class Dataset;
class MainWindow;

/// Receives images that are recorded live (in NewImageset()), processes and
/// potentially saves them, and sends visualizations to the UI.
class LiveImageConsumer : public ImageConsumer {
 Q_OBJECT
 public:
  LiveImageConsumer(
      bool live_detection,
      FeatureDetectorTaggedPattern* detector,
      Dataset* dataset,
      std::mutex* dataset_mutex,
      MainWindow* main_window,
      bool record_images,
      bool record_images_with_detections_only,
      const vector<QDir>& image_record_directories);
  
  void RecordOneImageset();
  
  void NewImageset(const vector<Image<Vec3u8>>& images) override;
  
  inline bool is_record_one_imageset_pending() const { return record_one_imageset; }
  
  void SetImageRecordDirectories(const vector<QDir>& directories) { image_record_directories = directories; }
  vector<QDir> get_image_record_directories() const { return image_record_directories; }
  
 signals:
  void NewImageSignal(int camera_index, QSharedPointer<Image<Vec3u8>> image);
  void NewDetectionsPerPixelSignal(int camera_index, QSharedPointer<Image<Vec3u8>> image);
  
 private:
  atomic<bool> record_one_imageset;
  
  bool live_detection;
  bool record_images;
  bool record_images_with_detections_only;
  int recorded_image_number = 0;
  vector<QDir> image_record_directories;
  
  vector<Image<u8>> detections_per_pixel;
  FeatureDetectorTaggedPattern* detector;
  Dataset* dataset;
  std::mutex* dataset_mutex;
};

}

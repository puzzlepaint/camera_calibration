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

#include "camera_calibration/ui/live_image_consumer.h"

#include "camera_calibration/dataset.h"
#include "camera_calibration/ui/main_window.h"

namespace vis {

LiveImageConsumer::LiveImageConsumer(
    bool live_detection,
    FeatureDetectorTaggedPattern* detector,
    Dataset* dataset,
    std::mutex* dataset_mutex,
    MainWindow* main_window,
    bool record_images,
    bool record_images_with_detections_only,
    const vector<QDir>& image_record_directories)
    : live_detection(live_detection),
      record_images(record_images),
      record_images_with_detections_only(record_images_with_detections_only),
      image_record_directories(image_record_directories),
      detector(detector),
      dataset(dataset),
      dataset_mutex(dataset_mutex) {
  record_one_imageset = false;
  detections_per_pixel.resize(dataset->num_cameras());
  if (main_window) {
    connect(this, &LiveImageConsumer::NewImageSignal, main_window, &MainWindow::NewImage, Qt::QueuedConnection);
    connect(this, &LiveImageConsumer::NewDetectionsPerPixelSignal, main_window, &MainWindow::NewDetectionsPerPixel, Qt::QueuedConnection);
  }
}

void LiveImageConsumer::RecordOneImageset() {
  if (record_one_imageset) {
    LOG(ERROR) << "record_one_imageset is already true in RecordOneImageset(), a previous call might get lost";
  }
  record_one_imageset = true;
}

void LiveImageConsumer::NewImageset(const vector<Image<Vec3u8>>& images) {
  bool live_detection_in_this_call = live_detection;
  bool record_images_in_this_call = record_images;
  bool disable_record_one_imageset = false;
  if (record_one_imageset) {
    live_detection_in_this_call = true;
    record_images_in_this_call = true;
    disable_record_one_imageset = true;
  }
  
  vector<QSharedPointer<Image<Vec3u8>>> detection_visualizations(images.size());
  
  // Run feature detection on the images, and add the result to the dataset.
  // Note that the lock duration could be shrunk to the actual storing of the
  // results in the dataset. However, the lock time is not very relevant here.
  dataset_mutex->lock();
  bool have_features = false;
  shared_ptr<Imageset> new_imageset = dataset->NewImageset();
  for (int camera_index = 0; camera_index < images.size(); ++ camera_index) {
    dataset->SetImageSize(camera_index, images[camera_index].size());
    
    if (detector && live_detection_in_this_call) {
      detection_visualizations[camera_index].reset(new Image<Vec3u8>());
      vector<PointFeature>& features = new_imageset->FeaturesOfCamera(camera_index);
      detector->DetectFeatures(images[camera_index], &features, detection_visualizations[camera_index].data());
      have_features |= !features.empty();
    }
  }
  if (!have_features) {
    // No need to store an empty imageset
    dataset->DeleteLastImageset();
  }
  dataset_mutex->unlock();
  
  // Update detection-count-per-pixel visualizations
  if (have_features) {
    for (int camera_index = 0; camera_index < images.size(); ++ camera_index) {
      const Image<Vec3u8>& image = images[camera_index];
      Image<u8>& detections = detections_per_pixel[camera_index];
      
      if (detections.empty()) {
        detections.SetSize(image.size());
        detections.SetTo(static_cast<u8>(0));
      }
      for (const PointFeature& feature : new_imageset->FeaturesOfCamera(camera_index)) {
        Vec2i xy = feature.xy.cast<int>().cwiseMax(Vec2i(0, 0)).cwiseMin(image.size().cast<int>() - Vec2i(1, 1));
        detections(xy) = std::min(255, detections(xy) + 1);
      }
      QSharedPointer<Image<Vec3u8>> visualization(new Image<Vec3u8>(detections.size()));
      for (int y = 0; y < image.height(); ++ y) {
        Vec3u8* ptr = visualization->row(y);
        for (int x = 0; x < image.width(); ++ x) {
          u8 count = detections(x, y);
          if (count == 0) {
            *ptr++ = Vec3u8(0, 0, 0);
          } else {
            constexpr int kMaxCount = 4;
            float factor = std::min(1.f, (count / (1.f * kMaxCount)));
            *ptr++ = Vec3u8(255 - 255 * factor, 255 * factor, 127);
          }
        }
      }
      emit NewDetectionsPerPixelSignal(camera_index, visualization);
    }
  }
  
  for (int camera_index = 0; camera_index < images.size(); ++ camera_index) {
    emit NewImageSignal(
        camera_index,
        (detector && live_detection_in_this_call) ?
            detection_visualizations[camera_index] :
            QSharedPointer<Image<Vec3u8>>(new Image<Vec3u8>(images[camera_index])));
  }
  
  // Save images?
  if (record_images_in_this_call && (!record_images_with_detections_only || have_features)) {
    // LOG(INFO) << "debug: LiveImageConsumer: writing images ...";
    for (int camera_index = 0; camera_index < images.size(); ++ camera_index) {
      QDir dir = image_record_directories[camera_index];
      if (!dir.exists()) {
        dir.mkpath(".");
      }
      QString filename = QString::number(recorded_image_number).rightJustified(6, '0') + ".png";
      QString path = dir.filePath(filename);
      
      if (images[camera_index].Write(path.toStdString())) {
        new_imageset->SetFilename(filename.toStdString());
      } else {
        LOG(ERROR) << "Could not write image to " << path.toStdString();
      }
    }
    
    ++ recorded_image_number;
  }
  
  if (disable_record_one_imageset) {
    record_one_imageset = false;
  }
}

}

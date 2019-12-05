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

#include <vector>

#include <libvis/image.h>
#include <libvis/image_display_qt_window.h>
#include <QComboBox>
#include <QMainWindow>
#include <QStackedLayout>
#include <QTabWidget>

namespace vis {

struct BAState;
class Dataset;

/// Window which shows a number of visualizations for running feature extraction
/// or calibration tasks.
class CalibrationWindow : public QMainWindow {
 Q_OBJECT
 public:
  CalibrationWindow(
      QWidget* parent = nullptr,
      Qt::WindowFlags flags = Qt::WindowFlags());
  
  void SetDataset(Dataset* dataset);
  
  void SetCurrentCameraIndex(int camera_index);
  
  void UpdateFeatureDetection(int camera_index, const Image<Vec3u8>& image);
  void UpdateInitialization(int camera_index, const Image<Vec3u8>& image);
  void UpdateObservationDirections(int camera_index, const Image<Vec3u8>& image);
  void UpdateErrorHistogram(int camera_index, const Image<u8>& image);
  void UpdateReprojectionErrors(int camera_index, const Image<Vec3u8>& image, const Dataset* dataset, const BAState* state);
  void UpdateErrorDirections(int camera_index, const Image<Vec3u8>& image);
  void UpdateRemovedOutliers(int camera_index, const Image<Vec3u8>& image);
  
 private:
  template <typename T>
  void CreateOrUpdateDisplay(
      const QString& tab_name,
      int camera_index,
      const Image<T>& image,
      vector<ImageDisplayQtWindow*>* displays) {
    RunInQtThreadBlocking([&]() {
      if (camera_index < 0 || camera_index >= displays->size()) {
        LOG(ERROR) << "Invalid camera index or too small displays vector. camera_index: " << camera_index << ", displays->size(): " << displays->size();
        return;
      }
      
      if (!(*displays)[camera_index]) {
        (*displays)[camera_index] = new ImageDisplayQtWindow(nullptr);
        (*displays)[camera_index]->SetDisplayAsWidget();
        (*displays)[camera_index]->FitContent();
        
        tab_widgets[camera_index]->addTab((*displays)[camera_index], tab_name);
      }
      
      (*displays)[camera_index]->SetImage(image);
    });
  }
  
  int current_camera_index;
  
  QComboBox* camera_selector;
  
  QStackedLayout* cameras_stack;
  vector<QString> camera_names;
  vector<QTabWidget*> tab_widgets;  // indexed by camera_index
  vector<ImageDisplayQtWindow*> feature_detection_displays;
  vector<ImageDisplayQtWindow*> initialization_displays;
  vector<ImageDisplayQtWindow*> observation_directions_displays;
  vector<ImageDisplayQtWindow*> error_histogram_displays;
  vector<ImageDisplayQtWindow*> reprojection_error_displays;
  vector<ImageDisplayQtWindow*> error_direction_displays;
  vector<ImageDisplayQtWindow*> removed_outliers_displays;
};

}

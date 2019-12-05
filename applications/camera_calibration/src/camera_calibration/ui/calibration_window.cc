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

#include "camera_calibration/ui/calibration_window.h"

#include <libvis/qt_thread.h>
#include <QBoxLayout>
#include <QLabel>
#include <QStatusBar>
#include <QTabWidget>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/dataset.h"

namespace vis {

CalibrationWindow::CalibrationWindow(
    QWidget* parent,
    Qt::WindowFlags flags)
    : QMainWindow(parent, flags) {
  current_camera_index = 0;
  
  QLabel* camera_selector_label = new QLabel(tr("Camera: "));
  camera_selector = new QComboBox();
  camera_selector->addItem(tr("current                   "));  // needs spaces at the end to reserve width in the widget layout for later changes
  QHBoxLayout* camera_selector_layout = new QHBoxLayout();
  camera_selector_layout->addStretch(1);
  camera_selector_layout->addWidget(camera_selector_label);
  camera_selector_layout->addWidget(camera_selector);
  
  cameras_stack = new QStackedLayout();
  
  QVBoxLayout* layout = new QVBoxLayout();
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addLayout(camera_selector_layout);
  layout->addLayout(cameras_stack);
  
  statusBar()->setSizeGripEnabled(true);
  statusBar()->show();
  
  QWidget* widget = new QWidget();
  widget->setLayout(layout);
  setCentralWidget(widget);
  setWindowTitle(tr("Camera calibration"));
  resize(800, 600);
  
  connect(camera_selector, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int index) {
    int selected_tab = tab_widgets[cameras_stack->currentIndex()]->currentIndex();
    
    int selected_camera = 0;
    if (index == 0) {
      selected_camera = current_camera_index;
    } else {
      selected_camera = index - 1;
    }
    cameras_stack->setCurrentIndex(selected_camera);
    
    tab_widgets[cameras_stack->currentIndex()]->setCurrentIndex(selected_tab);
  });
}

void CalibrationWindow::SetDataset(Dataset* dataset) {
  RunInQtThreadBlocking([&]() {
    camera_names.resize(dataset->num_cameras());
    
    tab_widgets.resize(dataset->num_cameras(), nullptr);
    
    feature_detection_displays.resize(dataset->num_cameras(), nullptr);
    initialization_displays.resize(dataset->num_cameras(), nullptr);
    observation_directions_displays.resize(dataset->num_cameras(), nullptr);
    error_histogram_displays.resize(dataset->num_cameras(), nullptr);
    reprojection_error_displays.resize(dataset->num_cameras(), nullptr);
    error_direction_displays.resize(dataset->num_cameras(), nullptr);
    removed_outliers_displays.resize(dataset->num_cameras(), nullptr);
    
    for (int camera_index = 0; camera_index < dataset->num_cameras(); ++ camera_index) {
      // TODO: Would be great to show the original camera names from the dataset
      //       metadata file here (if such a file was available when creating the dataset)
      ostringstream camera_name;
      camera_name << "camera" << camera_index;
      camera_names[camera_index] = QString::fromStdString(camera_name.str());
      
      camera_selector->addItem(camera_names[camera_index]);
      
      tab_widgets[camera_index] = new QTabWidget();
      cameras_stack->addWidget(tab_widgets[camera_index]);
    }
    
    cameras_stack->setCurrentIndex(0);
  });
}

void CalibrationWindow::SetCurrentCameraIndex(int camera_index) {
  RunInQtThreadBlocking([&]() {
    current_camera_index = camera_index;
    camera_selector->setItemText(0, tr("current: %1").arg(camera_names[camera_index]));
    if (camera_selector->currentIndex() == 0) {
      cameras_stack->setCurrentIndex(camera_index);
    }
  });
}

void CalibrationWindow::UpdateFeatureDetection(int camera_index, const Image<Vec3u8>& image) {
  CreateOrUpdateDisplay(
      tr("Feature detection"),
      camera_index,
      image,
      &feature_detection_displays);
}

void CalibrationWindow::UpdateInitialization(int camera_index, const Image<Vec3u8>& image) {
  CreateOrUpdateDisplay(
      tr("Initialization"),
      camera_index,
      image,
      &initialization_displays);
}

void CalibrationWindow::UpdateObservationDirections(int camera_index, const Image<Vec3u8>& image) {
  CreateOrUpdateDisplay(
      tr("Observation directions"),
      camera_index,
      image,
      &observation_directions_displays);
}

void CalibrationWindow::UpdateErrorHistogram(int camera_index, const Image<u8>& image) {
  CreateOrUpdateDisplay(
      tr("Error histogram"),
      camera_index,
      image,
      &error_histogram_displays);
}

void CalibrationWindow::UpdateReprojectionErrors(int camera_index, const Image<Vec3u8>& image, const Dataset* dataset, const BAState* state) {
  CreateOrUpdateDisplay(
      tr("Errors"),
      camera_index,
      image,
      &reprojection_error_displays);
  
  ImageDisplayQtWindow& display = *reprojection_error_displays[camera_index];
  display.Clear();
  
  for (int imageset_index = 0; imageset_index < dataset->ImagesetCount(); ++ imageset_index) {
    if (!state->image_used[imageset_index]) {
      continue;
    }
    
    const SE3d& image_tr_global = state->image_tr_global(camera_index, imageset_index);
    Mat3d image_r_global = image_tr_global.rotationMatrix();
    const Vec3d& image_t_global = image_tr_global.translation();
    
    shared_ptr<const Imageset> imageset = dataset->GetImageset(imageset_index);
    const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
    
    for (const PointFeature& feature : features) {
      Vec3d local_point = image_r_global * state->points[feature.index] + image_t_global;
      Vec2d pixel;
      if (state->intrinsics[camera_index]->Project(local_point, &pixel)) {
        Vec2d reprojection_error = pixel - feature.xy.cast<double>();
        
        Vec3u8 color;
        if (reprojection_error.norm() > 50) {
          color = Vec3u8(255, 0, 0);
        } else if (reprojection_error.norm() > 10) {
          color = Vec3u8(255, 100, 100);
        } else {
          color = Vec3u8(200, 200, 200);
        }
        display.AddSubpixelLinePixelCornerConv(pixel, feature.xy, color);
      }
    }
  }
}

void CalibrationWindow::UpdateErrorDirections(int camera_index, const Image<Vec3u8>& image) {
  CreateOrUpdateDisplay(
      tr("Error directions"),
      camera_index,
      image,
      &error_direction_displays);
}

void CalibrationWindow::UpdateRemovedOutliers(int camera_index, const Image<Vec3u8>& image) {
  CreateOrUpdateDisplay(
      tr("Removed outliers"),
      camera_index,
      image,
      &removed_outliers_displays);
}

}

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

#include <memory>
#include <vector>

#include <libvis/libvis.h>
#include <QCheckBox>
#include <QDialog>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>

#include "camera_calibration/image_input/image_input.h"

namespace vis {

class FeatureDetector;

/// Settings window that allows to select the cameras and settings for live
/// recording
class SettingsWindow : public QDialog {
 Q_OBJECT
 public:
  SettingsWindow(const shared_ptr<FeatureDetector>& detector, QWidget* parent = nullptr);
  
  vector<shared_ptr<AvailableInput>> GetChosenInputs();
  
  inline string PatternYAMLPaths() const {
    return patterns_edit->text().toStdString();
  }
  
  inline QWidget* GetSettingsWidget(AvailableInput::Type type) {
    return camera_settings_widgets[static_cast<int>(type)];
  }
  
  inline int FeatureWindowExtent() const {
    return atoi(feature_window_extent_edit->text().toStdString().c_str());
  }
  
  inline bool LiveDetection() const {
    return live_detection_checkbox->isChecked() && !show_pattern;
  }
  
  inline bool RecordImages() const {
    return record_checkbox->isChecked() && !show_pattern;
  }
  
  inline bool RecordImagesWithDetectionsOnly() const {
    return RecordImages() && record_detection_images_only_checkbox->isChecked();
  }
  
  inline QString RecordDirectory() const {
    return record_directory_edit->text();
  }
  
  inline bool SaveDatasetOnExit() const {
    return LiveDetection() && record_dataset_checkbox->isChecked();
  }
  
  inline bool show_pattern_clicked() const {
    return show_pattern;
  }
  
 public slots:
  bool TryAccept();
  void ShowPattern();
  void ChooseRecordDirectory();
  
 private:
  vector<shared_ptr<AvailableInput>> available_inputs;
  
  bool show_pattern = false;
  
  QListWidget* m_camera_list;
  QGroupBox* camera_settings_box;
  QWidget* camera_settings_widgets[static_cast<int>(AvailableInput::Type::NumTypes)];
  QLineEdit* patterns_edit;
  QCheckBox* live_detection_checkbox;
  QLineEdit* feature_window_extent_edit;
  QCheckBox* live_calibration_checkbox;
  QCheckBox* record_checkbox;
  QCheckBox* record_detection_images_only_checkbox;
  QLineEdit* record_directory_edit;
  QPushButton* record_directory_choose;
  QCheckBox* record_dataset_checkbox;
};

}

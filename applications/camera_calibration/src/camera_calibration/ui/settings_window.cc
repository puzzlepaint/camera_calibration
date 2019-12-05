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

#include "camera_calibration/ui/settings_window.h"

#include <libv4l2.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <libvis/logging.h>
#include <sstream>
#include <sys/ioctl.h>

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>

#include "camera_calibration/feature_detection/feature_detector.h"
#include "camera_calibration/image_input/image_input_realsense.h"
#include "camera_calibration/image_input/image_input_structure.h"
#include "camera_calibration/image_input/image_input_v4l2.h"

namespace vis {

SettingsWindow::SettingsWindow(const shared_ptr<FeatureDetector>& detector, QWidget* parent)
    : QDialog(parent) {
  QSettings settings;
  setWindowTitle(tr("Camera calibration - Settings"));
  
  auto layout = new QVBoxLayout(this);
  
  QLabel* choose_camera_label = new QLabel(tr("Choose camera(s):"));
  layout->addWidget(choose_camera_label);
  
  QHBoxLayout* camera_layout = new QHBoxLayout();
  
  m_camera_list = new QListWidget();
  
  // Let all image input implementations search for cameras that they can use
  // TODO: Those implementations should register themselves somewhere such that
  //       they don't need to be listed here
  QProgressDialog* progress = new QProgressDialog(tr("Searching for attached cameras ..."), tr("Abort"), 0, 3, nullptr);
  progress->setWindowModality(Qt::WindowModal);
  progress->setMinimumDuration(0);
  progress->setValue(0);
  // Workaround for the progress dialog not showing: change around its value a bit
  progress->setValue(1);
  progress->setValue(0);
  progress->setValue(1);
  progress->setValue(0);
  
  ImageInputRealSense::ListAvailableInputs(&available_inputs);
  
  progress->setValue(1);
  if (!progress->wasCanceled()) {
    ImageInputV4L2::ListAvailableInputs(&available_inputs);
  }
  
  progress->setValue(2);
  if (!progress->wasCanceled()) {
    ImageInputStructure::ListAvailableInputs(&available_inputs);
  }
  
  progress->setValue(3);
  progress->deleteLater();
  
  sort(available_inputs.begin(), available_inputs.end(),
       [&](const shared_ptr<AvailableInput>& a, const shared_ptr<AvailableInput>& b) {
    return a->display_text < b->display_text;
  });
  
  for (usize input_index = 0; input_index < available_inputs.size(); ++ input_index) {
    const AvailableInput& input = *available_inputs[input_index];
    QListWidgetItem* item = new QListWidgetItem(m_camera_list);
    item->setText(input.display_text);
    item->setCheckState(Qt::Unchecked);
    m_camera_list->addItem(item);
  }
  
  camera_layout->addWidget(m_camera_list);
  
  layout->addLayout(camera_layout, 1);
  
  QGridLayout* model_layout = new QGridLayout();
  int row = 0;
  
  // ---------------------------------------------------------------------------
  camera_settings_box = new QGroupBox(tr("Camera settings"));
  QVBoxLayout* camera_settings_layout = new QVBoxLayout();
  
  camera_settings_widgets[static_cast<int>(AvailableInput::Type::V4L2)] = ImageInputV4L2::CreateSettingsWidgets();
  camera_settings_widgets[static_cast<int>(AvailableInput::Type::RealSense)] = ImageInputRealSense::CreateSettingsWidgets();
  camera_settings_widgets[static_cast<int>(AvailableInput::Type::Structure)] = ImageInputStructure::CreateSettingsWidgets();
  
  for (int i = 0; i < static_cast<int>(AvailableInput::Type::NumTypes); ++ i) {
    if (camera_settings_widgets[i]) {
      camera_settings_layout->addWidget(camera_settings_widgets[i]);
    }
  }
  
  camera_settings_box->setLayout(camera_settings_layout);
  model_layout->addWidget(camera_settings_box, row, 0, 1, 2);
  ++ row;
  
  camera_settings_box->setVisible(false);
  // ---------------------------------------------------------------------------
  
  // ---------------------------------------------------------------------------
  QGroupBox* detection_group = new QGroupBox(tr("Feature detection"));
  QGridLayout* detection_layout = new QGridLayout();
  
  int detection_row = 0;
  
  QLabel* patterns_label = new QLabel(tr("Pattern YAML files (comma-separated):"));
  patterns_edit = new QLineEdit("");
  for (const string& path : detector->GetPatternYAMLPaths()) {
    if (!patterns_edit->text().isEmpty()) {
      patterns_edit->text() += ',';
    }
    patterns_edit->text() += QString::fromStdString(path);
  }
  if (patterns_edit->text().isEmpty()) {
    patterns_edit->setText(settings.value("settings_window/pattern_yaml_paths").toString());
  }
  detection_layout->addWidget(patterns_label, detection_row, 0);
  detection_layout->addWidget(patterns_edit, detection_row, 1);
  
  ++ detection_row;
  
  live_detection_checkbox = new QCheckBox(tr("Live feature detection"));
  live_detection_checkbox->setChecked(true);
  detection_layout->addWidget(live_detection_checkbox, detection_row, 0, 1, 2);
  
  ++ detection_row;
  
  QLabel* feature_window_extent_label = new QLabel(tr("Feature window half extent [px]:"));
  detection_layout->addWidget(feature_window_extent_label, detection_row, 0);
  
  feature_window_extent_edit = new QLineEdit(QString::number(detector->GetFeatureWindowHalfExtent()));
  detection_layout->addWidget(feature_window_extent_edit, detection_row, 1);
  
  ++ detection_row;
  
  detection_layout->setRowStretch(detection_row, 1);
  detection_group->setLayout(detection_layout);
  
  model_layout->addWidget(detection_group, row, 0, 1, 2);
  ++ row;
  // ---------------------------------------------------------------------------
  
  // ---------------------------------------------------------------------------
  QGroupBox* recording_group = new QGroupBox(tr("Data recording"));
  QGridLayout* recording_layout = new QGridLayout();
  
  int recording_row = 0;
  
  record_checkbox = new QCheckBox(tr("Record images"));
  record_checkbox->setChecked(true);
  recording_layout->addWidget(record_checkbox, recording_row, 0, 1, 2);
  
  ++ recording_row;
  
  record_detection_images_only_checkbox = new QCheckBox(tr("Only record those images in which features were detected"));
  record_detection_images_only_checkbox->setChecked(false);
  recording_layout->addWidget(record_detection_images_only_checkbox, recording_row, 0, 1, 2);
  
  ++ recording_row;
  
  record_dataset_checkbox = new QCheckBox(tr("Save dataset with extracted features on exit"));
  record_dataset_checkbox->setChecked(true);
  recording_layout->addWidget(record_dataset_checkbox, recording_row, 0, 1, 2);
  
  ++ recording_row;
  
  QLabel* record_location_label = new QLabel(tr("Directory to save dataset / images in:"));
  recording_layout->addWidget(record_location_label, recording_row, 0);
  
  QHBoxLayout* record_directory_layout = new QHBoxLayout();
  record_directory_edit = new QLineEdit("");
  record_directory_edit->setText(settings.value("settings_window/record_directory").toString());
  record_directory_layout->addWidget(record_directory_edit, 1);
  record_directory_choose = new QPushButton(tr("..."), this);
  {
    // Allow the "..." button to have a smaller width than it would usually have:
    // https://stackoverflow.com/a/19502467
    auto* button = record_directory_choose;
    auto text_size = button->fontMetrics().size(Qt::TextShowMnemonic, button->text());
    QStyleOptionButton opt;
    opt.initFrom(button);
    opt.rect.setSize(text_size);
    button->setMaximumWidth(
        2 * button->style()->sizeFromContents(
            QStyle::CT_PushButton, &opt, text_size, button).width());
  }
  connect(record_directory_choose, &QPushButton::clicked, this, &SettingsWindow::ChooseRecordDirectory);
  record_directory_layout->addWidget(record_directory_choose);
  recording_layout->addLayout(record_directory_layout, recording_row, 1);
  
  ++ recording_row;
  
  recording_layout->setRowStretch(recording_row, 1);
  recording_group->setLayout(recording_layout);
  
  model_layout->addWidget(recording_group, row, 0, 1, 2);
  ++ row;
  // ---------------------------------------------------------------------------
  
  model_layout->setRowStretch(row, 1);
  layout->addLayout(model_layout);
  
  QHBoxLayout* buttons_layout = new QHBoxLayout();
  
  buttons_layout->addStretch(1);
  
  QPushButton* show_pattern_button = new QPushButton(tr("Show pattern"));
  connect(show_pattern_button, &QPushButton::clicked, this, &SettingsWindow::ShowPattern);
  buttons_layout->addWidget(show_pattern_button);
  
  QPushButton* start_button = new QPushButton(tr("Start normal"));
  connect(start_button, &QPushButton::clicked, this, &SettingsWindow::TryAccept);
  buttons_layout->addWidget(start_button);
  
  layout->addLayout(buttons_layout);
  
  resize(800, 600);
  
  connect(record_checkbox, &QCheckBox::stateChanged, [&]() {
    record_detection_images_only_checkbox->setEnabled(live_detection_checkbox->isChecked() && record_checkbox->isChecked());
    record_directory_edit->setEnabled((live_detection_checkbox->isChecked() && record_dataset_checkbox->isChecked()) || record_checkbox->isChecked());
    record_directory_choose->setEnabled(record_directory_edit->isEnabled());
  });
  
  connect(live_detection_checkbox, &QCheckBox::stateChanged, [&]() {
    feature_window_extent_edit->setEnabled(live_detection_checkbox->isChecked());
    record_dataset_checkbox->setEnabled(live_detection_checkbox->isChecked());
    
    record_detection_images_only_checkbox->setEnabled(live_detection_checkbox->isChecked() && record_checkbox->isChecked());
    record_directory_edit->setEnabled((live_detection_checkbox->isChecked() && record_dataset_checkbox->isChecked()) || record_checkbox->isChecked());
    record_directory_choose->setEnabled(record_directory_edit->isEnabled());
  });
  
  connect(record_dataset_checkbox, &QCheckBox::stateChanged, [&]() {
    record_directory_edit->setEnabled((live_detection_checkbox->isChecked() && record_dataset_checkbox->isChecked()) || record_checkbox->isChecked());
    record_directory_choose->setEnabled(record_directory_edit->isEnabled());
  });
  
  connect(m_camera_list, &QListWidget::itemChanged, [&]() {
    // Update the visibility of the camera settings.
    vector<bool> have_types(static_cast<int>(AvailableInput::Type::NumTypes), false);
    
    for (int i = 0; i < m_camera_list->count(); ++ i) {
      if (m_camera_list->item(i)->checkState() == Qt::Checked) {
        have_types[static_cast<int>(available_inputs[i]->type)] = true;
      }
    }
    
    bool have_any_settings_widget = false;
    for (int i = 0; i < static_cast<int>(AvailableInput::Type::NumTypes); ++ i) {
      if (camera_settings_widgets[i]) {
        have_any_settings_widget |= have_types[i];
        camera_settings_widgets[i]->setVisible(have_types[i]);
      }
    }
    
    camera_settings_box->setVisible(have_any_settings_widget);
  });
}

vector<shared_ptr<AvailableInput>> SettingsWindow::GetChosenInputs() {
  vector<shared_ptr<AvailableInput>> result;
  
  for (int i = 0; i < m_camera_list->count(); ++ i) {
    if (m_camera_list->item(i)->checkState() == Qt::Checked) {
      result.push_back(available_inputs[i]);
    }
  }
  
  return result;
}

bool SettingsWindow::TryAccept() {
  if (RecordImages() || SaveDatasetOnExit()) {
    if (!QDir(RecordDirectory()).exists()) {
      QMessageBox::warning(this, tr("Error"), tr("Please select a valid directory to save the recorded file(s) in."));
      return false;
    }
  }
  
  bool any_camera_checked = false;
  for (int i = 0; i < m_camera_list->count(); ++ i) {
    if (m_camera_list->item(i)->checkState() == Qt::Checked) {
      any_camera_checked = true;
      break;
    }
  }
  if (!any_camera_checked) {
    QMessageBox::warning(this, tr("Error"), tr("Please select at least one camera to use."));
    return false;
  }
  
  QSettings settings;
  settings.setValue("settings_window/record_directory", record_directory_edit->text());
  settings.setValue("settings_window/pattern_yaml_paths", patterns_edit->text());
  
  QDialog::accept();
  return true;
}

void SettingsWindow::ShowPattern() {
  if (TryAccept()) {
    show_pattern = true;
  }
}

void SettingsWindow::ChooseRecordDirectory() {
  QString dir = QFileDialog::getExistingDirectory(
      this, tr("Choose directory to save files in"),
      RecordDirectory(), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty()) {
    record_directory_edit->setText(dir);
  }
}

}

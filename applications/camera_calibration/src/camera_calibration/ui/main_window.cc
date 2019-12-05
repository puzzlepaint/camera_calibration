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

#include "camera_calibration/ui/main_window.h"

#include <QLabel>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

namespace vis {

MainWindow::MainWindow(
    const vector<shared_ptr<AvailableInput>>& inputs,
    QWidget* parent,
    Qt::WindowFlags flags)
    : QMainWindow(parent, flags) {
  resolution_added.resize(inputs.size(), false);
  camera_labels.resize(inputs.size());
  image_displays.resize(inputs.size());
  detection_displays.resize(inputs.size());
  
  QVBoxLayout* layout = new QVBoxLayout();
  
  // Cameras
  QHBoxLayout* cameras_layout = new QHBoxLayout();
  
  for (usize i = 0; i < inputs.size(); ++ i) {
    QLabel*& camera_label = camera_labels[i];
    ImageDisplayQtWindow*& image_display = image_displays[i];
    ImageDisplayQtWindow*& detection_display = detection_displays[i];
    
    QVBoxLayout* camera_layout = new QVBoxLayout();
    
    camera_label = new QLabel(inputs[i]->display_text);
    camera_label->setAlignment(Qt::AlignHCenter);
    camera_layout->addWidget(camera_label);
    
    image_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
    image_display->SetDisplayAsWidget();
    image_display->setMinimumSize(640, 480);
    image_display->FitContent();
    camera_layout->addWidget(image_display);
    
    detection_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
    detection_display->SetDisplayAsWidget();
    detection_display->setMinimumSize(640, 480);
    detection_display->FitContent();
    camera_layout->addWidget(detection_display);
    
    cameras_layout->addLayout(camera_layout);
  }
  
  QWidget* cameras_widget = new QWidget();
  cameras_widget->setLayout(cameras_layout);
  
  QScrollArea* scroll_area = new QScrollArea();
  scroll_area->setWidget(cameras_widget);
  scroll_area->setWidgetResizable(true);
  layout->addWidget(scroll_area);
  
  QWidget* main_widget = new QWidget();
  main_widget->setLayout(layout);
  main_widget->setAutoFillBackground(false);
  setCentralWidget(main_widget);
  
  setWindowTitle(tr("Camera calibration"));
  
  // TODO: While this window is shown and image input is running, the application
  //       should prevent the display from timing out / the screensaver from starting.
  //       Unfortunately, I haven't found a Qt / cross-platform solution for that.
}

void MainWindow::SetImageDisplaysMinimumSize(int width, int height) {
  for (usize i = 0; i < image_displays.size(); ++ i) {
    image_displays[i]->setMinimumSize(width, height);
    detection_displays[i]->setMinimumSize(width, height);
  }
}

void MainWindow::NewImage(int camera_index, QSharedPointer<Image<Vec3u8>> image) {
  if (!resolution_added[camera_index]) {
    camera_labels[camera_index]->setText(
        camera_labels[camera_index]->text() +
        " (" + QString::number(image->width()) +
        " x " + QString::number(image->height()) + ")");
    resolution_added[camera_index] = true;
  }
  image_displays[camera_index]->SetImage(*image);
}

void MainWindow::NewDetectionsPerPixel(int camera_index, QSharedPointer<Image<Vec3u8>> image) {
  detection_displays[camera_index]->SetImage(*image);
}

}

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

#include <libvis/image_display_qt_window.h>
#include <QMainWindow>
#include <QLabel>
#include <QSharedPointer>

#include "camera_calibration/image_input/image_input.h"

namespace vis {

/// Shows views of live images and corresponding feature detection visualizations.
class MainWindow : public QMainWindow {
 Q_OBJECT
 public:
  MainWindow(
      const vector<shared_ptr<AvailableInput>>& inputs,
      QWidget* parent = nullptr,
      Qt::WindowFlags flags = Qt::WindowFlags());
  
  void SetImageDisplaysMinimumSize(int width, int height);
  
 public slots:
  void NewImage(int camera_index, QSharedPointer<Image<Vec3u8>> image);
  void NewDetectionsPerPixel(int camera_index, QSharedPointer<Image<Vec3u8>> image);
  
 private:
  vector<bool> resolution_added;
  vector<QLabel*> camera_labels;
  vector<ImageDisplayQtWindow*> image_displays;
  vector<ImageDisplayQtWindow*> detection_displays;
};

}

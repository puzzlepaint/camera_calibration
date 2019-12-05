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

#include <libvis/libvis.h>
#include <QWidget>
#include <QImage>

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"
#include "camera_calibration/ui/main_window.h"

namespace vis {

class LiveImageConsumer;

/// This widget is shown in fullscreen to show a pattern on the screen that can
/// be filmed by cameras.
class PatternDisplay : public QWidget {
 Q_OBJECT
 public:
  PatternDisplay(
      const vector<shared_ptr<AvailableInput>>& inputs,
      const shared_ptr<FeatureDetectorTaggedPattern>& detector,
      const char* apriltags_path,
      bool embed_main_window,
      QWidget* parent = nullptr);
  
  void SetLiveImageConsumer(LiveImageConsumer* forwarder);
  
  void RecordImages();
  
  inline MainWindow* main_window() { return embedded_main_window; }
  
 public slots:
  void RecordImageSlot();
  
 protected:
  void resizeEvent(QResizeEvent* event) override;
  void paintEvent(QPaintEvent* event) override;
  void keyPressEvent(QKeyEvent* event) override;
  
 private:
  void UpdateCachedImages();
  
  int current_pattern;
  bool recording_images;
  
  MainWindow* embedded_main_window;
  QCursor widget_cursor;
  
  LiveImageConsumer* image_consumer;
  
  Vec2i images_cached_for_size;
  vector<QImage> cached_images;
  vector<PatternData> data;
  string apriltags_path;
};

}

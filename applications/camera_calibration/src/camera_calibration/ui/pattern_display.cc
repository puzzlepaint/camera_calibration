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

#include "camera_calibration/ui/pattern_display.h"

#include <libvis/geometry.h>
#include <unistd.h>
#include <QPainter>
#include <QPaintEvent>
#include <QTimer>

#include "camera_calibration/ui/live_image_consumer.h"

namespace vis {

// The delay that we think that we have to wait upon changing the displayed
// pattern until it is shown on the screen and recorded in the camera images
// TODO: Rather than using a fixed delay, it would be better if we monitored the
//       camera images and just waited until we can detect the new pattern's
//       AprilTag in the image. Probably use the second such image (instead of
//       the first) to (likely) avoid having the previous pattern still mixed in.
constexpr int kImageRecordDelay = 350;  // in milliseconds

PatternDisplay::PatternDisplay(
    const vector<shared_ptr<AvailableInput>>& inputs,
    const shared_ptr<FeatureDetectorTaggedPattern>& detector,
    const char* apriltags_path,
    bool embed_main_window,
    QWidget* parent)
    : QWidget(parent),
      apriltags_path(apriltags_path) {
  images_cached_for_size = Vec2i(-1, -1);
  
  data.resize(detector->GetPatternCount());
  for (int i = 0; i < detector->GetPatternCount(); ++ i) {
    data[i] = detector->GetPatternData(i);
  }
  
  current_pattern = 0;
  recording_images = false;
  
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
  
  if (embed_main_window) {
    embedded_main_window = new MainWindow(inputs, this);
    embedded_main_window->setWindowFlags(Qt::Widget);
  } else {
    embedded_main_window = nullptr;
  }
}

void PatternDisplay::SetLiveImageConsumer(LiveImageConsumer* forwarder) {
  image_consumer = forwarder;
}

void PatternDisplay::RecordImages() {
  recording_images = true;
  if (embedded_main_window) {
    embedded_main_window->setVisible(false);
  }
  widget_cursor = cursor();
  setCursor(Qt::BlankCursor);
  
  current_pattern = 0;
  update(rect());
  QTimer::singleShot(kImageRecordDelay, this, &PatternDisplay::RecordImageSlot);
}

void PatternDisplay::RecordImageSlot() {
  vector<QDir> record_dirs = image_consumer->get_image_record_directories();
  vector<QDir> pattern_record_dirs(record_dirs.size());
  for (usize i = 0; i < record_dirs.size(); ++ i) {
    pattern_record_dirs[i] = record_dirs[i].filePath(QString("pattern%1").arg(current_pattern));
  }
  image_consumer->SetImageRecordDirectories(pattern_record_dirs);
  image_consumer->RecordOneImageset();
  while (image_consumer->is_record_one_imageset_pending()) {
    usleep(0);
  }
  image_consumer->SetImageRecordDirectories(record_dirs);
  
  if (current_pattern == cached_images.size() - 1) {
    // This was the last pattern, finish image recording.
    setCursor(widget_cursor);
    if (embedded_main_window) {
      embedded_main_window->setVisible(true);
    }
    recording_images = false;
  } else {
    // Proceed to the next pattern.
    ++ current_pattern;
    update(rect());
    QTimer::singleShot(kImageRecordDelay, this, &PatternDisplay::RecordImageSlot);
  }
}

void PatternDisplay::resizeEvent(QResizeEvent* /*event*/) {
  if (embedded_main_window) {
    embedded_main_window->setGeometry(0. * width(), 0. * height(), 0.3 * width(), 0.4 * height());
    embedded_main_window->SetImageDisplaysMinimumSize(
        std::max(0, embedded_main_window->width() - 40),
        std::max(0, embedded_main_window->height() - 40));
  }
}

void PatternDisplay::paintEvent(QPaintEvent* event) {
  // If no pattern images are cached for the current widget size, update them.
  Vec2i widget_size = Vec2i(width(), height());
  if (widget_size != images_cached_for_size) {
    UpdateCachedImages();
    images_cached_for_size = widget_size;
  }
  
  // Paint current image
  QPainter painter(this);
  QRect event_rect = event->rect();
  painter.setClipRect(event_rect);
  painter.setRenderHint(QPainter::Antialiasing, false);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
  painter.drawImage(QPoint(0, 0), cached_images[current_pattern]);
  painter.end();
}

void PatternDisplay::keyPressEvent(QKeyEvent* event) {
  if (recording_images) {
    return;
  }
  
  if (event->key() == Qt::Key::Key_Right || event->key() == Qt::Key::Key_Down) {
    current_pattern = (current_pattern + 1) % cached_images.size();
    update(rect());
  } else if (event->key() == Qt::Key::Key_Left || event->key() == Qt::Key::Key_Up) {
    -- current_pattern;
    if (current_pattern == -1) {
      current_pattern = cached_images.size() - 1;
    }
    update(rect());
  } else if (event->key() == Qt::Key::Key_Escape) {
    close();
  } else if (event->key() == Qt::Key::Key_Space) {
    RecordImages();
  }
}

void PatternDisplay::UpdateCachedImages() {
  cached_images.resize(data.size());
  for (int i = 0; i < data.size(); ++ i) {
    const PatternData& pattern = data[i];
    
    // Decide whether to render the pattern upright or rotated by 90 degrees.
    bool page_upright = pattern.page_height_mm >= pattern.page_width_mm;
    bool widget_upright = height() >= width();
    bool rotate_90deg = page_upright != widget_upright;
    
    int render_area_width = rotate_90deg ? height() : width();
    int render_area_height = rotate_90deg ? width() : height();
    
    // Get the pattern geometry.
    forward_list<vector<Vec2f>> geometry;
    pattern.ComputePatternGeometry(&geometry, apriltags_path.c_str());
    
    // Isotropically scale and translate the pattern geometry to be centered in
    // the widget and have maximum size.
    // In the coordinate system used by ComputePatternGeometry(), the pattern
    // bounds are:
    // X: (-1) to (pattern.squares_x - 1)
    // Y: (-1) to (pattern.squares_y - 1)
    double pattern_min_x = -1;
    double pattern_min_y = -1;
    double pattern_max_x = pattern.squares_x - 1;
    double pattern_max_y = pattern.squares_y - 1;
    
    double pattern_width = pattern_max_x - pattern_min_x;
    double pattern_height = pattern_max_y - pattern_min_y;
    
    double widget_s_pattern = std::min(render_area_width / pattern_width, render_area_height / pattern_height);  // s: scaling
    double widget_tx_pattern = (0.5 * render_area_width - 0.5 * widget_s_pattern * pattern_width) - (widget_s_pattern * pattern_min_x);  // tx: x translation
    double widget_ty_pattern = (0.5 * render_area_height - 0.5 * widget_s_pattern * pattern_height) - (widget_s_pattern * pattern_min_y);  // ty: y translation
    
    forward_list<vector<Vec2f>> transformed_geometry;
    for (vector<Vec2f>& polygon : geometry) {
      vector<Vec2f> transformed_polygon(polygon.size());
      for (usize i = 0; i < polygon.size(); ++ i) {
        Vec2f transformed_point = widget_s_pattern * polygon[i] + Vec2f(widget_tx_pattern, widget_ty_pattern);
        transformed_polygon[i] = transformed_point;
      }
      transformed_geometry.push_front(transformed_polygon);
    }
    
    // Render the polygons:
    // - Loop over all polygons and project their points into the image
    // - For a polygon, loop over all pixels in its bounding box
    // - Clip the polygon with the pixel's area as (convex) clipping path
    // - Darken the pixel by the amount of its area that is covered by the
    //   polygon (this works since the polygons are non-overlapping).
    Image<float> pattern_rendering(render_area_width, render_area_height);
    pattern_rendering.SetTo(1.f);
    for (vector<Vec2f>& polygon : transformed_geometry) {
      // Compute the bounding box of the polygon
      Eigen::AlignedBox2f bbox;
      for (const Vec2f& point : polygon) {
        bbox.extend(point);
      }
      
      // Loop over all pixels that intersect the bounding box
      int min_x = max<int>(0, bbox.min().x());
      int max_x = min<int>(pattern_rendering.width() - 1, bbox.max().x());
      int min_y = max<int>(0, bbox.min().y());
      int max_y = min<int>(pattern_rendering.height() - 1, bbox.max().y());
      for (int y = min_y; y <= max_y; ++ y) {
        for (int x = min_x; x <= max_x; ++ x) {
          // Intersect the pixel area and the projected polygon
          const vector<Vec2f> pixel_area = {Vec2f(x, y), Vec2f(x + 1, y), Vec2f(x + 1, y + 1), Vec2f(x, y + 1)};
          vector<Vec2f> intersection;
          ConvexClipPolygon(
              polygon,
              pixel_area,
              &intersection);
          pattern_rendering(x, y) -= PolygonArea(intersection);
        }
      }
    }
    
    // Write the relevant part of the rendering into a cached QImage, while
    // applying the 90 degree rotation if needed.
    QImage& image = cached_images[i];
    image = QImage(width(), height(), QImage::Format::Format_RGB888);
    // To let camera auto-exposure reach a good mean value in the middle, use a brightness value in the middle for the background
    image.fill(qRgb(127, 127, 127));
    
    Vec2f min_rendering_xy = widget_s_pattern * Vec2f(pattern_min_x, pattern_min_y) + Vec2f(widget_tx_pattern, widget_ty_pattern);
    int min_rendering_x = std::max<int>(0, std::floor(min_rendering_xy.x()));
    int min_rendering_y = std::max<int>(0, std::floor(min_rendering_xy.y()));
    
    Vec2f max_rendering_xy = widget_s_pattern * Vec2f(pattern_max_x, pattern_max_y) + Vec2f(widget_tx_pattern, widget_ty_pattern);
    int max_rendering_x = std::min<int>(render_area_width - 1, std::ceil(max_rendering_xy.x()) - 1);
    int max_rendering_y = std::min<int>(render_area_height - 1, std::ceil(max_rendering_xy.y()) - 1);
    
    for (int y = min_rendering_y; y < max_rendering_y; ++ y) {
      for (int x = min_rendering_x; x < max_rendering_x; ++ x) {
        u8 v = std::max<float>(0.f, 255.99f * pattern_rendering(x, y));
        
        if (rotate_90deg) {
          uchar* row = image.scanLine(x);
          int inv_y = image.width() - y;
          row[3 * inv_y + 0] = v;
          row[3 * inv_y + 1] = v;
          row[3 * inv_y + 2] = v;
        } else {
          uchar* row = image.scanLine(y);
          row[3 * x + 0] = v;
          row[3 * x + 1] = v;
          row[3 * x + 2] = v;
        }
      }
    }
  }
}

}

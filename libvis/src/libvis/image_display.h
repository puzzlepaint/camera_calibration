// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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

#include "libvis/libvis.h"

#ifdef LIBVIS_HAVE_QT

#include "libvis/image_display_qt_window.h"
#include "libvis/window_callbacks.h"

#include <QTimer>

namespace vis {

// Handle to a debug image display which allows to update the displayed image
// and close the display.
class ImageDisplay {
 public:
  ImageDisplay();
  
  // Detaches the display from the window (if the window exists). The window
  // will continue to be shown.
  ~ImageDisplay();
  
  // Updates the image that is displayed in the debug display.
  // Supports image types u8, Vec3u8, Vec4u8.
  template<typename T>
  void Update(const Image<T>& image,
              const string& title,
              const shared_ptr<ImageWindowCallbacks>& callbacks = shared_ptr<ImageWindowCallbacks>()) {
    Update(image,
           title,
           0,
           255,
           callbacks);
  }
  
  // Updates the image that is displayed in the debug display.
  // Supports all scalar image types.
  template<typename T>
  void Update(const Image<T>& image,
              const string& title,
              const double black_value,
              const double white_value,
              const shared_ptr<ImageWindowCallbacks>& callbacks = shared_ptr<ImageWindowCallbacks>()) {
    RunInQtThreadBlocking([&](){
      // Update the window.
      bool visible_before = window_->isVisible();
      window_->SetBlackWhiteValues(black_value, white_value);
      window_->SetImage(image);
      window_->SetCallbacks(callbacks);
      window_->setWindowTitle(QString::fromStdString(title) + " (" + QString::number(image.width()) + " x " + QString::number(image.height()) + ")");
      window_->show();
      if (!visible_before) {
        // It seems that the final status bar height is only available after the
        // event loop has started. Therefore, schedule resizing to after this.
        QTimer::singleShot(0, window_, SLOT(ZoomAndResizeToContent()));
      }
    });
  }
  
  // Redraws the widget, showing any changes made e.g. by calling AddSubpixelDotPixelCornerConv().
  inline void Update() {
    RunInQtThreadBlocking([&](){
      window_->widget().update(window_->widget().rect());
    });
  }
  
  // Adds a dot (displayed as a small circle) to the display, which is suitable to
  // mark subpixel positions. The coordinate origin convention is "pixel corner",
  // i.e., the coordinate (0, 0) is at the top left corner of the image.
  // This function does not cause a redraw. Call Update() after making all changes to redraw.
  template <typename DerivedA, typename DerivedB>
  void AddSubpixelDotPixelCornerConv(const MatrixBase<DerivedA>& position, const MatrixBase<DerivedB>& color) {
    RunInQtThreadBlocking([&](){
      window_->AddSubpixelDotPixelCornerConv(position.x(), position.y(), color.x(), color.y(), color.z());
    });
  }
  
  // Adds a line to the display, which is suitable to mark subpixel data.
  // The coordinate origin convention is "pixel corner",
  // i.e., the coordinate (0, 0) is at the top left corner of the image.
  // This function does not cause a redraw. Call Update() after making all changes to redraw.
  template <typename DerivedA, typename DerivedB, typename DerivedC>
  void AddSubpixelLinePixelCornerConv(const MatrixBase<DerivedA>& position_a, const MatrixBase<DerivedB>& position_b, const MatrixBase<DerivedC>& color) {
    RunInQtThreadBlocking([&](){
      window_->AddSubpixelLinePixelCornerConv(position_a.x(), position_a.y(), position_b.x(), position_b.y(), color.x(), color.y(), color.z());
    });
  }
  
  template <typename DerivedA, typename DerivedB>
  void AddSubpixelTextPixelCornerConv(const MatrixBase<DerivedA>& position, const MatrixBase<DerivedB>& color, const string& text) {
    RunInQtThreadBlocking([&](){
      window_->AddSubpixelTextPixelCornerConv(position.x(), position.y(), color.x(), color.y(), color.z(), text);
    });
  }
  
  // Removes all added subpixel elements.
  // This function does not cause a redraw. Call Update() after making all changes to redraw.
  void Clear();
  
  // Closes the debug display.
  void Close();
  
  // To be called from the Qt thread.
  void SetWindow(ImageDisplayQtWindow* window);
  
  bool IsOpen();

 private:
  // Pointer managed by Qt.
  // TODO: According to valgrind, it seems that this is leaked on exit.
  ImageDisplayQtWindow* window_;
};

}
#endif

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


#include "libvis/render_window_qt.h"

#include <QApplication>
#include <QDesktopWidget>

#include "libvis/qt_thread.h"

namespace vis {

RenderWindowQtWindow::RenderWindowQtWindow(int width, int height, QWidget* parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags) {
  // Note that the size given here should not exceed the available space,
  // otherwise unnecessary resize events can be generated which will resize
  // the window back to the available size.
  QDesktopWidget* desktop = QApplication::desktop();
  QRect available_rect = desktop->availableGeometry(desktop->primaryScreen());
  
  if (width > 0 && height > 0) {
    resize(width, height);
  } else if (width == 0 && height == 0) {
    // HACK: Hardcoded maximum values for the window frame size. This can be hard
    // to get prior to showing any window:
    // http://stackoverflow.com/questions/7408082/how-to-get-the-width-of-a-window-frame-before-creating-any-windows
    const int kFrameWidth = 10;
    const int kFrameHeight = 40;
    QSize available_window_size(
        available_rect.width() - kFrameWidth,
        available_rect.height() - kFrameHeight);
    
    vector<QSize> requested_size_list = {{1600, 1200}, {1200, 900}, {800, 600}, {640, 480}};
    bool found = false;
    for (const QSize& size : requested_size_list) {
      if (size.width() <= available_window_size.width() &&
          size.height() <= available_window_size.height()) {
        resize(size);
        found = true;
        break;
      }
    }
    if (!found) {
      resize(requested_size_list.back());
    }
  }
  
  // TODO: Allow users to add their own widgets next to the render widget.
}

RenderWindowQt::RenderWindowQt(
    const std::string& title,
    int width,
    int height,
    const shared_ptr<RenderWindowCallbacks>& callbacks,
    bool use_qt_thread,
    bool show)
    : RenderWindow(callbacks),
      use_qt_thread_(use_qt_thread) {
  auto init_function = [&](){
    window_ = new RenderWindowQtWindow(width, height);
    window_->setWindowTitle(QString::fromStdString(title));
    if (show) {
      window_->show();
    }
  };
  
  if (use_qt_thread_) {
    RunInQtThreadBlocking(init_function);
  } else {
    init_function();
  }
}

RenderWindowQt::~RenderWindowQt() {
  auto delete_function = [&](){
    window_->deleteLater();
    window_ = nullptr;
  };
  
  if (use_qt_thread_) {
    RunInQtThreadBlocking(delete_function);
  } else {
    delete_function();
  }
}

bool RenderWindowQt::IsOpen() {
  if (!use_qt_thread_) {
    return window_->isVisible();
  }
  
  atomic<bool> is_open;
  RunInQtThreadBlocking([&](){
    is_open = window_->isVisible();
  });
  return is_open;
}

}

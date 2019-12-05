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

#include <QMainWindow>

#include "libvis/libvis.h"
#include "libvis/render_window.h"

namespace vis {

// The actual window class for RenderWindowQt. This is separate since it may
// need to be created in another thread, the Qt thread, than the thread which
// the RenderWindowQt is created in.
class RenderWindowQtWindow : public QMainWindow {
 Q_OBJECT
 public:
  RenderWindowQtWindow(int width, int height,
                       QWidget* parent = nullptr,
                       Qt::WindowFlags flags = Qt::WindowFlags());
};

// A Qt based render window for Linux and Windows. This class does not provide
// an implementation of the rendering yet, but serves as a base class for
// Qt-based render window implementations in-between the generic RenderWindow
// class and the implementation. Its purpose is to allow library users to
// interact with Qt-based windows, for example by adding custom Qt widgets on
// the sides, without having to know whether the implementation is OpenGL or
// Vulkan based.
class RenderWindowQt : public RenderWindow {
 public:
  // Constructor, shows the window.
  RenderWindowQt(const std::string& title, int width, int height,
                 const shared_ptr<RenderWindowCallbacks>& callbacks,
                 bool use_qt_thread = true, bool show = true);
  
  // Destructor, closes the window.
  ~RenderWindowQt();
  
  // Returns whether the window is (still) shown, or it has been closed by the
  // user.
  virtual bool IsOpen() override;
  
  inline RenderWindowQtWindow* window() { return window_; }
  inline const RenderWindowQtWindow* window() const { return window_; }

 protected:
  bool use_qt_thread_;
  
  // Pointer managed by Qt.
  RenderWindowQtWindow* window_;
};

}

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

#include "libvis/opengl.h"

#include <QOpenGLWidget>
#include <QInputEvent>

#include "libvis/libvis.h"
#include "libvis/render_window_qt.h"

namespace vis {

class RenderWidgetOpenGL : public QOpenGLWidget {
 Q_OBJECT
 public:
  RenderWidgetOpenGL(const shared_ptr<RenderWindowCallbacks>& callbacks);
  ~RenderWidgetOpenGL();
   
 protected:
  virtual void initializeGL() override;
  virtual void paintGL() override;
  virtual void resizeGL(int width, int height) override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void keyPressEvent(QKeyEvent* event) override;
  virtual void keyReleaseEvent(QKeyEvent* event) override;
  
  bool initialized_ = false;
  shared_ptr<RenderWindowCallbacks> callbacks_;
};

// A Qt and OpenGL based render window implementation.
class RenderWindowQtOpenGL : public RenderWindowQt {
 public:
  RenderWindowQtOpenGL(
      const std::string& title,
      int width,
      int height,
      const shared_ptr<RenderWindowCallbacks>& callbacks,
      bool use_qt_thread = true,
      bool show = true);
  
  virtual void RenderFrame() override;
  
  virtual void MakeContextCurrent() override;
  
  virtual void ReleaseCurrentContext() override;
  
  inline RenderWidgetOpenGL* widget() { return render_widget_; }
  inline const RenderWidgetOpenGL* widget() const { return render_widget_; }

 private:
  // Pointer is managed by Qt.
  RenderWidgetOpenGL* render_widget_;
};

}

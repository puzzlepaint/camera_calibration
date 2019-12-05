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


#include "libvis/render_window_qt_opengl.h"

#include <QMouseEvent>

#include "libvis/qt_thread.h"

namespace vis {

RenderWidgetOpenGL::RenderWidgetOpenGL(const shared_ptr<RenderWindowCallbacks>& callbacks)
    : QOpenGLWidget(), callbacks_(callbacks) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
  setMinimumSize(200, 200);
  setMouseTracking(true);
  setFocusPolicy(Qt::ClickFocus);
  setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
}

RenderWidgetOpenGL::~RenderWidgetOpenGL() {
  makeCurrent();
  callbacks_->Deinitialize();
  doneCurrent();
}

void RenderWidgetOpenGL::initializeGL() {
  callbacks_->Initialize();
  initialized_ = true;
}

void RenderWidgetOpenGL::paintGL() {
  if (!initialized_) {
    return;
  }
  CHECK_OPENGL_NO_ERROR();
  
  callbacks_->Render();
}

void RenderWidgetOpenGL::resizeGL(int width, int height) {
  callbacks_->Resize(width, height);
}

void RenderWidgetOpenGL::mousePressEvent(QMouseEvent* event) {
  RenderWindowCallbacks::MouseButton button;
  if (event->button() == Qt::LeftButton) {
    button = RenderWindowCallbacks::MouseButton::kLeft;
  } else if (event->button() == Qt::MiddleButton) {
    button = RenderWindowCallbacks::MouseButton::kMiddle;
  } else if (event->button() == Qt::RightButton) {
    button = RenderWindowCallbacks::MouseButton::kRight;
  } else {
    return;
  }
  
  callbacks_->MouseDown(button, event->x(), event->y());
  event->accept();
}

void RenderWidgetOpenGL::mouseMoveEvent(QMouseEvent* event) {
  callbacks_->MouseMove(event->x(), event->y());
  event->accept();
}

void RenderWidgetOpenGL::mouseReleaseEvent(QMouseEvent* event) {
  RenderWindowCallbacks::MouseButton button;
  if (event->button() == Qt::LeftButton) {
    button = RenderWindowCallbacks::MouseButton::kLeft;
  } else if (event->button() == Qt::MiddleButton) {
    button = RenderWindowCallbacks::MouseButton::kMiddle;
  } else if (event->button() == Qt::RightButton) {
    button = RenderWindowCallbacks::MouseButton::kRight;
  } else {
    return;
  }
  
  callbacks_->MouseUp(button, event->x(), event->y());
  event->accept();
}

void RenderWidgetOpenGL::wheelEvent(QWheelEvent* event) {
  if (event->orientation() == Qt::Vertical) {
    callbacks_->WheelRotated(event->delta() / 8.0f, RenderWindowCallbacks::ConvertQtModifiers(event));
  }
}

void RenderWidgetOpenGL::keyPressEvent(QKeyEvent* event) {
  if (event->text().size() > 0) {
    callbacks_->KeyPressed(event->text()[0].toLatin1(), RenderWindowCallbacks::ConvertQtModifiers(event));
  }
}

void RenderWidgetOpenGL::keyReleaseEvent(QKeyEvent* event) {
  if (event->text().size() > 0) {
    callbacks_->KeyReleased(event->text()[0].toLatin1(), RenderWindowCallbacks::ConvertQtModifiers(event));
  }
}


RenderWindowQtOpenGL::RenderWindowQtOpenGL(
    const string& title,
    int width,
    int height,
    const shared_ptr<RenderWindowCallbacks>& callbacks,
    bool use_qt_thread,
    bool show)
    : RenderWindowQt(title, width, height, callbacks, use_qt_thread, show) {
  auto init_function = [&](){
    // Add the OpenGL render widget to the window created by the parent class.
    render_widget_ = new RenderWidgetOpenGL(callbacks);
    window_->setCentralWidget(render_widget_);
  };
  
  if (use_qt_thread) {
    RunInQtThreadBlocking(init_function);
  } else {
    init_function();
  }
}

void RenderWindowQtOpenGL::RenderFrame() {
  auto render_function = [&](){
    render_widget_->update(render_widget_->rect());
  };
  
  if (use_qt_thread_) {
    RunInQtThread(render_function);
  } else {
    render_function();
  }
}

void RenderWindowQtOpenGL::MakeContextCurrent() {
  render_widget_->makeCurrent();
}

void RenderWindowQtOpenGL::ReleaseCurrentContext() {
  render_widget_->doneCurrent();
}

}

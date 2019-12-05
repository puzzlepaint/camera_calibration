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

#include "libvis/opengl_context_qt.h"

#include <QCoreApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>

#include "libvis/glew.h"
#include "libvis/logging.h"
#include "libvis/qt_thread.h"

namespace vis {

OpenGLContextQt::~OpenGLContextQt() {
  // Delete the surface (since we always own it), but not the context (we might
  // own it, but then it should be deleted by Deinitialize()).
  if (surface) {
    RunInQtThreadBlocking([&](){
        delete surface;
    });
  }
}

bool OpenGLContextQt::InitializeWindowless(OpenGLContextImpl* sharing_context) {
  OpenGLContextQt* sharing_context_qt = dynamic_cast<OpenGLContextQt*>(sharing_context);
  if (sharing_context && !sharing_context_qt) {
    LOG(ERROR) << "Can only share names with an OpenGL context with the same implementation.";
    return false;
  }
  
  if (!qApp) {
    LOG(ERROR) << "A QApplication must be created before using OpenGLContextQt.";
    return false;
  }
  
  context = new QOpenGLContext();
  
  if (sharing_context_qt) {
    context->setShareContext(sharing_context_qt->context);
  }
  
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setAlphaBufferSize(8);
  context->setFormat(format);
  
  needs_glew_initialization = true;
  
  if (!context->create()) {
    delete context;
    context = nullptr;
    return false;
  }
  
  surface = new QOffscreenSurface();
  surface->setFormat(context->format());
  RunInQtThreadBlocking([&](){
    surface->create();
  });
  
  return true;
}

void OpenGLContextQt::Deinitialize() {
  delete context;
  context = nullptr;
  if (surface) {
    RunInQtThreadBlocking([&](){
        delete surface;
    });
  }
  surface = nullptr;
}

void OpenGLContextQt::AttachToCurrent() {
  context = QOpenGLContext::currentContext();
  
  if (context) {
    // TODO: Maybe do this surface allocation lazily (in MakeCurrent()) such
    //       that it can be avoided in cases where the context won't be used?
    
    surface = new QOffscreenSurface();
    surface->setFormat(context->format());
    RunInQtThreadBlocking([&](){
        surface->create();
    });
  }
}

void OpenGLContextQt::MakeCurrent() {
  if (!context) {
    QOpenGLContext* current_context = QOpenGLContext::currentContext();
    if (current_context) {
      current_context->doneCurrent();
    }
    return;
  }
  
  QCoreApplication::instance()->setAttribute(Qt::AA_DontCheckOpenGLContextThreadAffinity);
  
  if (!context->makeCurrent(surface)) {
    LOG(ERROR) << "Failed to make the context current.";
  }
  
  if (needs_glew_initialization) {
    // Initialize GLEW on first switch to a context.
    InitializeGLEW();
    needs_glew_initialization = false;
  }
}

}

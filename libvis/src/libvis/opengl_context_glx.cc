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

#include "libvis/opengl_context_glx.h"

#include "libvis/glew.h"
#include "libvis/logging.h"

namespace vis {
#ifndef WIN32

int XErrorHandler(Display* dsp, XErrorEvent* error) {
  constexpr int kBufferSize = 512;
  char error_string[kBufferSize];
  XGetErrorText(dsp, error->error_code, error_string, kBufferSize);

  LOG(FATAL) << "X Error:\n" << error_string;
  return 0;
}

bool OpenGLContextGLX::InitializeWindowless(OpenGLContextImpl* sharing_context) {
  OpenGLContextGLX* sharing_context_glx = dynamic_cast<OpenGLContextGLX*>(sharing_context);
  if (sharing_context && !sharing_context_glx) {
    LOG(ERROR) << "Can only share names with an OpenGL context with the same implementation.";
    return false;
  }
  
  GLint attributes[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, None};
  
  int (*old_error_handler)(Display*, XErrorEvent*) =
      XSetErrorHandler(XErrorHandler);
  
  Display* display = XOpenDisplay(NULL);
  if (!display) {
    LOG(ERROR) << "Cannot connect to X server.";
    return false;
  }
  
  Window root_window = DefaultRootWindow(display);
  XVisualInfo* visual = glXChooseVisual(display, 0, attributes);
  if (!visual) {
    LOG(ERROR) << "No appropriate visual found.";
    return false;
  }
  
  GLXContext glx_context =
      glXCreateContext(display, visual, sharing_context ? sharing_context_glx->context : nullptr, GL_TRUE);
  if (!glx_context) {
    LOG(ERROR) << "Cannot create GLX context.";
    return false;
  }
  XFree(visual);
  
  this->display = display;
  drawable = root_window;
  context = glx_context;
  needs_glew_initialization = true;
  
  XSetErrorHandler(old_error_handler);
  return true;
}

void OpenGLContextGLX::Deinitialize() {
  if (!context) {
    return;
  }
  
  glXDestroyContext(display, context);
  XCloseDisplay(display);
  
  drawable = None;
  context = nullptr;
}

void OpenGLContextGLX::AttachToCurrent() {
  display = glXGetCurrentDisplay();
  drawable = glXGetCurrentDrawable();
  context = glXGetCurrentContext();
  needs_glew_initialization = false;  // TODO: This is not clear.
}

void OpenGLContextGLX::MakeCurrent() {
  int (*old_error_handler)(Display*, XErrorEvent*) =
      XSetErrorHandler(XErrorHandler);
  
  if (glXMakeCurrent(display, drawable, context) == GL_FALSE) {
    LOG(FATAL) << "Cannot make GLX context current.";
  }
  
  if (needs_glew_initialization) {
    // Initialize GLEW on first switch to a context.
    InitializeGLEW();
    needs_glew_initialization = false;
  }
  
  XSetErrorHandler(old_error_handler);
}

#endif
}

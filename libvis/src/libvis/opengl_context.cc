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

#include "libvis/opengl_context_qt.h"  // must be first

#include "libvis/opengl_context.h"

#include "libvis/logging.h"

#include "libvis/opengl_context_glx.h"

namespace vis {

constexpr bool kUseQtImplementation = true;  // TODO: Make dependent on availability

OpenGLContext::OpenGLContext() { }

OpenGLContext::OpenGLContext(OpenGLContext&& other) { impl.swap(other.impl); }

OpenGLContext& OpenGLContext::operator=(OpenGLContext&& other) {
  impl.swap(other.impl);
  return *this;
}

OpenGLContext::~OpenGLContext() {
  Deinitialize();
}

bool OpenGLContext::InitializeWindowless(OpenGLContext* sharing_context) {
  if (kUseQtImplementation) {
    impl.reset(new OpenGLContextQt());
  } 
#ifndef WIN32
  else {
    impl.reset(new OpenGLContextGLX());
  }
#endif // !WIN32
  return impl->InitializeWindowless((sharing_context && sharing_context->impl) ? sharing_context->impl.get() : nullptr);
}

void OpenGLContext::Deinitialize() {
  if (impl) {
    impl->Deinitialize();
  }
}

void OpenGLContext::AttachToCurrent() {
  if (kUseQtImplementation) {
    impl.reset(new OpenGLContextQt());
  }
#ifndef WIN32
  else {
    impl.reset(new OpenGLContextGLX());
  }
#endif // !WIN32
  impl->AttachToCurrent();
}

void OpenGLContext::Detach() {
  impl.reset();
}

void OpenGLContext::MakeCurrent(OpenGLContext* old_context) {
  if (old_context) {
    old_context->AttachToCurrent();
  }
  impl->MakeCurrent();
}


void SwitchOpenGLContext(OpenGLContext& new_context, OpenGLContext* old_context) {
  new_context.MakeCurrent(old_context);
}

}

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


#include "libvis/render_window.h"

#ifdef LIBVIS_HAVE_QT
#include "libvis/render_window_qt_opengl.h"
#ifdef LIBVIS_HAVE_VULKAN
#include "libvis/render_window_qt_vulkan.h"
#endif
#endif

#ifdef WIN32
#undef CreateWindow
#endif

namespace vis {

shared_ptr<RenderWindow> RenderWindow::CreateWindow(const std::string& title, const shared_ptr<RenderWindowCallbacks>& callbacks) {
  (void) title;
  (void) callbacks;
  
  #ifdef LIBVIS_HAVE_QT
    #ifdef LIBVIS_HAVE_VULKAN
      return CreateWindow(title, API::kVulkan, callbacks);
    #else
      return CreateWindow(title, API::kOpenGL, callbacks);
    #endif
  #else
    // No implementation available.
    return nullptr;
  #endif
}

shared_ptr<RenderWindow> RenderWindow::CreateWindow(const std::string& title, API api, const shared_ptr<RenderWindowCallbacks>& callbacks) {
  return CreateWindow(title, -1, -1, api, callbacks);
}

shared_ptr<RenderWindow> RenderWindow::CreateWindow(const std::string& title, int width, int height, API api, const shared_ptr<RenderWindowCallbacks>& callbacks) {
  (void) title;
  (void) api;
  (void) width;
  (void) height;
  (void) callbacks;
  
  #ifdef LIBVIS_HAVE_QT
    if (api == API::kVulkan) {
      #ifdef LIBVIS_HAVE_VULKAN
        return shared_ptr<RenderWindow>(new RenderWindowQtVulkan(title, width, height, callbacks));
      #else
        return nullptr;
      #endif
    } else if (api == API::kOpenGL) {
      return shared_ptr<RenderWindow>(new RenderWindowQtOpenGL(title, width, height, callbacks));
    } else {
      return nullptr;
    }
  #else
    // No implementation available.
    return nullptr;
  #endif
}

RenderWindow::RenderWindow(const shared_ptr<RenderWindowCallbacks>& callbacks)
    : callbacks_(callbacks) {
  callbacks_->SetRenderWindow(this);
}

}

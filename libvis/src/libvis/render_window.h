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

#include <memory>
#include <string>

#include "libvis/libvis.h"
#include "libvis/window_callbacks.h"

#ifdef WIN32
#undef CreateWindow
#endif

namespace vis {

class RenderWindow;

// Provides a window to render 3D content. This class is independent of both
// the underlying platform and the underlying graphics API. It serves as a base
// class for specific implementations. Use the factory method CreateWindow() to
// create a window for the current environment.
class RenderWindow {
 public:
  enum class API {
    kOpenGL,
    kVulkan
  };
  
  virtual inline ~RenderWindow() {}
  
  // Returns false once the window has been closed by the user.
  virtual bool IsOpen() = 0;
  
  // Invokes the rendering of a frame.
  virtual void RenderFrame() = 0;
  
  // For graphics APIs with a context (i.e. OpenGL), makes the context current for this thread. Otherwise, does nothing.
  virtual void MakeContextCurrent() {}
  
  // For graphics APIs with a context (i.e. OpenGL), releases the context. Otherwise, does nothing.
  virtual void ReleaseCurrentContext() {}
  
  // Creates a render window with any API.
  static shared_ptr<RenderWindow> CreateWindow(const std::string& title, const shared_ptr<RenderWindowCallbacks>& callbacks);
  
  // Creates a render window with a specific API.
  static shared_ptr<RenderWindow> CreateWindow(const std::string& title, API api, const shared_ptr<RenderWindowCallbacks>& callbacks);
  
  // Creates a render window with a specific API and initial size.
  static shared_ptr<RenderWindow> CreateWindow(const std::string& title, int width, int height, API api, const shared_ptr<RenderWindowCallbacks>& callbacks);
  
 protected:
  RenderWindow(const shared_ptr<RenderWindowCallbacks>& callbacks);
  
  shared_ptr<RenderWindowCallbacks> callbacks_;
};

}

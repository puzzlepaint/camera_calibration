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

namespace vis {

class OpenGLContextImpl {
 public:
  virtual ~OpenGLContextImpl() = default;
  
  virtual bool InitializeWindowless(OpenGLContextImpl* sharing_context = nullptr) = 0;
  
  // Deinitializes the context.
  virtual void Deinitialize() = 0;
  
  // Attaches the context which is current to this thread to this object. You
  // may have to call Detach() manually after this to prevent deletion of the
  // context once this object is destructed.
  virtual void AttachToCurrent() = 0;
  
  // Makes this context current in this thread and stores the old context in
  // old_context (if given).
  virtual void MakeCurrent() = 0;
};


class OpenGLContext {
 public:
  // No-op constructor. Call InitializeWindowless() to initialize the context or
  // AttachToCurrent() to attach the object to the current context.
  OpenGLContext();
  
  // Swaps content with the other context.
  OpenGLContext& operator=(OpenGLContext&& other);
  
  // Swaps content with the other context.
  OpenGLContext(OpenGLContext&& other);
  
  // Deinitializes the context if it is initialized.
  ~OpenGLContext();
  
  
  // Initializes a windowless context. Call SwitchOpenGLContext() afterwards to
  // make it current.
  // 
  // \param sharing_context A context to share names with, or nullptr.
  // 
  // \returns true if successful.
  bool InitializeWindowless(OpenGLContext* sharing_context = nullptr);
  
  // Deinitializes the context.
  void Deinitialize();
  
  // Attaches the context which is current to this thread to this object. You
  // may have to call Detach() manually after this to prevent deletion of the
  // context once this object is destructed.
  void AttachToCurrent();
  
  // Detaches the contained context from this object without releasing it. Use
  // with care.
  void Detach();
  
  // Makes this context current in this thread and stores the old context in
  // old_context (if given).
  void MakeCurrent(OpenGLContext* old_context = nullptr);
  
  
  std::unique_ptr<OpenGLContextImpl> impl;
};


// Switches the current thread's OpenGL context to the given context, and
// returns the previously active context in old_context if it is non-null. One
// context can be current to only one thread at a time. Notice that this
// function creates a new OpenGLContext from the current one if old_context is
// given, which by default will get deinitialized once the OpenGLContext object
// is destructed. If this is undesired, call Detach() on the returned object.
// In case no context is current when this function is called, the returned
// object will represent no context and thus Detach() is unnecessary.
void SwitchOpenGLContext(OpenGLContext& new_context,
                         OpenGLContext* old_context = nullptr);

}  // namespace vis

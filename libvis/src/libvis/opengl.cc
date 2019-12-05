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


#include "libvis/opengl.h"

#ifdef WIN32
#include <windows.h>
#else
#include <GL/glx.h>
#endif

namespace vis {

std::string GetGLErrorName(GLenum error_code) {
  if (error_code == GL_NO_ERROR) {
    return "GL_NO_ERROR";
  } else if (error_code == GL_INVALID_ENUM) {
    return "GL_INVALID_ENUM";
  } else if (error_code == GL_INVALID_VALUE) {
    return "GL_INVALID_VALUE";
  } else if (error_code == GL_INVALID_OPERATION) {
    return "GL_INVALID_OPERATION";
  } else if (error_code == GL_INVALID_FRAMEBUFFER_OPERATION) {
    return "GL_INVALID_FRAMEBUFFER_OPERATION";
  } else if (error_code == GL_OUT_OF_MEMORY) {
    return "GL_OUT_OF_MEMORY";
  } else if (error_code == GL_STACK_UNDERFLOW) {
    return "GL_STACK_UNDERFLOW";
  } else if (error_code == GL_STACK_OVERFLOW) {
    return "GL_STACK_OVERFLOW";
  } else {
    return "UNKNOWN GL ERROR";
  }
}

std::string GetGLErrorDescription(GLenum error_code) {
  if (error_code == GL_NO_ERROR) {
    return "No error has been recorded.";
  } else if (error_code == GL_INVALID_ENUM) {
    return "An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.";
  } else if (error_code == GL_INVALID_VALUE) {
    return "A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.";
  } else if (error_code == GL_INVALID_OPERATION) {
    return "The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.";
  } else if (error_code == GL_INVALID_FRAMEBUFFER_OPERATION) {
    return "The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag.";
  } else if (error_code == GL_OUT_OF_MEMORY) {
    return "There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
  } else if (error_code == GL_STACK_UNDERFLOW) {
    return " An attempt has been made to perform an operation that would cause an internal stack to underflow.";
  } else if (error_code == GL_STACK_OVERFLOW) {
    return "An attempt has been made to perform an operation that would cause an internal stack to overflow.";
  } else {
    return "";
  }
}

bool IsOpenGLContextAvailable() {
#ifdef WIN32
  return (wglGetCurrentContext() != nullptr);
#else
  return (glXGetCurrentContext() != nullptr);
#endif
}

}

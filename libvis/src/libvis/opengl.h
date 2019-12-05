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

// Provides suitable OpenGL includes depending on the platform.

#include <string>

#include "libvis/logging.h"

#include "libvis/libvis.h"
#ifdef WIN32
#include <windows.h>
#endif

#ifdef ANDROID
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <glues/glu.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#endif


namespace vis {

std::string GetGLErrorName(GLenum error_code);

std::string GetGLErrorDescription(GLenum error_code);

// Tests whether a valid OpenGL context is current.
bool IsOpenGLContextAvailable();

// This uses a macro such that LOG(ERROR) picks up the correct file and line
// number.
#define CHECK_OPENGL_NO_ERROR() \
  do { \
    GLenum error = glGetError(); \
    if (error == GL_NO_ERROR) {  \
      break; \
    } \
    LOG(ERROR) << "OpenGL Error: " << GetGLErrorName(error) << " (" << error << "), description:" << std::endl << GetGLErrorDescription(error); \
  } while (true)


// Traits type allowing to get the OpenGL enum value for a given type as:
// GetGLType<Type>::value. For example, GetGLType<int>::value resoves to GL_INT.
// Not applicable for: GL_HALF_FLOAT, GL_FIXED, GL_INT_2_10_10_10_REV,
// GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_INT_10F_11F_11F_REV.
template <typename T> struct GetGLType {
  // Default value for unknown type.
  static const GLenum value = GL_FLOAT;
};

template<> struct GetGLType<i8> {
  static const GLenum value = GL_BYTE;
};

template<> struct GetGLType<u8> {
  static const GLenum value = GL_UNSIGNED_BYTE;
};

template<> struct GetGLType<i16> {
  static const GLenum value = GL_SHORT;
};

template<> struct GetGLType<u16> {
  static const GLenum value = GL_UNSIGNED_SHORT;
};

template<> struct GetGLType<int> {
  static const GLenum value = GL_INT;
};

template<> struct GetGLType<u32> {
  static const GLenum value = GL_UNSIGNED_INT;
};

template<> struct GetGLType<float> {
  static const GLenum value = GL_FLOAT;
};

template<> struct GetGLType<double> {
  static const GLenum value = GL_DOUBLE;
};

}

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

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/opengl.h"

namespace vis {

// Represents a shader program. At least a fragment and a vertex shader must be
// attached to a program to be complete. This class assumes some common
// attribute names in the shaders to simplify attribute handling:
// 
// in_position : Position input to the vertex shader.
// in_color    : Color input to the vertex shader.
//
// A current OpenGL context is required for calling each member function except
// the constructor. This includes the destructor.
class ShaderProgramOpenGL {
 public:
  enum class ShaderType {
    kVertexShader   = 1 << 0,
    kGeometryShader = 1 << 1,
    kFragmentShader = 1 << 2
  };
  
  // No-op constructor, no OpenGL context required.
  ShaderProgramOpenGL();
  
  // Deletes the program. Attention: Requires a current OpenGL context for
  // this thread! You may need to explicitly delete this object at a point where
  // such a context still exists.
  ~ShaderProgramOpenGL();
  
  // Attaches a shader to the program. Returns false if the shader does not
  // compile.
  bool AttachShader(const char* source_code, ShaderType type);
  
  // Links the program. Must be called after all shaders have been attached.
  // Returns true if successful.
  bool LinkProgram();
  
  // Makes this program the active program (calls glUseProgram()).
  void UseProgram() const;
  
  // Returns the location of the given uniform. If the uniform name does not
  // exist, returns -1.
  GLint GetUniformLocation(const char* name) const;
  
  // Same as GetUniformLocation(), but aborts the program if the uniform does
  // not exist.
  GLint GetUniformLocationOrAbort(const char* name) const;
  
  
  // Uniform setters.
  void SetUniformMatrix4f(GLint location, const Mat4f& matrix);
  void SetUniform1f(GLint location, float x);
  void SetUniform3f(GLint location, float x, float y, float z);
  void SetUniform4f(GLint location, float x, float y, float z, float w);
  
  
  // Attribute setters.
  void SetPositionAttribute(int component_count, GLenum component_type, GLsizei stride, usize offset);
  void SetColorAttribute(int component_count, GLenum component_type, GLsizei stride, usize offset);
  
  
  inline GLuint program_name() const { return program_; }
  
 private:
  // OpenGL name of the program. This is zero if the program has not been
  // successfully linked yet.
  GLuint program_;
  
  // OpenGL names of the shaders attached to the program. These are zero if not
  // attached.
  GLuint vertex_shader_;
  GLuint geometry_shader_;
  GLuint fragment_shader_;
  
  // Attribute locations. These are -1 if no attribute with the common name
  // exists.
  GLint position_attribute_location_;
  GLint color_attribute_location_;
};

}

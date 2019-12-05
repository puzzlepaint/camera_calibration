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


#include "libvis/shader_program_opengl.h"

#include <memory>

#include "libvis/logging.h"

namespace vis {

ShaderProgramOpenGL::ShaderProgramOpenGL()
    : program_(0),
      vertex_shader_(0),
      geometry_shader_(0),
      fragment_shader_(0),
      position_attribute_location_(-1),
      color_attribute_location_(-1) {}

ShaderProgramOpenGL::~ShaderProgramOpenGL() {
  if (vertex_shader_ != 0) {
    glDetachShader(program_, vertex_shader_);
    glDeleteShader(vertex_shader_);
  }
  
  if (geometry_shader_ != 0) {
    glDetachShader(program_, geometry_shader_);
    glDeleteShader(geometry_shader_);
  }
  
  if (fragment_shader_ != 0) {
    glDetachShader(program_, fragment_shader_);
    glDeleteShader(fragment_shader_);
  }

  if (program_ != 0) {
    glDeleteProgram(program_);
  }
}

bool ShaderProgramOpenGL::AttachShader(const char* source_code, ShaderType type) {
  CHECK(program_ == 0) << "Cannot attach a shader after linking the program.";
  
  GLenum shader_enum;
  GLuint* shader_name = nullptr;
  if (type == ShaderType::kVertexShader) {
    shader_enum = GL_VERTEX_SHADER;
    shader_name = &vertex_shader_;
  } else if (type == ShaderType::kGeometryShader) {
    shader_enum = GL_GEOMETRY_SHADER;
    shader_name = &geometry_shader_;
  } else if (type == ShaderType::kFragmentShader) {
    shader_enum = GL_FRAGMENT_SHADER;
    shader_name = &fragment_shader_;
  }
  CHECK(shader_name != nullptr) << "Unknown shader type.";
  
  *shader_name = glCreateShader(shader_enum);
  const GLchar* source_code_ptr =
      static_cast<const GLchar*>(source_code);
  glShaderSource(*shader_name, 1, &source_code_ptr, NULL);
  glCompileShader(*shader_name);

  GLint compiled;
  glGetShaderiv(*shader_name, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(*shader_name, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(*shader_name, length, &length, log.get());
    LOG(ERROR) << "GL Shader Compilation Error: " << log.get();
    return false;
  }
  return true;
}

bool ShaderProgramOpenGL::LinkProgram() {
  CHECK(program_ == 0) << "Program already linked.";
  
  program_ = glCreateProgram();
  if (fragment_shader_ != 0) {
    glAttachShader(program_, fragment_shader_);
  }
  if (geometry_shader_ != 0) {
    glAttachShader(program_, geometry_shader_);
  }
  if (vertex_shader_ != 0) {
    glAttachShader(program_, vertex_shader_);
  }
  glLinkProgram(program_);
  
  GLint linked;
  glGetProgramiv(program_, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLint length;
    glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetProgramInfoLog(program_, length, &length, log.get());
    LOG(ERROR) << "GL Program Linker Error: " << log.get();
    return false;
  }
  
  // Get attributes.
  position_attribute_location_ = glGetAttribLocation(program_, "in_position");
  color_attribute_location_ = glGetAttribLocation(program_, "in_color");
  return true;
}

void ShaderProgramOpenGL::UseProgram() const {
  glUseProgram(program_);
}

GLint ShaderProgramOpenGL::GetUniformLocation(const char* name) const {
  return glGetUniformLocation(program_, name);
}

GLint ShaderProgramOpenGL::GetUniformLocationOrAbort(const char* name) const {
  GLint result = glGetUniformLocation(program_, name);
  CHECK(result != -1) << "Uniform does not exist (might have been optimized out by the compiler): " << name;
  return result;
}

void ShaderProgramOpenGL::SetUniformMatrix4f(GLint location, const Mat4f& matrix) {
  glUniformMatrix4fv(location, 1, GL_FALSE, matrix.data());
}

void ShaderProgramOpenGL::SetUniform1f(GLint location, float x) {
  glUniform1f(location, x);
}

void ShaderProgramOpenGL::SetUniform3f(GLint location, float x, float y, float z) {
  glUniform3f(location, x, y, z);
}

void ShaderProgramOpenGL::SetUniform4f(GLint location, float x, float y, float z, float w) {
  glUniform4f(location, x, y, z, w);
}

void ShaderProgramOpenGL::SetPositionAttribute(int component_count, GLenum component_type, GLsizei stride, usize offset) {
  // CHECK(position_attribute_location_ != -1) << "SetPositionAttribute() called, but no attribute \"in_position\" found.";
  if (position_attribute_location_ == -1) {
    // Allow using an object with positions with a material that ignores the positions.
    return;
  }
  
  glEnableVertexAttribArray(position_attribute_location_);
  glVertexAttribPointer(
      position_attribute_location_,
      component_count,
      component_type,
      GL_FALSE,  // Whether fixed-point data values should be normalized.
      stride,
      reinterpret_cast<char*>(0) + offset);
  CHECK_OPENGL_NO_ERROR();
}

void ShaderProgramOpenGL::SetColorAttribute(int component_count, GLenum component_type, GLsizei stride, usize offset) {
  // CHECK(color_attribute_location_ != -1) << "SetColorAttribute() called, but no attribute \"in_color\" found.";
  if (color_attribute_location_ == -1) {
    // Allow using an object with colors with a material that ignores the colors.
    return;
  }
  
  glEnableVertexAttribArray(color_attribute_location_);
  glVertexAttribPointer(
      color_attribute_location_,
      component_count,
      component_type,
      GL_TRUE,  // Whether fixed-point data values should be normalized.
      stride,
      reinterpret_cast<char*>(0) + offset);
  CHECK_OPENGL_NO_ERROR();
}

}

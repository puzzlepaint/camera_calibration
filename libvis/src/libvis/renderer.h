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
#include <unordered_map>

#include <GL/glew.h>
#include <GL/gl.h>

#include "libvis/camera.h"
#include "libvis/shader_program_opengl.h"
#include "libvis/sophus.h"

namespace vis {

class RendererProgramBase {
 public:
  RendererProgramBase();
  virtual ~RendererProgramBase();
  
  void Initialize(
      bool render_color,
      bool render_depth);
  virtual const GLchar* GetShaderUniformDefinitions() const = 0;
  virtual const GLchar* GetShaderDistortionCode() const = 0;
  virtual void GetUniformLocations(const ShaderProgramOpenGL& shader_program) = 0;
  void SetUniformValues(const Camera& camera);
  
  inline ShaderProgramOpenGL& shader_program() { return shader_program_; }
  inline const ShaderProgramOpenGL& shader_program() const { return shader_program_; }
  inline GLint a_position_location() const { return a_position_location_; }
  inline GLint a_color_location() const { return a_color_location_; }
  inline GLint u_model_view_matrix_location() const { return u_model_view_matrix_location_;}
  inline GLint u_projection_matrix_location() const { return u_projection_matrix_location_;}
  
 private:
  // Shader names.
  ShaderProgramOpenGL shader_program_;
  
  // Common shader variable locations.
  GLint a_position_location_;
  GLint a_color_location_;
  GLint u_model_view_matrix_location_;
  GLint u_projection_matrix_location_;
};

typedef std::shared_ptr<RendererProgramBase>
    RendererProgramBasePtr;


// This class has a template specialization for each camera model, implementing
// the respective distortion.
template <class Camera>
class RendererProgram : public RendererProgramBase {};

template <>
class RendererProgram<PinholeCamera4f> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const PinholeCamera4f& /*camera*/) const;
};

template <>
class RendererProgram<RadtanCamera8d> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const RadtanCamera8d& /*camera*/) const;
};

template <>
class RendererProgram<RadtanCamera9d> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const RadtanCamera9d& /*camera*/) const;
};

template <>
class RendererProgram<ThinPrismFisheyeCamera12d> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const ThinPrismFisheyeCamera12d& /*camera*/) const;
};

template <>
class RendererProgram<NonParametricBicubicProjectionCamerad> : public RendererProgramBase {
 public:
  const GLchar* GetShaderUniformDefinitions() const override;
  const GLchar* GetShaderDistortionCode() const override;
  void GetUniformLocations(const ShaderProgramOpenGL& /*shader_program*/) override;
  void SetUniformValues(const NonParametricBicubicProjectionCamerad& /*camera*/) const;
};


// For each OpenGL context used, a vertex shader storage is required.
class RendererProgramStorage {
 friend class Renderer;
 public:
  RendererProgramStorage();
  
  RendererProgramBasePtr depth_program(Camera::Type type);
  RendererProgramBasePtr color_and_depth_program(Camera::Type type);
  
 private:
  std::unordered_map<int, RendererProgramBasePtr> depth_programs_;
  std::unordered_map<int, RendererProgramBasePtr> color_and_depth_programs_;
};

typedef std::shared_ptr<RendererProgramStorage>
    RendererProgramStoragePtr;


// Uses OpenGL for rendering depth maps from meshes. Radial camera distortion is
// applied on the vertex level, so the geometry should be represented by a dense
// mesh for accurate warping.
// TODO: Rename to RendererOpenGL
class Renderer {
 public:
  // Creates OpenGL objects (i.e., must be called with the correct OpenGL
  // context). The parameters specify the maximum size in pixels of images to be
  // rendered. Only the combinations "render_color && render_depth" and
  // "!render_color && render_depth" are supported.
  Renderer(
      bool render_color,
      bool render_depth,
      int max_width, int max_height,
      const RendererProgramStoragePtr& program_storage);

  // Destructor.
  ~Renderer();
  
  void BeginRendering(
      const Sophus::SE3f& transformation,
      const Camera& camera, float min_depth, float max_depth);
  void RenderTriangleList(
      GLuint vertex_buffer, GLuint index_buffer, uint32_t index_count);
  void RenderTriangleList(
      GLuint vertex_buffer, GLuint color_buffer, GLuint index_buffer, uint32_t index_count);
  void EndRendering();
  
  // Downloads the result to the CPU.
  void DownloadDepthResult(int width, int height, float* buffer);
  void DownloadColorResult(int width, int height, uint8_t* buffer);
  
  inline int max_width() const { return max_width_; }
  inline int max_height() const { return max_height_; }
  
  inline ShaderProgramOpenGL& shader_program() { return current_program_->shader_program(); }
  inline const ShaderProgramOpenGL& shader_program() const { return current_program_->shader_program(); }

 private:
  void CreateFrameBufferObject();

  void SetupProjection(
      const Sophus::SE3f& transformation, const Camera& camera,
      float min_depth, float max_depth);

  // Rendering target.
  GLuint frame_buffer_object_;
  GLuint depth_buffer_;
  GLuint depth_texture_;
  GLuint color_texture_;
  
  // Program storage and last program used in BeginRendering().
  RendererProgramStoragePtr program_storage_;
  RendererProgramBasePtr current_program_;

  // Target image size.
  int max_width_;
  int max_height_;
  
  // Settings.
  bool render_color_;
  bool render_depth_;
};

}

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


#include "libvis/renderer.h"

#include "libvis/logging.h"

#include "libvis/opengl.h"

namespace vis {

RendererProgramBase::RendererProgramBase() {}
RendererProgramBase::~RendererProgramBase() {}

void RendererProgramBase::Initialize(
    bool render_color,
    bool render_depth) {
  // Create fragment shader.
  std::ostringstream fragment_shader_src;
  
  fragment_shader_src << "#version 300 es\n";
  if (render_depth) {
    fragment_shader_src << "in highp float var_depth;\n";
  }
  if (render_color) {
    fragment_shader_src << "in lowp vec3 var_color;\n";
  }
  int output_location_index = 0;
  if (render_depth) {
    fragment_shader_src << "layout(location = " << output_location_index << ") out highp float out_depth;\n";
    ++ output_location_index;
  }
  if (render_color) {
    fragment_shader_src << "layout(location = " << output_location_index << ") out lowp vec3 out_color;\n";
    ++ output_location_index;
  }
  fragment_shader_src << "void main() {\n";
  if (render_depth) {
    fragment_shader_src << "   out_depth = var_depth;\n";
  }
  if (render_color) {
    fragment_shader_src << "   out_color = var_color;\n";
  }
  fragment_shader_src << "}\n";

  CHECK(shader_program_.AttachShader(fragment_shader_src.str().c_str(), ShaderProgramOpenGL::ShaderType::kFragmentShader));

  // Create vertex shader.
  std::ostringstream vertex_shader_src_part1;
  vertex_shader_src_part1 << "#version 300 es\n"
                             "uniform mat4 u_model_view_matrix;\n"
                             "uniform mat4 u_projection_matrix;\n"
                             "in vec4 in_position;\n";
  if (render_color) {
    vertex_shader_src_part1 << "in vec3 in_color;\n";
  }
  if (render_depth) {
    vertex_shader_src_part1 << "out float var_depth;\n";
  }
  if (render_color) {
    vertex_shader_src_part1 << "out vec3 var_color;\n";
  }
  
  std::ostringstream vertex_shader_src_part3;
  vertex_shader_src_part3 << "void main() {\n";
  if (render_color) {
    vertex_shader_src_part3 << "  var_color = in_color;\n";
  }
  vertex_shader_src_part3 << "  vec4 localPoint = u_model_view_matrix * in_position;\n"
                             "  localPoint.xyz /= localPoint.w;\n";
  if (render_depth) {
    vertex_shader_src_part3 << "  var_depth = localPoint.z;\n";
  }
  
  const std::string vertex_shader_src_part5 =
      "  localPoint.w = 1.0;\n"
      "  gl_Position = u_projection_matrix * localPoint;\n"
      "}\n";
  
  std::string vertex_shader_src =
      vertex_shader_src_part1.str() +
      GetShaderUniformDefinitions() +
      vertex_shader_src_part3.str() +
      GetShaderDistortionCode() +
      vertex_shader_src_part5;
  
  CHECK(shader_program_.AttachShader(vertex_shader_src.c_str(), ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  // Create program.
  CHECK(shader_program_.LinkProgram());
  
  shader_program_.UseProgram();
  CHECK_OPENGL_NO_ERROR();
  
  // Get attributes.
  a_position_location_ = glGetAttribLocation(shader_program_.program_name(), "in_position");
  CHECK_OPENGL_NO_ERROR();
  
  if (render_color) {
    a_color_location_ = glGetAttribLocation(shader_program_.program_name(), "in_color");
    CHECK_OPENGL_NO_ERROR();
  }
  
  u_model_view_matrix_location_ = shader_program_.GetUniformLocationOrAbort("u_model_view_matrix");
  CHECK_OPENGL_NO_ERROR();
  
  u_projection_matrix_location_ = shader_program_.GetUniformLocationOrAbort("u_projection_matrix");
  CHECK_OPENGL_NO_ERROR();
  
  GetUniformLocations(shader_program_);
}

void RendererProgramBase::SetUniformValues(const Camera& camera) {
  IDENTIFY_CAMERA(
      camera,
      (reinterpret_cast<RendererProgram<_camera_type>*>(this))
          ->SetUniformValues(_camera));
}


const GLchar*
RendererProgram<PinholeCamera4f>::GetShaderUniformDefinitions() const {
  return "";
}

const GLchar*
RendererProgram<PinholeCamera4f>::GetShaderDistortionCode() const {
  return "";
}

void RendererProgram<PinholeCamera4f>::GetUniformLocations(
    const ShaderProgramOpenGL& /*shader_program*/) {
  // No special values.
}

void RendererProgram<PinholeCamera4f>::SetUniformValues(
    const PinholeCamera4f& /*camera*/) const {
  // No special values.
}


const GLchar*
RendererProgram<RadtanCamera8d>::GetShaderUniformDefinitions() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

const GLchar*
RendererProgram<RadtanCamera8d>::GetShaderDistortionCode() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

void RendererProgram<RadtanCamera8d>::GetUniformLocations(
    const ShaderProgramOpenGL& /*shader_program*/) {
  LOG(FATAL) << "Not implemented yet";
}

void RendererProgram<RadtanCamera8d>::SetUniformValues(
    const RadtanCamera8d& /*camera*/) const {
  LOG(FATAL) << "Not implemented yet";
}


const GLchar*
RendererProgram<RadtanCamera9d>::GetShaderUniformDefinitions() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

const GLchar*
RendererProgram<RadtanCamera9d>::GetShaderDistortionCode() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

void RendererProgram<RadtanCamera9d>::GetUniformLocations(
    const ShaderProgramOpenGL& /*shader_program*/) {
  LOG(FATAL) << "Not implemented yet";
}

void RendererProgram<RadtanCamera9d>::SetUniformValues(
    const RadtanCamera9d& /*camera*/) const {
  LOG(FATAL) << "Not implemented yet";
}


const GLchar*
RendererProgram<ThinPrismFisheyeCamera12d>::GetShaderUniformDefinitions() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

const GLchar*
RendererProgram<ThinPrismFisheyeCamera12d>::GetShaderDistortionCode() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

void RendererProgram<ThinPrismFisheyeCamera12d>::GetUniformLocations(
    const ShaderProgramOpenGL& /*shader_program*/) {
  LOG(FATAL) << "Not implemented yet";
}

void RendererProgram<ThinPrismFisheyeCamera12d>::SetUniformValues(
    const ThinPrismFisheyeCamera12d& /*camera*/) const {
  LOG(FATAL) << "Not implemented yet";
}


const GLchar*
RendererProgram<NonParametricBicubicProjectionCamerad>::GetShaderUniformDefinitions() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

const GLchar*
RendererProgram<NonParametricBicubicProjectionCamerad>::GetShaderDistortionCode() const {
  LOG(FATAL) << "Not implemented yet";
  return "";
}

void RendererProgram<NonParametricBicubicProjectionCamerad>::GetUniformLocations(
    const ShaderProgramOpenGL& /*shader_program*/) {
  LOG(FATAL) << "Not implemented yet";
}

void RendererProgram<NonParametricBicubicProjectionCamerad>::SetUniformValues(
    const NonParametricBicubicProjectionCamerad& /*camera*/) const {
  LOG(FATAL) << "Not implemented yet";
}


RendererProgramStorage::RendererProgramStorage() {}

RendererProgramBasePtr
RendererProgramStorage::depth_program(Camera::Type type) {
  auto it = depth_programs_.find(static_cast<int>(type));
  if (it != depth_programs_.end()) {
    return it->second;
  } else {
    // Create new program.
    RendererProgramBasePtr new_program;
    IDENTIFY_CAMERA_TYPE(type,
                       new_program.reset(new RendererProgram<_type>()));
    new_program->Initialize(false, true);
    depth_programs_.insert(make_pair(static_cast<int>(type), new_program));
    return new_program;
  }
}

RendererProgramBasePtr
RendererProgramStorage::color_and_depth_program(Camera::Type type) {
  auto it = color_and_depth_programs_.find(static_cast<int>(type));
  if (it != color_and_depth_programs_.end()) {
    return it->second;
  } else {
    // Create new program.
    RendererProgramBasePtr new_program;
    IDENTIFY_CAMERA_TYPE(type,
                       new_program.reset(new RendererProgram<_type>()));
    new_program->Initialize(true, true);
    color_and_depth_programs_.insert(make_pair(static_cast<int>(type), new_program));
    return new_program;
  }
}


Renderer::Renderer(
    bool render_color,
    bool render_depth,
    int max_width,
    int max_height,
    const RendererProgramStoragePtr& program_storage) {
  render_color_ = render_color;
  render_depth_ = render_depth;
  max_width_ = max_width;
  max_height_ = max_height;
  program_storage_ = program_storage;
  
  CreateFrameBufferObject();
}

Renderer::~Renderer() {
  CHECK(IsOpenGLContextAvailable()) << "An OpenGL context must be current while the renderer destructor is called!";

  glDeleteTextures(1, &depth_texture_);
  if (render_color_) {
    glDeleteTextures(1, &color_texture_);
  }
  glDeleteRenderbuffers(1, &depth_buffer_);
  glDeleteFramebuffers(1, &frame_buffer_object_);

  CHECK_OPENGL_NO_ERROR();
}

void Renderer::BeginRendering(
    const Sophus::SE3f& transformation,
    const Camera& camera, float min_depth, float max_depth) {
  // Get or create the shader program for this camera.
  RendererProgramBasePtr program;
  if (!render_color_ && render_depth_) {
    program = program_storage_->depth_program(camera.type());
  } else if (render_color_ && render_depth_) {
    program = program_storage_->color_and_depth_program(camera.type());
  } else {
    LOG(FATAL) << "Requested depth / color rendering combination is not supported.";
  }
  current_program_ = program;

  // Set states.
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDisable(GL_CULL_FACE);
  glFrontFace(GL_CW);
  CHECK_OPENGL_NO_ERROR();

  // Setup framebuffer and shaders.
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  
  if (render_color_ && render_depth_) {
    GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, buffers);
    CHECK_OPENGL_NO_ERROR();
  }
  
  // Clear buffers.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  program->shader_program().UseProgram();
  program->SetUniformValues(camera);
  CHECK_OPENGL_NO_ERROR();

  // Setup projection.
  SetupProjection(transformation, camera, min_depth, max_depth);
}

void Renderer::RenderTriangleList(
    GLuint vertex_buffer, GLuint index_buffer, uint32_t index_count) {
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glEnableVertexAttribArray(current_program_->a_position_location());
  glVertexAttribPointer(current_program_->a_position_location(), 3, GL_FLOAT, GL_FALSE,
                        3 * sizeof(float), reinterpret_cast<char*>(0) + 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  
  glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT,
                 reinterpret_cast<char*>(0) + 0);

  glDisableVertexAttribArray(current_program_->a_position_location());
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  CHECK_OPENGL_NO_ERROR();
}

void Renderer::RenderTriangleList(
    GLuint vertex_buffer, GLuint color_buffer, GLuint index_buffer, uint32_t index_count) {
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glEnableVertexAttribArray(current_program_->a_position_location());
  glVertexAttribPointer(current_program_->a_position_location(), 3, GL_FLOAT, GL_FALSE,
                        3 * sizeof(float), reinterpret_cast<char*>(0) + 0);
  
  glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
  glEnableVertexAttribArray(current_program_->a_color_location());
  glVertexAttribPointer(current_program_->a_color_location(), 3, GL_UNSIGNED_BYTE, GL_TRUE,
                        3 * sizeof(uint8_t), reinterpret_cast<char*>(0) + 0);
  
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

  glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT,
                 reinterpret_cast<char*>(0) + 0);

  glDisableVertexAttribArray(current_program_->a_position_location());
  glDisableVertexAttribArray(current_program_->a_color_location());
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  CHECK_OPENGL_NO_ERROR();
}

void Renderer::EndRendering() {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::DownloadDepthResult(int width, int height, float* buffer) {
  CHECK(render_depth_);
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, buffer);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
}

void Renderer::DownloadColorResult(int width, int height, uint8_t* buffer) {
  CHECK(render_color_);
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  if (render_depth_ && render_color_) {
    glReadBuffer(GL_COLOR_ATTACHMENT1);
  } else {
    glReadBuffer(GL_COLOR_ATTACHMENT0);
  }
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
}

void Renderer::CreateFrameBufferObject() {
  glGenFramebuffers(1, &frame_buffer_object_);
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  CHECK_OPENGL_NO_ERROR();

  // Add a depth buffer to the frame buffer object.
  glGenRenderbuffers(1, &depth_buffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, max_width_, max_height_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, depth_buffer_);
  CHECK_OPENGL_NO_ERROR();
  
  int color_attachment_index = 0;

  // Add a color texture to the frame buffer object.
  // This class renders the depth to this color texture in addition to the depth
  // buffer because reading out the depth buffer did not seem to be supported in
  // OpenGL ES 2.0. This might have changed in later versions and
  // efficiency might benefit from removing the additional color texture.
  if (render_depth_) {
    glGenTextures(1, &depth_texture_);
    glBindTexture(GL_TEXTURE_2D, depth_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, max_width_, max_height_, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_OPENGL_NO_ERROR();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + color_attachment_index, GL_TEXTURE_2D, depth_texture_, 0);
    ++ color_attachment_index;
  }
  
  if (render_color_) {
    glGenTextures(1, &color_texture_);
    glBindTexture(GL_TEXTURE_2D, color_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, max_width_, max_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_OPENGL_NO_ERROR();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + color_attachment_index, GL_TEXTURE_2D, color_texture_, 0);
    ++ color_attachment_index;
  }

  // Verify frame buffer object creation.
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  CHECK_EQ(static_cast<int>(status), GL_FRAMEBUFFER_COMPLETE);
  CHECK_OPENGL_NO_ERROR();
}

void Renderer::SetupProjection(
    const Sophus::SE3f& transformation, const Camera& camera,
    float min_depth, float max_depth) {
  CHECK_GT(max_depth, min_depth);
  CHECK_GT(min_depth, 0);
  
  float fx, fy, cx, cy;
  if (camera.type() == Camera::Type::kPinholeCamera4f) {
    const PinholeCamera4f& pinhole_camera = reinterpret_cast<const PinholeCamera4f&>(camera);
    
    fx = pinhole_camera.parameters()[0];
    fy = pinhole_camera.parameters()[1];
    cx = pinhole_camera.parameters()[2];
    cy = pinhole_camera.parameters()[3];
  } else {
    LOG(FATAL) << "Camera type not supported.";
    return;
  }

  // Row-wise projection matrix construction.
  float matrix[16];
  matrix[0] = (2 * fx) / camera.width();
  matrix[4] = 0;
  matrix[8] = 2 * cx / camera.width() - 1.0f;
  matrix[12] = 0;

  matrix[1] = 0;
  matrix[5] = (2 * fy) / camera.height();
  matrix[9] = 2 * cy / camera.height() - 1.0f;
  matrix[13] = 0;

  matrix[2] = 0;
  matrix[6] = 0;
  matrix[10] = (max_depth + min_depth) / (max_depth - min_depth);
  matrix[14] = -(2 * max_depth * min_depth) / (max_depth - min_depth);

  matrix[3] = 0;
  matrix[7] = 0;
  matrix[11] = 1;
  matrix[15] = 0;

  glUniformMatrix4fv(current_program_->u_projection_matrix_location(), 1, GL_FALSE, matrix);
  CHECK_OPENGL_NO_ERROR();

  // Model-view matrix construction.
  Eigen::Matrix3f rotation = transformation.so3().matrix();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix[i + j * 4] = rotation(i, j);
    }
    matrix[i + 12] = transformation.translation()(i);
  }
  matrix[3] = 0;
  matrix[7] = 0;
  matrix[11] = 0;
  matrix[15] = 1;

  glUniformMatrix4fv(current_program_->u_model_view_matrix_location(), 1, GL_FALSE, matrix);
  CHECK_OPENGL_NO_ERROR();

  // Set viewport.
  glViewport(0, 0, camera.width(), camera.height());
  CHECK_OPENGL_NO_ERROR();
}

}

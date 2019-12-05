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


// Must be included before render_display.h to avoid errors
#include <QApplication>
#include <QClipboard>

#include "libvis/render_display.h"

namespace vis {

RenderDisplay::RenderDisplay() {
  width_ = 0;
  height_ = 0;
  
  dragging_ = false;
  pressed_mouse_buttons_ = 0;
  
  // Set default view parameters.
  min_depth_ = 0.01f;
  max_depth_ = 50.0f;
  
  camera_free_orbit_theta_ = 0.5;
  camera_free_orbit_phi_ = 1.57;
  camera_free_orbit_radius_ = 6;
  camera_free_orbit_offset_ = Vec3f(0, 0, 0);
  
  splat_half_extent_in_pixels_ = 1.5f;
  
  // up_direction_rotation_ = Mat3f::Identity();
  up_direction_rotation_ = AngleAxisf(M_PI, Vec3f(0, 0, 1)) * AngleAxisf(-M_PI / 2, Vec3f(1, 0, 0));
  
  visualization_cloud_size_ = numeric_limits<usize>::max();
  
  have_visualization_cloud_ = false;
  
  user_callbacks_ = nullptr;
}

void RenderDisplay::SetUserCallbacks(RenderWindowCallbacks* user_callbacks) {
  unique_lock<mutex> lock(user_callbacks_mutex_);
  user_callbacks_ = user_callbacks;
}

void RenderDisplay::Update(const std::shared_ptr<Point3fC3u8Cloud>& cloud, const string& /*name*/) {
  // TODO: Use the name parameter to associate the new input with an object. Allow multiple objects to co-exist.
  
  unique_lock<mutex> lock(visualization_cloud_mutex_);
  new_visualization_cloud_ = cloud;
  window_->RenderFrame();
}

void RenderDisplay::Update(const std::shared_ptr<Point3fCu8Cloud>& cloud, const string& /*name*/) {
  // TODO: Use the name parameter to associate the new input with an object. Allow multiple objects to co-exist.
  
  unique_lock<mutex> lock(visualization_cloud_mutex_);
  new_visualization_cloud_.reset(new Point3fC3u8Cloud(cloud->size()));
  for (usize i = 0; i < cloud->size(); ++ i) {
    new_visualization_cloud_->at(i).position() = cloud->at(i).position();
    new_visualization_cloud_->at(i).color() = Vec3u8::Constant(cloud->at(i).color());
  }
  window_->RenderFrame();
}

void RenderDisplay::Initialize() {
  GLenum glew_init_result = glewInit();
  CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
  glGetError();  // Ignore GL_INVALID_ENUM​ error caused by glew
  CHECK_OPENGL_NO_ERROR();
  
  CreateSplatProgram();
  CHECK_OPENGL_NO_ERROR();
  
  CreateConstantColorProgram();
  CHECK_OPENGL_NO_ERROR();
  
  // TODO: It would be preferable to handle this in a sane way instead of
  //       simply creating a global VAO at the beginning and then forgetting
  //       about it.
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->Initialize();
  }
}

void RenderDisplay::Resize(int width, int height) {
  width_ = width;
  height_ = height;
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->Resize(width, height);
  }
}

void RenderDisplay::Render() {
  CHECK_OPENGL_NO_ERROR();
  
  unique_lock<mutex> lock(render_mutex_);
  
  
  // ### Setup ###
  
  // Setup the render_camera_.
  SetCamera();
  
  unique_lock<mutex> camera_mutex_lock(camera_mutex_);
  
  // Compute projection_matrix_ from the camera.
  ComputeProjectionMatrix();
  
  // Set the rendering bounds (viewport) according to the camera dimensions.
  SetupViewport();
  
  // Set the camera_T_world_ transformation according to an orbiting scheme.
  SetViewpoint();
  
  camera_mutex_lock.unlock();
  
  CHECK_OPENGL_NO_ERROR();
  
  // Compute the model-view-projection matrix.
  Mat4f model_matrix = Mat4f::Identity();
  Mat4f model_view_matrix = camera_T_world_.matrix() * model_matrix;
  model_view_projection_matrix_ = projection_matrix_ * model_view_matrix;
  
  // Set states for rendering.
  glClearColor(0.9f, 0.9f, 0.9f, 1.0f);  // background color
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  // Render.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  //glShadeModel(GL_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  CHECK_OPENGL_NO_ERROR();
  
  
  // ### Rendering ###
  
  RenderPointCloud();
  
  
  // Take screenshot?
  unique_lock<mutex> screenshot_lock(screenshot_mutex_);
  if (!screenshot_path_.empty()) {
    Image<Vec3u8> image(width_, height_, width_ * sizeof(Vec3u8), 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    CHECK_OPENGL_NO_ERROR();
    
    image.FlipY();
    image.Write(screenshot_path_);
    
    screenshot_path_ = "";
    screenshot_lock.unlock();
    screenshot_condition_.notify_all();
  } else {
    screenshot_lock.unlock();
  }
  
  
  unique_lock<mutex> callbacks_lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->Render();
  }
};

void RenderDisplay::RenderPointCloud() {
  unique_lock<mutex> cloud_lock(visualization_cloud_mutex_);

  if (new_visualization_cloud_) {
    have_visualization_cloud_ = true;
    
    visualization_cloud_.TransferToGPU(*new_visualization_cloud_, GL_DYNAMIC_DRAW);
    CHECK_OPENGL_NO_ERROR();
    
    new_visualization_cloud_.reset();
  }
  cloud_lock.unlock();
  
  // Render the visualization cloud if a cloud is available.
  if (have_visualization_cloud_) {
    splat_program_.UseProgram();
    splat_program_.SetUniformMatrix4f(
        splat_u_model_view_projection_matrix_location_,
        model_view_projection_matrix_);
    splat_program_.SetUniform1f(splat_u_point_size_x_location_, splat_half_extent_in_pixels_ / width_);
    splat_program_.SetUniform1f(splat_u_point_size_y_location_, splat_half_extent_in_pixels_ / height_);
    CHECK_OPENGL_NO_ERROR();
    
    if (visualization_cloud_size_ != numeric_limits<usize>::max()) {
      visualization_cloud_.SetAttributes(&splat_program_);
      glDrawArrays(GL_POINTS, 0, visualization_cloud_size_);
    } else {
      visualization_cloud_.Render(&splat_program_);
    }
    CHECK_OPENGL_NO_ERROR();
  }
}

void RenderDisplay::MouseDown(MouseButton button, int x, int y) {
  pressed_mouse_buttons_ |= static_cast<int>(button);
  
  if (button == MouseButton::kLeft ||
      button == MouseButton::kMiddle) {
    dragging_ = true;
    last_drag_x_ = x;
    last_drag_y_ = y;
  }
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->MouseDown(button, x, y);
  }
}

void RenderDisplay::MouseMove(int x, int y) {
  if (dragging_) {
    bool move_camera = false;
    bool rotate_camera = false;
    
    move_camera = (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kMiddle)) ||
                  ((pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft)) &&
                    (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kRight)));
    rotate_camera = pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft);
    
    int x_distance = x - last_drag_x_;
    int y_distance = y - last_drag_y_;

    if (move_camera) {
      const float right_phi = camera_free_orbit_phi_ + 0.5f * M_PI;
      const Eigen::Vector3f right_vector =
          Eigen::Vector3f(cosf(right_phi), sinf(right_phi), 0.f);
      const float up_theta = camera_free_orbit_theta_ + 0.5f * M_PI;
      const float phi = camera_free_orbit_phi_;
      const Eigen::Vector3f up_vector =
          -1 * Eigen::Vector3f(sinf(up_theta) * cosf(phi),
                                sinf(up_theta) * sinf(phi), cosf(up_theta));
      
      // Camera move speed in units per pixel for 1 unit orbit radius.
      constexpr float kCameraMoveSpeed = 0.001f;
      unique_lock<mutex> lock(camera_mutex_);
      camera_free_orbit_offset_ -= x_distance * kCameraMoveSpeed *
                                    camera_free_orbit_radius_ * right_vector;
      camera_free_orbit_offset_ += y_distance * kCameraMoveSpeed *
                                    camera_free_orbit_radius_ * up_vector;
      lock.unlock();
      
      window_->RenderFrame();
    } else if (rotate_camera) {
      unique_lock<mutex> lock(camera_mutex_);
      camera_free_orbit_theta_ -= y_distance * 0.01f;
      camera_free_orbit_phi_ -= x_distance * 0.01f;

      camera_free_orbit_theta_ = fmin(camera_free_orbit_theta_, 3.14f);
      camera_free_orbit_theta_ = fmax(camera_free_orbit_theta_, 0.01f);
      lock.unlock();
      
      window_->RenderFrame();
    }
  }
  
  last_drag_x_ = x;
  last_drag_y_ = y;
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->MouseMove(x, y);
  }
}

void RenderDisplay::MouseUp(MouseButton button, int x, int y) {
  pressed_mouse_buttons_ &= ~static_cast<int>(button);
  
  if (button == MouseButton::kLeft) {
    dragging_ = false;
  }
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->MouseUp(button, x, y);
  }
}

void RenderDisplay::WheelRotated(float degrees, Modifier modifiers) {
  double num_steps = -1 * (degrees / 15.0);
  
  if (static_cast<int>(modifiers) & static_cast<int>(RenderWindowCallbacks::Modifier::kCtrl)) {
    // Change point size.
    splat_half_extent_in_pixels_ -= 0.5f * num_steps;
  } else {
    // Zoom camera.
    double scale_factor = powf(powf(2.0, 1.0 / 5.0), num_steps);
    camera_free_orbit_radius_ *= scale_factor;
  }
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->WheelRotated(degrees, modifiers);
  }
  
  window_->RenderFrame();
}

void RenderDisplay::KeyPressed(char key, Modifier modifiers) {
  if (key == 'c') {
    // Copy camera pose (as text).
    unique_lock<mutex> lock(camera_mutex_);
    
    QClipboard* clipboard = QApplication::clipboard();
    clipboard->setText(
        QString::number(camera_free_orbit_offset_.x()) + " " +
        QString::number(camera_free_orbit_offset_.y()) + " " +
        QString::number(camera_free_orbit_offset_.z()) + " " +
        QString::number(camera_free_orbit_radius_) + " " +
        QString::number(camera_free_orbit_theta_) + " " +
        QString::number(camera_free_orbit_phi_));
  } else if (key == 'v') {
    // Paste copied camera pose.
    QClipboard* clipboard = QApplication::clipboard();
    QString text = clipboard->text();
    QStringList list = text.split(' ');
    if (list.size() != 6) {
      LOG(ERROR) << "Cannot parse clipboard content as camera pose!";
    } else {
      unique_lock<mutex> lock(camera_mutex_);
      
      camera_free_orbit_offset_.x() = list[0].toFloat();
      camera_free_orbit_offset_.y() = list[1].toFloat();
      camera_free_orbit_offset_.z() = list[2].toFloat();
      camera_free_orbit_radius_ = list[3].toFloat();
      camera_free_orbit_theta_ = list[4].toFloat();
      camera_free_orbit_phi_ = list[5].toFloat();
      
      lock.unlock();
      window_->RenderFrame();
    }
  }
  
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->KeyPressed(key, modifiers);
  }
}

void RenderDisplay::KeyReleased(char key, Modifier modifiers) {
  unique_lock<mutex> lock(user_callbacks_mutex_);
  if (user_callbacks_) {
    user_callbacks_->KeyReleased(key, modifiers);
  }
}

void RenderDisplay::SetUpDirection(const Vec3f& direction) {
  unique_lock<mutex> lock(camera_mutex_);
  up_direction_rotation_ = Quaternionf::FromTwoVectors(direction, Vec3f(0, 0, 1)).toRotationMatrix();
}

void RenderDisplay::RenderFrame() {
  window_->RenderFrame();
}

void RenderDisplay::SaveScreenshot(const char* filepath) {
  // Use render_lock to make sure a new frame is rendered for the screenshot.
  // This way any previous calls to update the camera pose, for example, should
  // take effect.
  unique_lock<mutex> render_lock(render_mutex_);
  unique_lock<mutex> lock(screenshot_mutex_);
  screenshot_path_ = filepath;
  lock.unlock();
  render_lock.unlock();
  
  window_->RenderFrame();
  
  unique_lock<mutex> lock2(screenshot_mutex_);
  while (!screenshot_path_.empty()) {
    screenshot_condition_.wait(lock2);
  }
  lock2.unlock();
}

void RenderDisplay::SetCamera() {
  float camera_parameters[4];
  camera_parameters[0] = height_;  // fx
  camera_parameters[1] = height_;  // fy
  camera_parameters[2] = 0.5 * width_ - 0.5f;  // cx
  camera_parameters[3] = 0.5 * height_ - 0.5f;  // cy
  render_camera_ = PinholeCamera4f(width_, height_, camera_parameters);
}

void RenderDisplay::SetViewpoint() {
  Vec3f look_at = camera_free_orbit_offset_;
  float r = camera_free_orbit_radius_;
  float t = camera_free_orbit_theta_;
  float p = camera_free_orbit_phi_;
  Vec3f look_from =
      look_at + Vec3f(r * sinf(t) * cosf(p), r * sinf(t) * sinf(p),
                                r * cosf(t));
  
  Vec3f forward = (look_at - look_from).normalized();
  Vec3f up_temp = Vec3f(0, 0, 1);
  Vec3f right = forward.cross(up_temp).normalized();
  Vec3f up = right.cross(forward);
  
  Mat3f world_R_camera;
  world_R_camera.col(0) = right;
  world_R_camera.col(1) = -up;  // Y will be mirrored by the projection matrix to remove the discrepancy between OpenGL's and our coordinate system.
  world_R_camera.col(2) = forward;
  
  SE3f world_T_camera(world_R_camera, look_from);
  camera_T_world_ = world_T_camera.inverse();
  
  SE3f up_direction_rotation_transformation =
      SE3f(up_direction_rotation_, Vec3f::Zero());
  camera_T_world_ = camera_T_world_ * up_direction_rotation_transformation;
}

void RenderDisplay::ComputeProjectionMatrix() {
  CHECK_GT(max_depth_, min_depth_);
  CHECK_GT(min_depth_, 0);

  const float fx = render_camera_.parameters()[0];
  const float fy = render_camera_.parameters()[1];
  const float cx = render_camera_.parameters()[2];
  const float cy = render_camera_.parameters()[3];

  // Row-wise projection matrix construction.
  projection_matrix_(0, 0) = (2 * fx) / render_camera_.width();
  projection_matrix_(0, 1) = 0;
  projection_matrix_(0, 2) = 2 * (0.5f + cx) / render_camera_.width() - 1.0f;
  projection_matrix_(0, 3) = 0;
  
  projection_matrix_(1, 0) = 0;
  projection_matrix_(1, 1) = -1 * ((2 * fy) / render_camera_.height());
  projection_matrix_(1, 2) = -1 * (2 * (0.5f + cy) / render_camera_.height() - 1.0f);
  projection_matrix_(1, 3) = 0;
  
  projection_matrix_(2, 0) = 0;
  projection_matrix_(2, 1) = 0;
  projection_matrix_(2, 2) = (max_depth_ + min_depth_) / (max_depth_ - min_depth_);
  projection_matrix_(2, 3) = -(2 * max_depth_ * min_depth_) / (max_depth_ - min_depth_);
  
  projection_matrix_(3, 0) = 0;
  projection_matrix_(3, 1) = 0;
  projection_matrix_(3, 2) = 1;
  projection_matrix_(3, 3) = 0;
}

void RenderDisplay::SetupViewport() {
  glViewport(0, 0, render_camera_.width(), render_camera_.height());
}

void RenderDisplay::CreateSplatProgram() {
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out vec3 var1_color;\n"
      "void main() {\n"
      "  var1_color = in_color;\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "#extension GL_EXT_geometry_shader : enable\n"
      "layout(points) in;\n"
      "layout(triangle_strip, max_vertices = 4) out;\n"
      "\n"
      "uniform float u_point_size_x;\n"
      "uniform float u_point_size_y;\n"
      "\n"
      "in vec3 var1_color[];\n"
      "out vec3 var2_color;\n"
      "\n"
      "void main() {\n"
      "  var2_color = var1_color[0];\n"
      "  vec4 base_pos = vec4(gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w, 1.0);\n"
      "  gl_Position = base_pos + vec4(-u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(-u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  \n"
      "  EndPrimitive();\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kGeometryShader));
  
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "#extension GL_ARB_explicit_attrib_location : enable\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "in lowp vec3 var2_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = var2_color;\n"
      // For highlighting the splats in red:
//      "  out_color = vec3(1.0, 0.0, 0.0);\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(splat_program_.LinkProgram());
  
  splat_program_.UseProgram();
  
  splat_u_model_view_projection_matrix_location_ =
      splat_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  splat_u_point_size_x_location_ =
      splat_program_.GetUniformLocationOrAbort("u_point_size_x");
  splat_u_point_size_y_location_ =
      splat_program_.GetUniformLocationOrAbort("u_point_size_y");
}

void RenderDisplay::CreateConstantColorProgram() {
  CHECK(constant_color_program_.AttachShader(
      "#version 300 es\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "void main() {\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(constant_color_program_.AttachShader(
      "#version 300 es\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "uniform lowp vec3 u_constant_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = u_constant_color;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(constant_color_program_.LinkProgram());
  
  constant_color_program_.UseProgram();
  
  constant_color_u_model_view_projection_matrix_location_ =
      constant_color_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  constant_color_u_constant_color_location_ =
      constant_color_program_.GetUniformLocationOrAbort("u_constant_color");
}

}

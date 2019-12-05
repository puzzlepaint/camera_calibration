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

#include <condition_variable>
#include <mutex>

#include "libvis/camera.h"
#include "libvis/libvis.h"
#include "libvis/opengl.h"
#include "libvis/opengl_context.h"
#include "libvis/point_cloud.h"
#include "libvis/point_cloud_opengl.h"
#include "libvis/render_window.h"

namespace vis {

// Render window implementation which allows for easily displaying simple
// scenes, such as point clouds.
// 
// Create a RenderDisplay as follows (TODO: Make this easier!):
// 
//   shared_ptr<RenderDisplay> render_display = make_shared<RenderDisplay>();
//   shared_ptr<RenderWindow> render_window =
//       RenderWindow::CreateWindow("Window title", width, height, RenderWindow::API::kOpenGL, render_display);
class RenderDisplay : public RenderWindowCallbacks {
 public:
  RenderDisplay();
  
  void SetUserCallbacks(RenderWindowCallbacks* user_callbacks);
  
  // Updates the display to show the given colored point cloud.
  void Update(const std::shared_ptr<Point3fC3u8Cloud>& cloud, const string& name);
  void Update(const std::shared_ptr<Point3fCu8Cloud>& cloud, const string& name);
  
  void SetUpDirection(const Vec3f& direction);
  
  void RenderFrame();
  
  void SaveScreenshot(const char* filepath);
  
  inline const SE3f camera_T_world() const {
    unique_lock<mutex> camera_mutex_lock(camera_mutex_);
    return camera_T_world_;
  }
  
  inline const PinholeCamera4f& render_camera() const {
    unique_lock<mutex> camera_mutex_lock(camera_mutex_);
    return render_camera_;
  }
  
  
  // Implementations of necessary functions. Do not call these from outside.
  virtual void Initialize() override;
  
  virtual void Resize(int width, int height) override;
  
  virtual void Render() override;
  
  virtual void MouseDown(MouseButton button, int x, int y) override;
  virtual void MouseMove(int x, int y) override;
  virtual void MouseUp(MouseButton button, int x, int y) override;
  virtual void WheelRotated(float degrees, Modifier modifiers) override;
  virtual void KeyPressed(char key, Modifier modifiers) override;
  virtual void KeyReleased(char key, Modifier modifiers) override;
  
 private:
  void CreateSplatProgram();
  void CreateConstantColorProgram();
  
  void SetCamera();
  void SetViewpoint();
  void ComputeProjectionMatrix();
  void SetupViewport();
  
  void RenderPointCloud();
  
  std::mutex user_callbacks_mutex_;
  RenderWindowCallbacks* user_callbacks_;  // not owned.
  
  // Settings.
  float splat_half_extent_in_pixels_;
  
  int width_;
  int height_;
  
  // Input handling.
  bool dragging_;
  int last_drag_x_;
  int last_drag_y_;
  int pressed_mouse_buttons_;
  
  // Render camera and pose.
  SE3f camera_T_world_;
  PinholeCamera4f render_camera_;
  
  float min_depth_;
  float max_depth_;
  
  mutable std::mutex camera_mutex_;
  Mat3f up_direction_rotation_;
  Vec3f camera_free_orbit_offset_;
  float camera_free_orbit_radius_;
  float camera_free_orbit_theta_;
  float camera_free_orbit_phi_;
  
  Mat4f camera_matrix_;
  
  Mat4f projection_matrix_;
  Mat4f model_view_projection_matrix_;
  
  std::mutex render_mutex_;
  
  // Screenshot handling.
  string screenshot_path_;
  mutex screenshot_mutex_;
  condition_variable screenshot_condition_;
  
  // Vertex buffer handling.
  std::mutex visualization_cloud_mutex_;
  bool have_visualization_cloud_;
  Point3fC3u8CloudOpenGL visualization_cloud_;
  std::shared_ptr<Point3fC3u8Cloud> new_visualization_cloud_;
  usize visualization_cloud_size_;
  
  // Splat program.
  ShaderProgramOpenGL splat_program_;
  GLint splat_u_model_view_projection_matrix_location_;
  GLint splat_u_point_size_x_location_;
  GLint splat_u_point_size_y_location_;
  
  // Constant color program.
  ShaderProgramOpenGL constant_color_program_;
  GLint constant_color_u_model_view_projection_matrix_location_;
  GLint constant_color_u_constant_color_location_;
};

}

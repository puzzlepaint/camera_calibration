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

#include "libvis/camera.h"
#include "libvis/libvis.h"
#include "libvis/opengl.h"
#include "libvis/shader_program_opengl.h"

namespace vis {

// Represents a camera frustum, drawn as lines between the image corners and
// the projection center. The projection center is at (0, 0, 0), while the
// image corners are at (x, y, 1).
class CameraFrustumOpenGL {
 public:
  // No-op constructor, does not require a current OpenGL context.
  inline CameraFrustumOpenGL()
      : buffer_allocated_(false) {}
  
  // Creates a frustum for the given camera. Requires a current OpenGL context for this thread.
  inline CameraFrustumOpenGL(const Camera& camera)
      : buffer_allocated_(false) {
    Create(camera);
  }
  
  // Destroys the point cloud. Attention: Requires a current OpenGL context for
  // this thread! You may need to explicitly delete this object at a point where
  // such a context still exists.
  ~CameraFrustumOpenGL() {
    glDeleteBuffers(1, &vertex_buffer_);
    glDeleteBuffers(1, &index_buffer_);
  }
  
  // Creates a frustum for the given camera. Requires a current OpenGL context for this thread.
  inline void Create(const Camera& camera) {
    IDENTIFY_CAMERA(camera, _Create(_camera));
  }
  
  // Renders the point cloud. Requires a current OpenGL context for this thread.
  void Render(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before rendering.";
    
    SetAttributes(program);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glDrawElements(GL_LINES, 16, GetGLType<u32>::value,
                   reinterpret_cast<char*>(0) + 0);
  }
  
  // Sets the position attribute for the given program. Requires a current
  // OpenGL context for this thread.
  void SetAttributes(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before setting attributes.";
    
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    
    program->SetPositionAttribute(Vec3f::RowsAtCompileTime, GetGLType<typename Vec3f::Scalar>::value, sizeof(Vec3f), 0);
  }
  
 private:
  template <class CameraT>
  void _Create(const CameraT& camera) {
    // TODO: For different camera types than pinhole, the frustum could model
    //       their viewing cone more accurately than by using the image corners only.
    
    Vec3f vertices[5];
    vertices[0] = Vec3f(0, 0, 0);
    vertices[1] = camera.UnprojectFromPixelCornerConv(Vec2i(0, 0)).template cast<float>();  // Top left.
    vertices[2] = camera.UnprojectFromPixelCornerConv(Vec2i(camera.width(), 0)).template cast<float>();  // Top right.
    vertices[3] = camera.UnprojectFromPixelCornerConv(Vec2i(0, camera.height())).template cast<float>();  // Bottom left.
    vertices[4] = camera.UnprojectFromPixelCornerConv(Vec2i(camera.width(), camera.height())).template cast<float>();  // Bottom right.
    
    if (!buffer_allocated_) {
      glGenBuffers(1, &vertex_buffer_);
    }
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, 5 * sizeof(Vec3f), vertices, GL_STATIC_DRAW);
    
    u32 indices[16];
    // Lines from projection center to image corners
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 0;
    indices[3] = 2;
    indices[4] = 0;
    indices[5] = 3;
    indices[6] = 0;
    indices[7] = 4;
    // Image bounds
    indices[8] = 1;
    indices[9] = 2;
    indices[10] = 2;
    indices[11] = 4;
    indices[12] = 4;
    indices[13] = 3;
    indices[14] = 3;
    indices[15] = 1;
    
    if (!buffer_allocated_) {
      glGenBuffers(1, &index_buffer_);
      buffer_allocated_ = true;
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 16 * sizeof(u32), indices, GL_STATIC_DRAW);
  }
  
  // OpenGL name of the buffer containg the vertex data on the GPU.
  GLuint vertex_buffer_;
  
  // OpenGL name of the buffer containg the index data on the GPU.
  GLuint index_buffer_;
  
  bool buffer_allocated_;
};

}

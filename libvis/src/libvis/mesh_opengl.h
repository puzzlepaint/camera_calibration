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

#include "libvis/libvis.h"
#include "libvis/mesh.h"
#include "libvis/opengl.h"
#include "libvis/point_cloud_opengl.h"
#include "libvis/shader_program_opengl.h"

namespace vis {

// Stores mesh data on the GPU, suitable for rendering.
template <typename PointT, typename IndexT>
class MeshOpenGL {
 public:
  typedef Mesh<PointT, IndexT> MeshT;
  
  // No-op constructor, does not require a current OpenGL context.
  MeshOpenGL()
      : buffer_allocated_(false) {};
  
  // Transfers the data from the given mesh to the GPU. Requires a
  // current OpenGL context for this thread.
  MeshOpenGL(const MeshT& mesh)
      : buffer_allocated_(false) {
    TransferToGPU(mesh);
  };
  
  // Destroys the mesh. Attention: Requires a current OpenGL context for
  // this thread! You may need to explicitly delete this object at a point where
  // such a context still exists.
  ~MeshOpenGL() {
    glDeleteBuffers(1, &index_buffer_);
  }
  
  void TransferToGPU(const MeshT& mesh) {
    TransferToGPU(mesh, GL_STATIC_DRAW);
  }
  
  void TransferToGPU(const MeshT& mesh, GLenum usage) {
    index_count_ = 3 * mesh.triangles().size();
    
    if (!buffer_allocated_) {
      glGenBuffers(1, &index_buffer_);
      CHECK_OPENGL_NO_ERROR();
      buffer_allocated_ = true;
    }
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(typename MeshT::TriangleT) * mesh.triangles().size(),
                 mesh.triangles().data(), usage);
    CHECK_OPENGL_NO_ERROR();
    
    if (mesh.vertices()) {
      point_cloud_opengl_.TransferToGPU(*mesh.vertices());
    }
  }
  
  // Renders the point cloud. Requires a current OpenGL context for this thread.
  void Render(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated (e.g., by calling TransferToGPU()) before rendering.";
    
    if (point_cloud_opengl_.buffer_allocated()) {
      point_cloud_opengl_.SetAttributes(program);
    }
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glDrawElements(GL_TRIANGLES, index_count_, GetGLType<IndexT>::value,
                   reinterpret_cast<char*>(0) + 0);
  }
  
  
  inline PointCloudOpenGL<PointT> point_cloud_mutable() { return &point_cloud_opengl_; }
  inline const PointCloudOpenGL<PointT>& point_cloud() const { return point_cloud_opengl_; }
  
  // Returns the OpenGL buffer name of the index buffer. This is only valid if
  // it has been allocated before with one of the TransferToGPU() functions.
  GLuint index_buffer_name() const { return index_buffer_; }
  usize index_count() const { return index_count_; }
  
 private:
  // Vertex data on the GPU.
  PointCloudOpenGL<PointT> point_cloud_opengl_;
  
  // OpenGL name of the buffer containg the index data on the GPU.
  GLuint index_buffer_;
  
  usize index_count_;
  
  bool buffer_allocated_;
};

typedef MeshOpenGL<Point3f, u32> Mesh3fOpenGL;
typedef MeshOpenGL<Point3fC3u8, u32> Mesh3fC3u8OpenGL;

}

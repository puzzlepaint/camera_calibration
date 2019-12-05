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
#include "libvis/opengl.h"
#include "libvis/point_cloud.h"
#include "libvis/shader_program_opengl.h"

namespace vis {

// Must be a class template to allow for partial specialization.
// This contains the implementation for all types that do not have color.
template <typename PointT, bool has_color>
struct PointCloudOpenGL_SetColorAttribute {
  static void Call(ShaderProgramOpenGL* /*program*/) {}
};

// Partial specialization for all types that have color. This implementation can
// access members which exist only for color types.
template <typename PointT>
struct PointCloudOpenGL_SetColorAttribute<PointT, true> {
  static void Call(ShaderProgramOpenGL* program) {
    using ColorT = typename PointTraits<PointT>::ColorT;
    program->SetColorAttribute(
        ColorT::RowsAtCompileTime,
        GetGLType<typename ColorT::Scalar>::value,
        sizeof(PointT),
        reinterpret_cast<uintptr_t>(&reinterpret_cast<PointT*>(0)->color()));
  }
};


// Stores point cloud data on the GPU with OpenGL, suitable for rendering.
// TODO: Write a corresponding Vulkan class that can be used interchangably and
//       depending on the availability make a typedef PointCloudGPU which points
//       to one of the available types.
template <typename PointT>
class PointCloudOpenGL {
 public:
  using PositionT = typename PointTraits<PointT>::PositionT;
  
  // No-op constructor, does not require a current OpenGL context.
  PointCloudOpenGL()
      : buffer_allocated_(false) {};
  
  // Transfers the data from the given point cloud to the GPU. Requires a
  // current OpenGL context for this thread.
  PointCloudOpenGL(const PointCloud<PointT>& cloud)
      : buffer_allocated_(false) {
    TransferToGPU(cloud);
  };
  
  // Destroys the point cloud. Attention: Requires a current OpenGL context for
  // this thread! You may need to explicitly delete this object at a point in
  // your program's control flow before exiting where such a context still exists.
  ~PointCloudOpenGL() {
    if (buffer_allocated_) {
      glDeleteBuffers(1, &point_buffer_);
    }
  }
  
  // Allocates the point buffer on the GPU (but does not initialize it). If the
  // buffer is already allocated, resizes it.
  void AllocateBuffer(usize size_in_bytes, GLenum usage) {
    if (!buffer_allocated_) {
      glGenBuffers(1, &point_buffer_);
      buffer_allocated_ = true;
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, point_buffer_);
    glBufferData(GL_ARRAY_BUFFER, size_in_bytes, nullptr, usage);
  }
  
  // Updates the point cloud's data on the GPU with the given data on the CPU.
  // Requires a current OpenGL context for this thread.
  void TransferToGPU(usize element_size_in_bytes, usize cloud_size,
                     const float* data, GLenum usage = GL_STATIC_DRAW) {
    cloud_size_ = cloud_size;
    
    if (!buffer_allocated_) {
      glGenBuffers(1, &point_buffer_);
      buffer_allocated_ = true;
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, point_buffer_);
    glBufferData(GL_ARRAY_BUFFER, element_size_in_bytes * cloud_size, data, usage);
  }
  
  // Updates the point cloud's data on the GPU with the given data on the CPU.
  // Requires a current OpenGL context for this thread.
  void TransferToGPU(const PointCloud<PointT>& cloud, GLenum usage = GL_STATIC_DRAW) {
    cloud_size_ = cloud.size();
    
    if (!buffer_allocated_) {
      glGenBuffers(1, &point_buffer_);
      buffer_allocated_ = true;
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, point_buffer_);
    glBufferData(GL_ARRAY_BUFFER, cloud.SizeInBytes(), cloud.data(), usage);
  };
  
  // Transfers a continuous range of points to the GPU.
  // dest_first: Index of the first affected point in the GPU buffer.
  // src_first: Index of the first point read from the CPU buffer.
  // count: Number of transferred points.
  // The GPU buffer must be allocated before (using other functions) and the
  // destination range must fit within this buffer.
  void TransferPartToGPU(usize dest_first, usize src_first, usize count,
                         const PointCloud<PointT>& cloud) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before using TransferPartToGPU().";
    
    glBindBuffer(GL_ARRAY_BUFFER, point_buffer_);
    glBufferSubData(GL_ARRAY_BUFFER, dest_first * sizeof(PointT),
                    count * sizeof(PointT), &cloud[src_first]);
  }
  
  // Renders the point cloud. Requires a current OpenGL context for this thread.
  void Render(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before rendering.";
    
    SetAttributes(program);
    
    glDrawArrays(GL_POINTS, 0, cloud_size_);
  }
  
  // Renders the points as lines (for every i, the points 2*i and 2*i+1 form a
  // line). Requires a current OpenGL context for this thread.
  void RenderAsLines(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before rendering.";
    
    SetAttributes(program);
    
    glDrawArrays(GL_LINES, 0, cloud_size_);
  }
  
  // Renders the points as a line strip (all points are connected sequentially).
  // Requires a current OpenGL context for this thread.
  void RenderAsLineStrip(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before rendering.";
    
    SetAttributes(program);
    
    glDrawArrays(GL_LINE_STRIP, 0, cloud_size_);
  }
  
  // Sets the position and color (if applicable) attributes for the given
  // program. Requires a current OpenGL context for this thread.
  void SetAttributes(ShaderProgramOpenGL* program) {
    CHECK(buffer_allocated_)
        << "The buffer must be allocated before setting attributes.";
    
    glBindBuffer(GL_ARRAY_BUFFER, point_buffer_);
    
    program->SetPositionAttribute(
        PositionT::RowsAtCompileTime,
        GetGLType<typename PositionT::Scalar>::value,
        sizeof(PointT),
        reinterpret_cast<uintptr_t>(&reinterpret_cast<PointT*>(0)->position()));
    
    PointCloudOpenGL_SetColorAttribute<PointT, PointTraits<PointT>::has_color>::Call(program);
  }
  
  // Returns whether the buffer was allocated.
  inline bool buffer_allocated() const { return buffer_allocated_; }
  
  // Returns the OpenGL buffer name of the point buffer. This is only valid if
  // it has been allocated before with AllocateBuffer() or one of the
  // TransferToGPU() functions, which can be checked with buffer_allocated().
  inline GLuint buffer_name() const { return point_buffer_; }
  
  // Returns the number of points in the cloud. Attention, this is only valid
  // if the buffer was allocated, which can be checked with buffer_allocated().
  inline usize cloud_size() const { return cloud_size_; }
  
 private:
  // OpenGL name of the buffer containg the point data on the GPU.
  GLuint point_buffer_;
  
  // Number of points in the cloud.
  usize cloud_size_;
  
  bool buffer_allocated_;
};

typedef PointCloudOpenGL<Point3f> Point3fCloudOpenGL;
typedef PointCloudOpenGL<Point3fC3u8> Point3fC3u8CloudOpenGL;

}

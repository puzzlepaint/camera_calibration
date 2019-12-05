// Copyright 2019 ETH Zürich, Thomas Schöps
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

#include <fstream>
#include <memory>
#include <sstream>

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/point_cloud.h"

namespace vis {

// Represents a triangle in a mesh.
template <typename IndexT>
class Triangle {
 public:
  // Constructor which leaves the members uninitialized.
  inline Triangle() {}
  
  // Constructor which initializes the members.
  inline Triangle(IndexT i0, IndexT i1, IndexT i2)
      : indices_{i0, i1, i2} {}
  
  // Checks whether the triangle equals the triangle defined by the given
  // indices, possibly after rotating the indices (e.g, (1,2,3) may become
  // (2,3,1)), but while keeping the same (clockwise or counter-clockwise)
  // ordering.
  inline bool EqualsRotated(IndexT a, IndexT b, IndexT c) {
    return (indices_[0] == a && indices_[1] == b && indices_[2] == c) ||
           (indices_[0] == b && indices_[1] == c && indices_[2] == a) ||
           (indices_[0] == c && indices_[1] == a && indices_[2] == b);
  }
  
  // Provides read-write access to index i.
  inline IndexT& index(int i) { return indices_[i]; }
  // Provides read access to index i.
  inline IndexT index(int i) const { return indices_[i]; }
  
 private:
  IndexT indices_[3];
};

// Generic mesh type, storing a mesh in CPU memory.
template <typename PointT, typename IndexT>
class Mesh {
 public:
  typedef PointCloud<PointT> PointCloudT;
  typedef Triangle<IndexT> TriangleT;
  
  // Creates an empty mesh.
  Mesh() {}
  
  // Creates a deep copy of the other mesh.
  Mesh(const Mesh& other)
      : vertices_(other.vertices_ ? new PointCloudT(*other.vertices_) : nullptr),
        triangles_(other.triangles_) {}
  
  // Takes the existing point cloud as vertices, and copies the indices.
  Mesh(const shared_ptr<PointCloudT>& vertices, const vector<TriangleT>& triangles)
      : vertices_(vertices), triangles_(triangles) {}
  
  // Checks whether all indices are within the bound given by the number of
  // vertices.
  inline bool CheckIndexValidity() {
    usize vertices_size = vertices_->size();
    for (usize i = 0, size = triangles_.size(); i < size; ++ i) {
      const TriangleT& triangle = triangles_[i];
      if (triangle.index(0) >= vertices_size ||
          triangle.index(1) >= vertices_size ||
          triangle.index(2) >= vertices_size) {
        return false;
      }
    }
    return true;
  }
  
  bool WriteAsOBJ(const char* path) {
    ostringstream stream;
    if (!stream) {
      return false;
    }
    
    if (vertices_) {
      vertices_->WriteAsOBJ(&stream);
    }
    
    for (usize i = 0, size = triangles_.size(); i < size; ++ i) {
      const TriangleT& triangle = triangles_[i];
      stream << "f " << (triangle.index(0) + 1)
             << " " << (triangle.index(1) + 1)
             << " " << (triangle.index(2) + 1)
             << std::endl;
    }

    FILE* file = fopen(path, "wb");
    string str = stream.str();
    fwrite(str.c_str(), 1, str.size(), file);
    fclose(file);
    
    return true;
  }
  
  
  inline const shared_ptr<PointCloudT>& vertices() const { return vertices_; }
  inline shared_ptr<PointCloudT>* vertices_mutable() { return &vertices_; }
  
  inline const vector<TriangleT>& triangles() const { return triangles_; }
  inline vector<TriangleT>* triangles_mutable() { return &triangles_; }
  inline void AddTriangle(const TriangleT& triangle) { triangles_.push_back(triangle); }
  
 private:
  shared_ptr<PointCloudT> vertices_;
  vector<TriangleT> triangles_;
};

typedef Mesh<Point3f, u32> Mesh3f;
typedef Mesh<Point3fC3u8, u32> Mesh3fCu8;

}

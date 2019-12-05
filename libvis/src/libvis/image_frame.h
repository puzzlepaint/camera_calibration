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

#include "libvis/image_cache.h"
#include "libvis/libvis.h"
#include "libvis/sophus.h"

namespace vis {

// Stores an image, its pose and its timestamp. Templating allows for different kinds of poses
// to be used, for example SE3f, SE3d, Sim3f, or even rolling-shutter poses.
// The template parameter T specifies the image pixel type.
template<typename T, typename PoseType>
class ImageFrame : public ImageCache<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // TODO: Only required if PoseType requires alignment
  
  typedef ImageCache<T> Base;
  
  inline ImageFrame()
      : Base(), pose_valid_(false), timestamp_(-1) {}
  
  inline ImageFrame(const string& image_path)
      : Base(image_path), pose_valid_(false), timestamp_(-1) {}
      
  inline ImageFrame(const string& image_path, double timestamp)
      : Base(image_path), pose_valid_(false), timestamp_(timestamp) {}
  
  inline ImageFrame(const string& image_path, double timestamp, const string& timestamp_string)
      : Base(image_path), pose_valid_(false), timestamp_(timestamp), timestamp_string_(timestamp_string) {}
  
//   inline ImageFrame(const string& image_path, const PoseType& pose)
//       : Base(image_path), pose_valid_(true), pose_(pose), timestamp_(-1) {}
//   
//   inline ImageFrame(const string& image_path, const PoseType& pose, double timestamp)
//       : Base(image_path), pose_valid_(true), pose_(pose), timestamp_(timestamp) {}
  
  inline ImageFrame(const shared_ptr<Image<T>>& image)
      : Base(image), pose_valid_(false), timestamp_(-1) {}
  
//   inline ImageFrame(const shared_ptr<Image<T>>& image, const PoseType& pose)
//       : Base(image), pose_valid_(true), pose_(pose), timestamp_(-1) {}
//   
//   inline ImageFrame(const shared_ptr<Image<T>>& image, const PoseType& pose, double timestamp)
//       : Base(image), pose_valid_(true), pose_(pose), timestamp_(timestamp) {}
  
  inline ImageFrame(const string& image_path, const shared_ptr<Image<T>>& image)
      : Base(image_path, image), pose_valid_(false), timestamp_(-1) {}
  
//   inline ImageFrame(const string& image_path, const shared_ptr<Image<T>>& image, const PoseType& pose)
//       : Base(image_path, image), pose_valid_(true), pose_(pose), timestamp_(-1) {}
//   
//   inline ImageFrame(const string& image_path, const shared_ptr<Image<T>>& image, const PoseType& pose, double timestamp)
//       : Base(image_path, image), pose_valid_(true), pose_(pose), timestamp_(timestamp) {}
  
  inline void SetGlobalTFrame(const PoseType& global_T_frame) {
    global_T_frame_ = global_T_frame;
    frame_T_global_ = global_T_frame_.inverse();
    pose_valid_ = true;
  }
  
  inline void SetFrameTGlobal(const PoseType& frame_T_global) {
    frame_T_global_ = frame_T_global;
    global_T_frame_ = frame_T_global_.inverse();
    pose_valid_ = true;
  }
  
  inline void SetPoseInvalid() {
    pose_valid_ = false;
  }
  
  inline void SetTimestamp(double timestamp) {
    timestamp_ = timestamp;
  }
  
  inline bool pose_valid() const { return pose_valid_; }
  
  inline const PoseType& global_T_frame() const { return global_T_frame_; }
  
  inline const PoseType& frame_T_global() const { return frame_T_global_; }
  
  inline double timestamp() const { return timestamp_; }
  
  inline const string& timestamp_string() const { return timestamp_string_; }
  
 private:
  bool pose_valid_;
  PoseType global_T_frame_;
  PoseType frame_T_global_;
  double timestamp_;
  string timestamp_string_;
};

template<typename T, typename PoseType>
using ImageFramePtr = shared_ptr<ImageFrame<T, PoseType>>;
template<typename T, typename PoseType>
using ImageFrameConstPtr = shared_ptr<const ImageFrame<T, PoseType>>;

typedef ImageFrame<u8, SE3f> Image8FrameSE3f;
typedef ImageFramePtr<u8, SE3f> Image8FrameSE3fPtr;
typedef ImageFrameConstPtr<u8, SE3f> Image8FrameSE3fConstPtr;

typedef ImageFrame<u8, Sim3f> Image8FrameSim3f;
typedef ImageFramePtr<u8, Sim3f> Image8FrameSim3fPtr;
typedef ImageFrameConstPtr<u8, Sim3f> Image8FrameSim3fConstPtr;

}

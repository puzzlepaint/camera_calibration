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

#include "libvis/eigen.h"
#include "libvis/libvis.h"

namespace vis {

template <typename T>
class Image;

enum class ImageFormat {
  kPNG = 0,
  kPBM,  // netpbm portable bitmap file format
  kPGM,  // netpbm portable graymap file format
  kPPM,  // netpbm portable pixmap file format
  kPNM,  // netpbm portable anymap file format
  kOther
};

ImageFormat TryToDetermineImageFormat(const std::string& filename);

// Base class for image I/O classes. Note: All functions must be re-entrant and
// return true on success, respectively false if an error occurred.
class ImageIO {
 public:
  // Better values must have a higher value.
  enum class ImageFormatSupport {
    kComplete = 2,
    kIncomplete = 1,
    kNone = 0
  };
  
  virtual ~ImageIO() {}
  
  virtual ImageFormatSupport GetSupportForFormat(ImageFormat /*format*/) const {
    return ImageFormatSupport::kNone;
  }
  
  virtual bool Read(const std::string& /*image_file_name*/, Image<u8>* /*image*/) const {
    return false;
  }
  virtual bool Read(const std::string& /*image_file_name*/, Image<u16>* /*image*/) const {
    return false;
  }
  virtual bool Read(const std::string& /*image_file_name*/, Image<Vec3u8>* /*image*/) const {
    return false;
  }
  virtual bool Read(const std::string& /*image_file_name*/, Image<Vec4u8>* /*image*/) const {
    return false;
  }
  
  virtual bool Write(const std::string& /*image_file_name*/, const Image<u8>& /*image*/) const {
    return false;
  }
  virtual bool Write(const std::string& /*image_file_name*/, const Image<u16>& /*image*/) const {
    return false;
  }
  virtual bool Write(const std::string& /*image_file_name*/, const Image<Vec3u8>& /*image*/) const {
    return false;
  }
  virtual bool Write(const std::string& /*image_file_name*/, const Image<Vec4u8>& /*image*/) const {
    return false;
  }
};

// Image I/O class registry. All classes providing image I/O register themselves
// here, enabling to find them. The available classes depend on the available
// dependencies.
class ImageIORegistry {
 public:
  // Registers an image IO class.
  inline void Register(const shared_ptr<ImageIO>& image_io) {
    image_ios_.push_back(image_io);
    // LOG(INFO) << "Have " << image_ios_.size() << " handlers.";
  }
  
  inline ImageIO* Get(const std::string& filename) const {
    if (image_ios_.empty()) {
      return nullptr;
    }
    
    // Try to determine the image format.
    ImageFormat format = TryToDetermineImageFormat(filename);
    
    // Prioritize complete implementations for the format.
    ImageIO* best_implemenation = nullptr;
    ImageIO::ImageFormatSupport best_support = ImageIO::ImageFormatSupport::kNone;
    for (const shared_ptr<ImageIO>& image_io : image_ios_) {
      ImageIO::ImageFormatSupport support = image_io->GetSupportForFormat(format);
      if (static_cast<int>(support) > static_cast<int>(best_support)) {
        best_support = support;
        best_implemenation = image_io.get();
      }
    }
    return best_implemenation;
  }
  
  // Accessor to the global instance of this class.
  static ImageIORegistry* Instance() {
    static ImageIORegistry instance;
    return &instance;
  }

 private:
  vector<shared_ptr<ImageIO>> image_ios_;
};

}

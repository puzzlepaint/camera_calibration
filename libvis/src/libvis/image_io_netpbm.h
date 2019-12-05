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

#include "libvis/image_io.h"

namespace vis {

// Image IO for netpbm formats (pbm, pgm, ppm, pnm).
class ImageIONetPBM : public ImageIO {
 public:
  virtual ImageFormatSupport GetSupportForFormat(ImageFormat format) const override {
    if (format == ImageFormat::kPBM || format == ImageFormat::kPGM || format == ImageFormat::kPPM || format == ImageFormat::kPNM) {
      return ImageFormatSupport::kComplete;
    } else {
      return ImageFormatSupport::kNone;
    }
  }
  
  virtual bool Read(const std::string& image_file_name, Image<u8>* image) const override;
  virtual bool Read(const std::string& image_file_name, Image<u16>* image) const override;
  virtual bool Read(const std::string& image_file_name, Image<Vec3u8>* image) const override;
  virtual bool Read(const std::string& image_file_name, Image<Vec4u8>* image) const override;
  
  virtual bool Write(const std::string& image_file_name, const Image<u8>& image) const override;
  virtual bool Write(const std::string& image_file_name, const Image<u16>& image) const override;
  virtual bool Write(const std::string& image_file_name, const Image<Vec3u8>& image) const override;
  virtual bool Write(const std::string& image_file_name, const Image<Vec4u8>& image) const override;
};

}

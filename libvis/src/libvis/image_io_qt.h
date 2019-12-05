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
#ifdef LIBVIS_HAVE_QT

#include <QImageReader>
#include <QImageWriter>

#include "libvis/image_io.h"

namespace vis {

// Image IO using Qt.
class ImageIOQt : public ImageIO {
 public:
  virtual ImageFormatSupport GetSupportForFormat(ImageFormat format) const override {
    // Determine the format string(s) to check.
    QList<QByteArray> format_strings;
    bool is_incomplete = true;  // Cannot read or write 16 bit images with any format, so always return incomplete support. TODO: Implement more fine-grained read/write support info.
    
    switch (format) {
    case ImageFormat::kPNG:  format_strings.push_back("PNG");  break;
    case ImageFormat::kPBM:  format_strings.push_back("PBM");  break;
    case ImageFormat::kPGM:  format_strings.push_back("PGM");  break;
    case ImageFormat::kPPM:  format_strings.push_back("PPM");  break;
    case ImageFormat::kPNM:
      format_strings.push_back("PBM");
      format_strings.push_back("PGM");
      format_strings.push_back("PPM");
      break;
    case ImageFormat::kOther:
      // Guess that the format is supported (TODO: should not be done like this, check for support exhaustively).
      return is_incomplete ? ImageFormatSupport::kIncomplete : ImageFormatSupport::kComplete;
    }
    
    // Check for the format string(s).
    QList<QByteArray> read_formats = QImageReader::supportedImageFormats();
    QList<QByteArray> write_formats = QImageWriter::supportedImageFormats();
    
    for (const QByteArray& qformat : format_strings) {
      bool have_read_support = false;
      bool have_write_support = false;
      
      for (const QByteArray& read_format : read_formats) {
        if (read_format == qformat) {
          have_read_support = true;
          break;
        }
      }
      
      for (const QByteArray& write_format : write_formats) {
        if (write_format == qformat) {
          have_write_support = true;
          break;
        }
      }
      
      if (!have_read_support && !have_write_support) {
        return ImageFormatSupport::kNone;
      } else if (!have_read_support || !have_write_support) {
        is_incomplete = true;
      }
    }
    
    return is_incomplete ? ImageFormatSupport::kIncomplete : ImageFormatSupport::kComplete;
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

#endif

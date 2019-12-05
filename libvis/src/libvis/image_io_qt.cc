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


#ifdef LIBVIS_HAVE_QT

#include "libvis/image_io_qt.h"

#include "libvis/logging.h"
#include <QImageReader>
#include <QImageWriter>
#include <QString>

#include "libvis/image.h"

namespace vis {

bool ImageIOQt::Read(const std::string& image_file_name,
                     Image<u8>* image) const {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
  QImageReader reader(QString::fromStdString(image_file_name));
  QImage qimage = reader.read();
  if (qimage.isNull()) {
    return false;
  }
  
  if (qimage.format() != QImage::Format_Grayscale8) {
    qimage = qimage.convertToFormat(QImage::Format_Grayscale8);
  }
  
  image->SetSize(qimage.width(), qimage.height());
  for (u32 y = 0; y < image->height(); ++ y) {
    memcpy(image->row(y), qimage.scanLine(y), image->width() * sizeof(u8));
  }
  return true;
#else
  (void) image_file_name;
  (void) image;
  LOG(ERROR) << "u8 in the Qt image IO is only supported from Qt 5.5 on.";
  return false;
#endif
}

bool ImageIOQt::Read(const std::string& /*image_file_name*/,
                     Image<u16>* /*image*/) const {
  LOG(ERROR) << "u16 is not supported by the Qt image IO.";
  return false;
}

bool ImageIOQt::Read(const std::string& image_file_name,
                     Image<Vec3u8>* image) const {
  QImageReader reader(QString::fromStdString(image_file_name));
  QImage qimage = reader.read();
  if (qimage.isNull()) {
    return false;
  }
  
  if (qimage.format() != QImage::Format_RGB888) {
    qimage = qimage.convertToFormat(QImage::Format_RGB888);
  }
  
  image->SetSize(qimage.width(), qimage.height());
  for (u32 y = 0; y < image->height(); ++ y) {
    memcpy(static_cast<void*>(image->row(y)), qimage.scanLine(y), image->width() * sizeof(Vec3u8));
  }
  return true;
}

bool ImageIOQt::Read(const std::string& image_file_name,
                     Image<Vec4u8>* image) const {
  QImageReader reader(QString::fromStdString(image_file_name));
  QImage qimage = reader.read();
  if (qimage.isNull()) {
    return false;
  }
  
  if (qimage.format() != QImage::Format_RGBA8888) {
    qimage = qimage.convertToFormat(QImage::Format_RGBA8888);
  }
  
  image->SetSize(qimage.width(), qimage.height());
  for (u32 y = 0; y < image->height(); ++ y) {
    memcpy(static_cast<void*>(image->row(y)), qimage.scanLine(y), image->width() * sizeof(Vec4u8));
  }
  return true;
}

bool ImageIOQt::Write(const std::string& image_file_name,
                      const Image<u8>& image) const {
  QImage qImage = image.WrapInQImage();
  QImageWriter writer(QString::fromStdString(image_file_name));
  ImageFormat format = TryToDetermineImageFormat(image_file_name);
  if (format == ImageFormat::kPNG) {
    // This will effectively enable compression (without affecting the quality, which is always lossless).
    // See: https://bugreports.qt.io/browse/QTBUG-43618
    writer.setQuality(0);
  } else {
    writer.setQuality(100);  // TODO: This is for maximum-quality JPEGs. Make configurable.
  }
  return writer.write(qImage);
}

bool ImageIOQt::Write(const std::string& /*image_file_name*/,
                      const Image<u16>& /*image*/) const {
  LOG(ERROR) << "u16 is not supported by the Qt image IO.";
  return false;
}

bool ImageIOQt::Write(const std::string& image_file_name,
                      const Image<Vec3u8>& image) const {
  QImage qImage = image.WrapInQImage();
  QImageWriter writer(QString::fromStdString(image_file_name));
  ImageFormat format = TryToDetermineImageFormat(image_file_name);
  if (format == ImageFormat::kPNG) {
    // This will effectively enable compression (without affecting the quality, which is always lossless).
    // See: https://bugreports.qt.io/browse/QTBUG-43618
    writer.setQuality(0);
  } else {
    writer.setQuality(100);  // TODO: This is for maximum-quality JPEGs. Make configurable.
  }
  return writer.write(qImage);
}

bool ImageIOQt::Write(const std::string& image_file_name,
                      const Image<Vec4u8>& image) const {
  QImage qImage = image.WrapInQImage();
  QImageWriter writer(QString::fromStdString(image_file_name));
  ImageFormat format = TryToDetermineImageFormat(image_file_name);
  if (format == ImageFormat::kPNG) {
    // This will effectively enable compression (without affecting the quality, which is always lossless).
    // See: https://bugreports.qt.io/browse/QTBUG-43618
    writer.setQuality(0);
  } else {
    writer.setQuality(100);  // TODO: This is for maximum-quality JPEGs. Make configurable.
  }
  return writer.write(qImage);
}

}

#endif

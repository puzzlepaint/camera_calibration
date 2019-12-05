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


#include "libvis/image_io_netpbm.h"

#include "libvis/logging.h"

#include "libvis/image.h"

namespace vis {

// TODO: Put this into some util header?
bool IsLittleEndian() {
  short int number = 0x1;
  char* number_ptr = reinterpret_cast<char*>(&number);
  return (number_ptr[0] == 1);
}

template <typename T>
bool ReadNetPBMImage(
    const std::string& image_file_name,
    Image<T>* image) {
  FILE* file = fopen(image_file_name.c_str(), "rb");
  if (!file) {
    LOG(ERROR) << "File cannot be opened.";
    return false;
  }
  
  const int kRowBufferSize = 4096;
  char row[kRowBufferSize];
  
  u32 width = 0;
  u32 height = 0;
  u32 maximum_value = 0;
  bool is_binary_format = false;
  
  // Parse the text part.
  int parse_state = 0;  // 0: parse format header, 1: parse width, 2: parse height, 3: parse maximum value, 4: parse content.
  int cursor = 0;
  row[0] = 0;
  while (true) {
    // Check for new lines.
    if (row[cursor] == 0 || row[cursor] == '\r' || row[cursor] == '\n' || row[cursor] == '#') {
      // Read next line, skip over comment lines.
      if (std::fgets(row, kRowBufferSize, file) == nullptr) {
        LOG(ERROR) << "Cannot parse file content.";
        fclose(file);
        return false;
      }
      cursor = 0;
      continue;  // Try to parse the next line.
    }
    
    // Check for whitespace.
    if (row[cursor] == ' ' || row[cursor] == '\t') {
      // Skip over whitespace.
      ++ cursor;
      continue;  // Try to parse the next character.
    }
    
    // Parse the next word.
    if (parse_state == 0) {
      // Parse the file format header (P1 to P6).
      if (row[cursor] != 'P' || row[cursor + 1] < '1' || row[cursor + 1] > '6') {
        LOG(ERROR) << "Format seems incorrect.";
        fclose(file);
        return false;
      }
      is_binary_format = (row[cursor + 1] >= '4');
      cursor += 2;
      parse_state = 1;
    } else if (parse_state == 1) {
      // Parse width.
      width = atoi(row + cursor);
      while (row[cursor] >= '0' && row[cursor] <= '9') {
        ++ cursor;
      }
      parse_state = 2;
    } else if (parse_state == 2) {
      // Parse height.
      height = atoi(row + cursor);
      while (row[cursor] >= '0' && row[cursor] <= '9') {
        ++ cursor;
      }
      parse_state = 3;
      
      // Allocate space for the image content.
      image->SetSize(width, height);
    } else if (parse_state == 3) {
      // Parse maximum value.
      maximum_value = atoi(row + cursor);
      while (row[cursor] >= '0' && row[cursor] <= '9') {
        ++ cursor;
      }
      if (is_binary_format) {
        break;
      } else {
        parse_state = 4;
      }
    } else if (parse_state == 4) {
      // Parse text content.
      fclose(file);
      return false;  // TODO
    }
  }
  
  if (is_binary_format) {
    // Read the binary file content.
    // TODO: Very incomplete implementation.
    if (maximum_value == numeric_limits<T>::max()) {
      // TODO: Assumes that the image values are tightly packed.
      if (image->stride() != image->width() * sizeof(T)) {
        LOG(ERROR) << "This implementation only supports images with tightly packed values.";
        fclose(file);
        return false;
      }
      
      if (fread(image->data(), sizeof(T), width * height, file) != width * height) {
        LOG(ERROR) << "Cannot read image content.";
        fclose(file);
        return false;
      }
      
      // Convert to correct endianness if necessary.
      if (IsLittleEndian()) {
        for (u32 y = 0; y < height; ++ y) {
          T* ptr = image->row(y);
          const T* end = ptr + width;
          while (ptr < end) {
            *ptr = (reinterpret_cast<u8*>(ptr)[1] << 0) | (reinterpret_cast<u8*>(ptr)[0] << 8);
            ++ ptr;
          }
        }
      }
    } else {
      LOG(ERROR) << "Unsupported file format type.";
      fclose(file);
      return false;
    }
  }
  
  fclose(file);
  return true;
}

bool ImageIONetPBM::Read(
    const std::string& image_file_name,
    Image<u8>* image) const {
  return ReadNetPBMImage(image_file_name, image);
}

bool ImageIONetPBM::Read(
    const std::string& image_file_name,
    Image<u16>* image) const {
  return ReadNetPBMImage(image_file_name, image);
}

bool ImageIONetPBM::Read(
    const std::string& /*image_file_name*/,
    Image<Vec3u8>* /*image*/) const {
  LOG(FATAL) << "Function not implemented.";
  return false;  // TODO
//   return ReadImpl(image_file_name, image);
}

bool ImageIONetPBM::Read(
    const std::string& /*image_file_name*/,
    Image<Vec4u8>* /*image*/) const {
  LOG(ERROR) << "Vec4 is not supported by the NetPBM image IO.";
  return false;
}

bool ImageIONetPBM::Write(
    const std::string& /*image_file_name*/,
    const Image<u8>& /*image*/) const {
  LOG(FATAL) << "Function not implemented.";
  return false;  // TODO
//   return WriteImpl(image_file_name, image);
}

bool ImageIONetPBM::Write(
    const std::string& /*image_file_name*/,
    const Image<u16>& /*image*/) const {
  LOG(FATAL) << "Function not implemented.";
  return false;  // TODO
//   return WriteImpl(image_file_name, image);
}

bool ImageIONetPBM::Write(
    const std::string& /*image_file_name*/,
    const Image<Vec3u8>& /*image*/) const {
  LOG(FATAL) << "Function not implemented.";
  return false;  // TODO
//   return WriteImpl(image_file_name, image);
}

bool ImageIONetPBM::Write(
    const std::string& /*image_file_name*/,
    const Image<Vec4u8>& /*image*/) const {
  LOG(ERROR) << "Vec4 is not supported by the NetPBM image IO.";
  return false;
}

}

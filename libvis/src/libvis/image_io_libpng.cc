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


#include "libvis/image_io_libpng.h"

#include "libvis/logging.h"
#include <libpng/png.h> // TODO: Using the "libpng/" prefix is required on my Ubuntu 14.04 system to get a recent version of libpng. Remove this once not required anymore, since it might prevent finding the header at all!

#include "libvis/image.h"

namespace vis {

bool ImageIOLibPng::Read(const std::string& image_file_name,
                         Image<u8>* image) const {
  return ReadImpl(image_file_name, image);
}

bool ImageIOLibPng::Read(const std::string& image_file_name,
                         Image<u16>* image) const {
  return ReadImpl(image_file_name, image);
}

bool ImageIOLibPng::Read(const std::string& image_file_name,
                     Image<Vec3u8>* image) const {
  return ReadImpl(image_file_name, image);
}

bool ImageIOLibPng::Read(const std::string& image_file_name,
                     Image<Vec4u8>* image) const {
  return ReadImpl(image_file_name, image);
}

bool ImageIOLibPng::Write(const std::string& image_file_name,
                          const Image<u8>& image) const {
  return WriteImpl(image_file_name, image);
}

bool ImageIOLibPng::Write(const std::string& image_file_name,
                          const Image<u16>& image) const {
  return WriteImpl(image_file_name, image);
}

bool ImageIOLibPng::Write(const std::string& image_file_name,
                      const Image<Vec3u8>& image) const {
  return WriteImpl(image_file_name, image);
}

bool ImageIOLibPng::Write(const std::string& image_file_name,
                      const Image<Vec4u8>& image) const {
  return WriteImpl(image_file_name, image);
}

template<typename T>
bool ImageIOLibPng::ReadImpl(
    const std::string& image_file_name,
    Image<T>* image) const {
  const int output_bit_depth = 8 * image->bytes_per_pixel() / image->channel_count();
  const int output_channels = image->channel_count();
  
  // Open the file.
  FILE* file = fopen(image_file_name.c_str(), "rb");
  if (!file) {
    LOG(ERROR) << "Cannot open file: " << image_file_name;
    return false;
  }
  
  // Check the first bytes of the file against the PNG header signature to
  // verify that the file is a PNG file. 8 bytes is the maximum according to the
  // libpng documentation.
  constexpr int kBytesToCheck = 8;
  u8 header[8];
  if (fread(header, 1, kBytesToCheck, file) != kBytesToCheck) {
    LOG(ERROR) << "Cannot read first " << kBytesToCheck
               << " bytes for header validation of: " << image_file_name;
    return false;
  }
  bool is_png = !png_sig_cmp(header, 0, kBytesToCheck);
  if (!is_png) {
    LOG(ERROR) << "The file does not appear to be a PNG file: "
               << image_file_name;
    return false;
  }
  
  // Create PNG read and info structs.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    LOG(ERROR) << "png_create_read_struct() failed.";
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    LOG(ERROR) << "png_create_info_struct() failed.";
    return false;
  }
  
  // Set the error handler.
  if (setjmp(png_jmpbuf(png_ptr))) {
    LOG(ERROR) << "libpng's error handler was triggered.";
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(file);
    return false;
  }
  
  // Initialize I/O.
  png_init_io(png_ptr, file);
  png_set_sig_bytes(png_ptr, kBytesToCheck);
  
  // NOTE: libpng has a limit for the image dimensions of 1 million pixels by
  // default, while the theoretical maximum is 2^31-1. In case this ever needs
  // to be changed, use the following call here:
  //   png_set_user_limits(png_ptr, width_max, height_max);
  
  // NOTE: Quote from the libpng documentation:
  // Libpng imposes a limit of 8 Megabytes (8,000,000 bytes) on the amount of
  // memory that a compressed chunk other than IDAT can occupy, when
  // decompressed. You can change this limit with:
  //  png_set_chunk_malloc_max(png_ptr, user_chunk_malloc_max);
  
  // NOTE: gamma value coule be set here.
  // NOTE: alpha mode could be set here.
  // NOTE: According to the libpng documentation, the following is the default:
  //   png_set_alpha_mode(pp, PNG_ALPHA_PNG, PNG_DEFAULT_sRGB);
  // With it, the alpha channel is not pre-multiplied into the color components.
  
  // Read info.
  png_read_info(png_ptr, info_ptr);
  u32 width = png_get_image_width(png_ptr, info_ptr);
  u32 height = png_get_image_height(png_ptr, info_ptr);
  u8 bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  u8 color_type = png_get_color_type(png_ptr, info_ptr);
  u8 channel_count = png_get_channels(png_ptr, info_ptr);
  
  // NOTE: Valid values for bit_depth and color_type are as follows:
  // bit_depth: Holds the bit depth of one of the image channels:
  //            1, 2, 4, 8, 16
  // color_type: Describes which color/alpha channels are present:
  //             PNG_COLOR_TYPE_GRAY
  //               (bit depths 1, 2, 4, 8, 16)
  //             PNG_COLOR_TYPE_GRAY_ALPHA
  //               (bit depths 8, 16)
  //             PNG_COLOR_TYPE_PALETTE
  //               (bit depths 1, 2, 4, 8)
  //             PNG_COLOR_TYPE_RGB
  //               (bit_depths 8, 16)
  //             PNG_COLOR_TYPE_RGB_ALPHA
  //               (bit_depths 8, 16)
  //             
  //             PNG_COLOR_MASK_PALETTE
  //             PNG_COLOR_MASK_COLOR
  //             PNG_COLOR_MASK_ALPHA
  
  // Let libpng convert paletted data to RGB.
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png_ptr);
  }

  // Let libpng convert images with a transparent color to use an alpha channel.
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png_ptr);
  }

  // Let libpng expand grayscale images with less than 8 bits per pixel.
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }
  
  // Let libpng expand the color channels to 16 bit if requested, or scale them
  // down to 8 bit.
  if (output_bit_depth == 16 && bit_depth < 16) {
    png_set_expand_16(png_ptr);
  } else if (output_bit_depth == 8 && bit_depth == 16) {
    png_set_scale_16(png_ptr);
  }
  
  // Let libpng add or remove an alpha channel if necessary.
  if (output_channels == channel_count - 1 &&
      (color_type & PNG_COLOR_MASK_ALPHA)) {
    png_set_strip_alpha(png_ptr);
  } else if (output_channels == channel_count + 1 &&
             !(color_type & PNG_COLOR_MASK_ALPHA)) {
    png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
  } else if (output_channels == 3 && channel_count == 1) {
    png_set_gray_to_rgb(png_ptr);
  } else if (output_channels == 4 && channel_count == 1) {
    png_set_gray_to_rgb(png_ptr);
    png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
  } else if (output_channels == 1 && channel_count == 3) {
    // LOG(WARNING) << "ImageIOLibPng: implicitly converting an RGB image to grayscale on read";
    
    // error_action == 1: No warning if the image is not actually grayscale.
    // Negative weights result in default weights being used.
    png_set_rgb_to_gray(png_ptr, /*error_action*/ 1, -1, -1);
  } else if (output_channels == 1 && channel_count == 4) {
    // LOG(WARNING) << "ImageIOLibPng: implicitly converting an RGBA image to grayscale on read";
    
    png_set_strip_alpha(png_ptr);
    // error_action == 1: No warning if the image is not actually grayscale.
    // Negative weights result in default weights being used.
    png_set_rgb_to_gray(png_ptr, /*error_action*/ 1, -1, -1);
  } else if (output_channels == channel_count) {
    // Ok.
  } else {
    LOG(ERROR) << "Channel count (" << channel_count << ") does not match image buffer channel count (" << output_channels << ").";
    return false;
  }
  
  // Let libpng convert 16 bit values to little-endian.
  if (output_bit_depth == 16) {
    png_set_swap(png_ptr);
  }
  
  // Update the info struct with the transforms set above.
  png_read_update_info(png_ptr, info_ptr);
  
  // Read the image.
  image->SetSize(width, height);
  png_bytep* row_pointers = new png_bytep[height];
  for (u32 y = 0; y < height; ++ y) {
    row_pointers[y] = reinterpret_cast<png_bytep>(image->row(y));
  }
  png_read_image(png_ptr, row_pointers);
  delete[] row_pointers;
  
  // Clean up.
  // NOTE: png_read_end seems to be unnecessary as long as we don't intend to
  // read anything after the PNG data.
  // png_read_end(png_ptr, nullptr);
  
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  fclose(file);
  return true;
}

template<typename T>
bool ImageIOLibPng::WriteImpl(
    const std::string& image_file_name,
    const Image<T>& image) const {
  const int output_bit_depth = 8 * image.bytes_per_pixel() / image.channel_count();
  const int output_channels = image.channel_count();
  
  // Open the file.
  FILE* file = fopen(image_file_name.c_str(), "wb");
  if (!file) {
    LOG(ERROR) << "Cannot open file for writing: " << image_file_name;
    return false;
  }
  
  // Create PNG write and info structs.
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    LOG(ERROR) << "png_create_write_struct() failed.";
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    LOG(ERROR) << "png_create_info_struct() failed.";
    return false;
  }
  
  // Set the error handler.
  if (setjmp(png_jmpbuf(png_ptr))) {
    LOG(ERROR) << "libpng's error handler was triggered.";
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(file);
    return false;
  }
  
  // Initialize I/O.
  png_init_io(png_ptr, file);
  
  // Write info.
  int color_type;
  if (output_channels == 1) {
    color_type = PNG_COLOR_TYPE_GRAY;
  } else if (output_channels == 2) {
    color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
  } else if (output_channels == 3) {
    color_type = PNG_COLOR_TYPE_RGB;
  } else if (output_channels == 4) {
    color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  } else {
    LOG(ERROR) << "Invalid output channel count.";
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(file);
    return false;
  }
  png_set_IHDR(
    png_ptr,
    info_ptr,
    image.width(),
    image.height(),
    output_bit_depth,
    color_type,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png_ptr, info_ptr);
  
  if (output_bit_depth > 8) {
    png_set_swap(png_ptr);
  }
  
  // Write image.
  u32 height = image.height();
  png_const_bytep* row_pointers = new png_const_bytep[height];
  for (u32 y = 0; y < height; ++ y) {
    row_pointers[y] = reinterpret_cast<png_const_bytep>(image.row(y));
  }
  png_write_image(png_ptr, const_cast<png_bytep*>(row_pointers));
  delete[] row_pointers;
  
  png_write_end(png_ptr, nullptr);
  
  // Clean up.
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(file);
  return true;
}

}

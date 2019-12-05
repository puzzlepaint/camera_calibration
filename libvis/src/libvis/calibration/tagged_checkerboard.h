// Copyright 2018, 2019 ETH Zürich, Thomas Schöps
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

#include <vector>

#include "libvis/eigen.h"
#include "libvis/image.h"
#include "libvis/libvis.h"

namespace vis {

// Detects checkerboards which can be uniquely identified from tags embedded
// within them. The idea is that putting a checkerboard around a tag allows to
// use little space for the unique identification, while much space is available
// for the checkerboard corners (which provide well-localizable corners for
// accurate calibration).
// 
// For example, as an alternative, a grid of AprilTags would be even easier
// uniquely localizable, but provide less accurately localizable corners.
// 
// TODO: Which type of tag do we use, QR codes, AprilTags, something custom?
class TaggedCheckerboard {
 public:
  // Attempts to detect inner checkerboard corners of tagged checkerboards in
  // the given image (which must have a scalar type for pixels; color images
  // could be converted to grayscale to use them). Each checkerboard plane must
  // have exactly one tag embedded within it which uniquely identifies it.
  // TODO: which size should the tag have, how should it be positioned within
  //       the grid? Somehow the closest checkerboard corners next to the tag
  //       corners should be easily detectable.
  // 
  // For each detected tag, the tags output vector has an entry with the tag's
  // binary code (which should identity it uniquely as long as no tag is
  // duplicated in the scene). tag_start_indices has a corresponding entry which
  // specifies the index of the first corner of this tagged checkerboard in the
  // corners output vector. The corners are stored in this vector in row-major
  // order. If a corner was not found, its coordinates will be NaN.
  template <typename T>
  static void Detect(
      const Image<T>& image,
      vector<int>* tags,
      vector<int>* tag_start_indices,
      vector<Vec2f>* corners) {
    // Adaptively threshold the image to get a binary black-and-white image.
    constexpr int kHalfContextSize = 40;
    Image<u8> bw_image;
    AdaptivelyThresholdImageToBinary(image, kHalfContextSize, &bw_image);
    
    // Remove speckles to simplify corner detection
    RemoveSpeckles(&bw_image);
    
    // TODO: Continue here
//     Image<u64> white_integral_image;
  }
  
 private:
  static void RemoveSpeckles(
      Image<u8>* bw_image) {
    // Create an integral image of the black/white image.
    Image<u64> white_integral_image;
    bw_image->ComputeIntegralImage(&white_integral_image);
    
    // Perform speckle removal using the integral image.
    constexpr int kSpeckleRemovalWindowHalfSize = 1;  // TODO: Make parameter
    constexpr float kSpeckleRemovalAmountThreshold = 4.2 / 9.;  // TODO: Make parameter
    for (int y = 0; y < static_cast<int>(bw_image->height()) - 0; ++ y) {
      for (int x = 0; x < static_cast<int>(bw_image->width()) - 0; ++ x) {
        int left = std::max(0, x - kSpeckleRemovalWindowHalfSize);
        int top = std::max(0, y - kSpeckleRemovalWindowHalfSize);
        int right = std::min<int>(x + kSpeckleRemovalWindowHalfSize, bw_image->width() - 1);
        int bottom = std::min<int>(y + kSpeckleRemovalWindowHalfSize, bw_image->height() - 1);
        
        int area = (right - left + 1) * (bottom - top + 1);
        
        u64 top_left;
        if (x - kSpeckleRemovalWindowHalfSize - 1 < 0 ||
            y - kSpeckleRemovalWindowHalfSize - 1 < 0) {
          top_left = 0;
        } else {
          top_left = white_integral_image(x - kSpeckleRemovalWindowHalfSize - 1, y - kSpeckleRemovalWindowHalfSize - 1);
        }
        
        u64 top_right;
        if (y - kSpeckleRemovalWindowHalfSize - 1 < 0) {
          top_right = 0;
        } else {
          top_right = white_integral_image(right, y - kSpeckleRemovalWindowHalfSize - 1);
        }
        
        u64 bottom_left;
        if (x - kSpeckleRemovalWindowHalfSize - 1 < 0) {
          bottom_left = 0;
        } else {
          bottom_left = white_integral_image(x - kSpeckleRemovalWindowHalfSize - 1, bottom);
        }
        
        u64 bottom_right = white_integral_image(right, bottom);
        
        int white = bottom_right - top_right - bottom_left + top_left;
        float white_amount = white / static_cast<float>(area);
        float black_amount = 1 - white_amount;
        
        if (white_amount < kSpeckleRemovalAmountThreshold) {
          // set to black
          (*bw_image)(x, y) = 0;
        } else if (black_amount < kSpeckleRemovalAmountThreshold) {
          // set to white
          (*bw_image)(x, y) = 1;
        } else {
          // keep value
        }
      }
    }
  }
   
  // Using an integral image, the average intensity around each pixel within
  // the window of size [-half_context_size, +half_context_size] is quickly
  // computed and used as threshold to turn this pixel into black or white.
  // This robustly creates a black-and-white image by accounting for the local
  // image brightness.
  template <typename T>
  static void AdaptivelyThresholdImageToBinary(
      const Image<T>& image,
      int half_context_size,
      Image<u8>* bw_image) {
    // Ensure correct size of the output image
    bw_image->SetSize(image.width(), image.height());
    
    // Compute integral image of input
    Image<u64> raw_integral_image;
    image.ComputeIntegralImage(&raw_integral_image);
    
    // Adaptively threshold the input image based on the average intensity
    for (int y = 0; y < static_cast<int>(bw_image->height()) - 0; ++ y) {
      for (int x = 0; x < static_cast<int>(bw_image->width()) - 0; ++ x) {
        int left = std::max(0, x - half_context_size);
        int top = std::max(0, y - half_context_size);
        int right = std::min<int>(x + half_context_size, bw_image->width() - 1);
        int bottom = std::min<int>(y + half_context_size, bw_image->height() - 1);
        
        int area = (right - left + 1) * (bottom - top + 1);
        
        int left_outer = x - half_context_size - 1;
        int top_outer = y - half_context_size - 1;
        
        u64 top_left;
        if (left_outer < 0 ||
            top_outer < 0) {
          top_left = 0;
        } else {
          top_left = raw_integral_image(left_outer, top_outer);
        }
        
        u64 top_right;
        if (top_outer < 0) {
          top_right = 0;
        } else {
          top_right = raw_integral_image(right, top_outer);
        }
        
        u64 bottom_left;
        if (left_outer < 0) {
          bottom_left = 0;
        } else {
          bottom_left = raw_integral_image(left_outer, bottom);
        }
        
        u64 bottom_right = raw_integral_image(right, bottom);
        
        u64 sum = bottom_right - top_right - bottom_left + top_left;
        float average = static_cast<double>(sum) / area;
        
        (*bw_image)(x, y) = (image(x, y) < average) ? 0 : 1;
      }
    }
  }
};

}

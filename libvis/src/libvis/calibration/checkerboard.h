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

// TODO:
// Add the possibility to use information around the border of a checkerboard
// to uniquely identify it (and its orientation, if it is unclear from the board itself?).
// Sketch (with the checkerboard in the middle):
// 
// XXOOXOXOOX
// X        X
// O XOXOXO O
// O OXOXOX X
// X        X
// OXOOXOOXOO
// 
// The whitespace could be only half as wide as a normal checkerboard corner.
// The detector could read the binary code in the border. The checkerboard can
// then only be used if the detected code matches the known code (otherwise,
// the detection conditions are probably bad and the detection should be discarded
// instead of being matched with the closest known code).
// 
// Alternative: Put a QR-code / AprilTag in the middle and the checkerboard around it.
// Might be preferable since the tag then usually remains visible in close-ups.
// Maybe no checkerboard size needs to be given then a-priori, instead the
// grid is simply detected around the tag as far as possible. Could even put
// multiple tags into a single grid. Then it is even acceptable if the grid is
// only detected if a tag within it is seen.
// --> TODO: look at AprilTag detector (speed).

// Detects or tracks checkerboards in images.
// Currently, this class is only suited for pinhole or near-pinhole images.
class Checkerboard {
 public:
  // Attempts to detect the inner corners of a checkerboard of the given size
  // in the given image. rows and cols must specify the number of inner corners,
  // i.e., the number of squares minus one. top_left_square_is_black is used
  // to return the detections with a unique orientation (if possible). The
  // function returns true if the detection succeeded. In this case, the output
  // vector is resized to length rows * cols and its entries specify the corner
  // positions in row-major order. If a corner was not found, its coordinates
  // will be NaN.
  template <typename T>
  static bool Detect(int rows, int cols, bool top_left_square_is_black, const Image<T>& image, vector<Vec2f>* corners) {
    // TODO
  }
  
  // Similar to Detect(), but tries to track the checkerboard given its detection in
  // one (prev_corners) or two (prev_2_corners) previous images. If only one previous
  // detection is available, prev_2_corners can be set to nullptr. The function returns
  // true if the tracking succeeded.
  template <typename T>
  static bool Track(int rows, int cols, bool top_left_square_is_black, const Image<T>& image, vector<Vec2f>* corners,
                    const vector<Vec2f>* prev_corners, const vector<Vec2f>* prev_2_corners) {
    // TODO
  }
};

}

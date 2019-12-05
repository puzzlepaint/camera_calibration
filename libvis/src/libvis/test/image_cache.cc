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


#include "libvis/logging.h"
#include <gtest/gtest.h>

#include "libvis/image.h"
#include "libvis/image_cache.h"

using namespace vis;

// Basic test verifying that an image pyramid with 2 levels results in the
// expected image size.
TEST(ImageCache, ImagePyramid) {
  shared_ptr<Image<u8>> image(new Image<u8>(32, 16));
  image->SetTo(42);
  
  ImageCache<u8> image_cache(image);
  
  // Test computation.
  shared_ptr<Image<u8>> pyramid_image =
      ImagePyramid(&image_cache, 2).GetOrComputeResult();
  EXPECT_EQ(32u / 4, pyramid_image->width());
  EXPECT_EQ(16u / 4, pyramid_image->height());
  
  // Test access to computed result. Given the same computation, we expect to
  // get the same result object here as before (instead of a new one).
  shared_ptr<Image<u8>> pyramid_image_2 =
      ImagePyramid(&image_cache, 2).GetOrComputeResult();
  EXPECT_EQ(pyramid_image.get(), pyramid_image_2.get());
}

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


#include "libvis/image_io.h"

#include <boost/algorithm/string.hpp>

namespace vis {

ImageFormat TryToDetermineImageFormat(const std::string& filename) {
  if (filename.size() >= 4 &&
      boost::iequals(".png", filename.substr(filename.size() - 4))) {
    return ImageFormat::kPNG;
  }
  if (filename.size() >= 4 &&
      boost::iequals(".pbm", filename.substr(filename.size() - 4))) {
    return ImageFormat::kPBM;
  }
  if (filename.size() >= 4 &&
      boost::iequals(".pgm", filename.substr(filename.size() - 4))) {
    return ImageFormat::kPGM;
  }
  if (filename.size() >= 4 &&
      boost::iequals(".ppm", filename.substr(filename.size() - 4))) {
    return ImageFormat::kPPM;
  }
  if (filename.size() >= 4 &&
      boost::iequals(".pnm", filename.substr(filename.size() - 4))) {
    return ImageFormat::kPNM;
  }
  return ImageFormat::kOther;
}

}

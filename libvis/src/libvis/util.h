// Copyright 2019 ETH Zürich, Thomas Schöps
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

#include "libvis/libvis.h"

namespace vis {

/// This is a replacement for the following construct using C++ standard library
/// functionality:
/// 
/// some_vector.erase(std::remove_if(some_vector.begin(), some_vector.end(), condition), some_vector.end());
/// 
/// Using erase_if, an equivalent expression is this:
/// 
/// erase_if(some_vector, condition);
/// 
/// Currently the implementation is specific to vectors, but it could be
/// expanded easily.
template <typename T, typename Cond>
void erase_if(vector<T>& container, Cond condition) {
  const T* end_ptr = container.data() + container.size();
  
  T* write_ptr = container.data();
  while (write_ptr < end_ptr &&
         !condition(*write_ptr)) {
    ++ write_ptr;
  }
  
  const T* read_ptr = write_ptr + 1;
  while (read_ptr < end_ptr) {
    if (!condition(*read_ptr)) {
      *write_ptr = *read_ptr;
      ++ write_ptr;
    }
    ++ read_ptr;
  }
  
  // Should not use resize() instead of erase() since the former requires a
  // default constructor for T to be present while the latter does not.
  container.erase(
      container.begin() + static_cast<int>(write_ptr - container.data()),
      container.end());
}

}

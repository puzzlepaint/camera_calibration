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

#include <arpa/inet.h>
#include <stdio.h>

#include <libvis/libvis.h>

namespace vis {

inline void write_one(const u8* data, FILE* file) {
  fwrite(data, sizeof(u8), 1, file);
}

inline void write_one(const i8* data, FILE* file) {
  fwrite(data, sizeof(i8), 1, file);
}

inline void write_one(const u16* data, FILE* file) {
  u16 temp = htons(*data);
  fwrite(&temp, sizeof(u16), 1, file);
}

inline void write_one(const i16* data, FILE* file) {
  i16 temp = htons(*data);
  fwrite(&temp, sizeof(i16), 1, file);
}

inline void write_one(const u32* data, FILE* file) {
  u32 temp = htonl(*data);
  fwrite(&temp, sizeof(u32), 1, file);
}

inline void write_one(const i32* data, FILE* file) {
  i32 temp = htonl(*data);
  fwrite(&temp, sizeof(i32), 1, file);
}

inline void write_one(const float* data, FILE* file) {
  // TODO: Does this require a potential endian swap?
  fwrite(data, sizeof(float), 1, file);
}


// TODO: No proper error handling in these functions when reaching the end of the file
inline void read_one(u8* data, FILE* file) {
  if (fread(data, sizeof(u8), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
}

inline void read_one(i8* data, FILE* file) {
  if (fread(data, sizeof(i8), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
}

inline void read_one(u16* data, FILE* file) {
  u16 temp;
  if (fread(&temp, sizeof(u16), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
  *data = ntohs(temp);
}

inline void read_one(i16* data, FILE* file) {
  i16 temp;
  if (fread(&temp, sizeof(i16), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
  *data = ntohs(temp);
}

inline void read_one(u32* data, FILE* file) {
  u32 temp;
  if (fread(&temp, sizeof(u32), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
  *data = ntohl(temp);
}

inline void read_one(i32* data, FILE* file) {
  i32 temp;
  if (fread(&temp, sizeof(i32), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
  *data = ntohl(temp);
}

inline void read_one(float* data, FILE* file) {
  // TODO: Does this require a potential endian swap?
  if (fread(data, sizeof(float), 1, file) != 1) {
    LOG(ERROR) << "read_one() failed to read the element";
  }
}

}

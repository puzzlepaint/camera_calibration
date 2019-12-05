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


// This file should be included in every other file in libvis.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace vis {

// Import std namespace into vis namespace.
using namespace std;

// Type definitions which are more concise and thus easier to read and write (no
// underscore). int is used as-is.
typedef size_t usize;
typedef int64_t i64;
typedef uint64_t u64;
typedef int32_t i32;
typedef uint32_t u32;
typedef int16_t i16;
typedef uint16_t u16;
typedef int8_t i8;
typedef uint8_t u8;

// Helper object which allows to do some global initialization and destruction.
// Must be allocated at the start of the main() function (and destructed at its
// end). This should be done with the LIBVIS_APPLICATION() macro.
class LibvisApplication {
 public:
  LibvisApplication(int argc, char** argv);
  
  void SetDefaultQSurfaceFormat();
  
  int WrapQtEventLoopAround(int (*func)(int, char**), int argc, char** argv);
};


// Each application using libvis must use either LIBVIS_MAIN or LIBVIS_QT_MAIN
// (or do analogous things manually). This design arose from Qt's requirement
// to have GUI code run in the first (main) thread.
// 
// LIBVIS_MAIN() runs a Qt event loop in the main thread, while starting the
// actual program in a second thread.
// Pro: This ensures that GUI created by libvis has an event loop running for it
//      in the background.
// Con: If the application wants to create Qt widgets itself, it has to ensure
//      that this happens in the libvis-created thread, for example using
//      RunInQtThread[Blocking]().
// 
// LIBVIS_QT_MAIN() assumes that the application will run a Qt event loop itself
// in the main thread.
// Pro: The application can use Qt normally.
// Con: libvis' GUI functionality will only work as long as the main thread runs
//      a Qt event loop. This might not always be easy to achieve, for example,
//      if a secondary thread wants to visualize something while the main thread
//      is busy with something else.
// TODO: For this variant, we could provide a function that runs a temporary
//       event loop locally, analogous to OpenCV's waitKey(). But this will only
//       work if called from the main thread.

// This macro must replace the name of the main() function of every program
// using libvis; for Qt applications however which run a Qt event loop
// themselves, LIBVIS_QT_MAIN() must be used instead.
// 
// Usage:
// int LIBVIS_MAIN(int argc, char** argv) {
//   // Application code here
// }
#define LIBVIS_MAIN(argc_def, argv_def) \
  __libvis_main(int argc, char** argv); \
  \
  int main(int argc, char** argv) { \
    vis::LibvisApplication app(argc, argv); \
    app.SetDefaultQSurfaceFormat(); \
    return WrapQtEventLoopAround(&__libvis_main, argc, argv); \
  } \
  \
  int __libvis_main(argc_def, argv_def)

// Variant of the LIBVIS_MAIN() macro for Qt applications that run a Qt
// event loop themselves. Please note that libvis' GUI functionality might not
// work as long as no Qt event loop is running.
// 
// Usage:
// int LIBVIS_QT_MAIN(int argc, char** argv) {
//   // Application code here
// }
#define LIBVIS_QT_MAIN(argc_def, argv_def) \
  __libvis_main(int argc, char** argv); \
  \
  int main(int argc, char** argv) { \
    vis::LibvisApplication app(argc, argv); \
    app.SetDefaultQSurfaceFormat(); \
    return __libvis_main(argc, argv); \
  } \
  \
  int __libvis_main(argc_def, argv_def)

}

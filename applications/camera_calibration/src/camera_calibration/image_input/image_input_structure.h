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

#include <atomic>
#include <thread>

#include <libvis/libvis.h>
#ifdef HAVE_STRUCTURE
#include <ST/CaptureSession.h>
#endif

#include "camera_calibration/image_input/image_input.h"

class QWidget;

namespace vis {

struct AvailableInputStructure;
struct SessionDelegate;

/// Image input using the Structure SDK.
class ImageInputStructure : public ImageInput {
 friend struct SessionDelegate;
 public:
  ImageInputStructure(ImageConsumer* consumer, const vector<shared_ptr<AvailableInput>>& inputs, QWidget* settings_widget);
  ~ImageInputStructure();
  
  static void ListAvailableInputs(vector<shared_ptr<AvailableInput>>* list);
  static QWidget* CreateSettingsWidgets();
  
 private:
#ifdef HAVE_STRUCTURE
  shared_ptr<SessionDelegate> delegate_;
  shared_ptr<ST::CaptureSession> session_;
#endif
  
  bool both_infrared_enabled;
  
  vector<const AvailableInputStructure*> inputs_structure;
  ImageConsumer* m_consumer;
};

}

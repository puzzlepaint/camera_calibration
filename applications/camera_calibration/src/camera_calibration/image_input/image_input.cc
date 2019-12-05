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

#include "camera_calibration/image_input/image_input.h"

#include "camera_calibration/image_input/image_input_realsense.h"
#include "camera_calibration/image_input/image_input_structure.h"
#include "camera_calibration/image_input/image_input_v4l2.h"
#include "camera_calibration/ui/settings_window.h"

namespace vis {

ImageInput* ImageInput::CreateForInputs(ImageConsumer* consumer, vector<shared_ptr<AvailableInput>>* inputs, SettingsWindow* settings_window) {
  AvailableInput::Type prev_type = inputs->at(0)->type;
  
  for (usize i = 1; i < inputs->size(); ++ i) {
    if (inputs->at(i)->type != prev_type) {
      LOG(ERROR) << "Different input types chosen. This is not supported at the moment.";
      return nullptr;
    }
  }
  
  // ImageInputV4L2 only supports one camera
  if (prev_type == AvailableInput::Type::V4L2 && inputs->size() > 1) {
    LOG(ERROR) << "Only one concurrent V4L2 input is supported.";
    return nullptr;
  }
  
  if (prev_type == AvailableInput::Type::V4L2) {
    return new ImageInputV4L2(consumer, inputs->at(0).get());
  } else if (prev_type == AvailableInput::Type::RealSense) {
    return new ImageInputRealSense(consumer, *inputs, settings_window->GetSettingsWidget(AvailableInput::Type::RealSense));
  } else if (prev_type == AvailableInput::Type::Structure) {
    return new ImageInputStructure(consumer, *inputs, settings_window->GetSettingsWidget(AvailableInput::Type::Structure));
  } else {
    LOG(ERROR) << "Unsupported input type";
    return nullptr;
  }
}

}

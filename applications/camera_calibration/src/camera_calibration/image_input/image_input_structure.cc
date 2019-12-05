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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <sys/types.h>
#include <sys/time.h>
#include <libvis/logging.h>

#ifdef HAVE_STRUCTURE
#include <ST/Utilities.h>
#endif

#include "camera_calibration/image_input/image_input_structure.h"

namespace vis {

struct AvailableInputStructure : public AvailableInput {
#ifdef HAVE_STRUCTURE
  enum class Camera {
    InfraredLeft = 0,
    InfraredRight,
    Visible
  };
  Camera camera;
  char serial[256];
#endif
};


#ifdef HAVE_STRUCTURE
bool isMono(const ST::ColorFrame& visFrame) {
  return visFrame.width() * visFrame.height() == visFrame.rgbSize();
}

struct SessionDelegate : ST::CaptureSessionDelegate {
  SessionDelegate(ImageInputStructure* image_input)
      : image_input(image_input) {}
  
  ~SessionDelegate() {}
  
  void captureSessionEventDidOccur(ST::CaptureSession* session, ST::CaptureSessionEventId event) override {
    LOG(INFO) << "Received capture session event " << static_cast<int>(event) << " (" << ST::CaptureSessionSample::toString(event) << ")";
    switch (event) {
      case ST::CaptureSessionEventId::Booting: break;
      case ST::CaptureSessionEventId::Ready:
        LOG(INFO) << "Starting streams...";
        LOG(INFO) << "Sensor Serial Number is " << session->sensorSerialNumber();
        session->startStreaming();
        break;
      case ST::CaptureSessionEventId::Disconnected:
      case ST::CaptureSessionEventId::Error:
        LOG(ERROR) << "Capture session error";
        break;
      default:
        LOG(INFO) << "  Event unhandled";
    }
  }
  
  void captureSessionDidOutputSample(ST::CaptureSession* /*session*/, const ST::CaptureSessionSample& sample) override {
    vector<Image<Vec3u8>> images(image_input->inputs_structure.size());
    for (usize i = 0; i < images.size(); ++ i) {
      const auto& input = image_input->inputs_structure[i];
      
      if (input->camera == AvailableInputStructure::Camera::Visible) {
        if (!sample.visibleFrame.isValid()) {
          LOG(INFO) << "Incomplete frame received.";
          return;
        }
        
        images[i].SetSize(
            sample.visibleFrame.width(),
            sample.visibleFrame.height());
        if (!isMono(sample.visibleFrame)) {
          images[i].SetTo(reinterpret_cast<const Vec3u8*>(sample.visibleFrame.rgbData()));
        } else {
          for (int i = 0; i < images[i].pixel_count(); ++ i) {
            images[i].data()[i] = Vec3u8::Constant(sample.visibleFrame.yData()[i]);
          }
        }
      } else if (input->camera == AvailableInputStructure::Camera::InfraredLeft) {
        if (!sample.infraredFrame.isValid()) {
          LOG(INFO) << "Incomplete frame received.";
          return;
        }
        
        images[i].SetSize(
            sample.infraredFrame.width() / (image_input->both_infrared_enabled ? 2 : 1),
            sample.infraredFrame.height());
        // The right part of the combined image corresponds to the left IR image.
        int first_x = image_input->both_infrared_enabled ? (sample.infraredFrame.width() / 2) : 0;
        for (int y = 0; y < images[i].height(); ++ y) {
          for (int x = 0; x < images[i].width(); ++ x) {
            // TODO: Arbitrary scaling (here and below). It would be better to be able to use the raw 16-bit images.
            images[i](x, y) = Vec3u8::Constant(std::min(255.f, 1 / 3.f * sample.infraredFrame.data()[first_x + x + y * sample.infraredFrame.width()]));
          }
        }
      } else if (input->camera == AvailableInputStructure::Camera::InfraredRight) {
        if (!sample.infraredFrame.isValid()) {
          LOG(INFO) << "Incomplete frame received.";
          return;
        }
        
        images[i].SetSize(
            sample.infraredFrame.width() / (image_input->both_infrared_enabled ? 2 : 1),
            sample.infraredFrame.height());
        // The left part of the combined image corresponds to the right IR image.
        for (int y = 0; y < images[i].height(); ++ y) {
          for (int x = 0; x < images[i].width(); ++ x) {
            images[i](x, y) = Vec3u8::Constant(std::min(255.f, 1 / 3.f * sample.infraredFrame.data()[x + y * sample.infraredFrame.width()]));
          }
        }
      }
    }
    
    image_input->m_consumer->NewImageset(images);
  }
  
  ImageInputStructure* image_input;
};
#endif


ImageInputStructure::ImageInputStructure(ImageConsumer* consumer, const vector<shared_ptr<AvailableInput>>& inputs, QWidget* settings_widget)
    : m_consumer(consumer) {
#ifndef HAVE_STRUCTURE
  (void) consumer;
  (void) inputs;
  (void) settings_widget;
#else
  ST::CaptureSessionSettings settings;
  settings.source = ST::CaptureSessionSourceId::StructureCore;
  settings.frameSyncEnabled = true;
  settings.structureCore.depthEnabled = false;
  settings.structureCore.dynamicCalibrationMode = ST::StructureCoreDynamicCalibrationMode::Off;
  settings.structureCore.infraredAutoExposureEnabled = true;
  settings.structureCore.accelerometerEnabled = false;
  settings.structureCore.gyroscopeEnabled = false;
  settings.structureCore.infraredResolution = ST::StructureCoreInfraredResolution::_1280x960;
  // TODO: Is there any way to disable the IR projector?
  
  settings.structureCore.visibleEnabled = false;
  settings.structureCore.infraredEnabled = false;
  bool left_infrared = false;
  bool right_infrared = false;
  
  inputs_structure.resize(inputs.size());
  for (usize i = 0; i < inputs.size(); ++ i) {
    inputs_structure[i] = dynamic_cast<const AvailableInputStructure*>(inputs[i].get());
    CHECK(inputs_structure[i]);
    if (i > 0 && strcmp(inputs_structure[i]->serial, inputs_structure[0]->serial) != 0) {
      LOG(FATAL) << "Streaming from different devices at the same time is not implemented";
    }
    
    if (inputs_structure[i]->camera == AvailableInputStructure::Camera::Visible) {
      settings.structureCore.visibleEnabled = true;
      
      QLineEdit* rgb_exposure_edit = qobject_cast<QLineEdit*>(settings_widget->layout()->itemAt(1)->widget());
      bool ok;
      double rgb_exposure = rgb_exposure_edit->text().toDouble(&ok);
      if (ok) {
        settings.structureCore.initialVisibleExposure = rgb_exposure;
      } else {
        LOG(WARNING) << "Could not parse the text entered for the exposure time as a double: " << rgb_exposure_edit->text().toStdString();
      }
    } else if (inputs_structure[i]->camera == AvailableInputStructure::Camera::InfraredLeft) {
      settings.structureCore.infraredEnabled = true;
      left_infrared = true;
    } else if (inputs_structure[i]->camera == AvailableInputStructure::Camera::InfraredRight) {
      settings.structureCore.infraredEnabled = true;
      right_infrared = true;
    } else {
      LOG(FATAL) << "Unhandled camera type: " << static_cast<int>(inputs_structure[i]->camera);
    }
  }
  
  both_infrared_enabled = false;
  if (left_infrared && right_infrared) {
    settings.structureCore.infraredMode = ST::StructureCoreInfraredMode::BothCameras;
    both_infrared_enabled = true;
  } else if (left_infrared) {
    settings.structureCore.infraredMode = ST::StructureCoreInfraredMode::LeftCameraOnly;
  } else if (right_infrared) {
    settings.structureCore.infraredMode = ST::StructureCoreInfraredMode::RightCameraOnly;
  }
  settings.structureCore.sensorSerial = inputs_structure[0]->serial;
  
  delegate_.reset(new SessionDelegate(this));
  session_.reset(new ST::CaptureSession());
  session_->setDelegate(delegate_.get());
  if (!session_->startMonitoring(settings)) {
    LOG(FATAL) << "Failed to initialize capture session";
  }
#endif
}

ImageInputStructure::~ImageInputStructure() {
#ifdef HAVE_STRUCTURE
  if (session_) {
    session_->stopStreaming();
    session_.reset();  // make sure that the session is deleted before the delegate
  }
#endif
}

void ImageInputStructure::ListAvailableInputs(vector<shared_ptr<AvailableInput>>* list) {
#ifndef HAVE_STRUCTURE
  (void) list;
#else
  const ST::ConnectedSensorInfo* sensors[10];  // 10 is the maximum number of sensors that may be returned by enumerateConnectedSensors()
  int num_sensors;
  if (!ST::enumerateConnectedSensors(sensors, &num_sensors)) {
    LOG(ERROR) << "Failed to enumerate connected Structure Core sensors.";
    return;
  }
  
  for (int i = 0; i < num_sensors; ++ i) {
    shared_ptr<AvailableInputStructure> infrared_left_input(new AvailableInputStructure());
    infrared_left_input->type = AvailableInput::Type::Structure;
    infrared_left_input->camera = AvailableInputStructure::Camera::InfraredLeft;
    memcpy(infrared_left_input->serial, sensors[i]->serial, 256);
    infrared_left_input->display_text = QString("Structure SDK: Left infrared (%1)").arg(sensors[i]->serial);
    list->push_back(infrared_left_input);
    
    shared_ptr<AvailableInputStructure> infrared_right_input(new AvailableInputStructure());
    infrared_right_input->type = AvailableInput::Type::Structure;
    infrared_right_input->camera = AvailableInputStructure::Camera::InfraredRight;
    memcpy(infrared_right_input->serial, sensors[i]->serial, 256);
    infrared_right_input->display_text = QString("Structure SDK: Right infrared (%1)").arg(sensors[i]->serial);
    list->push_back(infrared_right_input);
    
    shared_ptr<AvailableInputStructure> visible_input(new AvailableInputStructure());
    visible_input->type = AvailableInput::Type::Structure;
    visible_input->camera = AvailableInputStructure::Camera::Visible;
    memcpy(visible_input->serial, sensors[i]->serial, 256);
    visible_input->display_text = QString("Structure SDK: Visible (%0)").arg(sensors[i]->serial);
    list->push_back(visible_input);
  }
#endif
}

QWidget* ImageInputStructure::CreateSettingsWidgets() {
#ifndef HAVE_STRUCTURE
  return nullptr;
#else
  QWidget* container_widget = new QWidget();
  QHBoxLayout* layout = new QHBoxLayout();
  
  QLabel* rgb_exposure_label = new QLabel(QObject::tr("Exposure time for RGB camera [s]:"));
  layout->addWidget(rgb_exposure_label);
  
  ST::CaptureSessionSettings settings;
  QLineEdit* rgb_exposure_edit = new QLineEdit(QString::number(settings.structureCore.initialVisibleExposure));
  layout->addWidget(rgb_exposure_edit);
  
  container_widget->setLayout(layout);
  return container_widget;
#endif
}

}

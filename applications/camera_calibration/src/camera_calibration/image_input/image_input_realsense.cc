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

#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#endif

#include <QCheckBox>
#include <QWidget>
#include <sys/types.h>
#include <sys/time.h>
#include <libvis/logging.h>

#include "camera_calibration/image_input/image_input_realsense.h"

namespace vis {

struct AvailableInputRealSense : public AvailableInput {
#ifdef HAVE_REALSENSE
  rs2::device device;
  rs2::sensor sensor;
  rs2_stream stream_type;
  int stream_index;
#endif
};


#ifdef HAVE_REALSENSE
static bool IsSupportedFormat(rs2_format format) {
  return format == RS2_FORMAT_Y8 ||
         format == RS2_FORMAT_RGB8;
}
#endif


ImageInputRealSense::ImageInputRealSense(ImageConsumer* consumer, const vector<shared_ptr<AvailableInput>>& inputs, QWidget* settings_widget)
    : m_quit_requested(false),
      m_consumer(consumer) {
  QCheckBox* enable_emitter = qobject_cast<QCheckBox*>(settings_widget);
  m_enable_emitter = enable_emitter->isChecked();
  
  m_input_thread.reset(new thread(bind(&ImageInputRealSense::ThreadMain, this, inputs)));
}

ImageInputRealSense::~ImageInputRealSense() {
  m_quit_requested = true;
  m_input_thread->join();
}

void ImageInputRealSense::ListAvailableInputs(vector<shared_ptr<AvailableInput>>* list) {
#ifndef HAVE_REALSENSE
  (void) list;
#else
  rs2::context ctx;
  rs2::device_list devices = ctx.query_devices();
  for (rs2::device device : devices) {
    // Get device name and serial number
    string device_name = "Unknown Device";
    if (device.supports(RS2_CAMERA_INFO_NAME)) {
      device_name = device.get_info(RS2_CAMERA_INFO_NAME);
    }
    if (device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
      device_name += string(" (#") + device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) + ")";
    }
    
    vector<rs2::sensor> sensors = device.query_sensors();
    for (rs2::sensor sensor : sensors) {
      // NOTE: A sensor in librealsense can consist of multiple actual cameras.
      //       E.g., "Stereo Module" is a sensor consisting of two infrared
      //       cameras for the D435 camera. So we have to list the "unique
      //       streams" instead.
      // Get sensor name
      string sensor_name = "Unknown Sensor";
      if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
        sensor_name = sensor.get_info(RS2_CAMERA_INFO_NAME);
      }
      
      vector<rs2::stream_profile> stream_profiles = sensor.get_stream_profiles();
      vector<pair<rs2_stream, int>> unique_streams;
      for (rs2::stream_profile& profile : stream_profiles) {
        // Check whether this is a video stream with a supported format.
        if (!profile.is<rs2::video_stream_profile>()) {
          continue;
        }
        rs2::video_stream_profile video_stream_profile = profile.as<rs2::video_stream_profile>();
        if (!IsSupportedFormat(video_stream_profile.format())) {
          continue;
        }
        
        // Check whether we have seen this type of stream before.
        bool found = false;
        for (auto& item : unique_streams) {
          if (item.first == profile.stream_type() &&
              item.second == profile.stream_index()) {
            found = true;
            break;
          }
        }
        
        // Found a new unique stream?
        if (!found) {
          unique_streams.push_back(make_pair(profile.stream_type(), profile.stream_index()));
          
          shared_ptr<AvailableInputRealSense> new_input(new AvailableInputRealSense());
          new_input->type = AvailableInput::Type::RealSense;
          ostringstream name;
          name << profile.stream_type() << " #" << profile.stream_index() << " - " << sensor_name << " - " << device_name;
          new_input->display_text = QString("librealsense: ") + QString::fromStdString(name.str());
          new_input->device = device;
          new_input->sensor = sensor;
          new_input->stream_type = profile.stream_type();
          new_input->stream_index = profile.stream_index();
          list->push_back(new_input);
        }
      }
    }
  }
#endif
}

QWidget* ImageInputRealSense::CreateSettingsWidgets() {
  QCheckBox* enable_emitter = new QCheckBox(QObject::tr("Enable infrared emitter"));
  enable_emitter->setChecked(false);
  return enable_emitter;
}

void ImageInputRealSense::ThreadMain(const vector<shared_ptr<AvailableInput>>& inputs) {
#ifndef HAVE_REALSENSE
  (void) inputs;
  LOG(ERROR) << "Attempting to use RealSense input, but the binary was compiled without librealsense.";
#else
  LOG(INFO) << "ImageInputRealSense::ThreadMain() starting ...";
  
  vector<const AvailableInputRealSense*> inputs_realsense(inputs.size());
  for (usize i = 0; i < inputs.size(); ++ i) {
    inputs_realsense[i] = dynamic_cast<const AvailableInputRealSense*>(inputs[i].get());
    CHECK(inputs_realsense[i]);
  }
  
  for (usize i = 0; i < inputs.size(); ++ i) {
    auto& input = inputs_realsense[i];
    LOG(INFO) << "Chosen camera [" << i << "]: " << input->display_text.toStdString();
    LOG(INFO) << "Chosen stream type [" << i << "]: " << input->stream_type << ", index: " << input->stream_index;
  }
  
  rs2::pipeline pipe;
  rs2::config cfg;
  
  cfg.disable_all_streams();
  
  // Activate all requested streams
  for (usize i = 0; i < inputs.size(); ++ i) {
    auto& input = inputs_realsense[i];
    
    if (input->device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
      cfg.enable_device(input->device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    }
    
    // Choose the stream profile with the largest image area (TODO: Allow the user to select the profile).
    std::vector<rs2::stream_profile> stream_profiles = input->sensor.get_stream_profiles();
    int best_image_area = 0;
    int best_fps = 0;
    rs2::stream_profile* best_profile = nullptr;
    for (rs2::stream_profile& profile : stream_profiles) {
      if (profile.stream_type() != input->stream_type ||
          profile.stream_index() != input->stream_index) {
        continue;
      }
      if (!profile.is<rs2::video_stream_profile>()) {
        continue;
      }
      rs2::video_stream_profile video_stream_profile = profile.as<rs2::video_stream_profile>();
      if (!IsSupportedFormat(video_stream_profile.format())) {
        continue;
      }
      
      int image_area = video_stream_profile.width() * video_stream_profile.height();
      if (image_area > best_image_area) {
        best_image_area = image_area;
        best_fps = video_stream_profile.fps();
        best_profile = &profile;
      } else if (image_area == best_image_area && video_stream_profile.fps() > best_fps) {
        best_fps = video_stream_profile.fps();
        best_profile = &profile;
      }
    }
    if (!best_profile) {
      LOG(ERROR) << "Cannot find a viable stream profile for input " << i << ".";
      return;
    }
    
    rs2::video_stream_profile video_stream_profile = best_profile->as<rs2::video_stream_profile>();
    LOG(INFO) << "Chosen stream settings [" << i << "]: " << video_stream_profile.width()
              << " x " << video_stream_profile.height()
              << " @ " << video_stream_profile.fps() << " Hz; format: " << video_stream_profile.format();
    cfg.enable_stream(best_profile->stream_type(),
                      best_profile->stream_index(),
                      video_stream_profile.width(),
                      video_stream_profile.height(),
                      video_stream_profile.format(),
                      video_stream_profile.fps());
  }
  
  rs2::pipeline_profile profile = pipe.start(cfg);
  
  // Set the white balance and exposure to auto
  for (rs2::sensor& sensor : profile.get_device().query_sensors()) {
    if (sensor.get_stream_profiles()[0].stream_type() == RS2_STREAM_COLOR) {
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, true);
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, true);
    }
  }
  
  // Disable the infrared emitter
  rs2::device selected_device = profile.get_device();
  auto depth_sensor = selected_device.first<rs2::depth_sensor>();
  if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
    depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, m_enable_emitter ? 1.f : 0.f);
  }
  if (depth_sensor.supports(RS2_OPTION_LASER_POWER)) {
    if (m_enable_emitter) {
      auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
      depth_sensor.set_option(RS2_OPTION_LASER_POWER, range.max);  // Set max power
    } else {
      depth_sensor.set_option(RS2_OPTION_LASER_POWER, 0.f);
    }
  }
  
  vector<Image<Vec3u8>> images(inputs_realsense.size());
  
  LOG(INFO) << "ImageInputRealSense::ThreadMain() going into main loop ...";
  
  bool have_frame = false;
  while (!m_quit_requested) {
    // Wait for a new frame from the camera
    rs2::frameset frameset = pipe.wait_for_frames();
    
    if (!have_frame) {
      LOG(INFO) << "ImageInputRealSense::ThreadMain() received first frame from camera.";
      have_frame = true;
    }
    
    // To reduce the frame rate, uncomment:
//     static int frame_rate_reducement = 0;
//     ++ frame_rate_reducement;
//     if (frame_rate_reducement % 10 != 0) {
//       continue;
//     }
    
    for (usize input_index = 0; input_index < inputs_realsense.size(); ++ input_index) {
      const AvailableInputRealSense& input = *inputs_realsense[input_index];
      rs2::video_frame frame =
          (input.stream_type == RS2_STREAM_COLOR) ?
          frameset.get_color_frame() :
          frameset.get_infrared_frame(input.stream_index);
      
      Image<Vec3u8>& image = images[input_index];
      if (frame.get_profile().format() == RS2_FORMAT_RGB8) {
        image.SetSize(frame.get_width(), frame.get_height());
        image.SetTo(reinterpret_cast<const Vec3u8*>(frame.get_data()), frame.get_stride_in_bytes());
      } else if (frame.get_profile().format() == RS2_FORMAT_Y8) {
        const u8* data = static_cast<const u8*>(frame.get_data());
        const u8* end = static_cast<const u8*>(frame.get_data()) + (frame.get_width() * frame.get_height());
        
        image.SetSize(frame.get_width(), frame.get_height());
        Vec3u8* ptr = image.data();
        while (data != end) {
          *ptr = Vec3u8::Constant(*data);
          ++ data;
          ++ ptr;
        }
      } else {
        LOG(ERROR) << "Image has unsupported format: " << frame.get_profile().format();
      }
    }
    
    m_consumer->NewImageset(images);
  }
#endif
}

}

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

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include <libvis/logging.h>

#include "camera_calibration/image_input/image_input_v4l2.h"

namespace vis {

#define CLEAR(x) memset(&(x), 0, sizeof(x))

struct buffer {
  void* start;
  size_t length;
};

static bool xioctl(int fh, int request, void* arg) {
  int r;
  
  do {
    r = v4l2_ioctl(fh, request, arg);
  } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));

  if (r == -1) {
    LOG(ERROR) << "error " << errno << ", " << strerror(errno);
    return false;
  }
  return true;
}


struct AvailableInputV4L2 : public AvailableInput {
  int camera_index;
};


ImageInputV4L2::ImageInputV4L2(ImageConsumer* consumer, const AvailableInput* input)
    : m_quit_requested(false),
      m_consumer(consumer) {
  const AvailableInputV4L2& input_v4l2 = *dynamic_cast<const AvailableInputV4L2*>(input);
  
  ostringstream path;
  path << "/dev/video" << input_v4l2.camera_index;
  m_input_thread.reset(new thread(bind(&ImageInputV4L2::ThreadMain, this, path.str())));
}

ImageInputV4L2::~ImageInputV4L2() {
  m_quit_requested = true;
  m_input_thread->join();
}

void ImageInputV4L2::ListAvailableInputs(vector<shared_ptr<AvailableInput>>* list) {
  int camera_index = 0;
  while (true) {
    ostringstream filename;
    filename << "/dev/video" << camera_index;
    
    struct v4l2_capability video_cap;
    
    int fd;
    if ((fd = v4l2_open(filename.str().c_str(), O_RDONLY)) == -1){
      break;
    }
    
    if (ioctl(fd, VIDIOC_QUERYCAP, &video_cap) == -1) {
      LOG(WARNING) << "Failed to get capabilities of " << filename.str();
      ::close(fd);
      continue;
    } else {
      int name_length = 0;
      for (; name_length < 32; ++ name_length) {
        if (video_cap.card[name_length] == 0) {
          break;
        }
      }
      
      shared_ptr<AvailableInputV4L2> new_input(new AvailableInputV4L2());
      new_input->type = AvailableInput::Type::V4L2;
      new_input->camera_index = camera_index;
      new_input->display_text =
          QString("video4linux2: ") +
          QString::fromUtf8(reinterpret_cast<const char*>(video_cap.card), name_length) +
          " (" + QString::fromStdString(filename.str()) + ")";
      list->push_back(new_input);
    }
    
    v4l2_close(fd);
    
    ++ camera_index;
  }
}

QWidget* ImageInputV4L2::CreateSettingsWidgets() {
  return nullptr;  // no settings at the moment
}

void LogError(int error_code) {
  if (error_code == EAGAIN || error_code == EWOULDBLOCK) {
    LOG(ERROR) << "The ioctl can't be handled because the device is in state where it can't perform it. This could happen for example in case where device is sleeping and ioctl is performed to query statistics. It is also returned when the ioctl would need to wait for an event, but the device was opened in non-blocking mode.";
  } else if (error_code == EBADF) {
    LOG(ERROR) << "The file descriptor is not a valid.";
  } else if (error_code == EBUSY) {
    LOG(ERROR) << "The ioctl can’t be handled because the device is busy. This is typically return while device is streaming, and an ioctl tried to change something that would affect the stream, or would require the usage of a hardware resource that was already allocated. The ioctl must not be retried without performing another action to fix the problem first (typically: stop the stream before retrying).";
  } else if (error_code == EFAULT) {
    LOG(ERROR) << "There was a failure while copying data from/to userspace, probably caused by an invalid pointer reference.";
  } else if (error_code == EINVAL) {
    LOG(ERROR) << "One or more of the ioctl parameters are invalid or out of the allowed range. This is a widely used error code. See the individual ioctl requests for specific causes.";
  } else if (error_code == ENODEV) {
    LOG(ERROR) << "Device not found or was removed.";
  } else if (error_code == ENOMEM) {
    LOG(ERROR) << "There’s not enough memory to handle the desired operation.";
  } else if (error_code == ENOTTY) {
    LOG(ERROR) << "The ioctl is not supported by the driver, actually meaning that the required functionality is not available, or the file descriptor is not for a media device.";
  } else if (error_code == ENOSPC) {
    LOG(ERROR) << "On USB devices, the stream ioctl’s can return this error, meaning that this request would overcommit the usb bandwidth reserved for periodic transfers (up to 80% of the USB bandwidth).";
  } else if (error_code == EPERM) {
    LOG(ERROR) << "Permission denied. Can be returned if the device needs write permission, or some special capabilities is needed (e.g. root)";
  } else if (error_code == EIO) {
    LOG(ERROR) << "I/O error. Typically used when there are problems communicating with a hardware device. This could indicate broken or flaky hardware. It’s a ‘Something is wrong, I give up!’ type of error.";
  } else if (error_code == ENXIO) {
    LOG(ERROR) << "No device corresponding to this device special file exists.";
  } else {
    LOG(ERROR) << "The given error code is unknown to LogError().";
  }
}

void ImageInputV4L2::ThreadMain(const string& path) {
  int result;
  
  int fd = v4l2_open(path.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd < 0) {
    LOG(ERROR) << "Cannot open device " << path;
    m_good = false;
    return;
  }
  
  // Enumerate video formats
  u32 best_format = 0;
  string best_format_description;
  bool have_best_format = false;
  
  struct v4l2_fmtdesc fmtdesc;
  CLEAR(fmtdesc);
  fmtdesc.index = 0;
  fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  while ((result = v4l2_ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)) == 0) {
    LOG(1) << "Supported image format: " << fmtdesc.description;
    if (fmtdesc.pixelformat == V4L2_PIX_FMT_GREY ||
        fmtdesc.pixelformat == V4L2_PIX_FMT_RGB24) {
      if (!have_best_format) {
        have_best_format = true;
        best_format = fmtdesc.pixelformat;
        best_format_description = string(reinterpret_cast<char*>(fmtdesc.description));
      } else if (fmtdesc.pixelformat == V4L2_PIX_FMT_RGB24) {
        // Prefer color over grayscale
        best_format = fmtdesc.pixelformat;
        best_format_description = string(reinterpret_cast<char*>(fmtdesc.description));
      }
    }
    ++ fmtdesc.index;
  }
  
  if (!have_best_format) {
    LOG(ERROR) << "No supported image format offered by the camera. Aborting.";
    m_good = false;
    return;
  }
  LOG(INFO) << "Chosen image format: " << best_format_description;
  
  // Choose the highest available resolution.
  // (TODO: The resolution should be selected previously by the user and passed in as a parameter here.)
  int selected_width = 0;
  int selected_height = 0;
  
  struct v4l2_frmsizeenum frmsize;
  CLEAR(frmsize);
  frmsize.index = 0;
  frmsize.pixel_format = best_format;
  while ((result = v4l2_ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize)) == 0) {
    if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
      LOG(1) << "Discrete frame size: " << frmsize.discrete.width << " x " << frmsize.discrete.height;
      if (frmsize.discrete.width * frmsize.discrete.height >
          selected_width * selected_height) {
        selected_width = frmsize.discrete.width;
        selected_height = frmsize.discrete.height;
      }
    } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE ||
               frmsize.type == V4L2_FRMSIZE_TYPE_CONTINUOUS) {
      LOG(1) << "Stepwise or continuous frame size. Max: " << frmsize.stepwise.max_width << " x " << frmsize.stepwise.max_height;
      selected_width = frmsize.stepwise.max_width;
      selected_height = frmsize.stepwise.max_height;
    } else {
      LOG(ERROR) << "Unknown type returned by VIDIOC_ENUM_FRAMESIZES ioctl: " << frmsize.type;
      break;
    }
    
    ++ frmsize.index;
  }
  
  struct v4l2_format fmt;
  CLEAR(fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = selected_width;
  fmt.fmt.pix.height = selected_height;
  fmt.fmt.pix.pixelformat = best_format;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  if (!xioctl(fd, VIDIOC_S_FMT, &fmt)) {
    m_good = false;
    return;
  }
  if (fmt.fmt.pix.pixelformat != best_format) {
    auto get_format_string = [](u32 fourcc_code) {
      string result;
      result.resize(4);
      result[0] = static_cast<char>(fourcc_code & 0x000000ff);
      result[1] = static_cast<char>(fourcc_code & 0x0000ff00);
      result[2] = static_cast<char>(fourcc_code & 0x00ff0000);
      result[3] = static_cast<char>(fourcc_code & 0xff000000);
      return result;
    };
    
    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_GREY || fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_RGB24) {
      LOG(WARNING) << "VIDIOC_S_FMT changed the pixel format to: " << get_format_string(fmt.fmt.pix.pixelformat);
    } else {
      LOG(ERROR) << "V4L2 did not accept the requested format and returned an unsupported format (" << get_format_string(fmt.fmt.pix.pixelformat) << ") instead. Aborting.";
      m_good = false;
      return;
    }
  }
  LOG(INFO) << "Chosen image width: " << fmt.fmt.pix.width;
  LOG(INFO) << "Chosen image height: " << fmt.fmt.pix.height;
  if ((fmt.fmt.pix.width != selected_width) || (fmt.fmt.pix.height != selected_height)) {
    LOG(ERROR) << "Selecting the desired resolution (" << selected_width << " x " << selected_height << ") did not work; returned resolution: " << fmt.fmt.pix.width << " x " << fmt.fmt.pix.height;
  }
  
  Image<Vec3u8> image(fmt.fmt.pix.width, fmt.fmt.pix.height);
  if (image.stride() != image.width() * sizeof(Vec3u8)) {
    LOG(FATAL) << "(stride != length of a row) is not supported.";
  }
  
  struct v4l2_requestbuffers req;
  CLEAR(req);
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  xioctl(fd, VIDIOC_REQBUFS, &req);
  
  struct v4l2_buffer buf;
  unsigned int n_buffers = req.count;
  vector<buffer> buffers;
  buffers.resize(n_buffers);
  for (int buffer_index = 0; buffer_index < req.count; ++buffer_index) {
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buffer_index;
    xioctl(fd, VIDIOC_QUERYBUF, &buf);
    
    buffers[buffer_index].length = buf.length;
    buffers[buffer_index].start = v4l2_mmap(
        NULL, buf.length,
        PROT_READ | PROT_WRITE, MAP_SHARED,
        fd, buf.m.offset);
    
    if (MAP_FAILED == buffers[buffer_index].start) {
      LOG(ERROR) << "mmap failed";
      m_good = false;
      return;
    }
  }
  
  for (int i = 0; i < n_buffers; ++i) {
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    xioctl(fd, VIDIOC_QBUF, &buf);
  }
  
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd, VIDIOC_STREAMON, &type);
  
  fd_set fds;
  struct timeval tv;
  while (!m_quit_requested) {
    int select_result;
    do {
      FD_ZERO(&fds);
      FD_SET(fd, &fds);
      
      /* Timeout. */
      tv.tv_sec = 2;
      tv.tv_usec = 0;
      
      select_result = select(fd + 1, &fds, NULL, NULL, &tv);
      
    } while ((select_result == -1 && (errno = EINTR)));
    if (select_result == -1) {
      LOG(ERROR) << "select failed";
      m_good = false;
      return;
    }
    
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    xioctl(fd, VIDIOC_DQBUF, &buf);
    
    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_GREY) {
      if (buf.bytesused != image.width() * image.height()) {
        LOG(FATAL) << "Received buffer does not have the expected length (buf.bytesused: " << buf.bytesused << ", image.width() * image.height(): " << (image.width() * image.height()) << ").";
      }
      
      const u8* src = static_cast<const u8*>(buffers[buf.index].start);
      Vec3u8* dest = image.data();
      for (int i = 0; i < image.pixel_count(); ++ i) {
        *dest = Vec3u8::Constant(*src);
        ++ src;
        ++ dest;
      }
    } else {  // fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_RGB24
      if (buf.bytesused != image.stride() * image.height()) {
        LOG(FATAL) << "Received buffer does not have the expected length (buf.bytesused: " << buf.bytesused << ", image.stride() * image.height(): " << (image.stride() * image.height()) << ").";
      }
      
      memcpy(static_cast<void*>(image.data()), buffers[buf.index].start, buf.bytesused);
    }
    
    m_consumer->NewImageset({image});
    
    xioctl(fd, VIDIOC_QBUF, &buf);
  }
  
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd, VIDIOC_STREAMOFF, &type);
  for (int i = 0; i < n_buffers; ++i) {
    v4l2_munmap(buffers[i].start, buffers[i].length);
  }
  v4l2_close(fd);
}

}

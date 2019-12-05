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

#include "camera_calibration/tools/tools.h"

#include <libvis/logging.h>

#include "camera_calibration/fitting_report.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/central_generic.h"

namespace vis {

int CompareCalibrations(
    const string& calibration_a,
    const string& calibration_b,
    const string& report_base_path) {
  if (calibration_a.empty() || calibration_b.empty() || report_base_path.empty()) {
    LOG(ERROR) << "For calibration comparison (--compare_calibrations), the input calibrations must be given with --calibration_a and --calibration_b, and the output base path with --report_base_path.";
    return EXIT_FAILURE;
  }
  
  shared_ptr<CameraModel> model_a = LoadCameraModel(calibration_a.c_str());
  if (!model_a) {
    LOG(ERROR) << "Cannot load file: " << calibration_a;
    return EXIT_FAILURE;
  }
  
  shared_ptr<CameraModel> model_b = LoadCameraModel(calibration_b.c_str());
  if (!model_b) {
    LOG(ERROR) << "Cannot load file: " << calibration_b;
    return EXIT_FAILURE;
  }
  
  // TODO: Make this available for all compatible models
  CentralGenericModel* bspline_model_a = dynamic_cast<CentralGenericModel*>(model_a.get());
  CentralGenericModel* bspline_model_b = dynamic_cast<CentralGenericModel*>(model_b.get());
  if (!bspline_model_a || !bspline_model_b) {
    LOG(ERROR) << "Calibration comparison is only implemented for CentralGenericModel at the moment.";
    return EXIT_FAILURE;
  }
  
  CreateFittingErrorReport(
      report_base_path.c_str(),
      *bspline_model_a,
      *bspline_model_b,
      Mat3d::Identity());  // TODO: Add an option to also optimize for this rotation here?
  return EXIT_SUCCESS;
}

}

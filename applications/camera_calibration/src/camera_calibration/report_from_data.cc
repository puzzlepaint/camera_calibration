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

#undef NDEBUG
#include <cassert>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <libvis/camera.h>
#include <libvis/command_line_parser.h>
#include <libvis/eigen.h>
#include <libvis/external_io/colmap_model.h>
#include <libvis/geometry.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/point_cloud.h>
#include <QSharedPointer>
#include <QtWidgets>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/joint_optimization.h"
#include "camera_calibration/calibration.h"
#include "camera_calibration/calibration_report.h"
#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"
#include "camera_calibration/fitting_report.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/image_input/image_input_realsense.h"
#include "camera_calibration/image_input/image_input_v4l2.h"
#include "camera_calibration/models/all_models.h"
#include "camera_calibration/tools/tools.h"
#include "camera_calibration/ui/calibration_window.h"
#include "camera_calibration/ui/live_image_consumer.h"
#include "camera_calibration/ui/main_window.h"
#include "camera_calibration/ui/pattern_display.h"
#include "camera_calibration/ui/settings_window.h"
#include "camera_calibration/util.h"

using namespace vis;


Q_DECLARE_METATYPE(QSharedPointer<Image<Vec3u8>>)

namespace rs = rapidjson;

void runReport(std::string const& input_filename) {
    ifstream ifs(input_filename);
    rapidjson::IStreamWrapper isw(ifs);

    rapidjson::Document d;
    d.ParseStream(isw);

    int const width = d["width"].GetInt();
    int const height = d["height"].GetInt();

    vector<Vec2d> errors;
    vector<Vec2f> features;

    assert(d["errors"].IsArray());
    assert(d["features"].IsArray());

    rs::Value errors_j = d["errors"].GetArray();

    rs::Value features_j = d["features"].GetArray();

    LOG(INFO) << "File " << input_filename << " has width/height " << width << ", " << height << " and " << errors_j.Size() << " errors";

    assert(errors_j.Size() > 0);
    assert(errors_j.Size() == features_j.Size());

    for (size_t ii = 0; ii < errors_j.Size(); ++ii) {
        assert(errors_j[ii].IsArray());
        assert(features_j[ii].IsArray());
        assert(errors_j[ii].Size() == 2);
        assert(features_j[ii].Size() == 2);

        errors.push_back(Vec2d(errors_j[ii][0].GetDouble(), errors_j[ii][1].GetDouble()));
        features.push_back(Vec2f(features_j[ii][0].GetDouble(), features_j[ii][1].GetDouble()));
    }

    LOG(INFO) << "Read all errors and features";

    CreateCalibrationReportForData((input_filename + "_report").c_str(), 0, width, height, errors, features);
    LOG(INFO) << "Finished writing report";
}

int LIBVIS_QT_MAIN(int argc, char** argv) {
  qRegisterMetaType<QSharedPointer<Image<Vec3u8>>>();
  srand(0);

  for (int ii = 1; ii < argc; ++ii) {
    runReport(argv[ii]);
  }

  return EXIT_SUCCESS;
}

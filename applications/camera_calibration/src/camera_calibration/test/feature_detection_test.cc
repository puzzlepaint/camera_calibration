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

#include <boost/filesystem.hpp>
#include <cuda_runtime.h>
#include <libvis/dlt.h>
#include <libvis/eigen.h>
#include <libvis/geometry.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>
#include <libvis/sophus.h>
#include <gtest/gtest.h>
#include <QApplication>
#include <yaml-cpp/yaml.h>

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"
#include "camera_calibration/feature_detection/feature_refinement.h"
#include "camera_calibration/test/main.h"

using namespace vis;

TEST(FeatureDetection, BiasEvaluation) {
  srand(0);
  
  // This is a symlink to the test_data folder installed via CMakeLists.txt
  string test_data_path = (boost::filesystem::path(g_binary_path).parent_path() / "test_data").string();
  string pattern_name = "pattern_resolution_17x24_segments_16_apriltag_0";
  
  // Load the pattern as an image
  string pattern_image_path = (boost::filesystem::path(test_data_path) / (pattern_name + ".png")).string();
  Image<u8> pattern_image(pattern_image_path);
  ASSERT_FALSE(pattern_image.empty()) << "Cannot load the pattern image from: " << pattern_image_path;
  
  Vec2f pattern_corners[4] = {
    Vec2f(0, 0),
    Vec2f(pattern_image.width(), 0),
    Vec2f(0, pattern_image.height()),
    Vec2f(pattern_image.width(), pattern_image.height()),
  };
  
  // Check for the presence of a CUDA device
  int num_cuda_devices;
  cudaGetDeviceCount(&num_cuda_devices);
  
  // Set up the feature detector
  int feature_window_extent = 10;
  string pattern_yaml_path = (boost::filesystem::path(test_data_path) / (pattern_name + ".yaml")).string();
  vector<string> patterns = {pattern_yaml_path};
  FeatureDetectorTaggedPattern detector(patterns, feature_window_extent, FeatureRefinement::Intensities, /*use_cuda*/ num_cuda_devices > 0);
  ASSERT_TRUE(detector.valid()) << "Could not load: " << pattern_yaml_path;
  
  // Load the page properties from the pattern YAML to be able to know where in
  // the image the features are located.
  YAML::Node pattern_node = YAML::LoadFile(pattern_yaml_path);
  if (pattern_node.IsNull()) {
    ASSERT_TRUE(false) << "Cannot read file: " << pattern_yaml_path;
  }
  int squares_x = pattern_node["squares_x"].as<int>();
  int squares_y = pattern_node["squares_y"].as<int>();
  YAML::Node page_node = pattern_node["page"];
  float page_width_mm = page_node["width_mm"].as<float>();
  float page_height_mm = page_node["height_mm"].as<float>();
  float pattern_start_x_mm = page_node["pattern_start_x_mm"].as<float>();
  float pattern_start_y_mm = page_node["pattern_start_y_mm"].as<float>();
  float pattern_end_x_mm = page_node["pattern_end_x_mm"].as<float>();
  float pattern_end_y_mm = page_node["pattern_end_y_mm"].as<float>();
  
  // Loop over test cases
  Image<u8> rendered_image(600, 800);
  
  constexpr int kBiasInImageDownsampling = 20;
  Image<Vec3u8> bias_in_image(rendered_image.width() / kBiasInImageDownsampling,
                              rendered_image.height() / kBiasInImageDownsampling);
  
  Vec2f half_variance_extent = rendered_image.size().cast<float>() / 7;
  
  constexpr int kNumTestCases = 15;
  bool any_test_case_successful = false;
  for (int test = 0; test < kNumTestCases; ++ test) {
    srand(test);
    
    // Determine a random homography to perspectively distort the pattern image with.
    // For the first test, fill the whole image without perspective distortion though
    // as a comparison.
    Vec2f corners[4];
    if (test == 0) {
      corners[0] = Vec2f(0, 0);
      corners[1] = Vec2f(rendered_image.width(), 0);
      corners[2] = Vec2f(0, rendered_image.height());
      corners[3] = Vec2f(rendered_image.width(), rendered_image.height());
    } else {
      for (int i = 0; i < 4; ++ i) {
        corners[i] = half_variance_extent + half_variance_extent.cwiseProduct(Vec2f::Random());
        if (i % 2 == 1) {
          corners[i].x() = rendered_image.width() - corners[i].x();
        }
        if (i / 2 == 1) {
          corners[i].y() = rendered_image.height() - corners[i].y();
        }
      }
    }
    
    Mat3f pattern_tr_rendering = NormalizedDLT(corners, pattern_corners, 4);
    Mat3f rendering_tr_pattern = pattern_tr_rendering.inverse();
    
    float distortion = (test == 0) ? 0 : (0.01f * ((rand() % 10000) / 10000.f));
    
    auto pattern_to_pixel_coord = [&](const Vec2f& pattern_coord) {
      Vec2f pattern_image_coord_mm =
          Vec2f(pattern_start_x_mm,
                pattern_start_y_mm) +
              (pattern_coord + Vec2f::Constant(1)).cwiseQuotient(Vec2f(squares_x, squares_y)).cwiseProduct(
              Vec2f(pattern_end_x_mm - pattern_start_x_mm,
                    pattern_end_y_mm - pattern_start_y_mm));
      Vec2f pattern_image_coord = pattern_image.size().cast<float>().cwiseQuotient(Vec2f(page_width_mm, page_height_mm)).cwiseProduct(pattern_image_coord_mm);
      Vec2f pixel = Vec3f(rendering_tr_pattern * pattern_image_coord.homogeneous()).hnormalized();  // "pixel corner" convention
      // Invert: rendering_coord.x() = rendering_coord.x() + distortion * rendering_coord.x() * rendering_coord.x();
      // rendering_coord.old_x = rendering_coord.new_x + distortion * rendering_coord.new_x * rendering_coord.new_x;
      // distortion * rendering_coord.new_x * rendering_coord.new_x + rendering_coord.new_x - rendering_coord.old_x = 0;
      // Quadratic function with:
      //   a = distortion
      //   b = 1
      //   c = -rendering_coord.old_x
      // rendering_coord.new_x = (-1 +- sqrt(1 + 4 * distortion * rendering_coord.old_x)) / (2 * distortion);
      if (distortion != 0) {
        pixel.x() = (-1 + sqrt(1 + 4 * distortion * pixel.x())) / (2 * distortion);
      }
      return pixel;
    };
    
    // Render and the image.
    // Generate the pattern by representing all black areas of it as
    // (non-overlapping) polygons.
    const PatternData& pattern = detector.GetPatternData(0);
    forward_list<vector<Vec2f>> geometry;
    pattern.ComputePatternGeometry(&geometry);
    
    // Densely subdivide the polygons and transform them to pixel space.
    // The subdivision enables to (approximately) deal with distortions.
    // The output polygons should still be non-overlapping.
    constexpr float max_edge_length = 0.1f;
    forward_list<vector<Vec2d>> subdivided_geometry;
    for (vector<Vec2f>& polygon : geometry) {
      vector<Vec2d> output_polygon;
      for (int i = 0; i < polygon.size(); ++ i) {
        int next_i = (i + 1) % polygon.size();
        float length = (polygon[next_i] - polygon[i]).norm();
        
        int subdivisions = max<int>(1, ceil(length / max_edge_length));
        for (int s = 0; s < subdivisions; ++ s) {
          Vec2f pattern_coord = polygon[i] + (s / (1.f * subdivisions)) * (polygon[next_i] - polygon[i]);
          output_polygon.emplace_back(pattern_to_pixel_coord(pattern_coord).cast<double>());
        }
      }
      subdivided_geometry.push_front(output_polygon);
    }
    
    // Render the polygons:
    // - Loop over all polygons
    // - For a polygon, loop over all pixels in its bounding box
    // - Clip the polygon with the pixel's area as (convex) clipping path
    // - Darken the pixel by the amount of its area that is covered by the
    //   polygon (this works since the polygons are non-overlapping).
    Image<float> pattern_rendering(rendered_image.size());
    pattern_rendering.SetTo(1.f);
    for (vector<Vec2d>& polygon : subdivided_geometry) {
      // Compute the bounding box of the polygon
      Eigen::AlignedBox2d bbox;
      for (const Vec2d& point : polygon) {
        bbox.extend(point);
      }
      
      // Loop over all pixels that intersect the bounding box
      int min_x = max<int>(0, bbox.min().x());
      int max_x = min<int>(pattern_rendering.width() - 1, bbox.max().x());
      int min_y = max<int>(0, bbox.min().y());
      int max_y = min<int>(pattern_rendering.height() - 1, bbox.max().y());
      for (int y = min_y; y <= max_y; ++ y) {
        for (int x = min_x; x <= max_x; ++ x) {
          // Intersect the pixel area and the polygon
          vector<Vec2d> pixel_area(4);
          pixel_area[0] = Vec2d(x, y);
          pixel_area[1] = Vec2d(x + 1, y);
          pixel_area[2] = Vec2d(x + 1, y + 1);
          pixel_area[3] = Vec2d(x, y + 1);
          vector<Vec2d> intersection;
          ConvexClipPolygon(
              polygon,
              pixel_area,
              &intersection);
          pattern_rendering(x, y) -= PolygonArea(intersection);
        }
      }
    }
    
    // Loop over all pixels in the image and either use the pattern rendering
    // or the loaded pattern image depending on if the pixel is within the
    // pattern or not.
    for (int y = 0; y < rendered_image.height(); ++ y) {
      for (int x = 0; x < rendered_image.width(); ++ x) {
        Vec2f rendering_coord(x + 0.5f, y + 0.5f);  // in "pixel corner" convention
        rendering_coord.x() = rendering_coord.x() + distortion * rendering_coord.x() * rendering_coord.x();
        Vec2f pattern_image_coord = Vec3f(pattern_tr_rendering * rendering_coord.homogeneous()).hnormalized();
        
        // Transform the coordinate from the pattern image into the pattern feature coordinate system.
        Vec2f pattern_image_coord_mm = Vec2f(page_width_mm, page_height_mm).cwiseQuotient(pattern_image.size().cast<float>()).cwiseProduct(pattern_image_coord);
        Vec2f pattern_coord =
            (pattern_image_coord_mm - Vec2f(pattern_start_x_mm, pattern_start_y_mm))
                .cwiseQuotient(Vec2f(pattern_end_x_mm - pattern_start_x_mm, pattern_end_y_mm - pattern_start_y_mm))
                .cwiseProduct(Vec2f(squares_x, squares_y)) -
            Vec2f::Constant(1);
        
        // If the pattern coordinate is within the area of the repeating
        // pattern, use the definition of the pattern to render it ourselves
        // (instead of looking up the pattern image). This ensures that no
        // potential biases are introduced by the rendering process of that
        // image or by its pixel sampling pattern.
        const PatternData& pattern = detector.GetPatternData(0);
        if (pattern.IsValidPatternCoord(pattern_coord.x(), pattern_coord.y())) {
          rendered_image(x, y) = std::max<float>(0.f, 255.99f * pattern_rendering(x, y));
        } else {
          Vec2f pattern_image_coord_pixel_center = pattern_image_coord - Vec2f::Constant(0.5f);
          if (pattern_image.ContainsPixelCenterConv(pattern_image_coord_pixel_center)) {
            rendered_image(x, y) = pattern_image.InterpolateBilinear<float>(pattern_image_coord_pixel_center);
          } else {
            rendered_image(x, y) = 0;
          }
        }
      }
    }
    
    // Run the feature detector on the distorted pattern image
    Image<Vec3u8> color_image(rendered_image.size());
    for (int y = 0; y < rendered_image.height(); ++ y) {
      for (int x = 0; x < rendered_image.width(); ++ x) {
        color_image(x, y) = Vec3u8::Constant(rendered_image(x, y));
      }
    }
    vector<PointFeature> features;
    detector.DetectFeatures(color_image, &features, nullptr);
    
    // Visualize the deviations of the feature detections from the ground truth
    // feature locations to see whether there is a bias on them
    unordered_map<int, Vec2i> feature_id_to_coord;
    detector.GetCorners(/*pattern_index*/ 0, &feature_id_to_coord);
    
    bias_in_image.SetTo(Vec3u8(0, 0, 0));
    
    static ImageDisplay rendering_display;
    rendering_display.Clear();
    static ImageDisplay residual_arrow_display;
    residual_arrow_display.Clear();
    static ImageDisplay bias_display;
    bias_display.Clear();
    static ImageDisplay bias_in_image_display;
    bias_in_image_display.Clear();
    
    bias_display.AddSubpixelLinePixelCornerConv(Vec2f(0.5f, 0), Vec2f(0.5f, 1), Vec3u8(127, 127, 127));
    bias_display.AddSubpixelLinePixelCornerConv(Vec2f(0, 0.5f), Vec2f(1, 0.5f), Vec3u8(127, 127, 127));
    
    float error_sum = 0;
    
    for (PointFeature& feature : features) {
      const Vec2i& pattern_coord = feature_id_to_coord.at(feature.id);
      Vec2f ground_truth_pixel = pattern_to_pixel_coord(pattern_coord.cast<float>());
      
      const Vec2f& estimated_pixel = feature.xy;  // "pixel corner" convention
      
      Vec2f offset = estimated_pixel - ground_truth_pixel;
      error_sum += offset.norm();
      
      bias_display.AddSubpixelDotPixelCornerConv(Vec2f::Constant(0.5f) + offset, Vec3u8(255, 0, 0));
      
      double dir = atan2(offset.y(), offset.x());  // from -M_PI to M_PI
      bias_in_image(feature.xy.x() / kBiasInImageDownsampling, feature.xy.y() / kBiasInImageDownsampling) =
          Vec3u8(
              127 + 127 * sin(dir),
              127 + 127 * cos(dir),
              127);
      
      rendering_display.AddSubpixelLinePixelCornerConv(ground_truth_pixel, estimated_pixel, Vec3u8(127, 127, 127));
      rendering_display.AddSubpixelDotPixelCornerConv(ground_truth_pixel, Vec3u8(0, 255, 0));
      rendering_display.AddSubpixelDotPixelCornerConv(estimated_pixel, Vec3u8(255, 0, 0));
      
      constexpr float kResidualEnhanceFactor = 1000;
      residual_arrow_display.AddSubpixelLinePixelCornerConv(ground_truth_pixel, ground_truth_pixel + kResidualEnhanceFactor * (estimated_pixel - ground_truth_pixel), Vec3u8(0, 0, 255));
    }
    
    float average_error = error_sum / features.size();
    
    // NOTE: This is a somewhat arbitrary criterion to test here
    if (!features.empty()) {
      LOG(INFO) << "Average error: " << average_error;
      
      EXPECT_LT(average_error, 0.1f);
      if (average_error < 0.1f) {
        any_test_case_successful = true;
      }
    } else {
      LOG(INFO) << "No feature detected in this image.";
    }
    
    Image<u8> background(1, 1);
    background(0, 0) = 0;
    bias_display.Update(background, "Bias plot");
    
    bias_in_image_display.Update(bias_in_image, "Bias directions in (downsampled) image");
    rendering_display.Update(rendered_image, "Results");
    
    Image<u8> white_image(rendered_image.size());
    white_image.SetTo(255);
    residual_arrow_display.Update(white_image, "Enhanced residuals");
    
    // Uncomment this to look at the visualizations:
    // std::getchar();
    
    if (test == kNumTestCases - 1) {
      rendering_display.Close();
      residual_arrow_display.Close();
      bias_display.Close();
      bias_in_image_display.Close();
    }
  }
  
  EXPECT_TRUE(any_test_case_successful);
}

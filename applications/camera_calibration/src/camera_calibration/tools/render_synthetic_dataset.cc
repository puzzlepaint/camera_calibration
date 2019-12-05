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

#include <fstream>

#include <boost/filesystem.hpp>
#include <libvis/camera.h>
#include <libvis/geometry.h>
#include <libvis/image.h>
#include <QDir>

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"

namespace vis {

int RenderSyntheticDataset(const char* binary_path, const string& path) {
  srand(time(nullptr));
  
  // Pinhole camera settings
  int width = 640;
  int height = 480;
  float fx = height;
  float fy = height;
  float cx = 0.5 * width;
  float cy = 0.5 * height;
  
  vector<float> camera_parameters = {fx, fy, cx, cy};
  PinholeCamera4f camera(width, height, camera_parameters.data());
  
  // TODO: Make the pattern to use configurable
  // This is a symlink to the test_data folder installed via CMakeLists.txt
  string test_data_path = (boost::filesystem::path(binary_path).parent_path() / "test_data").string();
  string pattern_name = "pattern_resolution_17x24_segments_16_apriltag_0";
  
  // Load the pattern as an image
  string pattern_image_path = (boost::filesystem::path(test_data_path) / (pattern_name + ".png")).string();
  Image<u8> pattern_image(pattern_image_path);
  CHECK(!pattern_image.empty()) << "Cannot load the pattern image from: " << pattern_image_path;
  
  // Set up the feature detector to obtain the pattern geometry from it
  int feature_window_extent = 10;
  string pattern_yaml_path = (boost::filesystem::path(test_data_path) / (pattern_name + ".yaml")).string();
  vector<string> patterns = {pattern_yaml_path};
  FeatureDetectorTaggedPattern detector(patterns, feature_window_extent, FeatureRefinement::GradientsXY, /*use_cuda*/ true);
  if (!detector.valid()) {
    LOG(ERROR) << "Failed to load: " << pattern_yaml_path;
    return EXIT_FAILURE;
  }
  
  const PatternData& pattern = detector.GetPatternData(0);
  
  auto pattern_to_pattern_image_coord = [&](const Vec2f& pattern_coord) {
    Vec2f pattern_image_coord_mm =
        Vec2f(pattern.pattern_start_x_mm,
              pattern.pattern_start_y_mm) +
            (pattern_coord + Vec2f::Constant(1)).cwiseQuotient(Vec2f(pattern.squares_x, pattern.squares_y)).cwiseProduct(
            Vec2f(pattern.pattern_end_x_mm - pattern.pattern_start_x_mm,
                  pattern.pattern_end_y_mm - pattern.pattern_start_y_mm));
    return Vec2f(pattern_image.size().cast<float>().cwiseQuotient(Vec2f(pattern.page_width_mm, pattern.page_height_mm)).cwiseProduct(pattern_image_coord_mm));
  };
  
  vector<Vec3f> apriltag_min_xy;
  vector<Vec3f> apriltag_max_xy;
  for (int i = 0; i < pattern.tags.size(); ++ i) {
    const auto& tag = pattern.tags[i];
    Vec2f min_xy_2d = pattern_to_pattern_image_coord(Vec2f(
        tag.x - 1,
        tag.y - 1));
    apriltag_min_xy.push_back(Vec3f(min_xy_2d.x(), min_xy_2d.y(), 0));
    Vec2f max_xy_2d = pattern_to_pattern_image_coord(Vec2f(
        tag.x - 1 + tag.width,
        tag.y - 1 + tag.height));
    apriltag_max_xy.push_back(Vec3f(max_xy_2d.x(), max_xy_2d.y(), 0));
  }
  
  // Get the pattern geometry in the pattern plane and lift it to 3D in pattern pixel coordinates.
  // NOTE: If we used any kind of lens distortion, we would also need to
  //       subsample the pattern here as in feature_detection_test.cc.
  forward_list<vector<Vec2f>> geometry;
  pattern.ComputePatternGeometry(&geometry);
  
  forward_list<vector<Vec3f>> geometry_3d;
  for (vector<Vec2f>& polygon : geometry) {
    vector<Vec3f> polygon_3d(polygon.size());
    for (usize i = 0; i < polygon.size(); ++ i) {
      Vec2f pattern_image_coord = pattern_to_pattern_image_coord(polygon[i]);
      polygon_3d[i] = Vec3f(pattern_image_coord.x(), pattern_image_coord.y(), 0);
    }
    geometry_3d.push_front(polygon_3d);
  }
  
  Eigen::Hyperplane<float, 3> pattern_plane(Vec3f(0, 0, -1), Vec3f(0, 0, 0));
  
  Vec2f pattern_corners[4] = {
    Vec2f(0, 0),
    Vec2f(pattern_image.width(), 0),
    Vec2f(0, pattern_image.height()),
    Vec2f(pattern_image.width(), pattern_image.height()),
  };
  Vec2f pattern_right_vector = (pattern_corners[1] - pattern_corners[0]).normalized();
  Vec2f pattern_down_vector = (pattern_corners[2] - pattern_corners[0]).normalized();
  
  // Create output directory
  QDir dataset_dir(QString::fromStdString(path));
  dataset_dir.mkpath(".");
  
  // Write dataset.yaml file
  std::string dataset_yaml_path = dataset_dir.absoluteFilePath("dataset.yaml").toStdString();
  std::ofstream dataset_yaml_file(dataset_yaml_path.c_str(), std::ios::out);
  if (dataset_yaml_file) {
    dataset_yaml_file << "- camera: \"Synthetic pinhole camera (fx: " << fx << ", fy: " << fy << ", cx: " << cx << ", cy: " << cy << ", 'pixel corner' coordinate origin convention)\"" << endl;
    dataset_yaml_file << "  path: \"images0\"" << endl;
  } else {
    LOG(ERROR) << "Failed to write dataset YAML file at: " << dataset_yaml_path;
    return EXIT_FAILURE;
  }
  
  // Create images directory
  QDir images_dir = dataset_dir.filePath("images0");
  images_dir.mkpath(".");
  
  // Render the images
  Image<float> pattern_rendering(width, height);
  Image<u8> rendered_image(width, height);
  int num_images = 500;
  for (int image_index = 0; image_index < num_images; ++ image_index) {
    LOG(INFO) << "Rendering image " << image_index << " ...";
    
    // Determine a random image pose in front of the pattern, ensuring that all
    // corners of the AprilTag are visible
    SE3d camera_tr_global;
    Mat3f camera_r_global;
    Vec3f camera_t_global;
    
    bool any_apriltag_visible = false;
    while (!any_apriltag_visible) {
      camera_tr_global = SE3d();
      camera_tr_global.translation().x() -= (-1 + 2 * ((rand() % 10000) / 10000.f)) * pattern_image.width();
      camera_tr_global.translation().y() -= (-1 + 2 * ((rand() % 10000) / 10000.f)) * pattern_image.height();
      camera_tr_global.translation().z() += 500 + 800 * ((rand() % 10000) / 10000.f);
      camera_tr_global = SE3d::exp(0.5 * SE3d::Tangent::Random()) * camera_tr_global;
      
      camera_r_global = camera_tr_global.rotationMatrix().cast<float>();
      camera_t_global = camera_tr_global.translation().cast<float>();
      
      for (int i = 0; i < apriltag_min_xy.size(); ++ i) {
        const Vec3f& top_left = apriltag_min_xy[i];
        const Vec3f& bottom_right = apriltag_max_xy[i];
        Vec3f top_right = Vec3f(bottom_right.x(), top_left.y(), bottom_right.z());
        Vec3f bottom_left = Vec3f(top_left.x(), bottom_right.y(), bottom_right.z());
        
        Vec2f dummy;
        if (!camera.ProjectToPixelCornerConvIfVisible(camera_r_global * top_left + camera_t_global, 0, &dummy)) {
          continue;
        }
        if (!camera.ProjectToPixelCornerConvIfVisible(camera_r_global * top_right + camera_t_global, 0, &dummy)) {
          continue;
        }
        if (!camera.ProjectToPixelCornerConvIfVisible(camera_r_global * bottom_left + camera_t_global, 0, &dummy)) {
          continue;
        }
        if (!camera.ProjectToPixelCornerConvIfVisible(camera_r_global * bottom_right + camera_t_global, 0, &dummy)) {
          continue;
        }
        
        any_apriltag_visible = true;
        break;
      }
    }
    
    SE3d global_tr_camera = camera_tr_global.inverse();
    Mat3f global_r_camera = global_tr_camera.rotationMatrix().cast<float>();
    Vec3f global_t_camera = global_tr_camera.translation().cast<float>();
    
    // Render the polygons:
    // - Loop over all polygons and project their points into the image
    // - For a polygon, loop over all pixels in its bounding box
    // - Clip the polygon with the pixel's area as (convex) clipping path
    // - Darken the pixel by the amount of its area that is covered by the
    //   polygon (this works since the polygons are non-overlapping).
    pattern_rendering.SetTo(1.f);
    for (vector<Vec3f>& polygon : geometry_3d) {
      // Project the polygon points into the image
      vector<Vec2d> projected(polygon.size());
      for (usize i = 0; i < polygon.size(); ++ i) {
        projected[i] = camera.ProjectToPixelCornerConv(camera_r_global * polygon[i] + camera_t_global).cast<double>();
      }
      
      // Compute the bounding box of the polygon
      Eigen::AlignedBox2d bbox;
      for (const Vec2d& point : projected) {
        bbox.extend(point);
      }
      
      // Loop over all pixels that intersect the bounding box
      int min_x = max<int>(0, bbox.min().x());
      int max_x = min<int>(pattern_rendering.width() - 1, bbox.max().x());
      int min_y = max<int>(0, bbox.min().y());
      int max_y = min<int>(pattern_rendering.height() - 1, bbox.max().y());
      for (int y = min_y; y <= max_y; ++ y) {
        for (int x = min_x; x <= max_x; ++ x) {
          // Intersect the pixel area and the projected polygon
          vector<Vec2d> pixel_area(4);
          pixel_area[0] = Vec2d(x, y);
          pixel_area[1] = Vec2d(x + 1, y);
          pixel_area[2] = Vec2d(x + 1, y + 1);
          pixel_area[3] = Vec2d(x, y + 1);
          vector<Vec2d> intersection;
          ConvexClipPolygon(
              projected,
              pixel_area,
              &intersection);
          pattern_rendering(x, y) -= PolygonArea(intersection);
        }
      }
    }
    
    // Loop over all pixels in the image and either use the pattern rendering
    // or the loaded pattern image depending on if the pixel is within the
    // pattern or not.
    for (int y = 0; y < height; ++ y) {
      for (int x = 0; x < width; ++ x) {
        // Determine the intersection of the center viewing ray of this pixel with the pattern plane
        Vec2f rendering_coord(x + 0.5f, y + 0.5f);  // in "pixel corner" convention
        Vec3f local_direction = camera.UnprojectFromPixelCornerConv(rendering_coord);  // not necessarily normalized
        
        Vec3f global_direction = global_r_camera * local_direction;
        const Vec3f& global_camera_position = global_t_camera;
        Eigen::ParametrizedLine<float, 3> viewing_line(global_camera_position, global_direction.normalized());
        
        Vec3f pattern_intersection = viewing_line.intersectionPoint(pattern_plane);
        
        // Compute the local 2D pattern coordinates of the intersection point
        Vec2f local_pattern_intersection_2d(pattern_intersection.x() - pattern_corners[0].x(), pattern_intersection.y() - - pattern_corners[0].y());
        Vec2f pattern_image_coord(
            pattern_right_vector.dot(local_pattern_intersection_2d),
            pattern_down_vector.dot(local_pattern_intersection_2d));
        
        // Transform the coordinate from the pattern image into the pattern feature coordinate system.
        Vec2f pattern_image_coord_mm = Vec2f(pattern.page_width_mm, pattern.page_height_mm).cwiseQuotient(pattern_image.size().cast<float>()).cwiseProduct(pattern_image_coord);
        Vec2f pattern_coord =
            (pattern_image_coord_mm - Vec2f(pattern.pattern_start_x_mm, pattern.pattern_start_y_mm))
                .cwiseQuotient(Vec2f(pattern.pattern_end_x_mm - pattern.pattern_start_x_mm, pattern.pattern_end_y_mm - pattern.pattern_start_y_mm))
                .cwiseProduct(Vec2f(pattern.squares_x, pattern.squares_y)) -
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
    
    // Save the rendered image.
    rendered_image.Write(images_dir.absoluteFilePath(QString::number(image_index).rightJustified(6, '0') + ".png").toStdString());
  }
  
  return EXIT_SUCCESS;
}

}

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

#include <boost/filesystem.hpp>
#ifdef LIBVIS_HAVE_CUDA
#include <libvis/cuda/patch_match_stereo.h>
#endif
#include <libvis/point_cloud.h>
#include <libvis/render_display.h>
#include <QApplication>
#include <QDir>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/models/central_thin_prism_fisheye.h"
#ifdef LIBVIS_HAVE_CUDA
#include "camera_calibration/models/cuda_camera_model.cuh"
#include "camera_calibration/models/cuda_central_generic_model.cuh"
#endif

namespace vis {

int StereoDepthEstimation(const string& state_base_path, const vector<string>& image_paths, const string& output_directory) {
  constexpr bool kDebug = false;
  
  CHECK_EQ(image_paths.size(), 2) << "Only two-view stereo is supported currently.";
  
  // Load state
  BAState state;
  if (!LoadBAState(state_base_path.c_str(), &state, nullptr)) {
    return EXIT_FAILURE;
  }
  CHECK_EQ(image_paths.size(), state.num_cameras()) << "For stereo depth estimation, the number of cameras and the number of given images must be the same.";
  
  
  // Load images
  vector<Image<u8>> images(image_paths.size());
  for (int i = 0; i < images.size(); ++ i) {
    if (!images[i].Read(image_paths[i])) {
      LOG(ERROR) << "Cannot read image: " << image_paths[i];
      return EXIT_FAILURE;
    }
  }
  
  
  // Create PixelCornerProjectors from cameras
  // TODO: This is somewhat of a mess, with special cases implemented for two
  //       different camera models. Would it be possible to make this a bit more generic?
  // TODO: Currently, only one projection implementation in the CUDA code of
  //       PixelCornerProjector is compiled at a time. So only this camera model
  //       can be used here.
  vector<shared_ptr<PixelCornerProjector>> projectors(image_paths.size());
  for (int i = 0; i < projectors.size(); ++ i) {
    CameraModel* camera = state.intrinsics[i].get();
    CentralGenericModel* cgbsp_model = dynamic_cast<CentralGenericModel*>(camera);
    CentralThinPrismFisheyeModel* ctpf_model = dynamic_cast<CentralThinPrismFisheyeModel*>(camera);
    
    if (cgbsp_model) {
      CUDACameraModel* cuda_model = cgbsp_model->CreateCUDACameraModel();
      CUDACentralGenericModel* cuda_cgbsp_model = /*dynamic_cast*/ reinterpret_cast<CUDACentralGenericModel*>(cuda_model);
      CHECK(cuda_cgbsp_model);
      
      CHECK_EQ(cgbsp_model->grid().width(), cuda_cgbsp_model->grid().width());
      CHECK_EQ(cgbsp_model->grid().height(), cuda_cgbsp_model->grid().height());
      Image<float3> cpu_grid(cgbsp_model->grid().size());
      cudaMemcpy2DAsync(cpu_grid.data(), cpu_grid.width() * sizeof(float3), cuda_cgbsp_model->grid().address(), cuda_cgbsp_model->grid().pitch(), cpu_grid.width(), cpu_grid.height(), cudaMemcpyDeviceToHost);
      projectors[i].reset(new PixelCornerProjector(
          cgbsp_model->width(), cgbsp_model->height(),
          cgbsp_model->calibration_min_x(), cgbsp_model->calibration_min_y(),
          cgbsp_model->calibration_max_x(), cgbsp_model->calibration_max_y(),
          cpu_grid));
      
      delete cuda_model;
    } else if (ctpf_model) {
      if (!ctpf_model->use_equidistant_projection()) {
        LOG(ERROR) << "use_equidistant_projection must be true for CentralThinPrismFisheyeModel in StereoDepthEstimation currently";
        return EXIT_FAILURE;
      }
      
      // Change parameter ordering from:
      // fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1.
      // to:
      // k1, k2, k3, k4, p1, p2, sx1, sy1, fx, fy, cx, cy.
      vector<double> camera_parameters(12);
      
      camera_parameters[0] = ctpf_model->parameters()(4);
      camera_parameters[1] = ctpf_model->parameters()(5);
      camera_parameters[2] = ctpf_model->parameters()(6);
      camera_parameters[3] = ctpf_model->parameters()(7);
      camera_parameters[4] = ctpf_model->parameters()(8);
      camera_parameters[5] = ctpf_model->parameters()(9);
      camera_parameters[6] = ctpf_model->parameters()(10);
      camera_parameters[7] = ctpf_model->parameters()(11);
      
      camera_parameters[8] = ctpf_model->parameters()(0);
      camera_parameters[9] = ctpf_model->parameters()(1);
      camera_parameters[10] = ctpf_model->parameters()(2);
      camera_parameters[11] = ctpf_model->parameters()(3);
      
      projectors[i].reset(new PixelCornerProjector(ThinPrismFisheyeCamera12d(
          images[i].width(), images[i].height(),
          camera_parameters.data())));
    } else {
      LOG(ERROR) << "This camera model is not supported in StereoDepthEstimation currently";
      return EXIT_FAILURE;
    }
  }
  
  
  // Create unprojection lookups
  vector<shared_ptr<CUDAUnprojectionLookup2D>> unprojection_lookups(images.size());
  vector<Image<float2>> lookup_buffer_cpu(images.size());
  for (int i = 0; i < images.size(); ++ i) {
    const CameraModel* camera = state.intrinsics[i].get();
    const CentralGenericModel* cgbsp_model = dynamic_cast<const CentralGenericModel*>(camera);
    const CentralThinPrismFisheyeModel* ctpf_model = dynamic_cast<const CentralThinPrismFisheyeModel*>(camera);
    if (!cgbsp_model && !ctpf_model) {
      LOG(ERROR) << "StereoDepthEstimation() currently only works with CentralGenericModel and CentralThinPrismFisheyeModel.";
      return EXIT_FAILURE;
    }
    
    lookup_buffer_cpu[i].SetSize(camera->width(), camera->height());
    for (int y = 0; y < camera->height(); ++ y) {
      for (int x = 0; x < camera->width(); ++ x) {
        Vec3d dir;
        bool valid;
        if (cgbsp_model) {
          valid = cgbsp_model->Unproject(x + 0.5, y + 0.5, &dir);
        } else if (ctpf_model) {
          valid = ctpf_model->Unproject(x + 0.5, y + 0.5, &dir);
        } else {
          CHECK(false);
        }
        
        if (valid) {
          lookup_buffer_cpu[i](x, y) = make_float2(dir.x() / dir.z(), dir.y() / dir.z());
        } else {
          lookup_buffer_cpu[i](x, y) = make_float2(numeric_limits<float>::quiet_NaN(), numeric_limits<float>::quiet_NaN());
        }
      }
    }
    
    unprojection_lookups[i].reset(new CUDAUnprojectionLookup2D(lookup_buffer_cpu[i], /*stream*/ 0));
  }
  
  
  // Run stereo depth estimation
  int reference_image_index = 0;
  int stereo_image_index = 1 - reference_image_index;
  
  PatchMatchStereoCUDA stereo(images[reference_image_index].width(), images[reference_image_index].height());
  
  stereo.SetMatchMetric(PatchMatchStereoCUDA::MatchMetric::kZNCC);
  stereo.SetContextRadius(10);  // TODO: tune / make configurable
  
  stereo.SetMinEpipolarGradientPerPixel(0.01f);  // almost disable; previous value: 1.f
  stereo.SetAngleThreshold(1.3f);
  stereo.SetCostThreshold(1.0f);  // disable; previous value: 0.2f
  stereo.SetMinComponentSize(50);
  stereo.SetSimilarDepthRatio(1.005f);
  stereo.SetSecondBestMinCostFactor(1);  // disable filtering based on second best cost
  
//   // To disable outlier filtering:
//   stereo.SetMinComponentSize(0);
//   stereo.SetCostThreshold(1.0f);
//   stereo.SetAngleThreshold(3.141592f / 180.f * 90);
//   stereo.SetSecondBestMinCostFactor(1);
  
  Image<u8> null_mask;
  Image<float> lr_consistency_depth_map;
  Image<float> inv_depth_map;
  
  // Estimate a depth map for the other image (low quality) to be used for left-right consistency checking.
  stereo.SetIterationCount(30);
  stereo.ComputeDepthMap(
      *unprojection_lookups[stereo_image_index],
      images[stereo_image_index],
      state.camera_tr_rig[stereo_image_index].cast<float>(),
      *projectors[reference_image_index],
      null_mask,
      images[reference_image_index],
      state.camera_tr_rig[reference_image_index].cast<float>(),
      &lr_consistency_depth_map);
  
  if (kDebug) {
    static ImageDisplay lr_inv_depth_display;
    lr_inv_depth_display.Update(lr_consistency_depth_map, "L/R consistency check inv depth", 0.f, 1.5f);
  }
  
  // Run stereo depth estimation to estimate a depth image for the right IR image.
  stereo.SetIterationCount(50);
  stereo.ComputeDepthMap(
      *unprojection_lookups[reference_image_index],
      images[reference_image_index],
      state.camera_tr_rig[reference_image_index].cast<float>(),
      *projectors[stereo_image_index],
      null_mask,
      images[stereo_image_index],
      state.camera_tr_rig[stereo_image_index].cast<float>(),
      &inv_depth_map,
      &lr_consistency_depth_map);
  
  // Create a point cloud from the result.
  shared_ptr<Point3fC3u8Cloud> cloud(new Point3fC3u8Cloud());
  
  const int context_radius = stereo.context_radius();
  
  // Count the valid pixels in the depth image.
  usize point_count = 0;
  for (u32 y = context_radius; y < inv_depth_map.height() - context_radius; ++ y) {
    const float* ptr = inv_depth_map.row(y) + context_radius;
    const float* end = inv_depth_map.row(y) + inv_depth_map.width() - context_radius;
    while (ptr < end) {
      if (*ptr != 0) {
        ++ point_count;
      }
      ++ ptr;
    }
  }
  
  cloud->Resize(point_count);
  
  // Create the points.
  point_count = 0;
  for (u32 y = context_radius; y < inv_depth_map.height() - context_radius; ++ y) {
    const float* d_ptr = inv_depth_map.row(y) + context_radius;
    const u8* rgb_ptr = images[reference_image_index].row(y) + context_radius;
    for (u32 x = context_radius; x < inv_depth_map.width() - context_radius; ++ x) {
      if (*d_ptr != 0) {
        const float2& dir = lookup_buffer_cpu[reference_image_index](x, y);
        cloud->at(point_count).position() =
            (1.f / *d_ptr) *
            Vec3f(dir.x, dir.y, 1);
        cloud->at(point_count).color() = Vec3u8::Constant(*rgb_ptr);
        ++ point_count;
      }
      ++ d_ptr;
      ++ rgb_ptr;
    }
  }
  CHECK_EQ(point_count, cloud->size());
  
  // Debug visualizations:
  if (kDebug) {
    static ImageDisplay inv_depth_estimate_display;
    inv_depth_estimate_display.Update(inv_depth_map, "inv depth estimate", 0.f, 1.5f);
    
//     Image<Vec3u8> reprojected_image_debug(inv_depth_map.width(), inv_depth_map.height());
//     for (u32 y = 0; y < reprojected_image_debug.height(); ++ y) {
//       for (u32 x = 0; x < reprojected_image_debug.width(); ++ x) {
//         float inv_depth = inv_depth_map(x, y);
//         if (inv_depth == 0) {
//           reprojected_image_debug(x, y) = Vec3u8(255, 0, 0);
//           continue;
//         }
//         
//         Vec3d dir = cameras[infrared_right_index]->UnprojectFromPixelCenterConv(Vec2d(x, y));
//         Vec3d infrared_right_point = (1.0 / inv_depth) * dir;
//         Vec3d infrared_left_point = (infrared_left_tr_rig * infrared_right_tr_rig.inverse()).cast<double>() * infrared_right_point;
//         Vec2d proj_pixel = cameras[infrared_left_index]->ProjectToPixelCenterConv(infrared_left_point);
//         if (proj_pixel.x() >= 0 && proj_pixel.y() >= 0 &&
//             proj_pixel.x() < ir_left.width() - 1 && proj_pixel.y() < ir_left.height() - 1) {
//           reprojected_image_debug(x, y) = Vec3u8::Constant(ir_left.InterpolateBilinear(proj_pixel.cast<float>()));
//         } else {
//           reprojected_image_debug(x, y) = Vec3u8(0, 0, 255);
//         }
//       }
//     }
//     reprojected_image_debug_display.Update(reprojected_image_debug, "left image reprojected to right");
    
    static ImageDisplay reference_image_display;
    reference_image_display.Update(images[reference_image_index], "Reference image");
    static ImageDisplay stereo_image_display;
    stereo_image_display.Update(images[stereo_image_index], "Stereo image");
    
    shared_ptr<RenderDisplay> render_display;
    shared_ptr<RenderWindow> generic_render_window;
    render_display = make_shared<RenderDisplay>();
    generic_render_window = RenderWindow::CreateWindow("Depth map debug", 800, 600, RenderWindow::API::kOpenGL, render_display);
    render_display->Update(cloud, "estimate cloud");
    
    std::getchar();
  }
  
  // Save the results.
  QDir(output_directory.c_str()).mkpath(".");
  
  // Save the metadata file.
  ofstream metadata_stream((boost::filesystem::path(output_directory) / "metadata.yaml").string(), std::ios::out);
  metadata_stream << "image_width: " << inv_depth_map.width() << "\n";
  metadata_stream << "image_height: " << inv_depth_map.height() << "\n";
  metadata_stream << "context_radius: " << context_radius << "\n";
  metadata_stream.close();
  
  // Save the depth image as a binary float buffer.
  Image<float> depth_image(inv_depth_map.size());
  for (int y = 0; y < depth_image.height(); ++ y) {
    for (int x = 0; x < depth_image.width(); ++ x) {
      float inv_depth = inv_depth_map(x, y);
      if (inv_depth == 0) {
        depth_image(x, y) = 0;
      } else {
        depth_image(x, y) = 1. / inv_depth;
      }
    }
  }
  
  string path = (boost::filesystem::path(output_directory) / "depth_image.bin").string();
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) {
    LOG(ERROR) << "Cannot open file for writing: " << path;
    return EXIT_FAILURE;
  }
  fwrite(depth_image.data(), depth_image.pixel_count(), sizeof(float), file);
  fclose(file);
  
  // Save the point cloud as binary PLY.
  path = (boost::filesystem::path(output_directory) / "point_cloud.ply").string();
  if (!cloud->WriteAsPLY(path)) {
    LOG(ERROR) << "Cannot save point cloud to: " << path;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}

}

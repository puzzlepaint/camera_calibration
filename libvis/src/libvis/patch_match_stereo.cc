// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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


#include "libvis/patch_match_stereo.h"

// #include "libvis/point_cloud.h"  // for debugging only
// #include "libvis/render_display.h"  // for debugging only

namespace vis {

constexpr float kMinInvDepth = 1e-5f;  // TODO: Make parameter

template <class CameraT>
inline float SampleAtProjectedPosition(
    const float x, const float y, const float z,
    const int stereo_width,
    const int stereo_height,
    const CameraT& projector,
    const Matrix<float, 3, 4>& stereo_tr_reference,
    const Image<u8>& stereo_image) {
  Vec3f pnxy = stereo_tr_reference * Vec3f(x, y, z).homogeneous();
  if (pnxy.z() <= 0.f) {
    return numeric_limits<float>::quiet_NaN();
  }
  
  const Vec2f pxy = projector.ProjectToPixelCenterConv(pnxy).template cast<float>();
  
  // NOTE: Written to catch NaNs
  if (!(pxy.x() >= 0.f) ||
      !(pxy.y() >= 0.f) ||
      !(pxy.x() < stereo_width - 1.0f) ||
      !(pxy.y() < stereo_height - 1.0f)) {
    return numeric_limits<float>::quiet_NaN();
  } else {
    return stereo_image.InterpolateBilinear(pxy);
  }
}


inline float CalculatePlaneDepth2(
    float d, const Vec2f& normal_xy, float normal_z,
    float query_x, float query_y) {
  return d / (query_x * normal_xy.x() + query_y * normal_xy.y() + normal_z);
}

inline float CalculatePlaneInvDepth2(
    float d, const Vec2f& normal_xy, float normal_z,
    float query_x, float query_y) {
  return (query_x * normal_xy.x() + query_y * normal_xy.y() + normal_z) / d;
}


template <class CameraT1, class CameraT2>
inline float ComputeCostsSSD(
    int x, int y,
    const Vec2f& normal_xy,
    const float inv_depth,
    const CameraT1& reference_camera,
    const Image<u8>& reference_image,
    const Matrix<float, 3, 4>& stereo_tr_reference,
    const CameraT2& stereo_camera,
    const Image<u8>& stereo_image,
    const int stereo_width,
    const int stereo_height,
    int context_radius) {
  if (inv_depth < kMinInvDepth) {
    return numeric_limits<float>::quiet_NaN();
  }
  
  const float normal_z =
      -sqrtf(1.f - normal_xy.x() * normal_xy.x() - normal_xy.y() * normal_xy.y());
  const float depth = 1.f / inv_depth;
  const Vec2f center_nxy =
      reference_camera.UnprojectFromPixelCenterConv(Vec2i(x, y)).template cast<float>().template topRows<2>();
  const float plane_d =
      (center_nxy.x() * depth) * normal_xy.x() +
      (center_nxy.y() * depth) * normal_xy.y() + depth * normal_z;
  
  float cost = 0;
  
  for (int dy = -context_radius; dy <= context_radius; ++ dy) {
    for (int dx = -context_radius; dx <= context_radius; ++ dx) {
      Vec2f nxy = reference_camera.UnprojectFromPixelCenterConv(Vec2i(x + dx, y + dy)).template cast<float>().template topRows<2>();
      float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x(), nxy.y());
      nxy.x() *= plane_depth;
      nxy.y() *= plane_depth;
      
      float sample =
            SampleAtProjectedPosition(nxy.x(), nxy.y(), plane_depth,
                                      stereo_width,
                                      stereo_height,
                                      stereo_camera,
                                      stereo_tr_reference,
                                      stereo_image);
      
      const float diff = sample - reference_image(x + dx, y + dy);
      cost += diff * diff;
    }
  }
  
  return cost;
}

// Computes 0.5f * (1 - ZNCC), so that the result can be used
// as a cost value with range [0; 1].
inline float ComputeZNCCBasedCost(
    const int context_radius,
    const float sum_a,
    const float squared_sum_a,
    const float sum_b,
    const float squared_sum_b,
    const float product_sum) {
  const float normalizer = 1.0f / ((float)(2.0f * context_radius + 1.0f)
                                   * (2.0f * context_radius + 1.0f));

  const float numerator =
      product_sum - normalizer * (sum_a * sum_b);
  const float denominator_reference =
      squared_sum_a - normalizer * sum_a * sum_a;
  const float denominator_other =
      squared_sum_b - normalizer * sum_b * sum_b;
  constexpr float kHomogeneousThreshold = 0.1f;
  if (denominator_reference < kHomogeneousThreshold ||
      denominator_other < kHomogeneousThreshold) {
    return 1.0f;
  } else {
    return 0.5f * (1.0f - numerator /
        sqrtf(denominator_reference * denominator_other));
  }
}

template <class CameraT1, class CameraT2>
inline float ComputeCostsZNCC(
    int x, int y,
    const Vec2f& normal_xy,
    const float inv_depth,
    const CameraT1& reference_camera,
    const Image<u8>& reference_image,
    const Matrix<float, 3, 4>& stereo_tr_reference,
    const CameraT2& stereo_camera,
    const Image<u8>& stereo_image,
    const int stereo_width,
    const int stereo_height,
    int context_radius) {
  if (inv_depth < kMinInvDepth) {
    return numeric_limits<float>::quiet_NaN();
  }
  
  const float normal_z =
      -sqrtf(1.f - normal_xy.x() * normal_xy.x() - normal_xy.y() * normal_xy.y());
  const float depth = 1.f / inv_depth;
  const Vec2f center_nxy =
      reference_camera.UnprojectFromPixelCenterConv(Vec2i(x, y)).template cast<float>().template topRows<2>();
  const float plane_d =
      (center_nxy.x() * depth) * normal_xy.x() +
      (center_nxy.y() * depth) * normal_xy.y() + depth * normal_z;
  
  float sum_a = 0;
  float squared_sum_a = 0;
  float sum_b = 0;
  float squared_sum_b = 0;
  float product_sum = 0;
  
  for (int dy = -context_radius; dy <= context_radius; ++ dy) {
    for (int dx = -context_radius; dx <= context_radius; ++ dx) {
      Vec2f nxy = reference_camera.UnprojectFromPixelCenterConv(Vec2i(x + dx, y + dy)).template cast<float>().template topRows<2>();
      float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x(), nxy.y());
      nxy.x() *= plane_depth;
      nxy.y() *= plane_depth;
      
      float stereo_value =
            SampleAtProjectedPosition(nxy.x(), nxy.y(), plane_depth,
                                      stereo_width,
                                      stereo_height,
                                      stereo_camera,
                                      stereo_tr_reference,
                                      stereo_image);
      
      sum_a += stereo_value;
      squared_sum_a += stereo_value * stereo_value;
      
      float reference_value = reference_image(x + dx, y + dy);
      
      sum_b += reference_value;
      squared_sum_b += reference_value * reference_value;
      
      product_sum += stereo_value * reference_value;
    }
  }
  
  return ComputeZNCCBasedCost(
      context_radius, sum_a, squared_sum_a, sum_b, squared_sum_b, product_sum);
}

template <class CameraT1, class CameraT2>
inline float ComputeCosts(
    int x, int y,
    const Vec2f& normal_xy,
    const float inv_depth,
    const CameraT1& reference_camera,
    const Image<u8>& reference_image,
    const Matrix<float, 3, 4>& stereo_tr_reference,
    const CameraT2& stereo_camera,
    const Image<u8>& stereo_image,
    const int stereo_width,
    const int stereo_height,
    int context_radius,
    PatchMatchStereoCPU::MatchMetric match_metric) {
  if (match_metric == PatchMatchStereoCPU::MatchMetric::kSSD) {
    return ComputeCostsSSD(
        x, y, normal_xy, inv_depth, reference_camera, reference_image,
        stereo_tr_reference, stereo_camera, stereo_image, stereo_width,
        stereo_height, context_radius);
  } else if (match_metric == PatchMatchStereoCPU::MatchMetric::kZNCC) {
    return ComputeCostsZNCC(
        x, y, normal_xy, inv_depth, reference_camera, reference_image,
        stereo_tr_reference, stereo_camera, stereo_image, stereo_width,
        stereo_height, context_radius);
  }
  
  // This should never be reached since all metrics should be handled above.
  return 0;
}


struct ConnectedComponent {
  ConnectedComponent(int _parent, int _pixel_count)
      : parent(_parent), pixel_count(_pixel_count) {}

  int parent;
  int pixel_count;
  bool should_be_removed;
};

void PatchMatchStereoCPU::RemoveSmallConnectedComponentsInInvDepthMap(
    float separator_value,
    int min_component_size,
    int min_x,
    int min_y,
    int max_x,
    int max_y,
    Image<float>* inv_depth_map) {
  // Find connected components and calculate number of certain pixels.
  vector<ConnectedComponent> components;
  constexpr size_t kPreallocationSize = 4096u;
  components.reserve(kPreallocationSize);
  Image<int> component_image(inv_depth_map->width(), inv_depth_map->height());
  
  for (int y = min_y; y <= max_y; ++ y) {
    for (int x = min_x; x <= max_x; ++ x) {
      // Mark pixel as invalid if it has the separator value.
      if (inv_depth_map->operator()(x, y) == separator_value) {
        component_image(x, y) = -1;
        continue;
      }
      
      if (x > min_x &&
          component_image(x - 1, y) != -1 &&
          DepthIsSimilar(inv_depth_map->operator()(x - 1, y), inv_depth_map->operator()(x, y))) {
        // Merge into left component.
        component_image(x, y) = component_image(x - 1, y);
        ConnectedComponent* const component = &components[component_image(x - 1, y)];
        ConnectedComponent* parent = component;
        while (&components[parent->parent] != parent) {
          ConnectedComponent* const higher_parent = &components[parent->parent];
          parent->parent = higher_parent->parent;
          parent = higher_parent;
        }
        parent->pixel_count += 1;
        
        if (y > min_y &&
            component_image(x, y - 1) != -1 &&
            DepthIsSimilar(inv_depth_map->operator()(x, y - 1), inv_depth_map->operator()(x, y))) {
          // Merge left into top component.
          // Notice: leaf, parent and target components may be the same.
          ConnectedComponent* const left_component = &components[component_image(x - 1, y)];
          ConnectedComponent* parent = left_component;
          while (&components[parent->parent] != parent) {
            ConnectedComponent* const higher_parent = &components[parent->parent];
            parent->parent = higher_parent->parent;
            parent = higher_parent;
          }
          int certain_pixels = left_component->pixel_count;
          left_component->pixel_count = 0;
          certain_pixels += parent->pixel_count;
          parent->pixel_count = 0;
          
          ConnectedComponent* const top_component = &components[component_image(x, y - 1)];
          parent->parent = top_component->parent;
          while (&components[parent->parent] != parent) {
            ConnectedComponent* const higher_parent = &components[parent->parent];
            parent->parent = higher_parent->parent;
            parent = higher_parent;
          }
          components[parent->parent].pixel_count += certain_pixels;
        }
      } else if (y > min_y &&
                 component_image(x, y - 1) != -1 &&
                 DepthIsSimilar(inv_depth_map->operator()(x, y - 1), inv_depth_map->operator()(x, y))) {
        // Merge into top component.
        component_image(x, y) = component_image(x, y - 1);
        ConnectedComponent* const component = &components[component_image(x, y - 1)];
        ConnectedComponent* parent = component;
        while (&components[parent->parent] != parent) {
          ConnectedComponent* const higher_parent = &components[parent->parent];
          parent->parent = higher_parent->parent;
          parent = higher_parent;
        }
        parent->pixel_count += 1;
      } else {
        // Create a new component.
        components.emplace_back(components.size(), 1);
        component_image(x, y) = components.size() - 1;
      }
    }
  }
  
  // Resolve parents until the root and decide on which components to remove.
  for (size_t i = 0u, end = components.size(); i < end; ++i) {
    ConnectedComponent* const component = &components[i];
    ConnectedComponent* parent = &components[component->parent];
    while (&components[parent->parent] != parent) {
      ConnectedComponent* const higher_parent = &components[parent->parent];
      parent->parent = higher_parent->parent;
      parent = higher_parent;
    }
    component->should_be_removed = parent->pixel_count < min_component_size;
  }
  
  // Remove bad connected components from image.
  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      if (component_image(x, y) >= 0 &&
          components[component_image(x, y)].should_be_removed) {
        inv_depth_map->operator()(x, y) = 0.f;
      }
    }
  }
}


PatchMatchStereoCPU::PatchMatchStereoCPU(int /*width*/, int /*height*/) {}

// // (Mostly) auto-generated function.
// typedef float Scalar;
// 
// // opcount = 243
// inline void ComputeResidualAndJacobian(
//     Scalar cx, Scalar cy, Scalar fx, Scalar fy,
//     Scalar inv_depth, Scalar n_x, Scalar n_y,
//     Scalar nx, Scalar ny,
//     Scalar other_nx, Scalar other_ny,
//     Scalar ref_intensity,
//     Scalar str_0_0, Scalar str_0_1, Scalar str_0_2, Scalar str_0_3,
//     Scalar str_1_0, Scalar str_1_1, Scalar str_1_2, Scalar str_1_3,
//     Scalar str_2_0, Scalar str_2_1, Scalar str_2_2, Scalar str_2_3,
//     const Image<u8>& stereo_image,
//     Scalar* residuals, Scalar* jacobian) {
//   const Scalar term0 = sqrt(-n_x*n_x - n_y*n_y + 1);
//   const Scalar term1 = n_x*other_nx + n_y*other_ny - term0;
//   const Scalar term2 = 1.0f/term1;
//   const Scalar term3 = str_1_2*term2;
//   const Scalar term4 = 1.0f/inv_depth;
//   const Scalar term5 = n_x*nx;
//   const Scalar term6 = n_y*ny;
//   const Scalar term7 = -term0*term4 + term4*term5 + term4*term6;
//   const Scalar term8 = other_nx*str_1_0*term2;
//   const Scalar term9 = other_ny*str_1_1*term2;
//   const Scalar term10 = str_1_3 + term3*term7 + term7*term8 + term7*term9;
//   const Scalar term11 = str_2_2*term2;
//   const Scalar term12 = other_nx*str_2_0*term2;
//   const Scalar term13 = other_ny*str_2_1*term2;
//   const Scalar term14 = str_2_3 + term11*term7 + term12*term7 + term13*term7;
//   const Scalar term15 = 1.0f/term14;
//   const Scalar term16 = fy*term15;
//   
//   float py = cy + term10*term16;
//   int iy = static_cast<int>(py);
//   const Scalar term17 = py - iy;
//   
//   const Scalar term18 = str_0_2*term2;
//   const Scalar term19 = other_nx*str_0_0*term2;
//   const Scalar term20 = other_ny*str_0_1*term2;
//   const Scalar term21 = str_0_3 + term18*term7 + term19*term7 + term20*term7;
//   const Scalar term22 = fx*term15;
//   
//   float px = cx + term21*term22;
//   int ix = static_cast<int>(px);
//   const Scalar term23 = px - ix;
//   
//   if (ix < 0 ||
//       iy < 0 ||
//       ix + 1 >= stereo_image.width() - 1 ||
//       iy + 1 >= stereo_image.height() - 1) {
//     *residuals = 0;
//     jacobian[1] = 0;
//     jacobian[2] = 0;
//     jacobian[3] = 0;
//     return;
//   }
//   
//   Scalar top_left = stereo_image.InterpolateBilinear(Vec2f(ix, iy));
//   Scalar top_right = stereo_image.InterpolateBilinear(Vec2f(ix + 1, iy));
//   Scalar bottom_left = stereo_image.InterpolateBilinear(Vec2f(ix, iy + 1));
//   Scalar bottom_right = stereo_image.InterpolateBilinear(Vec2f(ix + 1, iy + 1));
//   
//   const Scalar term24 = -term23 + 1;
//   const Scalar term25 = bottom_left*term24 + bottom_right*term23;
//   const Scalar term26 = -term17 + 1;
//   const Scalar term27 = term23*top_right;
//   const Scalar term28 = term24*top_left;
//   const Scalar term29 = -term17*(bottom_left - bottom_right) - term26*(top_left - top_right);
//   const Scalar term30 = term4 * term4;
//   const Scalar term31 = term0 - term5 - term6;
//   const Scalar term32 = term30*term31;
//   const Scalar term33 = term15 * term15;
//   const Scalar term34 = term30*term31*term33*(term11 + term12 + term13);
//   const Scalar term35 = term25 - term27 - term28;
//   const Scalar term36 = 1.0f/term0;
//   const Scalar term37 = n_x*term36;
//   const Scalar term38 = nx*term4 + term37*term4;
//   const Scalar term39 = -other_nx - term37;
//   const Scalar term40 = term2 * term2;
//   
//   const Scalar term40Xterm7 = term40*term7;
//   
//   const Scalar term41 = str_0_2*term40Xterm7;
//   const Scalar term42 = other_nx*str_0_0*term40Xterm7;
//   const Scalar term43 = other_ny*str_0_1*term40Xterm7;
//   const Scalar term44 = fx*term21*term33;
//   const Scalar term45 = str_2_2*term40Xterm7;
//   const Scalar term46 = other_nx*str_2_0*term40Xterm7;
//   const Scalar term47 = other_ny*str_2_1*term40Xterm7;
//   const Scalar term48 = -term11*term38 - term12*term38 - term13*term38 - term39*term45 - term39*term46 - term39*term47;
//   const Scalar term49 = str_1_2*term40Xterm7;
//   const Scalar term50 = other_nx*str_1_0*term40Xterm7;
//   const Scalar term51 = other_ny*str_1_1*term40Xterm7;
//   const Scalar term52 = fy*term10*term33;
//   const Scalar term53 = n_y*term36;
//   const Scalar term54 = ny*term4 + term4*term53;
//   const Scalar term55 = -other_ny - term53;
//   const Scalar term56 = -term11*term54 - term12*term54 - term13*term54 - term45*term55 - term46*term55 - term47*term55;
//   
//   *residuals = -ref_intensity + term17*term25 + term26*(term27 + term28);
//   jacobian[0] = term29*(-fx*term21*term34 + term22*(term18*term32 + term19*term32 + term20*term32)) + term35*(-fy*term10*term34 + term16*(term3*term32 + term32*term8 + term32*term9));
//   jacobian[1] = term29*(term22*(term18*term38 + term19*term38 + term20*term38 + term39*term41 + term39*term42 + term39*term43) + term44*term48) + term35*(term16*(term3*term38 + term38*term8 + term38*term9 + term39*term49 + term39*term50 + term39*term51) + term48*term52);
//   jacobian[2] = term29*(term22*(term18*term54 + term19*term54 + term20*term54 + term41*term55 + term42*term55 + term43*term55) + term44*term56) + term35*(term16*(term3*term54 + term49*term55 + term50*term55 + term51*term55 + term54*term8 + term54*term9) + term52*term56);
// }

void PatchMatchStereoCPU::ComputeDepthMap(
    const Camera& reference_camera,
    const Image<u8>& reference_image,
    const SE3f& reference_image_tr_global,
    const Camera& stereo_camera,
    const Image<u8>& stereo_image,
    const SE3f& stereo_image_tr_global,
    Image<float>* inv_depth_map) {
  IDENTIFY_CAMERA2(
      reference_camera,
      stereo_camera,
      ComputeDepthMap_(
          _reference_camera,
          reference_image,
          reference_image_tr_global,
          _stereo_camera,
          stereo_image,
          stereo_image_tr_global,
          inv_depth_map));
}

template <class CameraT1, class CameraT2>
void PatchMatchStereoCPU::ComputeDepthMap_(
    const CameraT1& reference_camera,
    const Image<u8>& reference_image,
    const SE3f& reference_image_tr_global,
    const CameraT2& stereo_camera,
    const Image<u8>& stereo_image,
    const SE3f& stereo_image_tr_global,
    Image<float>* inv_depth_map) {
  // NOTE: Allocating buffers each time this is called is slow. Cache them for speedup.
//   Image<float> inv_depth_map(reference_camera.width(), reference_camera.height());
//   Image<float> inv_depth_map_2(reference_camera.width(), reference_camera.height());
  
  Image<Vec2f> normals(reference_camera.width(), reference_camera.height());
  Image<float> costs(reference_camera.width(), reference_camera.height());
  Image<float> lambda(reference_camera.width(), reference_camera.height());
  
  inv_depth_map->SetSize(reference_image.width(), reference_image.height());
  
  Matrix<float, 3, 4> stereo_tr_reference = (stereo_image_tr_global * reference_image_tr_global.inverse()).matrix3x4();
  
  // Initialize the depth and normals randomly, and compute initial matching costs.
  for (u32 y = context_radius_; y < reference_camera.height() - context_radius_; ++ y) {
    for (u32 x = context_radius_; x < reference_camera.width() - context_radius_; ++ x) {
      // Initialize random initial normals
      constexpr float kNormalRange = 0.5f;
      Vec2f normal_xy;
      normal_xy.x() = kNormalRange * ((0.0001f * (rand() % 10001)) - 0.5f);
      normal_xy.y() = kNormalRange * ((0.0001f * (rand() % 10001)) - 0.5f);
      float length = normal_xy.norm();
      if (length > max_normal_2d_length_) {
        normal_xy *= max_normal_2d_length_ / length;
      }
      normals(x, y) = normal_xy;
      
      // Initialize random initial depths
      float inv_min_depth = 1.0f / min_initial_depth_;
      float inv_max_depth = 1.0f / max_initial_depth_;
      
      const float inv_depth = inv_max_depth + (inv_min_depth - inv_max_depth) * ((0.0001f * (rand() % 10001)));
      (*inv_depth_map)(x, y) = inv_depth;
      
      // Initialize lambda
      lambda(x, y) = 1.02f;  // TODO: tune
      
      // Compute initial costs
      costs(x, y) = ComputeCosts(
          x, y,
          normal_xy,
          inv_depth,
          reference_camera,
          reference_image,
          stereo_tr_reference,
          stereo_camera,
          stereo_image,
          inv_depth_map->width(),
          inv_depth_map->height(),
          context_radius_,
          match_metric_);
    }
  }
  
  // Perform PatchMatch iterations
  for (int iteration = 0; iteration < iteration_count_; ++ iteration) {
    float step_range = std::pow(0.5f, std::min(iteration + 1, 6 /*TODO: Make parameter*/)) * (1.0f / min_initial_depth_ - 1.0f / max_initial_depth_);
    
//     // An imaginary diagonal with the dimensions (height x height) is shifted through the image.
//     for (u32 row = 0; row < reference_camera.width() + reference_camera.height() - 1; ++ row) {
//       int rowstart = std::max(0, static_cast<int>(reference_camera.height()) - 1 - static_cast<int>(row));
//       int rowend = (row < reference_camera.width()) ? reference_camera.height() : (2 * reference_camera.width() - row - 1);
//       for (int i = rowstart; i < rowend; ++ i) {
//         int x = row + i - (static_cast<int>(reference_camera.height()) - 1);
//         int y = reference_camera.height() - 1 - i;
//         
//         CHECK_GE(x, 0);
//         CHECK_GE(y, 0);
//         CHECK_LT(x, reference_camera.width());
//         CHECK_LT(y, reference_camera.height());
    
//             if (x >= context_radius_ && y >= context_radius_ &&
//             x < reference_camera.width() - context_radius_ && y < reference_camera.height() - context_radius_) {
    
    LOG(INFO) << "iteration " << iteration;
    
    bool even_iteration = iteration % 2 == 0;
    for (int y = even_iteration ? context_radius_ : (reference_camera.height() - context_radius_ - 1);
         y >= context_radius_ && y < static_cast<int>(reference_camera.height()) - context_radius_;
         y += even_iteration ? 1 : -1) {
      for (int x = even_iteration ? context_radius_ : (reference_camera.width() - context_radius_ - 1);
           x >= context_radius_ && x < static_cast<int>(reference_camera.width()) - context_radius_;
           x += even_iteration ? 1 : -1) {
        // Attempt mutation.
        float proposed_inv_depth = (*inv_depth_map)(x, y);
        proposed_inv_depth =
            max(kMinInvDepth,
                fabs(proposed_inv_depth + step_range *
                    ((0.0001f * (rand() % 10001)) - 0.5f)));
        
        constexpr float kRandomNormalRange = 0.5f;
        Vec2f proposed_normal = normals(x, y);
        proposed_normal.x() += kRandomNormalRange * ((0.0001f * (rand() % 10001)) - 0.5f);
        proposed_normal.y() += kRandomNormalRange * ((0.0001f * (rand() % 10001)) - 0.5f);
        float length = proposed_normal.norm();
        if (length > max_normal_2d_length_) {
          proposed_normal.x() *= max_normal_2d_length_ / length;
          proposed_normal.y() *= max_normal_2d_length_ / length;
        }
        
        // Test whether to accept the proposal
        float proposal_costs = ComputeCosts(
            x, y,
            proposed_normal,
            proposed_inv_depth,
            reference_camera,
            reference_image,
            stereo_tr_reference,
            stereo_camera,
            stereo_image,
            inv_depth_map->width(),
            inv_depth_map->height(),
            context_radius_,
            match_metric_);
        
        if (!std::isnan(proposal_costs) && !(proposal_costs >= costs(x, y))) {
          costs(x, y) = proposal_costs;
          normals(x, y) = proposed_normal;
          (*inv_depth_map)(x, y) = proposed_inv_depth;
        }
        
        
//         // Optimize locally.
//         float inv_depth = (*inv_depth_map)(x, y);
//         Vec2f normal_xy = normals(x, y);
//         Vec2f nxy = reference_camera.UnprojectFromPixelCenterConv(Vec2i(x, y)).template cast<float>().template topRows<2>();
//         
//         // Gauss-Newton update equation coefficients.
//         float H[3 + 2 + 1] = {0, 0, 0, 0, 0, 0};
//         float b[3] = {0, 0, 0};
//         
//         #pragma unroll
//         for (int dy = -context_radius_; dy <= context_radius_; ++ dy) {
//           #pragma unroll
//           for (int dx = -context_radius_; dx <= context_radius_; ++ dx) {
//             float raw_residual;
//             float jacobian[3];
//             
//             Vec2f other_nxy = reference_camera.UnprojectFromPixelCenterConv(Vec2i(x + dx, y + dy)).template cast<float>().template topRows<2>();
//             
//             ComputeResidualAndJacobian(
//                 projector.cx - 0.5f, projector.cy - 0.5f, projector.fx, projector.fy,
//                 inv_depth, normal_xy.x, normal_xy.y,
//                 nxy.x, nxy.y,
//                 other_nxy.x, other_nxy.y,
//                 reference_image(y + dy, x + dx),
//                 stereo_tr_reference.row0.x, stereo_tr_reference.row0.y, stereo_tr_reference.row0.z, stereo_tr_reference.row0.w,
//                 stereo_tr_reference.row1.x, stereo_tr_reference.row1.y, stereo_tr_reference.row1.z, stereo_tr_reference.row1.w,
//                 stereo_tr_reference.row2.x, stereo_tr_reference.row2.y, stereo_tr_reference.row2.z, stereo_tr_reference.row2.w,
//                 stereo_image,
//                 &raw_residual, jacobian);
//             
//             // Accumulate
//             b[0] += raw_residual * jacobian[0];
//             b[1] += raw_residual * jacobian[1];
//             b[2] += raw_residual * jacobian[2];
//             
//             H[0] += jacobian[0] * jacobian[0];
//             H[1] += jacobian[0] * jacobian[1];
//             H[2] += jacobian[0] * jacobian[2];
//             
//             H[3] += jacobian[1] * jacobian[1];
//             H[4] += jacobian[1] * jacobian[2];
//             
//             H[5] += jacobian[2] * jacobian[2];
//           }
//         }
//         
//         /*// TEST: Optimize inv_depth only
//         b[0] = b[0] / H[0];
//         inv_depth -= b[0];*/
//         
//         // Levenberg-Marquardt
//         const float kDiagLambda = lambda(x, y);
//         H[0] *= kDiagLambda;
//         H[3] *= kDiagLambda;
//         H[5] *= kDiagLambda;
//         
//         // Solve for the update using Cholesky decomposition
//         // (H[0]          )   (H[0] H[1] H[2])   (x[0])   (b[0])
//         // (H[1] H[3]     ) * (     H[3] H[4]) * (x[1]) = (b[1])
//         // (H[2] H[4] H[5])   (          H[5])   (x[2])   (b[2])
//         H[0] = sqrtf(H[0]);
//         
//         H[1] = 1.f / H[0] * H[1];
//         H[3] = sqrtf(H[3] - H[1] * H[1]);
//         
//         H[2] = 1.f / H[0] * H[2];
//         H[4] = 1.f / H[3] * (H[4] - H[1] * H[2]);
//         H[5] = sqrtf(H[5] - H[2] * H[2] - H[4] * H[4]);
//         
//         // Re-use b for the intermediate vector
//         b[0] = (b[0] / H[0]);
//         b[1] = (b[1] - H[1] * b[0]) / H[3];
//         b[2] = (b[2] - H[2] * b[0] - H[4] * b[1]) / H[5];
//         
//         // Re-use b for the delta vector
//         b[2] = (b[2] / H[5]);
//         b[1] = (b[1] - H[4] * b[2]) / H[3];
//         b[0] = (b[0] - H[1] * b[1] - H[2] * b[2]) / H[0];
//         
//         // Apply the update, sanitize normal if necessary
//         inv_depth -= b[0];
//         normal_xy.x -= b[1];
//         normal_xy.y -= b[2];
//         
//         float length = sqrtf(normal_xy.x * normal_xy.x + normal_xy.y * normal_xy.y);
//         if (length > max_normal_2d_length) {
//           normal_xy.x *= max_normal_2d_length / length;
//           normal_xy.y *= max_normal_2d_length / length;
//         }
//         
//         // Test whether the update lowers the cost
//         float proposal_costs = ComputeCosts<context_radius_>(
//             x, y,
//             normal_xy,
//             inv_depth,
//             unprojector,
//             reference_image,
//             stereo_tr_reference,
//             projector,
//             stereo_image,
//             inv_depth_map.width(),
//             inv_depth_map.height(),
//             match_metric);
//         
//         if (!::isnan(proposal_costs) && !(proposal_costs >= costs(x, y))) {
//           costs(x, y) = proposal_costs;
//           normals(x, y) = make_char2(normal_xy.x * 127.f, normal_xy.y * 127.f);  // TODO: in this and similar places: rounding?
//           inv_depth_map(x, y) = inv_depth;
//           
//           lambda(x, y) *= 0.5f;
//         } else {
//           lambda(x, y) *= 2.f;
//         }
        
        // Attempt propagations ("pulling" the values inwards).
        Vec2f nxy = reference_camera.UnprojectFromPixelCenterConv(Vec2i(x, y)).template cast<float>().template topRows<2>();
        
        for (int dy = -1; dy <= 1; ++ dy) {
          for (int dx = -1; dx <= 1; ++ dx) {
            if ((dx == 0 && dy == 0) ||
                (dx != 0 && dy != 0)) {
              continue;
            }
            
            // Compute inv_depth for propagating the pixel at (x + dx, y + dy) to the center pixel.
            Vec2f other_nxy = reference_camera.UnprojectFromPixelCenterConv(Vec2i(x + dx, y + dy)).template cast<float>().template topRows<2>();
            
            float other_inv_depth = (*inv_depth_map)(x + dx, y + dy);
            if (std::isnan(other_inv_depth)) {
              continue;
            }
            float other_depth = 1.f / other_inv_depth;
            
            Vec2f other_normal_xy = normals(x + dx, y + dy);
            float other_normal_z = -sqrtf(1.f - other_normal_xy.x() * other_normal_xy.x() - other_normal_xy.y() * other_normal_xy.y());
            
            float plane_d = (other_nxy.x() * other_depth) * other_normal_xy.x() + (other_nxy.y() * other_depth) * other_normal_xy.y() + other_depth * other_normal_z;
            
            float inv_depth = CalculatePlaneInvDepth2(plane_d, other_normal_xy, other_normal_z, nxy.x(), nxy.y());
            
            // Test whether to propagate
            float proposal_costs = ComputeCosts(
                x, y,
                other_normal_xy,
                inv_depth,
                reference_camera,
                reference_image,
                stereo_tr_reference,
                stereo_camera,
                stereo_image,
                inv_depth_map->width(),
                inv_depth_map->height(),
                context_radius_,
                match_metric_);
            
            if (!std::isnan(proposal_costs) && !(proposal_costs >= costs(x, y))) {
              costs(x, y) = proposal_costs;
              normals(x, y) = other_normal_xy;
              (*inv_depth_map)(x, y) = inv_depth;
            }
          }
        }
        
        // end of loop over all pixels
      }
    }
    
//     // DEBUG
//     inv_depth_map_->DownloadAsync(0, inv_depth_map);
//     static ImageDisplay debug_display;
//     debug_display.Update(*inv_depth_map, "depth debug", 0.f, 1.5f);
//     
//     Image<char2> normals_cpu(normals_->width(), normals_->height());
//     normals_->DownloadAsync(0, &normals_cpu);
//     Image<Vec3u8> normals_visualization(normals_->width(), normals_->height());
//     for (int y = 0; y < normals_->height(); ++ y) {
//       for (int x = 0; x < normals_->width(); ++ x) {
//         const char2& n = normals_cpu(x, y);
//         normals_visualization(x, y) = Vec3u8(n.x + 127, n.y + 127, 127);
//       }
//     }
//     static ImageDisplay debug_display_2;
//     debug_display_2.Update(normals_visualization, "normals debug");
//     
//     // Show point cloud
//     static shared_ptr<RenderDisplay> render_display = make_shared<RenderDisplay>();
//     static shared_ptr<RenderWindow> render_window = RenderWindow::CreateWindow("PatchMatch debug Visualization", 1280, 720, RenderWindow::API::kOpenGL, render_display);
//     
//     shared_ptr<Point3fCu8Cloud> cloud(new Point3fCu8Cloud());
//     inv_depth_map_->DownloadAsync(0, inv_depth_map);
//     IDENTIFY_CAMERA(reference_camera,
//         cloud->SetFromRGBDImage(*inv_depth_map, true, 0.f, reference_image, _reference_camera));
//     
//     render_display->SetUpDirection(Vec3f(0, 0, 1));
//     render_display->Update(cloud, "visualization cloud");
//     
//     std::getchar();
//     // END DEBUG
  }  // end of loop over all PatchMatch iterations
  
  // Outlier filtering
//   const float cost_threshold =
//       (1 + 2 * context_radius_) *
//       (1 + 2 * context_radius_) *
//       cost_threshold_per_pixel_;
  
  // TODO
//   PatchMatchFilterOutliersCUDA(
//       /*stream*/ 0,
//       context_radius_,
//       1.f / max_initial_depth_,  // NOTE: max_initial_depth_ is reused for max depth cutoff here!
//       required_range_min_depth_,
//       required_range_max_depth_,
//       reference_camera,
//       *reference_image_,
//       stereo_tr_reference,
//       stereo_camera,
//       stereo_image,
//       inv_depth_map_.get(),
//       inv_depth_map_2_.get(),
//       normals.get(),
//       costs.get(),
//       cost_threshold,
//       min_patch_variance_);
  
  // Outlier filtering by removing small connected components.
  RemoveSmallConnectedComponentsInInvDepthMap(
      0, min_component_size_,
      context_radius_, context_radius_,
      inv_depth_map->width() - 1 - context_radius_,
      inv_depth_map->height() - 1 - context_radius_,
      inv_depth_map);
}

}

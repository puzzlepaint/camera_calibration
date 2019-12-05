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

#pragma once

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"

namespace vis {

// The positions are specified in pixel-center origin convention.
template <typename CostFunction, typename T, typename Derived>
bool RefineFeatureBySymmetry(
    int num_samples,
    const vector<Vec2f>& samples,
    const Image<T>& image,
    int window_half_size,
    const MatrixBase<Derived>& position,
    const Mat3f& local_pattern_tr_pixel,
    const Mat3f& local_pixel_tr_pattern,
    Vec2f* out_position,
    float* final_cost,
    bool debug) {
  constexpr bool kDebug = false;
  
  Vec2f original_position = position;
  *out_position = position;
  
  // Transform the samples from pixel space to pattern space (with (0, 0) being
  // the feature whose position is estimated) using the initial homography estimate.
  vector<Vec2f> pattern_samples(samples.size());
  for (usize i = 0; i < samples.size(); ++ i) {
    pattern_samples[i] = Vec3f(local_pattern_tr_pixel * (window_half_size * samples[i]).homogeneous()).hnormalized();
  }
  
  // Optimize the homography locally that maps the local pattern coordinates to
  // the pixel coordinates.
  Mat3f pixel_tr_pattern_samples = local_pixel_tr_pattern;
  Mat3f local_to_global_mapping = Mat3f::Identity();
  local_to_global_mapping(0, 2) = position.x();
  local_to_global_mapping(1, 2) = position.y();
  pixel_tr_pattern_samples = local_to_global_mapping * pixel_tr_pattern_samples;
  // Normalize pixel_tr_pattern_samples such that its bottom-right element is one.
  pixel_tr_pattern_samples /= pixel_tr_pattern_samples(2, 2);
  
  constexpr int kDim = 8;
  Matrix<float, kDim, kDim> H;
  Matrix<float, kDim, 1> b;
  float lambda = -1;
  
  float last_step_squared_norm = numeric_limits<float>::infinity();
  
  if (kDebug) {
    LOG(INFO) << "Initial pixel_tr_pattern_samples:\n" << pixel_tr_pattern_samples;
  }
  
  constexpr int kMaxIterationCount = 30;
  for (int iteration = 0; iteration < kMaxIterationCount; ++ iteration) {
    float cost;
    if (!CostFunction::ComputeCornerRefinementCostAndJacobian(
        pixel_tr_pattern_samples, image, num_samples, pattern_samples,
        &H, &b, &cost)) {
      if (debug || kDebug) {
        LOG(WARNING) << "Corner refinement failed because a sample was outside the image";
      }
      return false;
    }
    if (kDebug) {
      LOG(INFO) << "cost: " << cost;
    }
    if (final_cost) {
      *final_cost = cost;
    }
    
    // Initialize lambda?
    if (lambda < 0) {
      lambda = 0.001f * (1.f / kDim) * H.diagonal().sum();
    }
    
    bool applied_update = false;
    for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
      Matrix<float, kDim, kDim> H_LM;
      H_LM.triangularView<Eigen::Upper>() = H.triangularView<Eigen::Upper>();
      H_LM.diagonal().array() += lambda;
      
      // Solve for the update.
      // NOTE: Not sure if using double is helpful here
      Eigen::Matrix<float, kDim, 1> x = H_LM.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(b.cast<double>()).cast<float>();
//       // TEST: use pseudoinverse to ensure Gauge fixing
//       H_LM.template triangularView<Eigen::Lower>() = H_LM.template triangularView<Eigen::Upper>().transpose();
//       Eigen::Matrix<float, kDim, 1> x =
//           H_LM.cast<double>().completeOrthogonalDecomposition().solve(b.cast<double>()).cast<float>();
      if (kDebug) {
        LOG(INFO) << "  x in LM iteration " << lm_iteration << ": " << x.transpose();
      }
      
      // Test whether the update improves the cost.
      Mat3f test_pixel_tr_pattern_samples;
      test_pixel_tr_pattern_samples(0, 0) = pixel_tr_pattern_samples(0, 0) - x(0);
      test_pixel_tr_pattern_samples(0, 1) = pixel_tr_pattern_samples(0, 1) - x(1);
      test_pixel_tr_pattern_samples(0, 2) = pixel_tr_pattern_samples(0, 2) - x(2);
      test_pixel_tr_pattern_samples(1, 0) = pixel_tr_pattern_samples(1, 0) - x(3);
      test_pixel_tr_pattern_samples(1, 1) = pixel_tr_pattern_samples(1, 1) - x(4);
      test_pixel_tr_pattern_samples(1, 2) = pixel_tr_pattern_samples(1, 2) - x(5);
      test_pixel_tr_pattern_samples(2, 0) = pixel_tr_pattern_samples(2, 0) - x(6);
      test_pixel_tr_pattern_samples(2, 1) = pixel_tr_pattern_samples(2, 1) - x(7);
      test_pixel_tr_pattern_samples(2, 2) = pixel_tr_pattern_samples(2, 2);
      float test_cost;
      if (!CostFunction::ComputeCornerRefinementCost(test_pixel_tr_pattern_samples, image, num_samples, pattern_samples, &test_cost)) {
        if (kDebug) {
          LOG(INFO) << "  CostFunction::ComputeCornerRefinementCost() failed, aborting.";
        }
        return false;
      }
      
      if (kDebug) {
        LOG(INFO) << "  test_cost: " << test_cost << ", cost: " << cost;
      }
      
      if (test_cost < cost) {
        if (final_cost) {
          *final_cost = test_cost;
        }
        last_step_squared_norm = Vec2f(x(2), x(5)).squaredNorm();  // using the translation only
        pixel_tr_pattern_samples = test_pixel_tr_pattern_samples;
        lambda *= 0.5f;
        applied_update = true;
        break;
      } else {
        lambda *= 2.f;
      }
    }
    
    if (applied_update) {
      // Since the element at (2, 2) is always 1, we can directly assign the
      // translation values instead of computing:
      // *out_position = (pixel_tr_pattern_samples * Vec3f(0, 0, 1)).hnormalized();
      out_position->x() = pixel_tr_pattern_samples(0, 2);
      out_position->y() = pixel_tr_pattern_samples(1, 2);
      if (kDebug) {
        LOG(INFO) << "New position: " << out_position->transpose();
      }
    } else {
      // Cannot find an update that improves the cost. Treat this as converged.
      if (kDebug) {
        LOG(INFO) << "Cannot find an update to improve the cost. Returning convergence (iteration " << iteration << ").";
      }
      return true;
    }
    
    // Check for divergence.
    if (fabs(original_position.x() - out_position->x()) >= window_half_size ||
        fabs(original_position.y() - out_position->y()) >= window_half_size) {
      // The result is probably not the originally intended corner,
      // since it is not within the original search window.
      if (debug || kDebug) {
        LOG(WARNING) << "Corner refinement failed because the refined position left the original window";
      }
      return false;
    }
    
    // TODO: Why was this commented out? For parity with the CUDA version?
//     // Check for convergence.
//     if (x.squaredNorm() < numeric_limits<float>::epsilon()) {
//       return true;
//     }
  }
  
  // If the last step had a very small norm, we consider this to be converged,
  // even if technically it is not. But at this point, the optimization error is
  // likely so small that the true (detection) error is dominated by other factors.
  if (last_step_squared_norm < 1e-4f) {
    if (kDebug) {
      LOG(INFO) << "Number of iterations exceeded, but last update was tiny. Returning convergence.";
    }
    return true;
  }
  
  // No convergence after exceeding the maximum iteration count.
  if (debug || kDebug) {
    LOG(WARNING) << "Corner refinement failed because the optimization did not converge. last_step_squared_norm: " << last_step_squared_norm;
  }
  return false;
}


struct SymmetryCostFunction_GradientsXY {
  // The position is specified in pixel-center origin convention.
  template <typename T>
  static bool ComputeCornerRefinementCost(
      const Mat3f& pixel_tr_pattern_samples,
      const Image<T>& image,
      int num_samples,
      const vector<Vec2f>& pattern_samples,
      float* out_cost) {
    *out_cost = 0;
    for (int i = 0; i < num_samples; ++ i) {
      const Vec2f& sample = pattern_samples[i];
      
      // Get sample in one direction
      Vec2f sample_pos = Vec3f(pixel_tr_pattern_samples * sample.homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      Vec2f intensity_a = image.InterpolateBilinear(sample_pos);
      
      // Get sample in opposite direction
      sample_pos = Vec3f(pixel_tr_pattern_samples * (-1 * sample).homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      Vec2f intensity_b = image.InterpolateBilinear(sample_pos);
      
      Vec2f residual = intensity_a + intensity_b;
      *out_cost += residual.dot(residual);
    }
    return true;
  }
  
  // The position is specified in pixel-center origin convention.
  template <typename T>
  static bool ComputeCornerRefinementCostAndJacobian(
      const Mat3f& pixel_tr_pattern_samples,
      const Image<T>& image,
      int num_samples,
      const vector<Vec2f>& pattern_samples,
      Matrix<float, 8, 8>* H,
      Matrix<float, 8, 1>* b,
      float* out_cost) {
    H->triangularView<Eigen::Upper>().setZero();
    b->setZero();
    *out_cost = 0;
    
    for (int i = 0; i < num_samples; ++ i) {
      const Vec2f& sample = pattern_samples[i];
      
      // Get sample in one direction
      Vec2f sample_pos = Vec3f(pixel_tr_pattern_samples * sample.homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      Vec2f intensity_a;
      Mat2f gradient_a;
      image.InterpolateBilinearWithJacobian(sample_pos, &intensity_a, &gradient_a);
      
      // Get sample in opposite direction
      sample_pos = Vec3f(pixel_tr_pattern_samples * (-1 * sample).homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      Vec2f intensity_b;
      Mat2f gradient_b;
      image.InterpolateBilinearWithJacobian(sample_pos, &intensity_b, &gradient_b);
      
      // Compute residual and Jacobian.
      // 
      // The intensity calculation (of the half of the residual with positive sample offset) is:
      //   I((H00 * sample.x() + H01 * sample.y() + H02) / (H20 * sample.x() + H21 * sample.y() + 1),
      //     (H10 * sample.x() + H11 * sample.y() + H12) / (H20 * sample.x() + H21 * sample.y() + 1));
      // 
      // The corresponding Jacobian wrt. (H00, H01, H02, H10, H11, H12, H20, H21) is:
      //   grad_I * [J0, J1, J2, 0, 0, 0, J3, J4]
      //            [0, 0, 0, J0, J1, J2, J5, J6]
      // with:
      //   term0 = 1 / (H20 * sample.x() + H21 * sample.y() + 1)
      //   J0 = sample.x() * term0
      //   J1 = sample.y() * term0
      //   J2 =          1 * term0
      //   term1 = -1 / pow(H20*sample.x() + H21*sample.y() + 1, 2)
      //   term2 = (H00*sample.x() + H01*sample.y() + H02) * term1
      //   J3 = sample.x() * term2
      //   J4 = sample.y() * term2
      // 
      //   term3 = (H10*sample.x() + H11*sample.y() + H12) * term1
      //   J5 = sample.x() * term3
      //   J6 = sample.y() * term3
      // 
      // For the opposite sample (with negative sample offset), there is simply a minus in front of all sample.x() and sample.y().
      // 
      // In octave / MatLab, these formulas can be derived with:
      //   pkg load symbolic;
      //   syms H00 H01 H02 H10 H11 H12 H20 H21;
      //   syms sample_x sample_y;
      //   ccode(diff((H10 * sample_x + H11 * sample_y + H12) / (H20 * sample_x + H21 * sample_y +1), H21 <or other H elements here ...> ))
      Vec2f residual = intensity_a + intensity_b;
      
      const float& H00 = pixel_tr_pattern_samples(0, 0);
      const float& H01 = pixel_tr_pattern_samples(0, 1);
      const float& H02 = pixel_tr_pattern_samples(0, 2);
      const float& H10 = pixel_tr_pattern_samples(1, 0);
      const float& H11 = pixel_tr_pattern_samples(1, 1);
      const float& H12 = pixel_tr_pattern_samples(1, 2);
      const float& H20 = pixel_tr_pattern_samples(2, 0);
      const float& H21 = pixel_tr_pattern_samples(2, 1);
      
      // Sample in first direction
      Matrix<float, 2, 8> position_wrt_homography;
      float term0 = 1 / (H20 * sample.x() + H21 * sample.y() + 1);
      float term1 = -1 * term0 * term0;
      float term2 = (H00 * sample.x() + H01 * sample.y() + H02) * term1;
      float term3 = (H10 * sample.x() + H11 * sample.y() + H12) * term1;
      position_wrt_homography(0, 0) = sample.x() * term0;
      position_wrt_homography(0, 1) = sample.y() * term0;
      position_wrt_homography(0, 2) = term0;
      position_wrt_homography(0, 3) = 0;
      position_wrt_homography(0, 4) = 0;
      position_wrt_homography(0, 5) = 0;
      position_wrt_homography(0, 6) = sample.x() * term2;
      position_wrt_homography(0, 7) = sample.y() * term2;
      position_wrt_homography(1, 0) = 0;
      position_wrt_homography(1, 1) = 0;
      position_wrt_homography(1, 2) = 0;
      position_wrt_homography(1, 3) = position_wrt_homography(0, 0);
      position_wrt_homography(1, 4) = position_wrt_homography(0, 1);
      position_wrt_homography(1, 5) = position_wrt_homography(0, 2);
      position_wrt_homography(1, 6) = sample.x() * term3;
      position_wrt_homography(1, 7) = sample.y() * term3;
      
      Matrix<float, 2, 8> jacobian = gradient_a * position_wrt_homography;  // the second component will be added below
      
      // Sample in opposite direction
      float minus_sample_x = -1 * sample.x();
      float minus_sample_y = -1 * sample.y();
      term0 = 1 / (H20 * minus_sample_x + H21 * minus_sample_y + 1);
      term1 = -1 * term0 * term0;
      term2 = (H00*minus_sample_x + H01*minus_sample_y + H02) * term1;
      term3 = (H10*minus_sample_x + H11*minus_sample_y + H12) * term1;
      position_wrt_homography(0, 0) = minus_sample_x * term0;
      position_wrt_homography(0, 1) = minus_sample_y * term0;
      position_wrt_homography(0, 2) = term0;
      position_wrt_homography(0, 3) = 0;
      position_wrt_homography(0, 4) = 0;
      position_wrt_homography(0, 5) = 0;
      position_wrt_homography(0, 6) = minus_sample_x * term2;
      position_wrt_homography(0, 7) = minus_sample_y * term2;
      position_wrt_homography(1, 0) = 0;
      position_wrt_homography(1, 1) = 0;
      position_wrt_homography(1, 2) = 0;
      position_wrt_homography(1, 3) = position_wrt_homography(0, 0);
      position_wrt_homography(1, 4) = position_wrt_homography(0, 1);
      position_wrt_homography(1, 5) = position_wrt_homography(0, 2);
      position_wrt_homography(1, 6) = minus_sample_x * term3;
      position_wrt_homography(1, 7) = minus_sample_y * term3;
      
      jacobian += gradient_b * position_wrt_homography;
      
      // Accumulate update equation coefficients.
      float weight = 1;
      
      Matrix<float, 1, 8> jacobian_weighted = weight * jacobian.row(0);
      H->triangularView<Eigen::Upper>() += jacobian.row(0).transpose() * jacobian_weighted;
      *b += residual(0) * jacobian_weighted;
      
      jacobian_weighted = weight * jacobian.row(1);
      H->triangularView<Eigen::Upper>() += jacobian.row(1).transpose() * jacobian_weighted;
      *b += residual(1) * jacobian_weighted;
      
      *out_cost += residual.dot(residual);
    }
    return true;
  }
};


struct SymmetryCostFunction_SingleChannel {
  // The position is specified in pixel-center origin convention.
  template <typename T>
  static bool ComputeCornerRefinementCost(
      const Mat3f& pixel_tr_pattern_samples,
      const Image<T>& image,
      int num_samples,
      const vector<Vec2f>& pattern_samples,
      float* out_cost) {
    *out_cost = 0;
    for (int i = 0; i < num_samples; ++ i) {
      const Vec2f& sample = pattern_samples[i];
      
      // Get sample in one direction
      Vec2f sample_pos = Vec3f(pixel_tr_pattern_samples * sample.homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      float intensity_a = image.InterpolateBilinear(sample_pos);
      
      // Get sample in opposite direction
      sample_pos = Vec3f(pixel_tr_pattern_samples * (-1 * sample).homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      float intensity_b = image.InterpolateBilinear(sample_pos);
      
      float residual = intensity_a - intensity_b;
      *out_cost += residual * residual;
    }
    return true;
  }
  
  // The position is specified in pixel-center origin convention.
  template <typename T>
  static bool ComputeCornerRefinementCostAndJacobian(
      const Mat3f& pixel_tr_pattern_samples,
      const Image<T>& image,
      int num_samples,
      const vector<Vec2f>& pattern_samples,
      Matrix<float, 8, 8>* H,
      Matrix<float, 8, 1>* b,
      float* out_cost) {
    H->triangularView<Eigen::Upper>().setZero();
    b->setZero();
    *out_cost = 0;
    
    for (int i = 0; i < num_samples; ++ i) {
      const Vec2f& sample = pattern_samples[i];
      
      // Get sample in one direction
      Vec2f sample_pos = Vec3f(pixel_tr_pattern_samples * sample.homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      float intensity_a;
      Matrix<float, 1, 2> gradient_a;
      image.InterpolateBilinearWithJacobian(sample_pos, &intensity_a, &gradient_a);
      
      // Get sample in opposite direction
      sample_pos = Vec3f(pixel_tr_pattern_samples * (-1 * sample).homogeneous()).hnormalized();
      if (!image.ContainsPixelCenterConv(sample_pos)) {
        *out_cost = numeric_limits<float>::infinity();
        return false;
      }
      
      float intensity_b;
      Matrix<float, 1, 2> gradient_b;
      image.InterpolateBilinearWithJacobian(sample_pos, &intensity_b, &gradient_b);
      
      // Compute residual and Jacobian.
      // 
      // The intensity calculation (of the half of the residual with positive sample offset) is:
      //   I((H00 * sample.x() + H01 * sample.y() + H02) / (H20 * sample.x() + H21 * sample.y() + 1),
      //     (H10 * sample.x() + H11 * sample.y() + H12) / (H20 * sample.x() + H21 * sample.y() + 1));
      // 
      // The corresponding Jacobian wrt. (H00, H01, H02, H10, H11, H12, H20, H21) is:
      //   grad_I * [J0, J1, J2, 0, 0, 0, J3, J4]
      //            [0, 0, 0, J0, J1, J2, J5, J6]
      // with:
      //   term0 = 1 / (H20 * sample.x() + H21 * sample.y() + 1)
      //   J0 = sample.x() * term0
      //   J1 = sample.y() * term0
      //   J2 =          1 * term0
      //   term1 = -1 / pow(H20*sample.x() + H21*sample.y() + 1, 2)
      //   term2 = (H00*sample.x() + H01*sample.y() + H02) * term1
      //   J3 = sample.x() * term2
      //   J4 = sample.y() * term2
      // 
      //   term3 = (H10*sample.x() + H11*sample.y() + H12) * term1
      //   J5 = sample.x() * term3
      //   J6 = sample.y() * term3
      // 
      // For the opposite sample (with negative sample offset), there is simply a minus in front of all sample.x() and sample.y().
      // 
      // In octave / MatLab, these formulas can be derived with:
      //   pkg load symbolic;
      //   syms H00 H01 H02 H10 H11 H12 H20 H21;
      //   syms sample_x sample_y;
      //   ccode(diff((H10 * sample_x + H11 * sample_y + H12) / (H20 * sample_x + H21 * sample_y +1), H21 <or other H elements here ...> ))
      float residual = intensity_a - intensity_b;
      
      const float& H00 = pixel_tr_pattern_samples(0, 0);
      const float& H01 = pixel_tr_pattern_samples(0, 1);
      const float& H02 = pixel_tr_pattern_samples(0, 2);
      const float& H10 = pixel_tr_pattern_samples(1, 0);
      const float& H11 = pixel_tr_pattern_samples(1, 1);
      const float& H12 = pixel_tr_pattern_samples(1, 2);
      const float& H20 = pixel_tr_pattern_samples(2, 0);
      const float& H21 = pixel_tr_pattern_samples(2, 1);
      
      // Sample in first direction
      Matrix<float, 2, 8> position_wrt_homography;
      float term0 = 1 / (H20 * sample.x() + H21 * sample.y() + 1);
      float term1 = -1 * term0 * term0;
      float term2 = (H00 * sample.x() + H01 * sample.y() + H02) * term1;
      float term3 = (H10 * sample.x() + H11 * sample.y() + H12) * term1;
      position_wrt_homography(0, 0) = sample.x() * term0;
      position_wrt_homography(0, 1) = sample.y() * term0;
      position_wrt_homography(0, 2) = term0;
      position_wrt_homography(0, 3) = 0;
      position_wrt_homography(0, 4) = 0;
      position_wrt_homography(0, 5) = 0;
      position_wrt_homography(0, 6) = sample.x() * term2;
      position_wrt_homography(0, 7) = sample.y() * term2;
      position_wrt_homography(1, 0) = 0;
      position_wrt_homography(1, 1) = 0;
      position_wrt_homography(1, 2) = 0;
      position_wrt_homography(1, 3) = position_wrt_homography(0, 0);
      position_wrt_homography(1, 4) = position_wrt_homography(0, 1);
      position_wrt_homography(1, 5) = position_wrt_homography(0, 2);
      position_wrt_homography(1, 6) = sample.x() * term3;
      position_wrt_homography(1, 7) = sample.y() * term3;
      
      Matrix<float, 1, 8> jacobian = gradient_a * position_wrt_homography;  // the second component will be added below
      
      // Sample in opposite direction
      float minus_sample_x = -1 * sample.x();
      float minus_sample_y = -1 * sample.y();
      term0 = 1 / (H20 * minus_sample_x + H21 * minus_sample_y + 1);
      term1 = -1 * term0 * term0;
      term2 = (H00*minus_sample_x + H01*minus_sample_y + H02) * term1;
      term3 = (H10*minus_sample_x + H11*minus_sample_y + H12) * term1;
      position_wrt_homography(0, 0) = minus_sample_x * term0;
      position_wrt_homography(0, 1) = minus_sample_y * term0;
      position_wrt_homography(0, 2) = term0;
      position_wrt_homography(0, 3) = 0;
      position_wrt_homography(0, 4) = 0;
      position_wrt_homography(0, 5) = 0;
      position_wrt_homography(0, 6) = minus_sample_x * term2;
      position_wrt_homography(0, 7) = minus_sample_y * term2;
      position_wrt_homography(1, 0) = 0;
      position_wrt_homography(1, 1) = 0;
      position_wrt_homography(1, 2) = 0;
      position_wrt_homography(1, 3) = position_wrt_homography(0, 0);
      position_wrt_homography(1, 4) = position_wrt_homography(0, 1);
      position_wrt_homography(1, 5) = position_wrt_homography(0, 2);
      position_wrt_homography(1, 6) = minus_sample_x * term3;
      position_wrt_homography(1, 7) = minus_sample_y * term3;
      
      jacobian -= gradient_b * position_wrt_homography;
      
      // Accumulate update equation coefficients.
      float weight = 1;
      
      Matrix<float, 1, 8> jacobian_weighted = weight * jacobian;
      H->triangularView<Eigen::Upper>() += jacobian.transpose() * jacobian_weighted;
      *b += residual * jacobian_weighted;
      
      *out_cost += residual * residual;
    }
    return true;
  }
};

}

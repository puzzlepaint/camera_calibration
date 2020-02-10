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

// Given the initial alignment, initializes factor and bias such as to minimize
// the cost.
// The position is specified in pixel-center origin convention.
// 
// Derivation:
// goal: minimize:
//   sum_i (factor * p_i + bias - q_i)^2
// 
// d / d bias == 0:
//   2 * sum_i (factor * p_i + bias - q_i) == 0
//   sum_i (factor * p_i) + n * bias - sum_i q_i == 0
//   factor * sum_i p_i + n * bias - sum_i q_i == 0
//   with:
//     sum_p := sum_i p_i
//     sum_q := sum_i q_i
//   factor * sum_p + n * bias - sum_q == 0
//   bias = (1 / n) * (sum_q - factor * sum_p)
// 
// d / d factor == 0:
//   2 * sum_i ((factor * p_i + bias - q_i) * p_i) == 0
//   sum_i (factor * p_i * p_i + bias * p_i - q_i * p_i) == 0
//   sum_i (factor * p_i * p_i) + sum_i (bias * p_i) - sum_i (q_i * p_i) == 0
//     sum_qp := sum_i (q_i * p_i)
//   sum_i (factor * p_i * p_i) + sum_i (bias * p_i) == sum_qp
//   factor * sum_i (p_i * p_i) + bias * sum_p == sum_qp
//     sum_pp := sum_i (p_i * p_i)
//   factor * sum_pp + bias * sum_p == sum_qp
// 
// Solve: set in bias = ... in equation at the bottom and solve for factor:
// 
// factor * sum_pp + (1 / n) * (sum_q - factor * sum_p) * sum_p == sum_qp
// factor * sum_pp + (sum_p / n) * (sum_q - factor * sum_p) == sum_qp
// factor * sum_pp + (sum_p / n) * sum_q - factor * (sum_p^2 / n) == sum_qp
// factor * (sum_pp - (sum_p^2 / n)) == sum_qp - (sum_p / n) * sum_q
// 
// --> factor = (sum_qp - (sum_p / n) * sum_q) / (sum_pp - (sum_p^2 / n))
// --> bias = (1 / n) * (sum_q - factor * sum_p)
template <typename T>
static bool ComputeFeatureRefinementAgainstPatternInitialization(
    const Vec2f& position,
    float* factor,
    float* bias,
    int window_half_size,
    const Image<T>& image,
    int num_samples,
    const vector<Vec2f>& samples,
    const vector<float>& rendered_samples) {
  float sum_qp = 0;
  float sum_p = 0;
  float sum_q = 0;
  float sum_pp = 0;
  
  for (int i = 0; i < num_samples; ++ i) {
    Vec2f sample_pos = position + window_half_size * samples[i];
    if (!image.ContainsPixelCenterConv(sample_pos)) {
      return false;
    }
    
    float p = image.InterpolateBilinear(sample_pos);
    float q = rendered_samples[i];
    
    sum_qp += q * p;
    sum_p += p;
    sum_q += q;
    sum_pp += p * p;
  }
  
  float denominator = sum_pp - (sum_p * sum_p / num_samples);
  if (fabs(denominator) > 1e-6f) {
    *factor = (sum_qp - (sum_p / num_samples) * sum_q) / denominator;
  } else {
    *factor = 1.f;
  }
  *bias = (1.f / num_samples) * (sum_q - (*factor) * sum_p);
  return true;
}

// The position is specified in pixel-center origin convention.
template <typename T>
static bool ComputeFeatureRefinementAgainstPatternCostAndJacobian(
    const Vec2f& position,
    float factor,
    float bias,
    int window_half_size,
    const Image<T>& image,
    int num_samples,
    const vector<Vec2f>& samples,
    const vector<float>& rendered_samples,
    Matrix<float, 4, 4>* H,
    Matrix<float, 4, 1>* b,
    float* out_cost) {
  H->triangularView<Eigen::Upper>().setZero();
  b->setZero();
  *out_cost = 0;
  
  for (int i = 0; i < num_samples; ++ i) {
    Vec2f sample_pos = position + window_half_size * samples[i];
    if (!image.ContainsPixelCenterConv(sample_pos)) {
      *out_cost = numeric_limits<float>::infinity();
      return false;
    }
    
    float intensity;
    Matrix<double, 1, 2> gradient;
    image.InterpolateBilinearWithJacobian(sample_pos, &intensity, &gradient);
    
    // Compute residual and Jacobian.
    float raw_residual = factor * intensity + bias - rendered_samples[i];
    
    Matrix<float, 1, 4> jacobian;
    jacobian.leftCols<2>() = factor * gradient.cast<float>();  // Jac. wrt. position
    jacobian(2) = intensity;  // Jac. wrt. factor
    jacobian(3) = 1;  // Jac. wrt. bias
    
    // Accumulate update equation coefficients.
    constexpr float weight = 1;
    Matrix<float, 1, 4> jacobian_weighted = weight * jacobian;
    H->triangularView<Eigen::Upper>() += jacobian.transpose() * jacobian_weighted;
    *b += raw_residual * jacobian_weighted.transpose();
    
    *out_cost += raw_residual * raw_residual;
  }
  
//   // DEBUG: Visualize the transformed intensities.
//   constexpr int kVisualizationResolution = 40;
//   Image<float> visualization(kVisualizationResolution, kVisualizationResolution);
//   visualization.SetTo(0.f);
//   Image<int> visualization_count(kVisualizationResolution, kVisualizationResolution);
//   visualization_count.SetTo(0);
//   for (int i = 0; i < num_samples; ++ i) {
//     Vec2i pixel = (Vec2f::Constant(0.5f * kVisualizationResolution) + 0.5f * kVisualizationResolution * samples[i]).cast<int>();
//     if (pixel.x() < 0 || pixel.y() < 0 ||
//         pixel.x() >= visualization.width() || pixel.y() >= visualization.height()) {
//       // NOTE: Could clamp the coordinate into the image as an alternative
//       continue;
//     }
//     
//     Vec2f sample_pos = position + window_half_size * samples[i];
//     float intensity = image.InterpolateBilinear(sample_pos);
//     float transformed_intensity = factor * intensity + bias;
//     
//     visualization(pixel) += transformed_intensity;
//     visualization_count(pixel) += 1;
//   }
//   for (int y = 0; y < visualization.height(); ++ y) {
//     for (int x = 0; x < visualization.width(); ++ x) {
//       int count = visualization_count(x, y);
//       if (count > 0) {
//         visualization(x, y) /= count;
//       }
//     }
//   }
//   LOG(INFO) << "Debug: Showing current transformed intensities with factor: " << factor << ", bias: " << bias;
//   static ImageDisplay display;
//   display.Update(visualization, "Rasterized transformed intensities", 0.f, 16.f);
//   std::getchar();
//   // END DEBUG
  
  return true;
}

// The position is specified in pixel-center origin convention.
template <typename T>
static bool ComputeFeatureRefinementAgainstPatternCost(
    const Vec2f& position,
    float factor,
    float bias,
    int window_half_size,
    const Image<T>& image,
    int num_samples,
    const vector<Vec2f>& samples,
    const vector<float>& rendered_samples,
    float* out_cost) {
  *out_cost = 0;
  
  for (int i = 0; i < num_samples; ++ i) {
    Vec2f sample_pos = position + window_half_size * samples[i];
    if (!image.ContainsPixelCenterConv(sample_pos)) {
      *out_cost = numeric_limits<float>::infinity();
      return false;
    }
    
    float intensity = image.InterpolateBilinear(sample_pos);
    
    float residual = factor * intensity + bias - rendered_samples[i];
    
    *out_cost += residual * residual;
  }
  return true;
}

// The positions are specified in pixel-center origin convention.
template <typename T, typename DerivedA, typename DerivedB>
bool RefineFeatureByMatching(
    int num_samples,
    const vector<Vec2f>& samples,
    const Image<T>& image,
    int window_half_size,
    const MatrixBase<DerivedA>& position,
    const MatrixBase<DerivedB>& local_pattern_tr_pixel,
    const PatternData& pattern,
    Vec2f* out_position,
    float* final_cost,
    bool debug) {
  constexpr int kNumAntiAliasSamples = 16;
  
  // Using the local homography, render the known pattern at the samples. Each
  // sample defines a pixel offset from the feature position.
  vector<float> rendered_samples(num_samples);
  for (int i = 0; i < num_samples; ++ i) {
    float sum = 0;
    for (int s = 0; s < kNumAntiAliasSamples; ++ s) {
      // Samples spread in [-0.5, 0.5], i.e., within the range of one pixel.
      Vec2f pixel_offset = window_half_size * samples[i] +
                           Vec2f(-0.5 + 1 / 8.f + 1 / 4.f * (s % 4),
                                 -0.5 + 1 / 8.f + 1 / 4.f * (s / 4));
      Vec2f pattern_offset = Vec3f(local_pattern_tr_pixel * pixel_offset.homogeneous()).hnormalized();
      sum += pattern.PatternIntensityAt(pattern_offset);
    }
    // Normalization by kNumSubpixelSamples is not necessary here since an
    // affine intensity transformation is optimized for later.
    rendered_samples[i] = sum;
  }
  
//   // DEBUG: Visualize the rendered samples.
//   constexpr int kVisualizationResolution = 40;
//   Image<float> visualization(kVisualizationResolution, kVisualizationResolution);
//   visualization.SetTo(0.f);
//   Image<int> visualization_count(kVisualizationResolution, kVisualizationResolution);
//   visualization_count.SetTo(0);
//   for (int i = 0; i < num_samples; ++ i) {
//     Vec2i pixel = (Vec2f::Constant(0.5f * kVisualizationResolution) + 0.5f * kVisualizationResolution * samples[i]).cast<int>();
//     if (pixel.x() < 0 || pixel.y() < 0 ||
//         pixel.x() >= visualization.width() || pixel.y() >= visualization.height()) {
//       // NOTE: Could clamp the coordinate into the image as an alternative
//       continue;
//     }
//     
//     visualization(pixel) += rendered_samples[i];
//     visualization_count(pixel) += 1;
//   }
//   for (int y = 0; y < visualization.height(); ++ y) {
//     for (int x = 0; x < visualization.width(); ++ x) {
//       int count = visualization_count(x, y);
//       if (count > 0) {
//         visualization(x, y) /= count;
//       }
//     }
//   }
//   static ImageDisplay display;
//   display.Update(visualization, "Rasterized sample rendering", 0.f, static_cast<float>(kNumAntiAliasSamples));
//   std::getchar();
//   // END DEBUG
  
  // Initialize factor and bias.
  float factor;
  float bias;
  if (!ComputeFeatureRefinementAgainstPatternInitialization(
          position, &factor, &bias,
          window_half_size, image, num_samples,
          samples, rendered_samples)) {
    if (debug) {
      LOG(WARNING) << "Corner refinement failed because a sample was outside the image";
    }
    return false;
  }
  
  // Use direct image alignment to align the rendered
  // samples with the measured image. For simplicity, optimize for x/y offset
  // only instead of for the whole homography.
  Vec2f original_position = position;
  *out_position = position;
  
  Matrix<float, 4, 4> H;
  Matrix<float, 4, 1> b;
  float lambda = -1;
  
  float last_step_squared_norm = numeric_limits<float>::infinity();
  
  bool converged = false;
  constexpr int kMaxIterationCount = 50;
  for (int iteration = 0; iteration < kMaxIterationCount; ++ iteration) {
    float cost;
    if (!ComputeFeatureRefinementAgainstPatternCostAndJacobian(
        *out_position, factor, bias, window_half_size, image, num_samples, samples, rendered_samples,
        &H, &b, &cost)) {
      if (debug) {
        LOG(WARNING) << "Corner refinement failed because a sample was outside the image";
      }
      return false;
    }
    if (final_cost) {
      *final_cost = cost;
    }
    
    // Initialize lambda?
    if (lambda < 0) {
      lambda = 0.001f * 0.5f * (H(0, 0) + H(1, 1) + H(2, 2) + H(3, 3));
    }
    
    bool applied_update = false;
    for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
      Matrix<float, 4, 4> H_LM;
      H_LM.triangularView<Eigen::Upper>() = H.triangularView<Eigen::Upper>();
      H_LM.diagonal().array() += lambda;
      
      // Solve for the update.
      // NOTE: Not sure if using double is helpful here
      Eigen::Matrix<float, 4, 1> x = H_LM.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(b.cast<double>()).cast<float>();
      
      // Test whether the update improves the cost.
      Vec2f test_position = *out_position - x.topRows<2>();
      float test_factor = factor - x(2);
      float test_bias = bias - x(3);
      
      float test_cost;
      if (!ComputeFeatureRefinementAgainstPatternCost(test_position, test_factor, test_bias, window_half_size, image, num_samples, samples, rendered_samples, &test_cost)) {
        return false;
      }
      
      if (test_cost < cost) {
        if (final_cost) {
          *final_cost = test_cost;
        }
        last_step_squared_norm = x.squaredNorm();
        *out_position = test_position;
        factor = test_factor;
        bias = test_bias;
        lambda *= 0.5f;
        applied_update = true;
        break;
      } else {
        lambda *= 2.f;
      }
    }
    
    if (!applied_update) {
      // Cannot find an update that improves the cost. Treat this as converged.
      converged = true;
      break;
    }
    
    // Check for divergence.
    if (fabs(original_position.x() - out_position->x()) >= window_half_size ||
        fabs(original_position.y() - out_position->y()) >= window_half_size) {
      // The result is probably not the originally intended corner,
      // since it is not within the original search window.
      if (debug) {
        LOG(WARNING) << "Corner refinement failed because the refined position left the original window";
      }
      return false;
    }
  }
  
  // If the last step had a very small norm, we consider this to be converged,
  // even if technically it is not. But at this point, the error is likely so
  // small that the true error is dominated by other factors.
  if (last_step_squared_norm < 1e-8) {
    converged = true;
  }
  
  // No convergence after exceeding the maximum iteration count?
  if (!converged) {
    if (debug) {
      LOG(WARNING) << "Corner refinement failed because the optimization did not converge";
    }
    return false;
  }
  
  // Bad factor in affine intensity transformation?
  if (factor <= 0) {
    if (debug) {
      LOG(WARNING) << "Corner refinement failed because of negative affine factor (" << factor << ")";
    }
    return false;
  }
  
  return true;
}

}

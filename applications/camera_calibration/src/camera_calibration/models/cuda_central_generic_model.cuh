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

#include <cuda_runtime.h>
#include <libvis/cuda/cuda_buffer.cuh>
#include <libvis/libvis.h>
#include <math_constants.h>

#include "camera_calibration/cuda/cuda_matrix.cuh"
#include "camera_calibration/models/cuda_camera_model.cuh"

namespace vis {

// TODO: Move the functions below to a better place

__forceinline__ __device__ float3 NormalizeNoCheck(const float3& input) {
  float length = sqrtf(input.x * input.x + input.y * input.y + input.z * input.z);
  return make_float3(input.x / length, input.y / length, input.z / length);
}

__forceinline__ __device__ float3 CrossProduct(const float3& a, const float3& b) {
  return make_float3(a.y * b.z - b.y * a.z,
                     b.x * a.z - a.x * b.z,
                     a.x * b.y - b.x * a.y);
}

struct CUDALineTangents {
  float3 t1;
  float3 t2;
};

/// Computes tangent vectors to the direction which are used to define the
/// local parametrization.
__forceinline__ __device__ void ComputeTangentsForDirectionOrLine(
    const float3& direction,
    CUDALineTangents* tangents) {
  tangents->t1 = NormalizeNoCheck(CrossProduct(direction, (fabs(direction.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0)));
  tangents->t2 = CrossProduct(direction, tangents->t1);  // is already normalized
}

__forceinline__ __device__ void ApplyLocalUpdateToDirection(
    float3* direction,
    const CUDALineTangents& tangents,
    float offset1,
    float offset2) {
  // Projection onto the sphere in the direction towards the origin.
  // NOTE: We could theoretically divide by sqrt(1 + offset1 * offset1 + offset2 * offset2) to normalize here,
  //       but we do a full renormalization to prevent error accumulation.
  *direction = NormalizeNoCheck(
      make_float3(direction->x + offset1 * tangents.t1.x + offset2 * tangents.t2.x,
                  direction->y + offset1 * tangents.t1.y + offset2 * tangents.t2.y,
                  direction->z + offset1 * tangents.t1.z + offset2 * tangents.t2.z));
}



// opcount = 486
template <typename Scalar>
__forceinline__ __device__ void CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(Scalar frac_x, Scalar frac_y, float3 p[4][4], float3* result, CUDAMatrix<Scalar, 3, 2>* dresult_dxy) {
  const Scalar term0 = 0.166666666666667f*frac_y;
  const Scalar term1 = -term0 + 0.666666666666667f;
  const Scalar term2 = (frac_y - 4) * (frac_y - 4);
  const Scalar term3 = (frac_x - 4) * (frac_x - 4);
  const Scalar term4 = 0.166666666666667f*frac_x;
  const Scalar term5 = -term4 + 0.666666666666667f;
  const Scalar term6 = p[0][0].x*term5;
  const Scalar term7 = (frac_x - 3) * (frac_x - 3);
  const Scalar term8 = term4 - 0.5f;
  const Scalar term9 = p[0][3].x*term8;
  const Scalar term10 = frac_x * frac_x;
  const Scalar term11 = 0.5*frac_x*term10;
  const Scalar term12 = 19.5f*frac_x - 5.5*term10 + term11 - 21.8333333333333f;
  const Scalar term13 = -16*frac_x + 5*term10 - term11 + 16.6666666666667f;
  const Scalar term14 = p[0][1].x*term12 + p[0][2].x*term13 + term3*term6 + term7*term9;
  const Scalar term15 = term14*term2;
  const Scalar term16 = term1*term15;
  const Scalar term17 = term0 - 0.5f;
  const Scalar term18 = (frac_y - 3) * (frac_y - 3);
  const Scalar term19 = p[3][0].x*term5;
  const Scalar term20 = p[3][3].x*term8;
  const Scalar term21 = p[3][1].x*term12 + p[3][2].x*term13 + term19*term3 + term20*term7;
  const Scalar term22 = term18*term21;
  const Scalar term23 = term17*term22;
  const Scalar term24 = frac_y * frac_y;
  const Scalar term25 = 0.5f*frac_y*term24;
  const Scalar term26 = -16*frac_y + 5*term24 - term25 + 16.6666666666667f;
  const Scalar term27 = p[2][0].x*term5;
  const Scalar term28 = p[2][3].x*term8;
  const Scalar term29 = p[2][1].x*term12 + p[2][2].x*term13 + term27*term3 + term28*term7;
  const Scalar term30 = term26*term29;
  const Scalar term31 = 19.5f*frac_y - 5.5f*term24 + term25 - 21.8333333333333f;
  const Scalar term32 = p[1][0].x*term5;
  const Scalar term33 = p[1][3].x*term8;
  const Scalar term34 = p[1][1].x*term12 + p[1][2].x*term13 + term3*term32 + term33*term7;
  const Scalar term35 = term31*term34;
  const Scalar term36 = term16 + term23 + term30 + term35;
  const Scalar term37 = p[0][0].y*term5;
  const Scalar term38 = p[0][3].y*term8;
  const Scalar term39 = p[0][1].y*term12 + p[0][2].y*term13 + term3*term37 + term38*term7;
  const Scalar term40 = term2*term39;
  const Scalar term41 = term1*term40;
  const Scalar term42 = p[3][0].y*term5;
  const Scalar term43 = p[3][3].y*term8;
  const Scalar term44 = p[3][1].y*term12 + p[3][2].y*term13 + term3*term42 + term43*term7;
  const Scalar term45 = term18*term44;
  const Scalar term46 = term17*term45;
  const Scalar term47 = p[2][0].y*term5;
  const Scalar term48 = p[2][3].y*term8;
  const Scalar term49 = p[2][1].y*term12 + p[2][2].y*term13 + term3*term47 + term48*term7;
  const Scalar term50 = term26*term49;
  const Scalar term51 = p[1][0].y*term5;
  const Scalar term52 = p[1][3].y*term8;
  const Scalar term53 = p[1][1].y*term12 + p[1][2].y*term13 + term3*term51 + term52*term7;
  const Scalar term54 = term31*term53;
  const Scalar term55 = term41 + term46 + term50 + term54;
  const Scalar term56 = p[0][0].z*term5;
  const Scalar term57 = p[0][3].z*term8;
  const Scalar term58 = p[0][1].z*term12 + p[0][2].z*term13 + term3*term56 + term57*term7;
  const Scalar term59 = term2*term58;
  const Scalar term60 = term1*term59;
  const Scalar term61 = p[3][0].z*term5;
  const Scalar term62 = p[3][3].z*term8;
  const Scalar term63 = p[3][1].z*term12 + p[3][2].z*term13 + term3*term61 + term62*term7;
  const Scalar term64 = term18*term63;
  const Scalar term65 = term17*term64;
  const Scalar term66 = p[2][0].z*term5;
  const Scalar term67 = p[2][3].z*term8;
  const Scalar term68 = p[2][1].z*term12 + p[2][2].z*term13 + term3*term66 + term67*term7;
  const Scalar term69 = term26*term68;
  const Scalar term70 = p[1][0].z*term5;
  const Scalar term71 = p[1][3].z*term8;
  const Scalar term72 = p[1][1].z*term12 + p[1][2].z*term13 + term3*term70 + term7*term71;
  const Scalar term73 = term31*term72;
  const Scalar term74 = term60 + term65 + term69 + term73;
  const Scalar term75 = (term36 * term36) + (term55 * term55) + (term74 * term74);
  const Scalar term76 = 1.f / sqrt(term75);
  const Scalar term77 = term1*term2;
  const Scalar term78 = 0.166666666666667f*term3;
  const Scalar term79 = 0.166666666666667f*term7;
  const Scalar term80 = 1.5f*term10;
  const Scalar term81 = -11.0f*frac_x + term80 + 19.5f;
  const Scalar term82 = 10*frac_x - term80 - 16;
  const Scalar term83 = 2*frac_x;
  const Scalar term84 = term83 - 8;
  const Scalar term85 = term83 - 6;
  const Scalar term86 = term17*term18;
  const Scalar term87 = term26*(-p[2][0].x*term78 + p[2][1].x*term81 + p[2][2].x*term82 + p[2][3].x*term79 + term27*term84 + term28*term85) + term31*(-p[1][0].x*term78 + p[1][1].x*term81 + p[1][2].x*term82 + p[1][3].x*term79 + term32*term84 + term33*term85) + term77*(-p[0][0].x*term78 + p[0][1].x*term81 + p[0][2].x*term82 + p[0][3].x*term79 + term6*term84 + term85*term9) + term86*(-p[3][0].x*term78 + p[3][1].x*term81 + p[3][2].x*term82 + p[3][3].x*term79 + term19*term84 + term20*term85);
  const Scalar term88b = 1.f / sqrt(term75);
  const Scalar term88 = term88b * term88b * term88b;
  const Scalar term89 = (1.0f/2.0f)*term16 + (1.0f/2.0f)*term23 + (1.0f/2.0f)*term30 + (1.0f/2.0f)*term35;
  const Scalar term90 = (1.0f/2.0f)*term41 + (1.0f/2.0f)*term46 + (1.0f/2.0f)*term50 + (1.0f/2.0f)*term54;
  const Scalar term91 = term26*(-p[2][0].y*term78 + p[2][1].y*term81 + p[2][2].y*term82 + p[2][3].y*term79 + term47*term84 + term48*term85) + term31*(-p[1][0].y*term78 + p[1][1].y*term81 + p[1][2].y*term82 + p[1][3].y*term79 + term51*term84 + term52*term85) + term77*(-p[0][0].y*term78 + p[0][1].y*term81 + p[0][2].y*term82 + p[0][3].y*term79 + term37*term84 + term38*term85) + term86*(-p[3][0].y*term78 + p[3][1].y*term81 + p[3][2].y*term82 + p[3][3].y*term79 + term42*term84 + term43*term85);
  const Scalar term92 = (1.0f/2.0f)*term60 + (1.0f/2.0f)*term65 + (1.0f/2.0f)*term69 + (1.0f/2.0f)*term73;
  const Scalar term93 = term26*(-p[2][0].z*term78 + p[2][1].z*term81 + p[2][2].z*term82 + p[2][3].z*term79 + term66*term84 + term67*term85) + term31*(-p[1][0].z*term78 + p[1][1].z*term81 + p[1][2].z*term82 + p[1][3].z*term79 + term70*term84 + term71*term85) + term77*(-p[0][0].z*term78 + p[0][1].z*term81 + p[0][2].z*term82 + p[0][3].z*term79 + term56*term84 + term57*term85) + term86*(-p[3][0].z*term78 + p[3][1].z*term81 + p[3][2].z*term82 + p[3][3].z*term79 + term61*term84 + term62*term85);
  const Scalar term94 = 2*term88*(term87*term89 + term90*term91 + term92*term93);
  const Scalar term95 = 1.5f*term24;
  const Scalar term96 = 10*frac_y - term95 - 16;
  const Scalar term97 = term29*term96;
  const Scalar term98 = -11.0f*frac_y + term95 + 19.5f;
  const Scalar term99 = term34*term98;
  const Scalar term100 = 2*frac_y;
  const Scalar term101 = term1*(term100 - 8);
  const Scalar term102 = term101*term14;
  const Scalar term103 = term17*(term100 - 6);
  const Scalar term104 = term103*term21;
  const Scalar term105 = term49*term96;
  const Scalar term106 = term53*term98;
  const Scalar term107 = term101*term39;
  const Scalar term108 = term103*term44;
  const Scalar term109 = term68*term96;
  const Scalar term110 = term72*term98;
  const Scalar term111 = term101*term58;
  const Scalar term112 = term103*term63;
  const Scalar term113 = term88*(term89*(2*term102 + 2*term104 - 0.333333333333333f*term15 + 0.333333333333333f*term22 + 2*term97 + 2*term99) + term90*(2*term105 + 2*term106 + 2*term107 + 2*term108 - 0.333333333333333f*term40 + 0.333333333333333f*term45) + term92*(2*term109 + 2*term110 + 2*term111 + 2*term112 - 0.333333333333333f*term59 + 0.333333333333333f*term64));
  
  (*result).x = term36*term76;
  (*result).y = term55*term76;
  (*result).z = term74*term76;
  (*dresult_dxy)(0, 0) = -term36*term94 + term76*term87;
  (*dresult_dxy)(0, 1) = -term113*term36 + term76*(term102 + term104 - 0.166666666666667f*term15 + 0.166666666666667f*term22 + term97 + term99);
  (*dresult_dxy)(1, 0) = -term55*term94 + term76*term91;
  (*dresult_dxy)(1, 1) = -term113*term55 + term76*(term105 + term106 + term107 + term108 - 0.166666666666667f*term40 + 0.166666666666667f*term45);
  (*dresult_dxy)(2, 0) = -term74*term94 + term76*term93;
  (*dresult_dxy)(2, 1) = -term113*term74 + term76*(term109 + term110 + term111 + term112 - 0.166666666666667f*term59 + 0.166666666666667f*term64);
}

class CUDACentralGenericModel : public CUDACameraModel {
 friend class CentralGenericModel;
 public:
  template <bool have_replacement>
  __forceinline__ __device__ bool UnprojectWithJacobian(float x, float y, float3* result, CUDAMatrix<float, 3, 2>* dresult_dxy, int gx = -9999, int gy = -9999, float3* replacement_direction = nullptr) const {
    if (!IsInCalibratedArea(x, y)) {
      return false;
    }
    
    float2 grid_point = PixelCornerConvToGridPoint(x, y);
    grid_point.x += 2;
    grid_point.y += 2;
    
    int ix = ::floor(grid_point.x);
    int iy = ::floor(grid_point.y);
    
    float frac_x = grid_point.x - (ix - 3);
    float frac_y = grid_point.y - (iy - 3);
    
    float3 p[4][4];
    for (int y = 0; y < 4; ++ y) {
      for (int x = 0; x < 4; ++ x) {
        if (have_replacement && ix - 3 + x == gx && iy - 3 + y == gy) {
          p[y][x] = *replacement_direction;
        } else {
          p[y][x] = m_grid(iy - 3 + y, ix - 3 + x);
        }
      }
    }
    
    CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, p, result, dresult_dxy);
    (*dresult_dxy)(0, 0) = PixelScaleToGridScaleX((*dresult_dxy)(0, 0));
    (*dresult_dxy)(0, 1) = PixelScaleToGridScaleY((*dresult_dxy)(0, 1));
    (*dresult_dxy)(1, 0) = PixelScaleToGridScaleX((*dresult_dxy)(1, 0));
    (*dresult_dxy)(1, 1) = PixelScaleToGridScaleY((*dresult_dxy)(1, 1));
    (*dresult_dxy)(2, 0) = PixelScaleToGridScaleX((*dresult_dxy)(2, 0));
    (*dresult_dxy)(2, 1) = PixelScaleToGridScaleY((*dresult_dxy)(2, 1));
    return true;
  }
  
  __forceinline__ __device__ bool ProjectWithInitialEstimate(const float3& point, float2* result) const {
    // NOTE: We are not caring for the special case of ||point|| == 0 here,
    //       as the resulting NaN/Inf should lead to the point not being
    //       projected anyway.
    float length = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
    return ProjectDirectionWithInitialEstimate</*have_replacement*/ false>(
        make_float3(point.x / length, point.y / length, point.z / length),
        result);
  }
  
  /// NOTE: This function allows to replace one grid value at the given coordinate (gx, gy) with replacement_direction.
  template <bool have_replacement>
  __forceinline__ __device__ bool ProjectDirectionWithInitialEstimate(const float3& point_direction, float2* result, int gx = -9999, int gy = -9999, float3* replacement_direction = nullptr) const {
    // Levenberg-Marquardt optimization algorithm.
    constexpr float kEpsilon = 1e-10f;  // NOTE: This threshold has been increased compared to the CPU version, which uses 1e-12f.
    const usize kMaxIterations = 100;
    
    double lambda = -1;
    for (usize i = 0; i < kMaxIterations; ++i) {
      CUDAMatrix<float, 3, 2> ddxy_dxy;
      float3 direction;
      /*CHECK(*/ UnprojectWithJacobian<have_replacement>(result->x, result->y, &direction, &ddxy_dxy, gx, gy, replacement_direction);
      
      // (Non-squared) residuals.
      float dx = direction.x - point_direction.x;
      float dy = direction.y - point_direction.y;
      float dz = direction.z - point_direction.z;
      
      float cost = dx * dx + dy * dy + dz * dz;
      
      // Accumulate H and b.
      float H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0) + ddxy_dxy(2, 0) * ddxy_dxy(2, 0);
      float H_1_0_and_0_1 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1) + ddxy_dxy(2, 0) * ddxy_dxy(2, 1);
      float H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1) + ddxy_dxy(2, 1) * ddxy_dxy(2, 1);
      float b_0 = dx * ddxy_dxy(0, 0) + dy * ddxy_dxy(1, 0) + dz * ddxy_dxy(2, 0);
      float b_1 = dx * ddxy_dxy(0, 1) + dy * ddxy_dxy(1, 1) + dz * ddxy_dxy(2, 1);
      
      if (lambda < 0) {
        constexpr double kInitialLambdaFactor = 0.01;
        lambda = kInitialLambdaFactor * 0.5 * (H_0_0 + H_1_1);
      }
      
      bool update_accepted = false;
      for (int lm_iteration = 0; lm_iteration < 10; ++ lm_iteration) {
        double H_0_0_LM = H_0_0 + lambda;
        double H_1_1_LM = H_1_1 + lambda;
        
        // Solve the system.
        double x_1 = (b_1 - H_1_0_and_0_1 / H_0_0_LM * b_0) /
                     (H_1_1_LM - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0_LM);
        double x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0_LM;
        
//       // Perform in-place Cholesky decomposition of H
//       H_0_0 = sqrtf(H_0_0);
//       H_1_0_and_0_1 = H_1_0_and_0_1 / H_0_0;
//       H_1_1 = sqrtf(H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1);
//       
//       // Solve H * x = b for x.
//       //
//       // (H_0_0     0)   (H_0_0 H_0_1)   (x0)   (b0)
//       // (H_1_0 H_1_1) * (    0 H_1_1) * (x1) = (b1)
//       //
//       // Naming the result of the second multiplication y, we get:
//       //
//       // (H_0_0     0)   (y0)   (b0)
//       // (H_1_0 H_1_1) * (y1) = (b1)
//       // 
//       // and:
//       // 
//       // (H_0_0 H_0_1) * (x0) = (y0)
//       // (    0 H_1_1)   (x1) = (y1)
//       
//       float y_0 = b_0 / H_0_0;
//       float y_1 = (b_1 - H_1_0_and_0_1 * y_0) / H_1_1;
//       
//       float x_1 = y_1 / H_1_1;
//       float x_0 = (y_0 - H_1_0_and_0_1 * x_1) / H_0_0;
        
        // Compute the test state.
        float2 test_result = make_float2(result->x - x_0, result->y - x_1);
        
        // Compute the test cost.
#ifdef __CUDA_ARCH__
        float test_cost = CUDART_INF_F;
#else
        float test_cost = 999999;
#endif
        float3 test_direction;
        // TODO: The Jacobian is not needed here
        if (UnprojectWithJacobian<have_replacement>(test_result.x, test_result.y, &test_direction, &ddxy_dxy, gx, gy, replacement_direction)) {
          float test_dx = test_direction.x - point_direction.x;
          float test_dy = test_direction.y - point_direction.y;
          float test_dz = test_direction.z - point_direction.z;
          
          test_cost = test_dx * test_dx + test_dy * test_dy + test_dz * test_dz;
        }
        
        if (test_cost < cost) {
          lambda *= 0.5;
          *result = test_result;
          update_accepted = true;
          break;
        } else {
          lambda *= 2;
        }
      }
      
      if (!update_accepted) {
        // if (cost >= kEpsilon) {
        //   LOG(WARNING) << "No update found and not converged. Current state: " << result->transpose();
        // }
        
        return cost < kEpsilon;
      }
      
      if (cost < kEpsilon) {
        return true;
      }
    }
    
    // LOG(WARNING) << "Not converged. Current state: " << result->transpose();
    return false;
  }
  
  __forceinline__ __device__ int calibration_min_x() const {
    return m_calibration_min_x;
  }
  __forceinline__ __device__ int calibration_min_y() const {
    return m_calibration_min_y;
  }
  __forceinline__ __device__ int calibration_max_x() const {
    return m_calibration_max_x;
  }
  __forceinline__ __device__ int calibration_max_y() const {
    return m_calibration_max_y;
  }
  
  __forceinline__ __device__ bool IsInCalibratedArea(float x, float y) const {
    return x >= m_calibration_min_x && y >= m_calibration_min_y &&
           x < m_calibration_max_x + 1 && y < m_calibration_max_y + 1;
  }
  
  __forceinline__ __device__ bool is_central_camera_model() const {
    return true;
  }
  
  /// For x and y in [0, n_grid.width/height()[, returns the location of that
  /// grid point in pixel-corner coordinate origin convention.
  __forceinline__ __device__ float2 GridPointToPixelCornerConv(int x, int y) const {
    return make_float2(
        m_calibration_min_x + ((x - 1.f) / (m_grid.width() - 3.f)) * (m_calibration_max_x + 1 - m_calibration_min_x),
        m_calibration_min_y + ((y - 1.f) / (m_grid.height() - 3.f)) * (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  __forceinline__ __device__ float GridScaleToPixelScaleX(float length) const {
    return length * ((m_calibration_max_x + 1 - m_calibration_min_x) / (m_grid.width() - 3.f));
  }
  __forceinline__ __device__ float GridScaleToPixelScaleY(float length) const {
    return length * ((m_calibration_max_y + 1 - m_calibration_min_y) / (m_grid.height() - 3.f));
  }
  
  /// Inverse of GridPointToPixelCornerConv().
  __forceinline__ __device__ float2 PixelCornerConvToGridPoint(float x, float y) const {
    return make_float2(
        1.f + (m_grid.width() - 3.f) * (x - m_calibration_min_x) / (m_calibration_max_x + 1 - m_calibration_min_x),
        1.f + (m_grid.height() - 3.f) * (y - m_calibration_min_y) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  __forceinline__ __device__ float PixelScaleToGridScaleX(float length) const {
    return length * ((m_grid.width() - 3.f) / (m_calibration_max_x + 1 - m_calibration_min_x));
  }
  __forceinline__ __device__ float PixelScaleToGridScaleY(float length) const {
    return length * ((m_grid.height() - 3.f) / (m_calibration_max_y + 1 - m_calibration_min_y));
  }
  
  static const int IntrinsicsJacobianSize = 2 * 16;
  
  __forceinline__ __device__ bool ProjectionJacobianWrtIntrinsics(
      const float3& local_point,
      const float2& projected_pixel,
      float numerical_diff_delta,
      u32* grid_update_indices,
      float* intrinsic_jac_x,
      float* intrinsic_jac_y) {
    float length = sqrtf(local_point.x * local_point.x + local_point.y * local_point.y + local_point.z * local_point.z);
    float3 point_direction =
        make_float3(local_point.x / length, local_point.y / length, local_point.z / length);
    
    float2 grid_point = PixelCornerConvToGridPoint(projected_pixel.x, projected_pixel.y);
    
    int ix = ::floor(grid_point.x);
    int iy = ::floor(grid_point.y);
    if (!(ix >= 1 && iy >= 1 && ix + 2 < m_grid.width() && iy + 2 < m_grid.height())) {  // catches NaNs
      return false;
    }
    
    int local_index = 0;
    for (int y = 0; y < 4; ++ y) {
      int gy = iy + y - 1;
      // CHECK_GE(gy, 0);
      // CHECK_LT(gy, m_grid.height());
      
      for (int x = 0; x < 4; ++ x) {
        int gx = ix + x - 1;
        // CHECK_GE(gx, 0);
        // CHECK_LT(gx, m_grid.width());
        
        int sequential_index = gx + gy * m_grid.width();
        grid_update_indices[local_index + 0] = 2 * sequential_index + 0;
        grid_update_indices[local_index + 1] = 2 * sequential_index + 1;
        
        CUDALineTangents tangents;
        ComputeTangentsForDirectionOrLine(m_grid(gy, gx), &tangents);
        
        #pragma unroll
        for (int d = 0; d < 2; ++ d) {
          float3 test_direction = m_grid(gy, gx);
          ApplyLocalUpdateToDirection(
              &test_direction, tangents,
              (d == 0) ? numerical_diff_delta : 0,
              (d == 1) ? numerical_diff_delta : 0);
          
          float2 test_projected_pixel = projected_pixel;
          bool success = ProjectDirectionWithInitialEstimate</*have_replacement*/ true>(point_direction, &test_projected_pixel, gx, gy, &test_direction);
          
          if (!success) {
            return false;
          }
          
          intrinsic_jac_x[local_index + d] = (test_projected_pixel.x - projected_pixel.x) / numerical_diff_delta;
          intrinsic_jac_y[local_index + d] = (test_projected_pixel.y - projected_pixel.y) / numerical_diff_delta;
        }
        
        local_index += 2;
      }
    }
//     CHECK_EQ(local_index, grid_update_indices->size());
    return true;
  }
  
  const CUDABuffer_<float3>& grid() const { return m_grid; }
  
 private:
  /// Size of the camera images in pixels.
  int m_width;
  int m_height;
  
  /// Extents of the calibrated image area within the image bounds.
  int m_calibration_min_x;
  int m_calibration_min_y;
  int m_calibration_max_x;
  int m_calibration_max_y;
  
  CUDABuffer_<float3> m_grid;
};

}

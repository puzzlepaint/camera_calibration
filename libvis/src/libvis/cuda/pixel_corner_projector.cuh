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


#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include "libvis/cuda/cuda_buffer.cuh"
#include "libvis/logging.h"

namespace vis {

// Helper for point projection using the "pixel corner" origin convention, in CUDA code.
struct PixelCornerProjector_ {
  PixelCornerProjector_() = default;
  
  // Host-only copy (should not run on device since it would be inefficient)
  __host__ PixelCornerProjector_(const PixelCornerProjector_& other)
      : resolution_x(other.resolution_x),
        resolution_y(other.resolution_y),
        min_nx(other.min_nx),
        min_ny(other.min_ny),
        max_nx(other.max_nx),
        max_ny(other.max_ny),
        grid2(other.grid2),
        grid3(other.grid3),
        omega(other.omega),
        two_tan_omega_half(other.two_tan_omega_half),
        fx(other.fx), fy(other.fy), cx(other.cx), cy(other.cy),
        k1(other.k1), k2(other.k2), k3(other.k3), k4(other.k4),
        p1(other.p1), p2(other.p2), sx1(other.sx1), sy1(other.sy1),
        type(other.type), width(other.width), height(other.height) {}
  
//   __forceinline__ __device__ float2 CubicHermiteSpline(
//       const float2& p0,
//       const float2& p1,
//       const float2& p2,
//       const float2& p3,
//       const float x) const {
//     const float2 a = make_float2(
//         static_cast<float>(0.5) * (-p0.x + static_cast<float>(3.0) * p1.x - static_cast<float>(3.0) * p2.x + p3.x),
//         static_cast<float>(0.5) * (-p0.y + static_cast<float>(3.0) * p1.y - static_cast<float>(3.0) * p2.y + p3.y));
//     const float2 b = make_float2(
//         static_cast<float>(0.5) * (static_cast<float>(2.0) * p0.x - static_cast<float>(5.0) * p1.x + static_cast<float>(4.0) * p2.x - p3.x),
//         static_cast<float>(0.5) * (static_cast<float>(2.0) * p0.y - static_cast<float>(5.0) * p1.y + static_cast<float>(4.0) * p2.y - p3.y));
//     const float2 c = make_float2(
//         static_cast<float>(0.5) * (-p0.x + p2.x),
//         static_cast<float>(0.5) * (-p0.y + p2.y));
//     const float2 d = p1;
//     
//     // Use Horner's rule to evaluate the function value and its
//     // derivative.
//     
//     // f = ax^3 + bx^2 + cx + d
//     return make_float2(
//         d.x + x * (c.x + x * (b.x + x * a.x)),
//         d.y + x * (c.y + x * (b.y + x * a.y)));
//   }
  
  // opcount = 486
  template <typename Scalar>
  __forceinline__ __device__ void CentralGenericBSpline_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian(
      Scalar frac_x, Scalar frac_y, float3 p[4][4], float3* result,
      float* dresult_dxy_0_0, float* dresult_dxy_0_1,
      float* dresult_dxy_1_0, float* dresult_dxy_1_1,
      float* dresult_dxy_2_0, float* dresult_dxy_2_1) const {
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
    *dresult_dxy_0_0 = -term36*term94 + term76*term87;
    *dresult_dxy_0_1 = -term113*term36 + term76*(term102 + term104 - 0.166666666666667f*term15 + 0.166666666666667f*term22 + term97 + term99);
    *dresult_dxy_1_0 = -term55*term94 + term76*term91;
    *dresult_dxy_1_1 = -term113*term55 + term76*(term105 + term106 + term107 + term108 - 0.166666666666667f*term40 + 0.166666666666667f*term45);
    *dresult_dxy_2_0 = -term74*term94 + term76*term93;
    *dresult_dxy_2_1 = -term113*term74 + term76*(term109 + term110 + term111 + term112 - 0.166666666666667f*term59 + 0.166666666666667f*term64);
  }
  
  __forceinline__ __device__ bool IsInCalibratedImageArea(float x, float y) const {
    return x >= min_nx && y >= min_ny &&
           x < max_nx + 1 && y < max_ny + 1;
  }
  
  /// Inverse of GridPointToPixelCornerConv().
  __forceinline__ __device__ float2 PixelCornerConvToGridPoint(float x, float y) const {
    return make_float2(
        1.f + (grid3.width() - 3.f) * (x - min_nx) / (max_nx + 1 - min_nx),
        1.f + (grid3.height() - 3.f) * (y - min_ny) / (max_ny + 1 - min_ny));
  }
  
  __forceinline__ __device__ float PixelScaleToGridScaleX(float length) const {
    return length * ((grid3.width() - 3.f) / (max_nx + 1 - min_nx));
  }
  __forceinline__ __device__ float PixelScaleToGridScaleY(float length) const {
    return length * ((grid3.height() - 3.f) / (max_ny + 1 - min_ny));
  }
  
  __forceinline__ __device__ bool UnprojectFromPixelCornerConvWithJacobian(
      float x, float y, float3* result,
      float* dresult_dxy_0_0, float* dresult_dxy_0_1,
      float* dresult_dxy_1_0, float* dresult_dxy_1_1,
      float* dresult_dxy_2_0, float* dresult_dxy_2_1) const {
    if (!IsInCalibratedImageArea(x, y)) {
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
        p[y][x] = grid3(iy - 3 + y, ix - 3 + x);
      }
    }
    
    CentralGenericBSpline_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian(frac_x, frac_y, p, result, dresult_dxy_0_0, dresult_dxy_0_1, dresult_dxy_1_0, dresult_dxy_1_1, dresult_dxy_2_0, dresult_dxy_2_1);
    *dresult_dxy_0_0 = PixelScaleToGridScaleX(*dresult_dxy_0_0);
    *dresult_dxy_0_1 = PixelScaleToGridScaleY(*dresult_dxy_0_1);
    *dresult_dxy_1_0 = PixelScaleToGridScaleX(*dresult_dxy_1_0);
    *dresult_dxy_1_1 = PixelScaleToGridScaleY(*dresult_dxy_1_1);
    *dresult_dxy_2_0 = PixelScaleToGridScaleX(*dresult_dxy_2_0);
    *dresult_dxy_2_1 = PixelScaleToGridScaleY(*dresult_dxy_2_1);
    return true;
  }
  
  // Assumes that position.z > 0.
  __forceinline__ __device__ float2 Project(float3 position) const {
    // Pinhole camera.  NOTE: Commented out for shorter compile times. Find a better solution.
//     // if (type == Camera::Type::kPinholeCamera4f) {
//       return make_float2(fx * (position.x / position.z) + cx,
//                          fy * (position.y / position.z) + cy);
//     // }
    
    // RadtanCamera8d.  NOTE: Commented out for shorter compile times. Find a better solution.
//     // if (type == Camera::Type::kRadtanCamera8d) {
//       float2 undistorted_point = make_float2(position.x / position.z,
//                                              position.y / position.z);
//       const float mx2_u = undistorted_point.x * undistorted_point.x;
//       const float my2_u = undistorted_point.y * undistorted_point.y;
//       const float mxy_u = undistorted_point.x * undistorted_point.y;
//       const float rho2_u = mx2_u + my2_u;
//       const float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
//       float2 distorted_point = make_float2(undistorted_point.x + undistorted_point.x * rad_dist_u + 2.0f * p1 * mxy_u + p2 * (rho2_u + 2.0f * mx2_u),
//                                            undistorted_point.y + undistorted_point.y * rad_dist_u + 2.0f * p2 * mxy_u + p1 * (rho2_u + 2.0f * my2_u));
//       return make_float2(fx * distorted_point.x + cx,
//                          fy * distorted_point.y + cy);
//     // }
    
    // -------------------------------------------------------------------------
    
    // FovCamera5f.
//     if (type == 3 /*Camera::Type::FovCamera5f*/) {
    
//       float2 nxy = make_float2(position.x / position.z,
//                                position.y / position.z);
//     
//       const float r = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
//       const float kEpsilon = static_cast<float>(1e-6);
//       const float factor =
//           (r < kEpsilon) ?
//           1.f :
//           (atanf(r * two_tan_omega_half) / (r * omega));
//       return make_float2(fx * factor * nxy.x + cx,
//                          fy * factor * nxy.y + cy);
    
    // -------------------------------------------------------------------------
    
    // ThinPrismFisheyeCamera12d.
//     } else if (type == 3 /*Camera::Type::kThinPrismFisheyeCamera12d*/) {

//       float2 undistorted_nxy = make_float2(position.x / position.z,
//                                            position.y / position.z);
//       
//       float r = sqrtf(undistorted_nxy.x * undistorted_nxy.x + undistorted_nxy.y * undistorted_nxy.y);
//       
// //       if (r > radius_cutoff_) {
// //         return Eigen::Vector2f((undistorted_nxy.x < 0) ? -100 : 100,
// //                           (undistorted_nxy.y < 0) ? -100 : 100);
// //       }
//       
//       float fisheye_x, fisheye_y;
//       const float kEpsilon = static_cast<float>(1e-6);
//       if (r > kEpsilon) {
//         float theta_by_r = atanf(r) / r;
//         fisheye_x = theta_by_r * undistorted_nxy.x;
//         fisheye_y = theta_by_r * undistorted_nxy.y;
//       } else {
//         fisheye_x = undistorted_nxy.x;
//         fisheye_y = undistorted_nxy.y;
//       }
//       
//       const float x2 = fisheye_x * fisheye_x;
//       const float xy = fisheye_x * fisheye_y;
//       const float y2 = fisheye_y * fisheye_y;
//       const float r2 = x2 + y2;
//       const float r4 = r2 * r2;
//       const float r6 = r4 * r2;
//       const float r8 = r6 * r2;
//       
//       const float radial =
//           k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
//       const float dx = static_cast<float>(2) * p1 * xy + p2 * (r2 + static_cast<float>(2) * x2) + sx1 * r2;
//       const float dy = static_cast<float>(2) * p2 * xy + p1 * (r2 + static_cast<float>(2) * y2) + sy1 * r2;
//       
//       float nx = fisheye_x + radial * fisheye_x + dx;
//       float ny = fisheye_y + radial * fisheye_y + dy;
//       
//       return make_float2(fx * nx + cx,
//                          fy * ny + cy);

//     } else if (type == 0 /*Camera::Type::kInvalid*/) {
      
      // -----------------------------------------------------------------------
      
      // TODO: HACK for the CentralGenericBSplineModel from the camera_calibration project.
      //       There should instead be a sane possibility for passing in external projection models.
      
      // NOTE: We are not caring for the special case of ||position|| == 0 here,
      //       as the resulting NaN/Inf should not lead to the position being
      //       projected anyway.
      float length = sqrtf(position.x * position.x + position.y * position.y + position.z * position.z);
      float3 point_direction = make_float3(position.x / length, position.y / length, position.z / length);
      
      // Define initial estimate
      float2 result = make_float2(0.5f * (min_nx + max_nx + 1),
                                  0.5f * (min_ny + max_ny + 1));
      
      // Gauss-Newton optimization algorithm.
      constexpr float kEpsilon = 1e-10f;  // NOTE: This threshold has been increased compared to the CPU version, which uses 1e-12f.
      const usize kMaxIterations = 100;
      
      bool left_calibrated_area_before = false;
      (void) left_calibrated_area_before;
      bool converged = false;
      for (usize i = 0; i < kMaxIterations; ++i) {
        float ddxy_dxy_0_0;
        float ddxy_dxy_0_1;
        float ddxy_dxy_1_0;
        float ddxy_dxy_1_1;
        float ddxy_dxy_2_0;
        float ddxy_dxy_2_1;
        float3 direction;
        UnprojectFromPixelCornerConvWithJacobian(result.x, result.y, &direction, &ddxy_dxy_0_0, &ddxy_dxy_0_1, &ddxy_dxy_1_0, &ddxy_dxy_1_1, &ddxy_dxy_2_0, &ddxy_dxy_2_1);
        
        // (Non-squared) residuals.
        float dx = direction.x - point_direction.x;
        float dy = direction.y - point_direction.y;
        float dz = direction.z - point_direction.z;
        
        // Accumulate H and b.
        float H_0_0 = ddxy_dxy_0_0 * ddxy_dxy_0_0 + ddxy_dxy_1_0 * ddxy_dxy_1_0 + ddxy_dxy_2_0 * ddxy_dxy_2_0;
        float H_1_0_and_0_1 = ddxy_dxy_0_0 * ddxy_dxy_0_1 + ddxy_dxy_1_0 * ddxy_dxy_1_1 + ddxy_dxy_2_0 * ddxy_dxy_2_1;
        float H_1_1 = ddxy_dxy_0_1 * ddxy_dxy_0_1 + ddxy_dxy_1_1 * ddxy_dxy_1_1 + ddxy_dxy_2_1 * ddxy_dxy_2_1;
        float b_0 = dx * ddxy_dxy_0_0 + dy * ddxy_dxy_1_0 + dz * ddxy_dxy_2_0;
        float b_1 = dx * ddxy_dxy_0_1 + dy * ddxy_dxy_1_1 + dz * ddxy_dxy_2_1;
        
        // Solve the system and update the parameters.
        // Make sure that the matrix is positive definite
        // (instead of only semi-positive definite).
        constexpr float kDiagEpsilon = 1e-6f;
        H_0_0 += kDiagEpsilon;
        H_1_1 += kDiagEpsilon;
        
        // Perform in-place Cholesky decomposition of H
        H_0_0 = sqrtf(H_0_0);
        H_1_0_and_0_1 = H_1_0_and_0_1 / H_0_0;
        H_1_1 = sqrtf(H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1);
        
        // Solve H * x = b for x.
        //
        // (H_0_0     0)   (H_0_0 H_0_1)   (x0)   (b0)
        // (H_1_0 H_1_1) * (    0 H_1_1) * (x1) = (b1)
        //
        // Naming the result of the second multiplication y, we get:
        //
        // (H_0_0     0)   (y0)   (b0)
        // (H_1_0 H_1_1) * (y1) = (b1)
        // 
        // and:
        // 
        // (H_0_0 H_0_1) * (x0) = (y0)
        // (    0 H_1_1)   (x1) = (y1)
        
        float y_0 = b_0 / H_0_0;
        float y_1 = (b_1 - H_1_0_and_0_1 * y_0) / H_1_1;
        
        float x_1 = y_1 / H_1_1;
        float x_0 = (y_0 - H_1_0_and_0_1 * x_1) / H_0_0;
        
        result.x -= x_0;
        result.y -= x_1;
        
        // Check whether the estimated projection has left the calibrated image
        // area. This check should catch NaNs as well. We do not return false
        // immediately when this happens, but only if it happens for two iterations
        // in a row. This is because the Gauss-Newton step may overestimate the
        // step size and thus leave the image area slightly for points that project
        // close to the border of the image.
        if (!IsInCalibratedImageArea(result.x, result.y)) {
#ifdef __CUDA_ARCH__
          if (left_calibrated_area_before || ::isnan(result.x)) {
            return make_float2(-99999, -99999);
          }
#else
          LOG(FATAL) << "Must never be called.";
#endif
          left_calibrated_area_before = true;
          
          // Clamp projection back into the calibrated area for the next step.
          // The #ifdef avoids trouble with CUDA's min/max apparently not being
          // visible outside of nvcc.
#ifdef __CUDA_ARCH__
          result = make_float2(
              ::min(max_nx + 0.999f, ::max(result.x, static_cast<float>(min_nx))),
              ::min(max_ny + 0.999f, ::max(result.y, static_cast<float>(min_ny))));
#else
          LOG(FATAL) << "Must never be called.";
#endif
        } else {
          left_calibrated_area_before = false;
          if (dx * dx + dy * dy + dz * dz < kEpsilon) {
            converged = true;
            break;
          }
        }
      }
      
      return converged ? result : make_float2(-99999, -99999);
//     }  // -------------------------------------------------------------------
    
    // kNonParametricBicubicProjectionCamerad.  NOTE: Commented out for shorter compile times. Find a better solution.
//     // For nonparametric bicubic projection camera:
//     float2 undistorted_nxy = make_float2(position.x / position.z,
//                                          position.y / position.z);
//     
//     float fc = (undistorted_nxy.x - min_nx) * ((resolution_x - 1) / (max_nx - min_nx));
//     float fr = (undistorted_nxy.y - min_ny) * ((resolution_y - 1) / (max_ny - min_ny));
//     const int row = ::floor(fr);
//     const int col = ::floor(fc);
//     float r_frac = fr - row;
//     float c_frac = fc - col;
//     
//     int c[4];
//     int r[4];
//     for (int i = 0; i < 4; ++ i) {
//       c[i] = min(max(0, col - 1 + i), resolution_x - 1);
//       r[i] = min(max(0, row - 1 + i), resolution_y - 1);
//     }
//     
//     float2 f[4];
//     for (int wrow = 0; wrow < 4; ++ wrow) {
//       float2 p0 = grid(r[wrow], c[0]);
//       float2 p1 = grid(r[wrow], c[1]);
//       float2 p2 = grid(r[wrow], c[2]);
//       float2 p3 = grid(r[wrow], c[3]);
//       
//       f[wrow] = CubicHermiteSpline(p0, p1, p2, p3, c_frac);
//     }
//     
//     return CubicHermiteSpline(f[0], f[1], f[2], f[3], r_frac);
  }
  
  int resolution_x;
  int resolution_y;
  float min_nx;
  float min_ny;
  float max_nx;
  float max_ny;
  
  CUDABuffer_<float2> grid2;
  CUDABuffer_<float3> grid3;
  
  float omega;
  float two_tan_omega_half;
  float fx, fy, cx, cy;
  float k1, k2, k3, k4, p1, p2;
  float sx1, sy1;
  
  int type;  // from Camera::Type enum
  int width;
  int height;
};

}

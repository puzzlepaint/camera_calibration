// TODO: This code is currently unused. Update the implementation to work with ZNCC-based cost?
//       To have the residuals conform to what works well with Gauss-Newton, could use an affine brightness mapping (with optimized factor & bias parameters)
//       instead of the ZNCC computation, which should achieve the same affine invariance.

// // (Mostly) auto-generated function.
// typedef float Scalar;
// 
// // opcount = 243
// __forceinline__ __device__ void ComputeResidualAndJacobian(
//     Scalar cx, Scalar cy, Scalar fx, Scalar fy,
//     Scalar inv_depth, Scalar n_x, Scalar n_y,
//     Scalar nx, Scalar ny,
//     Scalar other_nx, Scalar other_ny,
//     Scalar ref_intensity,
//     Scalar str_0_0, Scalar str_0_1, Scalar str_0_2, Scalar str_0_3,
//     Scalar str_1_0, Scalar str_1_1, Scalar str_1_2, Scalar str_1_3,
//     Scalar str_2_0, Scalar str_2_1, Scalar str_2_2, Scalar str_2_3,
//     cudaTextureObject_t stereo_texture,
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
//   Scalar top_left = 255.0f * tex2D<float>(stereo_texture, ix + 0.5f, iy + 0.5f);
//   Scalar top_right = 255.0f * tex2D<float>(stereo_texture, ix + 1.5f, iy + 0.5f);
//   Scalar bottom_left = 255.0f * tex2D<float>(stereo_texture, ix + 0.5f, iy + 1.5f);
//   Scalar bottom_right = 255.0f * tex2D<float>(stereo_texture, ix + 1.5f, iy + 1.5f);
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
// 
// template <int kContextRadius>
// __global__ void PatchMatchOptimizationStepCUDAKernel(
//     int match_metric,
//     float max_normal_2d_length,
//     CUDAUnprojectionLookup2D_ unprojector,
//     CUDABuffer_<u8> reference_image,
//     cudaTextureObject_t reference_texture,
//     CUDAMatrix3x4 stereo_tr_reference,
//     PixelCornerProjector projector,
//     cudaTextureObject_t stereo_image,
//     CUDABuffer_<float> inv_depth_map,
//     CUDABuffer_<char2> normals,
//     CUDABuffer_<float> costs,
//     CUDABuffer_<curandState> random_states,
//     CUDABuffer_<float> lambda) {
//   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//   
//   if (x >= kContextRadius && y >= kContextRadius &&
//       x < inv_depth_map.width() - kContextRadius && y < inv_depth_map.height() - kContextRadius) {
//     float inv_depth = inv_depth_map(y, x);
//     char2 normal_xy_char = normals(y, x);
//     float2 normal_xy = make_float2(
//         normal_xy_char.x * (1 / 127.f), normal_xy_char.y * (1 / 127.f));
//     float2 nxy = unprojector.UnprojectPoint(x, y);
//     
//     // Gauss-Newton update equation coefficients.
//     float H[3 + 2 + 1] = {0, 0, 0, 0, 0, 0};
//     float b[3] = {0, 0, 0};
//     
//     #pragma unroll
//     for (int dy = -kContextRadius; dy <= kContextRadius; ++ dy) {
//       #pragma unroll
//       for (int dx = -kContextRadius; dx <= kContextRadius; ++ dx) {
//         float raw_residual;
//         float jacobian[3];
//         
//         float2 other_nxy = unprojector.UnprojectPoint(x + dx, y + dy);
//         
//         ComputeResidualAndJacobian(
//             projector.cx - 0.5f, projector.cy - 0.5f, projector.fx, projector.fy,
//             inv_depth, normal_xy.x, normal_xy.y,
//             nxy.x, nxy.y,
//             other_nxy.x, other_nxy.y,
//             reference_image(y + dy, x + dx),
//             stereo_tr_reference.row0.x, stereo_tr_reference.row0.y, stereo_tr_reference.row0.z, stereo_tr_reference.row0.w,
//             stereo_tr_reference.row1.x, stereo_tr_reference.row1.y, stereo_tr_reference.row1.z, stereo_tr_reference.row1.w,
//             stereo_tr_reference.row2.x, stereo_tr_reference.row2.y, stereo_tr_reference.row2.z, stereo_tr_reference.row2.w,
//             stereo_image,
//             &raw_residual, jacobian);
//         
//         // Accumulate
//         b[0] += raw_residual * jacobian[0];
//         b[1] += raw_residual * jacobian[1];
//         b[2] += raw_residual * jacobian[2];
//         
//         H[0] += jacobian[0] * jacobian[0];
//         H[1] += jacobian[0] * jacobian[1];
//         H[2] += jacobian[0] * jacobian[2];
//         
//         H[3] += jacobian[1] * jacobian[1];
//         H[4] += jacobian[1] * jacobian[2];
//         
//         H[5] += jacobian[2] * jacobian[2];
//       }
//     }
//     
//     /*// TEST: Optimize inv_depth only
//     b[0] = b[0] / H[0];
//     inv_depth -= b[0];*/
//     
//     // Levenberg-Marquardt
//     const float kDiagLambda = lambda(y, x);
//     H[0] *= kDiagLambda;
//     H[3] *= kDiagLambda;
//     H[5] *= kDiagLambda;
//     
//     // Solve for the update using Cholesky decomposition
//     // (H[0]          )   (H[0] H[1] H[2])   (x[0])   (b[0])
//     // (H[1] H[3]     ) * (     H[3] H[4]) * (x[1]) = (b[1])
//     // (H[2] H[4] H[5])   (          H[5])   (x[2])   (b[2])
//     H[0] = sqrtf(H[0]);
//     
//     H[1] = 1.f / H[0] * H[1];
//     H[3] = sqrtf(H[3] - H[1] * H[1]);
//     
//     H[2] = 1.f / H[0] * H[2];
//     H[4] = 1.f / H[3] * (H[4] - H[1] * H[2]);
//     H[5] = sqrtf(H[5] - H[2] * H[2] - H[4] * H[4]);
//     
//     // Re-use b for the intermediate vector
//     b[0] = (b[0] / H[0]);
//     b[1] = (b[1] - H[1] * b[0]) / H[3];
//     b[2] = (b[2] - H[2] * b[0] - H[4] * b[1]) / H[5];
//     
//     // Re-use b for the delta vector
//     b[2] = (b[2] / H[5]);
//     b[1] = (b[1] - H[4] * b[2]) / H[3];
//     b[0] = (b[0] - H[1] * b[1] - H[2] * b[2]) / H[0];
//     
//     // Apply the update, sanitize normal if necessary
//     inv_depth -= b[0];
//     normal_xy.x -= b[1];
//     normal_xy.y -= b[2];
//     
//     float length = sqrtf(normal_xy.x * normal_xy.x + normal_xy.y * normal_xy.y);
//     if (length > max_normal_2d_length) {
//       normal_xy.x *= max_normal_2d_length / length;
//       normal_xy.y *= max_normal_2d_length / length;
//     }
//     
//     // Test whether the update lowers the cost
//     float proposal_costs = ComputeCosts<kContextRadius>(
//         x, y,
//         normal_xy,
//         inv_depth,
//         unprojector,
//         reference_image,
//         reference_texture,
//         stereo_tr_reference,
//         projector,
//         stereo_image,
//         match_metric,
//         0,  // TODO: Update if using this function again
//         CUDABuffer_<float>());  // TODO: Update if using this function again
//     
//     if (!::isnan(proposal_costs) && !(proposal_costs >= costs(y, x))) {
//       costs(y, x) = proposal_costs;
//       normals(y, x) = make_char2(normal_xy.x * 127.f, normal_xy.y * 127.f);  // TODO: in this and similar places: rounding?
//       inv_depth_map(y, x) = inv_depth;
//       
//       lambda(y, x) *= 0.5f;
//     } else {
//       lambda(y, x) *= 2.f;
//     }
//   }
// }
// 
// void PatchMatchOptimizationStepCUDA(
//     cudaStream_t stream,
//     int match_metric,
//     int context_radius,
//     float max_normal_2d_length,
//     cudaTextureObject_t reference_unprojection_lookup,
//     const CUDABuffer_<u8>& reference_image,
//     cudaTextureObject_t reference_texture,
//     const CUDAMatrix3x4& stereo_tr_reference,
//     const PixelCornerProjector_& stereo_camera,
//     const cudaTextureObject_t stereo_image,
//     CUDABuffer_<float>* inv_depth_map,
//     CUDABuffer_<char2>* normals,
//     CUDABuffer_<float>* costs,
//     CUDABuffer_<curandState>* random_states,
//     CUDABuffer_<float>* lambda) {
//   CHECK_CUDA_NO_ERROR();
//   COMPILE_INT_4_OPTIONS(context_radius, 5, 8, 10, 15, CUDA_AUTO_TUNE_2D(
//       PatchMatchOptimizationStepCUDAKernel<_context_radius>,
//       16, 16,
//       inv_depth_map->width(), inv_depth_map->height(),
//       0, stream,
//       /* kernel parameters */
//       match_metric,
//       max_normal_2d_length,
//       CUDAUnprojectionLookup2D_(reference_unprojection_lookup),
//       reference_image,
//       reference_texture,
//       stereo_tr_reference,
//       stereo_camera,
//       stereo_image,
//       stereo_camera.width(),
//       stereo_camera.height(),
//       *inv_depth_map,
//       *normals,
//       *costs,
//       *random_states,
//       *lambda));
//   cudaDeviceSynchronize();
//   CHECK_CUDA_NO_ERROR();
// }

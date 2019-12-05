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

#include <Eigen/Core>

// opcount = 486
template <typename Scalar>
inline void CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(Scalar frac_x, Scalar frac_y, Eigen::Matrix<Scalar, 3, 1> p[4][4], Eigen::Matrix<Scalar, 3, 1>* result, Eigen::Matrix<Scalar, 3, 2>* dresult_dxy) {
  const Scalar term0 = 0.166666666666667*frac_y;
  const Scalar term1 = -term0 + 0.666666666666667;
  const Scalar term2 = (frac_y - 4) * (frac_y - 4);
  const Scalar term3 = (frac_x - 4) * (frac_x - 4);
  const Scalar term4 = 0.166666666666667*frac_x;
  const Scalar term5 = -term4 + 0.666666666666667;
  const Scalar term6 = p[0][0].x()*term5;
  const Scalar term7 = (frac_x - 3) * (frac_x - 3);
  const Scalar term8 = term4 - 0.5;
  const Scalar term9 = p[0][3].x()*term8;
  const Scalar term10 = frac_x * frac_x;
  const Scalar term11 = 0.5*frac_x*term10;
  const Scalar term12 = 19.5*frac_x - 5.5*term10 + term11 - 21.8333333333333;
  const Scalar term13 = -16*frac_x + 5*term10 - term11 + 16.6666666666667;
  const Scalar term14 = p[0][1].x()*term12 + p[0][2].x()*term13 + term3*term6 + term7*term9;
  const Scalar term15 = term14*term2;
  const Scalar term16 = term1*term15;
  const Scalar term17 = term0 - 0.5;
  const Scalar term18 = (frac_y - 3) * (frac_y - 3);
  const Scalar term19 = p[3][0].x()*term5;
  const Scalar term20 = p[3][3].x()*term8;
  const Scalar term21 = p[3][1].x()*term12 + p[3][2].x()*term13 + term19*term3 + term20*term7;
  const Scalar term22 = term18*term21;
  const Scalar term23 = term17*term22;
  const Scalar term24 = frac_y * frac_y;
  const Scalar term25 = 0.5*frac_y*term24;
  const Scalar term26 = -16*frac_y + 5*term24 - term25 + 16.6666666666667;
  const Scalar term27 = p[2][0].x()*term5;
  const Scalar term28 = p[2][3].x()*term8;
  const Scalar term29 = p[2][1].x()*term12 + p[2][2].x()*term13 + term27*term3 + term28*term7;
  const Scalar term30 = term26*term29;
  const Scalar term31 = 19.5*frac_y - 5.5*term24 + term25 - 21.8333333333333;
  const Scalar term32 = p[1][0].x()*term5;
  const Scalar term33 = p[1][3].x()*term8;
  const Scalar term34 = p[1][1].x()*term12 + p[1][2].x()*term13 + term3*term32 + term33*term7;
  const Scalar term35 = term31*term34;
  const Scalar term36 = term16 + term23 + term30 + term35;
  const Scalar term37 = p[0][0].y()*term5;
  const Scalar term38 = p[0][3].y()*term8;
  const Scalar term39 = p[0][1].y()*term12 + p[0][2].y()*term13 + term3*term37 + term38*term7;
  const Scalar term40 = term2*term39;
  const Scalar term41 = term1*term40;
  const Scalar term42 = p[3][0].y()*term5;
  const Scalar term43 = p[3][3].y()*term8;
  const Scalar term44 = p[3][1].y()*term12 + p[3][2].y()*term13 + term3*term42 + term43*term7;
  const Scalar term45 = term18*term44;
  const Scalar term46 = term17*term45;
  const Scalar term47 = p[2][0].y()*term5;
  const Scalar term48 = p[2][3].y()*term8;
  const Scalar term49 = p[2][1].y()*term12 + p[2][2].y()*term13 + term3*term47 + term48*term7;
  const Scalar term50 = term26*term49;
  const Scalar term51 = p[1][0].y()*term5;
  const Scalar term52 = p[1][3].y()*term8;
  const Scalar term53 = p[1][1].y()*term12 + p[1][2].y()*term13 + term3*term51 + term52*term7;
  const Scalar term54 = term31*term53;
  const Scalar term55 = term41 + term46 + term50 + term54;
  const Scalar term56 = p[0][0].z()*term5;
  const Scalar term57 = p[0][3].z()*term8;
  const Scalar term58 = p[0][1].z()*term12 + p[0][2].z()*term13 + term3*term56 + term57*term7;
  const Scalar term59 = term2*term58;
  const Scalar term60 = term1*term59;
  const Scalar term61 = p[3][0].z()*term5;
  const Scalar term62 = p[3][3].z()*term8;
  const Scalar term63 = p[3][1].z()*term12 + p[3][2].z()*term13 + term3*term61 + term62*term7;
  const Scalar term64 = term18*term63;
  const Scalar term65 = term17*term64;
  const Scalar term66 = p[2][0].z()*term5;
  const Scalar term67 = p[2][3].z()*term8;
  const Scalar term68 = p[2][1].z()*term12 + p[2][2].z()*term13 + term3*term66 + term67*term7;
  const Scalar term69 = term26*term68;
  const Scalar term70 = p[1][0].z()*term5;
  const Scalar term71 = p[1][3].z()*term8;
  const Scalar term72 = p[1][1].z()*term12 + p[1][2].z()*term13 + term3*term70 + term7*term71;
  const Scalar term73 = term31*term72;
  const Scalar term74 = term60 + term65 + term69 + term73;
  const Scalar term75 = (term36 * term36) + (term55 * term55) + (term74 * term74);
  const Scalar term76 = 1. / sqrt(term75);
  const Scalar term77 = term1*term2;
  const Scalar term78 = 0.166666666666667*term3;
  const Scalar term79 = 0.166666666666667*term7;
  const Scalar term80 = 1.5*term10;
  const Scalar term81 = -11.0*frac_x + term80 + 19.5;
  const Scalar term82 = 10*frac_x - term80 - 16;
  const Scalar term83 = 2*frac_x;
  const Scalar term84 = term83 - 8;
  const Scalar term85 = term83 - 6;
  const Scalar term86 = term17*term18;
  const Scalar term87 = term26*(-p[2][0].x()*term78 + p[2][1].x()*term81 + p[2][2].x()*term82 + p[2][3].x()*term79 + term27*term84 + term28*term85) + term31*(-p[1][0].x()*term78 + p[1][1].x()*term81 + p[1][2].x()*term82 + p[1][3].x()*term79 + term32*term84 + term33*term85) + term77*(-p[0][0].x()*term78 + p[0][1].x()*term81 + p[0][2].x()*term82 + p[0][3].x()*term79 + term6*term84 + term85*term9) + term86*(-p[3][0].x()*term78 + p[3][1].x()*term81 + p[3][2].x()*term82 + p[3][3].x()*term79 + term19*term84 + term20*term85);
  const Scalar term88b = 1. / sqrt(term75);
  const Scalar term88 = term88b * term88b * term88b;
  const Scalar term89 = (1.0L/2.0L)*term16 + (1.0L/2.0L)*term23 + (1.0L/2.0L)*term30 + (1.0L/2.0L)*term35;
  const Scalar term90 = (1.0L/2.0L)*term41 + (1.0L/2.0L)*term46 + (1.0L/2.0L)*term50 + (1.0L/2.0L)*term54;
  const Scalar term91 = term26*(-p[2][0].y()*term78 + p[2][1].y()*term81 + p[2][2].y()*term82 + p[2][3].y()*term79 + term47*term84 + term48*term85) + term31*(-p[1][0].y()*term78 + p[1][1].y()*term81 + p[1][2].y()*term82 + p[1][3].y()*term79 + term51*term84 + term52*term85) + term77*(-p[0][0].y()*term78 + p[0][1].y()*term81 + p[0][2].y()*term82 + p[0][3].y()*term79 + term37*term84 + term38*term85) + term86*(-p[3][0].y()*term78 + p[3][1].y()*term81 + p[3][2].y()*term82 + p[3][3].y()*term79 + term42*term84 + term43*term85);
  const Scalar term92 = (1.0L/2.0L)*term60 + (1.0L/2.0L)*term65 + (1.0L/2.0L)*term69 + (1.0L/2.0L)*term73;
  const Scalar term93 = term26*(-p[2][0].z()*term78 + p[2][1].z()*term81 + p[2][2].z()*term82 + p[2][3].z()*term79 + term66*term84 + term67*term85) + term31*(-p[1][0].z()*term78 + p[1][1].z()*term81 + p[1][2].z()*term82 + p[1][3].z()*term79 + term70*term84 + term71*term85) + term77*(-p[0][0].z()*term78 + p[0][1].z()*term81 + p[0][2].z()*term82 + p[0][3].z()*term79 + term56*term84 + term57*term85) + term86*(-p[3][0].z()*term78 + p[3][1].z()*term81 + p[3][2].z()*term82 + p[3][3].z()*term79 + term61*term84 + term62*term85);
  const Scalar term94 = 2*term88*(term87*term89 + term90*term91 + term92*term93);
  const Scalar term95 = 1.5*term24;
  const Scalar term96 = 10*frac_y - term95 - 16;
  const Scalar term97 = term29*term96;
  const Scalar term98 = -11.0*frac_y + term95 + 19.5;
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
  const Scalar term113 = term88*(term89*(2*term102 + 2*term104 - 0.333333333333333*term15 + 0.333333333333333*term22 + 2*term97 + 2*term99) + term90*(2*term105 + 2*term106 + 2*term107 + 2*term108 - 0.333333333333333*term40 + 0.333333333333333*term45) + term92*(2*term109 + 2*term110 + 2*term111 + 2*term112 - 0.333333333333333*term59 + 0.333333333333333*term64));
  
  (*result)[0] = term36*term76;
  (*result)[1] = term55*term76;
  (*result)[2] = term74*term76;
  (*dresult_dxy)(0, 0) = -term36*term94 + term76*term87;
  (*dresult_dxy)(0, 1) = -term113*term36 + term76*(term102 + term104 - 0.166666666666667*term15 + 0.166666666666667*term22 + term97 + term99);
  (*dresult_dxy)(1, 0) = -term55*term94 + term76*term91;
  (*dresult_dxy)(1, 1) = -term113*term55 + term76*(term105 + term106 + term107 + term108 - 0.166666666666667*term40 + 0.166666666666667*term45);
  (*dresult_dxy)(2, 0) = -term74*term94 + term76*term93;
  (*dresult_dxy)(2, 1) = -term113*term74 + term76*(term109 + term110 + term111 + term112 - 0.166666666666667*term59 + 0.166666666666667*term64);
}

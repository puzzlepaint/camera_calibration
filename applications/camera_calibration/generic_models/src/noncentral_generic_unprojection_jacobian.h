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
#include <Eigen/Geometry>

// opcount = 803
template <typename Scalar>
inline void NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(Scalar frac_x, Scalar frac_y, Eigen::Matrix<Scalar, 6, 1> l[4][4], Eigen::ParametrizedLine<Scalar, 3>* result, Eigen::Matrix<Scalar, 6, 2>* dresult_dxy) {
  const Scalar term0 = 0.166666666666667*frac_y;
  const Scalar term1 = -term0 + 0.666666666666667;
  const Scalar term2 = (frac_y - 4) * (frac_y - 4);
  const Scalar term3 = (frac_x - 4) * (frac_x - 4);
  const Scalar term4 = 0.166666666666667*frac_x;
  const Scalar term5 = -term4 + 0.666666666666667;
  const Scalar term6 = l[0][0].x()*term5;
  const Scalar term7 = (frac_x - 3) * (frac_x - 3);
  const Scalar term8 = term4 - 0.5;
  const Scalar term9 = l[0][3].x()*term8;
  const Scalar term10 = frac_x * frac_x;
  const Scalar term11 = 0.5*term10*frac_x;
  const Scalar term12 = 19.5*frac_x - 5.5*term10 + term11 - 21.8333333333333;
  const Scalar term13 = -16*frac_x + 5*term10 - term11 + 16.6666666666667;
  const Scalar term14 = l[0][1].x()*term12 + l[0][2].x()*term13 + term3*term6 + term7*term9;
  const Scalar term15 = term14*term2;
  const Scalar term16 = term1*term15;
  const Scalar term17 = term0 - 0.5;
  const Scalar term18 = (frac_y - 3) * (frac_y - 3);
  const Scalar term19 = l[3][0].x()*term5;
  const Scalar term20 = l[3][3].x()*term8;
  const Scalar term21 = l[3][1].x()*term12 + l[3][2].x()*term13 + term19*term3 + term20*term7;
  const Scalar term22 = term18*term21;
  const Scalar term23 = term17*term22;
  const Scalar term24 = frac_y * frac_y;
  const Scalar term25 = 0.5*term24*frac_y;
  const Scalar term26 = -16*frac_y + 5*term24 - term25 + 16.6666666666667;
  const Scalar term27 = l[2][0].x()*term5;
  const Scalar term28 = l[2][3].x()*term8;
  const Scalar term29 = l[2][1].x()*term12 + l[2][2].x()*term13 + term27*term3 + term28*term7;
  const Scalar term30 = term26*term29;
  const Scalar term31 = 19.5*frac_y - 5.5*term24 + term25 - 21.8333333333333;
  const Scalar term32 = l[1][0].x()*term5;
  const Scalar term33 = l[1][3].x()*term8;
  const Scalar term34 = l[1][1].x()*term12 + l[1][2].x()*term13 + term3*term32 + term33*term7;
  const Scalar term35 = term31*term34;
  const Scalar term36 = term16 + term23 + term30 + term35;
  const Scalar term37 = l[0][0].y()*term5;
  const Scalar term38 = l[0][3].y()*term8;
  const Scalar term39 = l[0][1].y()*term12 + l[0][2].y()*term13 + term3*term37 + term38*term7;
  const Scalar term40 = term2*term39;
  const Scalar term41 = term1*term40;
  const Scalar term42 = l[3][0].y()*term5;
  const Scalar term43 = l[3][3].y()*term8;
  const Scalar term44 = l[3][1].y()*term12 + l[3][2].y()*term13 + term3*term42 + term43*term7;
  const Scalar term45 = term18*term44;
  const Scalar term46 = term17*term45;
  const Scalar term47 = l[2][0].y()*term5;
  const Scalar term48 = l[2][3].y()*term8;
  const Scalar term49 = l[2][1].y()*term12 + l[2][2].y()*term13 + term3*term47 + term48*term7;
  const Scalar term50 = term26*term49;
  const Scalar term51 = l[1][0].y()*term5;
  const Scalar term52 = l[1][3].y()*term8;
  const Scalar term53 = l[1][1].y()*term12 + l[1][2].y()*term13 + term3*term51 + term52*term7;
  const Scalar term54 = term31*term53;
  const Scalar term55 = term41 + term46 + term50 + term54;
  const Scalar term56 = l[0][0].z()*term5;
  const Scalar term57 = l[0][3].z()*term8;
  const Scalar term58 = l[0][1].z()*term12 + l[0][2].z()*term13 + term3*term56 + term57*term7;
  const Scalar term59 = term2*term58;
  const Scalar term60 = term1*term59;
  const Scalar term61 = l[3][0].z()*term5;
  const Scalar term62 = l[3][3].z()*term8;
  const Scalar term63 = l[3][1].z()*term12 + l[3][2].z()*term13 + term3*term61 + term62*term7;
  const Scalar term64 = term18*term63;
  const Scalar term65 = term17*term64;
  const Scalar term66 = l[2][0].z()*term5;
  const Scalar term67 = l[2][3].z()*term8;
  const Scalar term68 = l[2][1].z()*term12 + l[2][2].z()*term13 + term3*term66 + term67*term7;
  const Scalar term69 = term26*term68;
  const Scalar term70 = l[1][0].z()*term5;
  const Scalar term71 = l[1][3].z()*term8;
  const Scalar term72 = l[1][1].z()*term12 + l[1][2].z()*term13 + term3*term70 + term7*term71;
  const Scalar term73 = term31*term72;
  const Scalar term74 = term60 + term65 + term69 + term73;
  const Scalar term75 = term36 * term36 + term55 * term55 + term74 * term74;
  const Scalar term76 = 1 / sqrtf(term75);
  const Scalar term77 = term1*term2;
  const Scalar term78 = l[0][0](3)*term5;
  const Scalar term79 = l[0][3](3)*term8;
  const Scalar term80 = l[0][1](3)*term12 + l[0][2](3)*term13 + term3*term78 + term7*term79;
  const Scalar term81 = term17*term18;
  const Scalar term82 = l[3][0](3)*term5;
  const Scalar term83 = l[3][3](3)*term8;
  const Scalar term84 = l[3][1](3)*term12 + l[3][2](3)*term13 + term3*term82 + term7*term83;
  const Scalar term85 = l[2][0](3)*term5;
  const Scalar term86 = l[2][3](3)*term8;
  const Scalar term87 = l[2][1](3)*term12 + l[2][2](3)*term13 + term3*term85 + term7*term86;
  const Scalar term88 = l[1][0](3)*term5;
  const Scalar term89 = l[1][3](3)*term8;
  const Scalar term90 = l[1][1](3)*term12 + l[1][2](3)*term13 + term3*term88 + term7*term89;
  const Scalar term91 = l[0][0](4)*term5;
  const Scalar term92 = l[0][3](4)*term8;
  const Scalar term93 = l[0][1](4)*term12 + l[0][2](4)*term13 + term3*term91 + term7*term92;
  const Scalar term94 = l[3][0](4)*term5;
  const Scalar term95 = l[3][3](4)*term8;
  const Scalar term96 = l[3][1](4)*term12 + l[3][2](4)*term13 + term3*term94 + term7*term95;
  const Scalar term97 = l[2][0](4)*term5;
  const Scalar term98 = l[2][3](4)*term8;
  const Scalar term99 = l[2][1](4)*term12 + l[2][2](4)*term13 + term3*term97 + term7*term98;
  const Scalar term100 = l[1][0](4)*term5;
  const Scalar term101 = l[1][3](4)*term8;
  const Scalar term102 = l[1][1](4)*term12 + l[1][2](4)*term13 + term100*term3 + term101*term7;
  const Scalar term103 = l[0][0](5)*term5;
  const Scalar term104 = l[0][3](5)*term8;
  const Scalar term105 = l[0][1](5)*term12 + l[0][2](5)*term13 + term103*term3 + term104*term7;
  const Scalar term106 = l[3][0](5)*term5;
  const Scalar term107 = l[3][3](5)*term8;
  const Scalar term108 = l[3][1](5)*term12 + l[3][2](5)*term13 + term106*term3 + term107*term7;
  const Scalar term109 = l[2][0](5)*term5;
  const Scalar term110 = l[2][3](5)*term8;
  const Scalar term111 = l[2][1](5)*term12 + l[2][2](5)*term13 + term109*term3 + term110*term7;
  const Scalar term112 = l[1][0](5)*term5;
  const Scalar term113 = l[1][3](5)*term8;
  const Scalar term114 = l[1][1](5)*term12 + l[1][2](5)*term13 + term112*term3 + term113*term7;
  const Scalar term115 = 0.166666666666667*term3;
  const Scalar term116 = 0.166666666666667*term7;
  const Scalar term117 = 1.5*term10;
  const Scalar term118 = -11.0*frac_x + term117 + 19.5;
  const Scalar term119 = 10*frac_x - term117 - 16;
  const Scalar term120 = 2*frac_x;
  const Scalar term121 = term120 - 8;
  const Scalar term122 = term120 - 6;
  const Scalar term123 = term26*(-l[2][0].x()*term115 + l[2][1].x()*term118 + l[2][2].x()*term119 + l[2][3].x()*term116 + term121*term27 + term122*term28) + term31*(-l[1][0].x()*term115 + l[1][1].x()*term118 + l[1][2].x()*term119 + l[1][3].x()*term116 + term121*term32 + term122*term33) + term77*(-l[0][0].x()*term115 + l[0][1].x()*term118 + l[0][2].x()*term119 + l[0][3].x()*term116 + term121*term6 + term122*term9) + term81*(-l[3][0].x()*term115 + l[3][1].x()*term118 + l[3][2].x()*term119 + l[3][3].x()*term116 + term121*term19 + term122*term20);
  const Scalar term124_temp = sqrtf(term75);
  const Scalar term124 = 1 / (term124_temp * term124_temp * term124_temp);
  const Scalar term125 = (1.0L/2.0L)*term16 + (1.0L/2.0L)*term23 + (1.0L/2.0L)*term30 + (1.0L/2.0L)*term35;
  const Scalar term126 = (1.0L/2.0L)*term41 + (1.0L/2.0L)*term46 + (1.0L/2.0L)*term50 + (1.0L/2.0L)*term54;
  const Scalar term127 = term26*(-l[2][0].y()*term115 + l[2][1].y()*term118 + l[2][2].y()*term119 + l[2][3].y()*term116 + term121*term47 + term122*term48) + term31*(-l[1][0].y()*term115 + l[1][1].y()*term118 + l[1][2].y()*term119 + l[1][3].y()*term116 + term121*term51 + term122*term52) + term77*(-l[0][0].y()*term115 + l[0][1].y()*term118 + l[0][2].y()*term119 + l[0][3].y()*term116 + term121*term37 + term122*term38) + term81*(-l[3][0].y()*term115 + l[3][1].y()*term118 + l[3][2].y()*term119 + l[3][3].y()*term116 + term121*term42 + term122*term43);
  const Scalar term128 = (1.0L/2.0L)*term60 + (1.0L/2.0L)*term65 + (1.0L/2.0L)*term69 + (1.0L/2.0L)*term73;
  const Scalar term129 = term26*(-l[2][0].z()*term115 + l[2][1].z()*term118 + l[2][2].z()*term119 + l[2][3].z()*term116 + term121*term66 + term122*term67) + term31*(-l[1][0].z()*term115 + l[1][1].z()*term118 + l[1][2].z()*term119 + l[1][3].z()*term116 + term121*term70 + term122*term71) + term77*(-l[0][0].z()*term115 + l[0][1].z()*term118 + l[0][2].z()*term119 + l[0][3].z()*term116 + term121*term56 + term122*term57) + term81*(-l[3][0].z()*term115 + l[3][1].z()*term118 + l[3][2].z()*term119 + l[3][3].z()*term116 + term121*term61 + term122*term62);
  const Scalar term130 = 2*term124*(term123*term125 + term126*term127 + term128*term129);
  const Scalar term131 = 1.5*term24;
  const Scalar term132 = 10*frac_y - term131 - 16;
  const Scalar term133 = term132*term29;
  const Scalar term134 = -11.0*frac_y + term131 + 19.5;
  const Scalar term135 = term134*term34;
  const Scalar term136 = 2*frac_y;
  const Scalar term137 = term1*(term136 - 8);
  const Scalar term138 = term137*term14;
  const Scalar term139 = term17*(term136 - 6);
  const Scalar term140 = term139*term21;
  const Scalar term141 = term132*term49;
  const Scalar term142 = term134*term53;
  const Scalar term143 = term137*term39;
  const Scalar term144 = term139*term44;
  const Scalar term145 = term132*term68;
  const Scalar term146 = term134*term72;
  const Scalar term147 = term137*term58;
  const Scalar term148 = term139*term63;
  const Scalar term149 = term124*(term125*(2*term133 + 2*term135 + 2*term138 + 2*term140 - 0.333333333333333*term15 + 0.333333333333333*term22) + term126*(2*term141 + 2*term142 + 2*term143 + 2*term144 - 0.333333333333333*term40 + 0.333333333333333*term45) + term128*(2*term145 + 2*term146 + 2*term147 + 2*term148 - 0.333333333333333*term59 + 0.333333333333333*term64));
  const Scalar term150 = 0.166666666666667*term2;
  const Scalar term151 = 0.166666666666667*term18;
  
  result->direction().x() = term36*term76;
  result->direction().y() = term55*term76;
  result->direction().z() = term74*term76;
  result->origin().x() = term26*term87 + term31*term90 + term77*term80 + term81*term84;
  result->origin().y() = term102*term31 + term26*term99 + term77*term93 + term81*term96;
  result->origin().z() = term105*term77 + term108*term81 + term111*term26 + term114*term31;
  (*dresult_dxy)(0, 0) = term123*term76 - term130*term36;
  (*dresult_dxy)(0, 1) = -term149*term36 + term76*(term133 + term135 + term138 + term140 - 0.166666666666667*term15 + 0.166666666666667*term22);
  (*dresult_dxy)(1, 0) = term127*term76 - term130*term55;
  (*dresult_dxy)(1, 1) = -term149*term55 + term76*(term141 + term142 + term143 + term144 - 0.166666666666667*term40 + 0.166666666666667*term45);
  (*dresult_dxy)(2, 0) = term129*term76 - term130*term74;
  (*dresult_dxy)(2, 1) = -term149*term74 + term76*(term145 + term146 + term147 + term148 - 0.166666666666667*term59 + 0.166666666666667*term64);
  (*dresult_dxy)(3, 0) = term26*(-l[2][0](3)*term115 + l[2][1](3)*term118 + l[2][2](3)*term119 + l[2][3](3)*term116 + term121*term85 + term122*term86) + term31*(-l[1][0](3)*term115 + l[1][1](3)*term118 + l[1][2](3)*term119 + l[1][3](3)*term116 + term121*term88 + term122*term89) + term77*(-l[0][0](3)*term115 + l[0][1](3)*term118 + l[0][2](3)*term119 + l[0][3](3)*term116 + term121*term78 + term122*term79) + term81*(-l[3][0](3)*term115 + l[3][1](3)*term118 + l[3][2](3)*term119 + l[3][3](3)*term116 + term121*term82 + term122*term83);
  (*dresult_dxy)(3, 1) = term132*term87 + term134*term90 + term137*term80 + term139*term84 - term150*term80 + term151*term84;
  (*dresult_dxy)(4, 0) = term26*(-l[2][0](4)*term115 + l[2][1](4)*term118 + l[2][2](4)*term119 + l[2][3](4)*term116 + term121*term97 + term122*term98) + term31*(-l[1][0](4)*term115 + l[1][1](4)*term118 + l[1][2](4)*term119 + l[1][3](4)*term116 + term100*term121 + term101*term122) + term77*(-l[0][0](4)*term115 + l[0][1](4)*term118 + l[0][2](4)*term119 + l[0][3](4)*term116 + term121*term91 + term122*term92) + term81*(-l[3][0](4)*term115 + l[3][1](4)*term118 + l[3][2](4)*term119 + l[3][3](4)*term116 + term121*term94 + term122*term95);
  (*dresult_dxy)(4, 1) = term102*term134 + term132*term99 + term137*term93 + term139*term96 - term150*term93 + term151*term96;
  (*dresult_dxy)(5, 0) = term26*(-l[2][0](5)*term115 + l[2][1](5)*term118 + l[2][2](5)*term119 + l[2][3](5)*term116 + term109*term121 + term110*term122) + term31*(-l[1][0](5)*term115 + l[1][1](5)*term118 + l[1][2](5)*term119 + l[1][3](5)*term116 + term112*term121 + term113*term122) + term77*(-l[0][0](5)*term115 + l[0][1](5)*term118 + l[0][2](5)*term119 + l[0][3](5)*term116 + term103*term121 + term104*term122) + term81*(-l[3][0](5)*term115 + l[3][1](5)*term118 + l[3][2](5)*term119 + l[3][3](5)*term116 + term106*term121 + term107*term122);
  (*dresult_dxy)(5, 1) = term105*term137 - term105*term150 + term108*term139 + term108*term151 + term111*term132 + term114*term134;
}

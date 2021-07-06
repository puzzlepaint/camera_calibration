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

#include "../cuda_shims.h"

#include <libvis/libvis.h>

namespace vis {

// Jacobian of the local point position wrt. pose parameters (quaternion,
// translation) and the global point position.
// opcount = 69
template <typename Scalar>
__forceinline__ __host__ __device__ void ComputeJacobian(
    Scalar itp_0, Scalar itp_1, Scalar itp_2, Scalar itp_3,
    Scalar p_0, Scalar p_1, Scalar p_2,
    Scalar* jacobian_row_0, Scalar* jacobian_row_1, Scalar* jacobian_row_2) {
  const Scalar term0 = 2*itp_2;
  const Scalar term1 = p_2*term0;
  const Scalar term2 = itp_3*p_1;
  const Scalar term3 = 2*term2;
  const Scalar term4 = p_1*term0;
  const Scalar term5 = 2*itp_3*p_2;
  const Scalar term6 = 2*itp_0;
  const Scalar term7 = p_2*term6;
  const Scalar term8 = 2*itp_1;
  const Scalar term9 = p_1*term8;
  const Scalar term10 = 4*itp_2;
  const Scalar term11 = p_1*term6;
  const Scalar term12 = p_2*term8;
  const Scalar term13 = itp_3*p_0;
  const Scalar term14 = -2*itp_2*itp_2;
  const Scalar term15 = -2*itp_3*itp_3;
  const Scalar term16 = itp_3*term6;
  const Scalar term17 = itp_2*term8;
  const Scalar term18 = itp_2*term6;
  const Scalar term19 = itp_3*term8;
  const Scalar term20 = 2*term13;
  const Scalar term21 = 4*itp_1;
  const Scalar term22 = p_0*term0;
  const Scalar term23 = p_0*term8;
  const Scalar term24 = p_0*term6;
  const Scalar term25 = -2*itp_1*itp_1 + 1;
  const Scalar term26 = itp_1*term6;
  const Scalar term27 = itp_3*term0;
  
  // Change of the local point's x ...
  // ... depending on quaternion:
  jacobian_row_0[0] = term1 - term3;
  jacobian_row_0[1] = term4 + term5;
  jacobian_row_0[2] = -p_0*term10 + term7 + term9;
  jacobian_row_0[3] = -term11 + term12 - 4*term13;
  // ... depending on translation:
  jacobian_row_0[4] = 1;
  jacobian_row_0[5] = 0;
  jacobian_row_0[6] = 0;
  // ... depending on global point coordinates:
  jacobian_row_0[7] = term14 + term15 + 1;
  jacobian_row_0[8] = -term16 + term17;
  jacobian_row_0[9] = term18 + term19;
  
  // Change of the local point's y ...
  // ... depending on quaternion:
  jacobian_row_1[0] = -term12 + term20;
  jacobian_row_1[1] = -p_1*term21 + term22 - term7;
  jacobian_row_1[2] = term23 + term5;
  jacobian_row_1[3] = term1 - 4*term2 + term24;
  // ... depending on translation:
  jacobian_row_1[4] = 0;
  jacobian_row_1[5] = 1;
  jacobian_row_1[6] = 0;
  // ... depending on global point coordinates:
  jacobian_row_1[7] = term16 + term17;
  jacobian_row_1[8] = term15 + term25;
  jacobian_row_1[9] = -term26 + term27;
  
  // Change of the local point's z ...
  // ... depending on quaternion:
  jacobian_row_2[0] = -term22 + term9;
  jacobian_row_2[1] = -p_2*term21 + term11 + term20;
  jacobian_row_2[2] = -p_2*term10 - term24 + term3;
  jacobian_row_2[3] = term23 + term4;
  // ... depending on translation:
  jacobian_row_2[4] = 0;
  jacobian_row_2[5] = 0;
  jacobian_row_2[6] = 1;
  // ... depending on global point coordinates:
  jacobian_row_2[7] = -term18 + term19;
  jacobian_row_2[8] = term26 + term27;
  jacobian_row_2[9] = term14 + term25;
}

// opcount = 381
template <typename Scalar>
__forceinline__ __host__ __device__ void ComputeRigJacobian(
    Scalar ctr_0, Scalar ctr_1, Scalar ctr_2, Scalar ctr_3,
    Scalar p_0, Scalar p_1, Scalar p_2,
    Scalar rtg_0, Scalar rtg_1, Scalar rtg_2, Scalar rtg_3, Scalar rtg_4, Scalar rtg_5, Scalar rtg_6,
    Scalar* jacobian_row_0, Scalar* jacobian_row_1, Scalar* jacobian_row_2) {
  const Scalar term0 = 2*p_1;
  const Scalar term1 = rtg_3*term0;
  const Scalar term2 = -term1;
  const Scalar term3 = p_2*rtg_2;
  const Scalar term4 = 2*term3;
  const Scalar term5 = p_0*rtg_2;
  const Scalar term6 = p_1*rtg_1;
  const Scalar term7 = term5 - term6;
  const Scalar term8 = ctr_0*ctr_2;
  const Scalar term9 = 4*term8;
  const Scalar term10 = ctr_1*ctr_3;
  const Scalar term11 = 4*term10;
  const Scalar term12 = term11 + term9;
  const Scalar term13 = p_0*rtg_3;
  const Scalar term14 = p_2*rtg_1;
  const Scalar term15 = -term14;
  const Scalar term16 = term13 + term15;
  const Scalar term17 = ctr_0*ctr_3;
  const Scalar term18 = 4*term17;
  const Scalar term19 = ctr_1*ctr_2;
  const Scalar term20 = 4*term19;
  const Scalar term21 = term18 - term20;
  const Scalar term22 = p_1*rtg_3;
  const Scalar term23 = term22 - term3;
  const Scalar term24 = ctr_2 * ctr_2;
  const Scalar term25 = 4*term24;
  const Scalar term26 = ctr_3 * ctr_3;
  const Scalar term27 = 4*term26;
  const Scalar term28 = term25 + term27;
  const Scalar term29 = rtg_2*term0;
  const Scalar term30 = p_2*rtg_3;
  const Scalar term31 = 2*term30;
  const Scalar term32 = p_1*rtg_2;
  const Scalar term33 = term30 + term32;
  const Scalar term34 = p_1*rtg_0;
  const Scalar term35 = 2*term14;
  const Scalar term36 = -term35;
  const Scalar term37 = term13 + term34 + term36;
  const Scalar term38 = p_2*rtg_0;
  const Scalar term39 = rtg_1*term0;
  const Scalar term40 = term38 + term39 - term5;
  const Scalar term41 = 2*term38;
  const Scalar term42 = p_0*rtg_1;
  const Scalar term43 = term30 + term42;
  const Scalar term44 = p_0*rtg_0;
  const Scalar term45 = -term22 + term4 + term44;
  const Scalar term46 = 2*term5;
  const Scalar term47 = -term46;
  const Scalar term48 = term38 + term47 + term6;
  const Scalar term49 = rtg_0*term0;
  const Scalar term50 = term32 + term42;
  const Scalar term51 = term2 + term3 + term44;
  const Scalar term52 = 2*term13;
  const Scalar term53 = term15 + term34 + term52;
  const Scalar term54 = -2*term24;
  const Scalar term55 = -2*term26;
  const Scalar term56 = 2*term17;
  const Scalar term57 = 2*term19;
  const Scalar term58 = 2*term8;
  const Scalar term59 = 2*term10;
  const Scalar term60 = rtg_0*rtg_1;
  const Scalar term61 = rtg_2*rtg_3;
  const Scalar term62 = term60 + term61;
  const Scalar term63 = 2*p_2;
  const Scalar term64 = rtg_1 * rtg_1;
  const Scalar term65 = rtg_2 * rtg_2;
  const Scalar term66 = 2*p_0;
  const Scalar term67 = rtg_0*rtg_2;
  const Scalar term68 = rtg_1*rtg_3;
  const Scalar term69 = term67 - term68;
  const Scalar term70 = p_2 + rtg_6 + term0*term62 - term63*(term64 + term65) - term66*term69;
  const Scalar term71 = ctr_2*term70;
  const Scalar term72 = 2*term71;
  const Scalar term73 = rtg_0*rtg_3;
  const Scalar term74 = rtg_1*rtg_2;
  const Scalar term75 = term73 + term74;
  const Scalar term76 = rtg_3 * rtg_3;
  const Scalar term77 = term64 + term76;
  const Scalar term78 = term60 - term61;
  const Scalar term79 = 4*p_0*term75 - 4*p_1*term77 - 4*p_2*term78 + 2*rtg_5 + term0;
  const Scalar term80 = ctr_3*term79;
  const Scalar term81 = 2*ctr_3*term70;
  const Scalar term82 = ctr_2*term79;
  const Scalar term83 = 2*ctr_0;
  const Scalar term84 = term70*term83;
  const Scalar term85 = 2*ctr_1;
  const Scalar term86 = p_1 + rtg_5 - term0*term77 - term63*term78 + term66*term75;
  const Scalar term87 = term85*term86;
  const Scalar term88 = term67 + term68;
  const Scalar term89 = term73 - term74;
  const Scalar term90 = p_0 + rtg_4 - term0*term89 + term63*term88 - term66*(term65 + term76);
  const Scalar term91 = ctr_2*term90;
  const Scalar term92 = term83*term86;
  const Scalar term93 = term70*term85;
  const Scalar term94 = ctr_3*term90;
  const Scalar term95 = -2*term65;
  const Scalar term96 = -2*term76;
  const Scalar term97 = 4*term65;
  const Scalar term98 = 4*term76;
  const Scalar term99 = term97 + term98 - 2;
  const Scalar term100 = 2*term73;
  const Scalar term101 = 2*term74;
  const Scalar term102 = 4*term64 - 2;
  const Scalar term103 = term102 + term98;
  const Scalar term104 = 2*term67;
  const Scalar term105 = 2*term68;
  const Scalar term106 = term102 + term97;
  const Scalar term107 = ctr_0*ctr_1;
  const Scalar term108 = 4*term107;
  const Scalar term109 = ctr_2*ctr_3;
  const Scalar term110 = 4*term109;
  const Scalar term111 = term108 - term110;
  const Scalar term112 = term18 + term20;
  const Scalar term113 = ctr_1 * ctr_1;
  const Scalar term114 = 4*term113;
  const Scalar term115 = term114 + term27;
  const Scalar term116 = 2*term42;
  const Scalar term117 = 2*term44;
  const Scalar term118 = -2*term113 + 1;
  const Scalar term119 = 2*term107;
  const Scalar term120 = 2*term109;
  const Scalar term121 = 2*term94;
  const Scalar term122 = 4*ctr_1;
  const Scalar term123 = 2*term91;
  const Scalar term124 = term85*term90;
  const Scalar term125 = term83*term90;
  const Scalar term126 = -2*term64 + 1;
  const Scalar term127 = 2*term60;
  const Scalar term128 = 2*term61;
  const Scalar term129 = term108 + term110;
  const Scalar term130 = -term11 + term9;
  const Scalar term131 = term114 + term25;
  
  // Change of the local point's x ...
  // ... depending on camera_tr_rig quaternion:
  jacobian_row_0[0] = -term12*term7 - term16*term21 + term2 + term23*term28 + term4;
  jacobian_row_0[1] = term12*term37 + term21*term40 - term28*term33 + term29 + term31;
  jacobian_row_0[2] = -term12*term45 - term21*term43 - term28*term48 + term39 + term41 - 4*term5;
  jacobian_row_0[3] = term12*term50 - 4*term13 - term21*term51 + term28*term53 + term35 - term49;
  
  // ... depending on camera_tr_rig translation:
  jacobian_row_0[4] = term54 + term55 + 1;
  jacobian_row_0[5] = -term56 + term57;
  jacobian_row_0[6] = term58 + term59;
  
  // ... depending on rig_tr_global quaternion:
  jacobian_row_0[7] = term72 - term80;
  jacobian_row_0[8] = term81 + term82;
  jacobian_row_0[9] = term84 + term87 - 4*term91;
  jacobian_row_0[10] = -term92 + term93 - 4*term94;
  
  // ... depending on rig_tr_global translation:
  jacobian_row_0[11] = 1;
  jacobian_row_0[12] = 0;
  jacobian_row_0[13] = 0;
  
  // ... depending on global point coordinates:
  jacobian_row_0[14] = -term12*term69 - term21*term75 + term95 + term96 + term99*(term24 + term26) + 1;
  jacobian_row_0[15] = -term100 + term101 + term103*(term17 - term19) + term12*term62 + term28*term89;
  jacobian_row_0[16] = term104 + term105 - term106*(term10 + term8) + term21*term78 - term28*term88;
  
  // Change of the local point's y ...
  // ... depending on camera_tr_rig quaternion:
  jacobian_row_1[0] = term111*term7 - term112*term23 - term115*term16 + term36 + term52;
  jacobian_row_1[1] = -term111*term37 + term112*term33 + term115*term40 - term41 + term46 - 4*term6;
  jacobian_row_1[2] = term111*term45 + term112*term48 - term115*term43 + term116 + term31;
  jacobian_row_1[3] = -term111*term50 - term112*term53 - term115*term51 + term117 - 4*term22 + term4;
  
  // ... depending on camera_tr_rig translation:
  jacobian_row_1[4] = term56 + term57;
  jacobian_row_1[5] = term118 + term55;
  jacobian_row_1[6] = -term119 + term120;
  
  // ... depending on rig_tr_global quaternion:
  jacobian_row_1[7] = term121 - term93;
  jacobian_row_1[8] = -term122*term86 + term123 - term84;
  jacobian_row_1[9] = term124 + term81;
  jacobian_row_1[10] = -4*ctr_3*term86 + term125 + term72;
  
  // ... depending on rig_tr_global translation:
  jacobian_row_1[11] = 0;
  jacobian_row_1[12] = 1;
  jacobian_row_1[13] = 0;
  
  // ... depending on global point coordinates:
  jacobian_row_1[14] = term100 + term101 + term111*term69 - term115*term75 - term99*(term17 + term19);
  jacobian_row_1[15] = term103*(term113 + term26) - term111*term62 - term112*term89 + term126 + term96;
  jacobian_row_1[16] = term106*(term107 - term109) + term112*term88 + term115*term78 - term127 + term128;
  
  // Change of the local point's z ...
  // ... depending on camera_tr_rig quaternion:
  jacobian_row_2[0] = term129*term16 + term130*term23 + term131*term7 + term39 + term47;
  jacobian_row_2[1] = -term129*term40 - term130*term33 - term131*term37 - 4*term14 + term49 + term52;
  jacobian_row_2[2] = term1 - term117 + term129*term43 - term130*term48 + term131*term45 - 4*term3;
  jacobian_row_2[3] = term116 + term129*term51 + term130*term53 - term131*term50 + term29;
  
  // ... depending on camera_tr_rig translation:
  jacobian_row_2[4] = -term58 + term59;
  jacobian_row_2[5] = term119 + term120;
  jacobian_row_2[6] = term118 + term54;
  
  // ... depending on rig_tr_global quaternion:
  jacobian_row_2[7] = -term123 + term87;
  jacobian_row_2[8] = term121 - term122*term70 + term92;
  jacobian_row_2[9] = -term125 - 4*term71 + term80;
  jacobian_row_2[10] = term124 + term82;
  
  // ... depending on rig_tr_global translation:
  jacobian_row_2[11] = 0;
  jacobian_row_2[12] = 0;
  jacobian_row_2[13] = 1;
  
  // ... depending on global point coordinates:
  jacobian_row_2[14] = -term104 + term105 + term129*term75 + term131*term69 + term99*(-term10 + term8);
  jacobian_row_2[15] = -term103*(term107 + term109) + term127 + term128 + term130*term89 - term131*term62;
  jacobian_row_2[16] = term106*(term113 + term24) + term126 - term129*term78 - term130*term88 + term95;
}

// opcount = 39
template <typename Scalar>
__forceinline__ __host__ __device__  void ComputeBorderRegularizationJacobian(Scalar i1_0, Scalar i1_1, Scalar i1_2, Scalar i2_0, Scalar i2_1, Scalar i2_2, Scalar* jacobian_row_0, Scalar* jacobian_row_1, Scalar* jacobian_row_2) {
  const Scalar term0 = 2*i1_1;
  const Scalar term1 = i2_1*term0;
  const Scalar term2 = 2*i1_2;
  const Scalar term3 = i2_2*term2;
  const Scalar term4 = 2*i1_0;
  const Scalar term5 = i1_1*term4;
  const Scalar term6 = i1_2*term4;
  const Scalar term7 = i2_0*term4;
  const Scalar term8 = i1_2*term0;
  
  jacobian_row_0[0] = -1;
  jacobian_row_0[1] = 0;
  jacobian_row_0[2] = 0;
  jacobian_row_0[3] = 4*i1_0*i2_0 + term1 + term3;
  jacobian_row_0[4] = i2_1*term4;
  jacobian_row_0[5] = i2_2*term4;
  jacobian_row_0[6] = 2*i1_0*i1_0 - 1;
  jacobian_row_0[7] = term5;
  jacobian_row_0[8] = term6;
  jacobian_row_1[0] = 0;
  jacobian_row_1[1] = -1;
  jacobian_row_1[2] = 0;
  jacobian_row_1[3] = i2_0*term0;
  jacobian_row_1[4] = 4*i1_1*i2_1 + term3 + term7;
  jacobian_row_1[5] = i2_2*term0;
  jacobian_row_1[6] = term5;
  jacobian_row_1[7] = 2*i1_1*i1_1 - 1;
  jacobian_row_1[8] = term8;
  jacobian_row_2[0] = 0;
  jacobian_row_2[1] = 0;
  jacobian_row_2[2] = -1;
  jacobian_row_2[3] = i2_0*term2;
  jacobian_row_2[4] = i2_1*term2;
  jacobian_row_2[5] = 4*i1_2*i2_2 + term1 + term7;
  jacobian_row_2[6] = term6;
  jacobian_row_2[7] = term8;
  jacobian_row_2[8] = 2*i1_2*i1_2 - 1;
}

}

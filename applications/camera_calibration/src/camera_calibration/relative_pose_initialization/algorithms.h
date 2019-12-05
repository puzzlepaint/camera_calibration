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
#include <libvis/libvis.h>
#include <libvis/point_cloud.h>
#include <libvis/sophus.h>

namespace vis {

// This file contains algorithms for estimating the relative poses of three
// point clouds (having the same number of points) where each triple of points
// with the same index from each cloud must be collinear (i.e., on the same
// line). This is useful to initialize non-parametric camera calibrations.


// Algorithm for non-central cameras with 3D calibration objects from:
// "A Unifying Model for Camera Calibration", S. Ramalingam and P. Sturm, PAMI 2017.
// This is not applicable to planar calibration objects or cameras which are near-central!
// The pose of cloud[2] is fixed to the identity, and the estimated poses for
// clouds[0] and clouds[1] are returned in cloud2_tr_cloud[0], and cloud2_tr_cloud[1] such that
// cloud2_tr_cloud * point transforms the point into the frame of clouds[2].
bool NonCentralCamera3DCalibrationObjectRelativePose(const Point3fCloud clouds[3], SE3d cloud2_tr_cloud[2]);

// Algorithm for non-central cameras with planar calibration objects from:
// "A Unifying Model for Camera Calibration", S. Ramalingam and P. Sturm, PAMI 2017.
// This is not applicable to 3D calibration objects or cameras which are near-central!
// For points in clouds[2], the z coordinate must be zero.
// The pose of cloud[2] is fixed to the identity, and the estimated poses for
// clouds[0] and clouds[1] are returned in cloud2_tr_cloud[0], and cloud2_tr_cloud[1] such that
// cloud2_tr_cloud * point transforms the point into the frame of clouds[2].
// TODO: Currently, an ambiguity in the output is resolved by looking at gt_cloud2_tr_cloud[0].
bool NonCentralCameraPlanarCalibrationObjectRelativePose(const Point3fCloud clouds[3], SE3d cloud2_tr_cloud[2], SE3d gt_cloud2_tr_cloud[2] = nullptr);

// Algorithm for central cameras with 3D calibration objects from:
// "A Unifying Model for Camera Calibration", S. Ramalingam and P. Sturm, PAMI 2017.
// This is not applicable to planar calibration objects or cameras which are non-central!
// The pose of cloud[1] is fixed to the identity, and the estimated pose for
// clouds[0] is returned in cloud1_tr_cloud[0], such that
// cloud1_tr_cloud * point transforms the point into the frame of clouds[1].
bool CentralCamera3DCalibrationObjectRelativePose(const Point3fCloud clouds[2], SE3d cloud1_tr_cloud[1], Vec3d* optical_center = nullptr);

// Algorithm for central cameras with planar calibration objects from:
// "A Unifying Model for Camera Calibration", S. Ramalingam and P. Sturm, PAMI 2017.
// This is not applicable to 3D calibration objects or cameras which are non-central!
// NOTE: The z-coordinate of all points in all clouds must be zero.
// The pose of cloud[2] is fixed to the identity, and the estimated poses for
// clouds[0] and clouds[1] are returned in cloud2_tr_cloud[0], and cloud2_tr_cloud[1] such that
// cloud2_tr_cloud * point transforms the point into the frame of clouds[2].
bool CentralCameraPlanarCalibrationObjectRelativePose(const Point3fCloud clouds[3], SE3d cloud2_tr_cloud[2], Vec3d* optical_center = nullptr);

}

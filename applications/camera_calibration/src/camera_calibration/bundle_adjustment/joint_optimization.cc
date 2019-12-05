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

#include "camera_calibration/bundle_adjustment/joint_optimization.h"

#include <libvis/image_display.h>
#include <libvis/lm_optimizer.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/joint_optimization_jacobians.h"
#include "camera_calibration/calibration_report.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/local_parametrizations/quaternion_parametrization.h"
#include "camera_calibration/models/all_models.h"
#include "camera_calibration/util.h"

namespace vis {

/// Represents the state used in joint optimization (bundle adjustment). This
/// contains only the members of BAState that are non-constant (and thus avoids
/// copying the constant members around). In particular, it also excludes
/// entries of rig_tr_global for which image_used is false.
/// 
/// If eliminate_points == true, the variable ordering for updates is:
/// - points [3 * point_count]
/// - rig_tr_global [6 * used_imageset_count]
/// - camera_tr_rig [6 * camera_count, or 0 if camera_count == 1]
/// - intrinsics [depends on camera model(s)]
/// 
/// If eliminate_points == false, the imageset poses are elimintated, and the variable ordering for updates is:
/// - rig_tr_global [6 * used_imageset_count]
/// - camera_tr_rig [6 * camera_count, or 0 if camera_count == 1]
/// - points [3 * point_count]
/// - intrinsics [depends on camera model(s)]
struct JointOptimizationState {
  inline JointOptimizationState() {}
  
  inline JointOptimizationState(
      const BAState& other,
      vector<int>* original_to_seq_index,
      vector<int>* seq_to_original_index,
      bool localize_only,
      bool eliminate_points)
      : camera_tr_rig(other.camera_tr_rig),
        intrinsics(other.intrinsics.size()),
        points(other.points),
        localize_only(localize_only),
        eliminate_points(eliminate_points) {
    CHECK_EQ(other.image_used.size(), other.rig_tr_global.size());
    
    for (int i = 0; i < intrinsics.size(); ++ i) {
      intrinsics[i].reset(other.intrinsics[i]->duplicate());
    }
    
    original_to_seq_index->resize(other.rig_tr_global.size());
    seq_to_original_index->clear();
    for (usize i = 0; i < other.rig_tr_global.size(); ++ i) {
      if (other.image_used[i]) {
        rig_tr_global.push_back(other.rig_tr_global[i]);
        (*original_to_seq_index)[i] = rig_tr_global.size() - 1;
        seq_to_original_index->push_back(i);
      } else {
        (*original_to_seq_index)[i] = -1;
      }
    }
  }
  
  inline JointOptimizationState(const JointOptimizationState& other)
      : camera_tr_rig(other.camera_tr_rig),
        rig_tr_global(other.rig_tr_global),
        intrinsics(other.intrinsics.size()),
        points(other.points),
        localize_only(other.localize_only),
        eliminate_points(other.eliminate_points) {
    for (int i = 0; i < intrinsics.size(); ++ i) {
      intrinsics[i].reset(other.intrinsics[i]->duplicate());
    }
  }
  
  inline JointOptimizationState& operator= (const JointOptimizationState& other) {
    camera_tr_rig = other.camera_tr_rig;
    rig_tr_global = other.rig_tr_global;
    intrinsics.resize(other.intrinsics.size());
    for (int i = 0; i < intrinsics.size(); ++ i) {
      intrinsics[i].reset(other.intrinsics[i]->duplicate());
    }
    points = other.points;
    localize_only = other.localize_only;
    eliminate_points = other.eliminate_points;
    return *this;
  }
  
  inline int degrees_of_freedom() const {
    int num_intrinsics_parameters = 0;
    if (!localize_only) {
      for (int i = 0; i < intrinsics.size(); ++ i) {
        num_intrinsics_parameters += intrinsics[i]->update_parameter_count();
      }
    }
    
    // If there is only one camera, we exclude the camera_tr_rig transformation
    // as it is unneeded. Otherwise, we keep all the camera_tr_rig transformations
    // for simplicity (and let the optimization deal with the resulting gauge freedom).
    return (are_camera_tr_rig_in_state() ? (camera_tr_rig.size() * SE3d::DoF) : 0) +
           rig_tr_global.size() * SE3d::DoF +
           num_intrinsics_parameters +
           points.size() * 3;
  }
  
  static constexpr bool is_reversible() { return false; }
  
  inline bool are_camera_tr_rig_in_state() const {
    return camera_tr_rig.size() > 1;
  }
  
  /// Returns the offset of the first variable for rig_tr_global in the update vector.
  inline int first_rig_tr_global_offset() const {
    return eliminate_points ? (points.size() * 3) : 0;
  }
  inline int first_camera_tr_rig_offset() const {
    return first_rig_tr_global_offset() + rig_tr_global.size() * SE3d::DoF;
  }
  inline int first_points_offset() const {
    if (eliminate_points) {
      return 0;
    } else {
      return first_camera_tr_rig_offset() + (are_camera_tr_rig_in_state() ? (camera_tr_rig.size() * SE3d::DoF) : 0);
    }
  }
  inline int first_intrinsics_offset() const {
    if (localize_only) {
      LOG(ERROR) << "first_intrinsics_offset() called with localize_only";
    }
    return eliminate_points ? (first_camera_tr_rig_offset() + (are_camera_tr_rig_in_state() ? (camera_tr_rig.size() * SE3d::DoF) : 0)) : (first_points_offset() + points.size() * 3);
  }
  inline int intrinsics_offset(int camera_index) const {
    if (localize_only) {
      LOG(ERROR) << "intrinsics_offset() called with localize_only";
    }
    int offset = first_intrinsics_offset();
    for (int i = 0; i < camera_index; ++ i) {
      offset += intrinsics[i]->update_parameter_count();
    }
    return offset;
  }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    usize delta_index = first_rig_tr_global_offset();
    for (usize pose_index = 0; pose_index < rig_tr_global.size(); ++ pose_index) {
      rig_tr_global[pose_index] =
          SE3d(ApplyLocalUpdateToQuaternion(
                   rig_tr_global[pose_index].unit_quaternion(),
                   -delta.template segment<3>(delta_index)),
               rig_tr_global[pose_index].translation() -
                   Vec3d(delta[delta_index + 3], delta[delta_index + 4], delta[delta_index + 5]));
      delta_index += SE3d::DoF;
    }
    
    if (are_camera_tr_rig_in_state()) {
      delta_index = first_camera_tr_rig_offset();
      for (usize pose_index = 0; pose_index < camera_tr_rig.size(); ++ pose_index) {
        camera_tr_rig[pose_index] =
            SE3d(ApplyLocalUpdateToQuaternion(
                    camera_tr_rig[pose_index].unit_quaternion(),
                    -delta.template segment<3>(delta_index)),
                camera_tr_rig[pose_index].translation() -
                    Vec3d(delta[delta_index + 3], delta[delta_index + 4], delta[delta_index + 5]));
        delta_index += SE3d::DoF;
      }
    }
    
    delta_index = first_points_offset();
    for (usize point_index = 0; point_index < points.size(); ++ point_index) {
      points[point_index] -= delta.template segment<3>(delta_index).template cast<double>();
      delta_index += 3;
    }
    
    if (!localize_only) {
      delta_index = first_intrinsics_offset();
      for (int i = 0; i < intrinsics.size(); ++ i) {
        CameraModel& model_ref = *intrinsics[i];
        IDENTIFY_CAMERA_MODEL(model_ref,
          _model_ref.SubtractDelta(delta.segment(delta_index, intrinsics[i]->update_parameter_count()));
        )
        delta_index += intrinsics[i]->update_parameter_count();
      }
    }
  }
  
  /// See BAState for documentation on these parameters, they are directly
  /// copied from and to there.
  vector<SE3d> camera_tr_rig;
  vector<SE3d> rig_tr_global;
  vector<shared_ptr<CameraModel>> intrinsics;
  vector<Vec3d> points;
  
  bool localize_only;
  bool eliminate_points;
};

struct JointOptimizationCostFunction {
  /// For a grid of directions, computes the LineTangents for each direction.
  void ComputeTangentsImage(const Image<Vec3d>& directions, Image<LineTangents>* output) const {
    output->SetSize(directions.size());
    for (u32 y = 0; y < directions.height(); ++ y) {
      for (u32 x = 0; x < directions.width(); ++ x) {
        ComputeTangentsForDirectionOrLine(
            directions.at(x, y),
            &(*output)(x, y));
      }
    }
  }
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const JointOptimizationState& state,
      Accumulator* accumulator) const {
    vector<bool> is_central_camera_model(state.intrinsics.size());
    for (int camera_index = 0; camera_index < state.intrinsics.size(); ++ camera_index) {
      const CameraModel* model = state.intrinsics[camera_index].get();
      is_central_camera_model[camera_index] = CameraModel::IsCentral(model->type());
    }
    
    // Cache tangents.
    // TODO: Do this earlier! When done here, it can be needlessly re-computed several times.
    // TODO: This is duplicated in several places.
    /// Indexed by: [camera_index].
    vector<Image<LineTangents>> tangents_images(state.intrinsics.size());
    if (compute_jacobians) {
      for (int camera_index = 0; camera_index < state.intrinsics.size(); ++ camera_index) {
        const CameraModel* model = state.intrinsics[camera_index].get();
        if (model->type() == CameraModel::Type::CentralGeneric) {
          ComputeTangentsImage(dynamic_cast<const CentralGenericModel*>(model)->grid(), &tangents_images[camera_index]);
        } else if (model->type() == CameraModel::Type::NoncentralGeneric) {
          ComputeTangentsImage(dynamic_cast<const NoncentralGenericModel*>(model)->direction_grid(), &tangents_images[camera_index]);
        } else if (model->type() == CameraModel::Type::CentralThinPrismFisheye ||
                   model->type() == CameraModel::Type::CentralOpenCV ||
                   model->type() == CameraModel::Type::CentralRadial) {
          // No preprocessing to be done for this model.
        } else {
          LOG(ERROR) << "Not implemented for this camera model, assuming no preprocessing is necessary.";
        }
      }
    }
    
    // Add all reprojection error residuals.
    for (int seq_pose_index = 0; seq_pose_index < state.rig_tr_global.size(); ++ seq_pose_index) {
      for (int camera_index = 0; camera_index < state.intrinsics.size(); ++ camera_index) {
        CameraModel& model = *state.intrinsics[camera_index];
        
        const SE3d& image_tr_global = state.camera_tr_rig[camera_index] * state.rig_tr_global[seq_pose_index];
        Quaterniond image_q_global = image_tr_global.unit_quaternion();
        Mat3d image_r_global = image_tr_global.rotationMatrix().cast<double>();
        const Vec3d& image_t_global = image_tr_global.translation();
        
        int original_pose_index = seq_to_original_index[seq_pose_index];
        vector<PointFeature>& matches = dataset->GetImageset(original_pose_index)->FeaturesOfCamera(camera_index);
        
        for (PointFeature& feature : matches) {
          AddReprojectionResidual<compute_jacobians>(
              state, is_central_camera_model, camera_index, model,
              tangents_images[camera_index], seq_pose_index, image_q_global,
              image_r_global, image_t_global, feature, accumulator);
        }  // loop over features in image
      }  // loop over cameras
      
      if (on_the_fly_block_processing) {
        accumulator->FinishedBlockForSchurComplement();
      }
    }  // loop over images
    
    // Optionally, add regularization residuals.
    if (regularization_weight > 0) {
      LOG(ERROR) << "Regularization is disabled at the moment since it is untested with the current version.";
      
      // for (int camera_index = 0; camera_index < state.intrinsics.size(); ++ camera_index) {
      //   AddRegularizationResiduals<compute_jacobians>(state, camera_index, tangents_images[camera_index], accumulator);
      // }
    }
  }
  
  template<bool compute_jacobians, class Accumulator>
  inline void AddReprojectionResidual(
      const JointOptimizationState& state,
      const vector<bool>& is_central_camera_model,
      int camera_index,
      CameraModel& model,
      const Image<LineTangents>& tangents_image,
      int seq_pose_index,
      const Quaterniond& image_q_global,
      const Mat3d& image_r_global,
      const Vec3d& image_t_global,
      PointFeature& feature,
      Accumulator* accumulator) const {
    const Vec3d& point = state.points[feature.index];
    Vec3d local_point = image_r_global * point + image_t_global;
    
    // Cache the projection result of the point over successive optimization iterations to speed up this projection by using a good initial estimate
    Vec2d pixel = feature.last_projection;
    if (!(pixel.x() >= model.calibration_min_x() &&
          pixel.y() >= model.calibration_min_y() &&
          pixel.x() < model.calibration_max_x() + 1 &&
          pixel.y() < model.calibration_max_y() + 1) ||
        pixel.hasNaN()) {
      pixel = Vec2d(0.5f * (model.calibration_min_x() + model.calibration_max_x() + 1),
                    0.5f * (model.calibration_min_y() + model.calibration_max_y() + 1));
    }
    if (!model.ProjectWithInitialEstimate(local_point, &pixel)) {
      // Try backup: re-initialize the initial estimate at the center of the calibrated area
      pixel = Vec2d(0.5f * (model.calibration_min_x() + model.calibration_max_x() + 1),
                    0.5f * (model.calibration_min_y() + model.calibration_max_y() + 1));
      if (!model.ProjectWithInitialEstimate(local_point, &pixel)) {
        accumulator->AddInvalidResidual();
        return;
      }
    }
    feature.last_projection = pixel;
    
    if (!compute_jacobians) {
      accumulator->AddResidual(pixel - feature.xy.cast<double>(), HuberLoss<double>(1.0));
      return;
    }
    
    // Compute Jacobian wrt. image pose, optionally camera_tr_rig, and point position [2 x (6 + (rig ? 6 : 0) + 3)].
    // Residual: Project(exp(delta) * image_tr_pattern * pattern_point) - measurement
    
    // Compute Jacobian as follows:
    //   (d pixel) / (d local_point)                  [2 x 3], numerical
    // * (d local_point) / (d pose and global_point)  [3 x (7 + (rig ? 7 : 0) + 3)], analytical
    
    // Numerical part:
    Matrix<double, 2, 3> pixel_wrt_local_point;
    const double kDelta = numerical_diff_delta * (is_central_camera_model[camera_index] ? local_point.norm() : 0.1);
    bool ok = true;
    for (int dimension = 0; dimension < 3; ++ dimension) {
      Vec3d offset_point = local_point;
      offset_point(dimension) += kDelta;
      Vec2d offset_pixel = pixel;
      if (!model.ProjectWithInitialEstimate(offset_point, &offset_pixel)) {
        ok = false;
        break;
      }
      
      pixel_wrt_local_point(0, dimension) = (offset_pixel.x() - pixel.x()) / kDelta;
      pixel_wrt_local_point(1, dimension) = (offset_pixel.y() - pixel.y()) / kDelta;
    }
    if (!ok) {
      accumulator->AddResidual(pixel - feature.xy.cast<double>(), HuberLoss<double>(1.0));
      return;
    }
    
    // Analytical part:
    Matrix<double, 3, 7 + 7 + 3, Eigen::RowMajor> local_point_wrt_poses_and_global_point;
    if (state.are_camera_tr_rig_in_state()) {
      const Quaterniond& camera_q_rig = state.camera_tr_rig[camera_index].unit_quaternion();
      const Quaterniond& rig_q_global = state.rig_tr_global[seq_pose_index].unit_quaternion();
      const Vec3d& rig_t_global = state.rig_tr_global[seq_pose_index].translation();
      ComputeRigJacobian(
          camera_q_rig.w(), camera_q_rig.x(), camera_q_rig.y(), camera_q_rig.z(),
          point.x(), point.y(), point.z(),
          rig_q_global.w(), rig_q_global.x(), rig_q_global.y(), rig_q_global.z(), rig_t_global.x(), rig_t_global.y(), rig_t_global.z(),
          local_point_wrt_poses_and_global_point.row(0).data(),
          local_point_wrt_poses_and_global_point.row(1).data(),
          local_point_wrt_poses_and_global_point.row(2).data());
    } else {
      ComputeJacobian(
          image_q_global.w(), image_q_global.x(), image_q_global.y(), image_q_global.z(),
          point.x(), point.y(), point.z(),
          local_point_wrt_poses_and_global_point.row(0).data(),
          local_point_wrt_poses_and_global_point.row(1).data(),
          local_point_wrt_poses_and_global_point.row(2).data());
    }
    
    Matrix<double, 2, 6> pose_jacobian;
    Matrix<double, 2, 6> rig_jacobian;
    Matrix<double, 2, 3> point_jacobian;
    
    if (state.are_camera_tr_rig_in_state()) {
      // local_point_wrt_poses_and_global_point contains the Jacobian wrt.:
      // - rig_tr_global (indices 0 .. 6)
      // - camera_tr_rig (indices 7 .. 13)
      // - global_point (indices 14 .. 16)
      
      const Quaterniond& camera_q_rig = state.camera_tr_rig[camera_index].unit_quaternion();
      const Quaterniond& rig_q_global = state.rig_tr_global[seq_pose_index].unit_quaternion();
      
      Matrix<double, 4, 3> camera_q_rig_wrt_update;
      QuaternionJacobianWrtLocalUpdate(camera_q_rig, &camera_q_rig_wrt_update);
      
      Matrix<double, 4, 3> rig_q_global_wrt_update;
      QuaternionJacobianWrtLocalUpdate(rig_q_global, &rig_q_global_wrt_update);
      
      pose_jacobian.leftCols<3>() = (pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 4>(0, 0)) * rig_q_global_wrt_update;
      pose_jacobian.rightCols<3>() = pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 3>(0, 0 + 4);
      
      rig_jacobian.leftCols<3>() = (pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 4>(0, 7)) * camera_q_rig_wrt_update;
      rig_jacobian.rightCols<3>() = pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 3>(0, 7 + 4);
      
      point_jacobian = pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 3>(0, 14);
    } else {
      // local_point_wrt_poses_and_global_point contains the Jacobian wrt.:
      // - rig_tr_global (indices 0 .. 6)
      // - global_point (indices 7 .. 9)
      
      Matrix<double, 4, 3> quaternion_wrt_update;
      QuaternionJacobianWrtLocalUpdate(image_q_global, &quaternion_wrt_update);
      
      pose_jacobian.leftCols<3>() = pixel_wrt_local_point * (local_point_wrt_poses_and_global_point.leftCols<4>() * quaternion_wrt_update);
      pose_jacobian.rightCols<3>() = pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 3>(0, 4);
      
      point_jacobian = pixel_wrt_local_point * local_point_wrt_poses_and_global_point.block<3, 3>(0, 7);
    }
    
    // Get the model Jacobian:
    IDENTIFY_CAMERA_MODEL(model,
      ok = AccumulateModelJacobian<_model_type::IntrinsicsJacobianSize>(
          state, camera_index, &_model, accumulator, tangents_image,
          seq_pose_index, pose_jacobian, rig_jacobian, point_jacobian, feature, local_point, pixel);
    )
    if (!ok) {
      accumulator->AddResidual(pixel - feature.xy.cast<double>(), HuberLoss<double>(1.0));
    }
  }
  
  template <int num_variables, typename ModelT, class Accumulator>
  inline bool AccumulateModelJacobian(
      const JointOptimizationState& state,
      int camera_index,
      ModelT* model,
      Accumulator* accumulator,
      const Image<LineTangents>& tangents_image,
      int seq_pose_index,
      const Matrix<double, 2, 6>& pose_jacobian,
      const Matrix<double, 2, 6>& rig_jacobian,
      const Matrix<double, 2, 3>& point_jacobian,
      const PointFeature& feature,
      const Vec3d& local_point,
      const Vec2d& pixel) const {
    Matrix<int, num_variables, 1> grid_update_indices;
    Matrix<double, 2, num_variables, Eigen::RowMajor> pixel_wrt_grid_updates;
    if (!localize_only) {
      if (!model->ProjectionJacobianWrtIntrinsics(
          local_point,
          pixel,
          tangents_image,
          numerical_diff_delta,
          &grid_update_indices,
          &pixel_wrt_grid_updates)) {
        return false;
      }
    }
    
    // Accumulate:
    usize pose_jac_index = state.first_rig_tr_global_offset() + seq_pose_index * 6;
    usize rig_jac_index = state.first_camera_tr_rig_offset() + camera_index * 6;
    usize point_jac_index = state.first_points_offset() + feature.index * 3;
    if (!localize_only) {
      usize model_jac_start_index = state.intrinsics_offset(camera_index);
      for (int i = 0; i < num_variables; ++ i) {
        grid_update_indices(i) += model_jac_start_index;
      }
    }
    
    if (localize_only) {
      if (eliminate_points) {
        if (state.are_camera_tr_rig_in_state()) {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              point_jac_index,
              point_jacobian,
              pose_jac_index,
              pose_jacobian,
              rig_jac_index,
              rig_jacobian,
              /*enable0*/ true, /*enable1*/ true, /*enable2*/ true,
              HuberLoss<double>(1.0));
        } else {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              point_jac_index,
              point_jacobian,
              pose_jac_index,
              pose_jacobian,
              /*enable0*/ true, /*enable1*/ true,
              HuberLoss<double>(1.0));
        }
      } else {
        if (state.are_camera_tr_rig_in_state()) {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              pose_jac_index,
              pose_jacobian,
              rig_jac_index,
              rig_jacobian,
              point_jac_index,
              point_jacobian,
              /*enable0*/ true, /*enable1*/ true, /*enable2*/ true,
              HuberLoss<double>(1.0));
        } else {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              pose_jac_index,
              pose_jacobian,
              point_jac_index,
              point_jacobian,
              /*enable0*/ true, /*enable1*/ true,
              HuberLoss<double>(1.0));
        }
      }
    } else {
      if (eliminate_points) {
        if (state.are_camera_tr_rig_in_state()) {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              point_jac_index,
              point_jacobian,
              pose_jac_index,
              pose_jacobian,
              rig_jac_index,
              rig_jacobian,
              grid_update_indices,
              pixel_wrt_grid_updates,
              /*enable0*/ true, /*enable1*/ true, /*enable2*/ true,
              HuberLoss<double>(1.0));
        } else {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              point_jac_index,
              point_jacobian,
              pose_jac_index,
              pose_jacobian,
              grid_update_indices,
              pixel_wrt_grid_updates,
              /*enable0*/ true, /*enable1*/ true,
              HuberLoss<double>(1.0));
        }
      } else {
        if (state.are_camera_tr_rig_in_state()) {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              pose_jac_index,
              pose_jacobian,
              rig_jac_index,
              rig_jacobian,
              point_jac_index,
              point_jacobian,
              grid_update_indices,
              pixel_wrt_grid_updates,
              /*enable0*/ true, /*enable1*/ true, /*enable2*/ true,
              HuberLoss<double>(1.0));
        } else {
          accumulator->AddResidualWithJacobian(
              pixel - feature.xy.cast<double>(),
              pose_jac_index,
              pose_jacobian,
              point_jac_index,
              point_jacobian,
              grid_update_indices,
              pixel_wrt_grid_updates,
              /*enable0*/ true, /*enable1*/ true,
              HuberLoss<double>(1.0));
        }
      }
    }
    
    return true;
  }
  
  template<bool compute_jacobians, class Accumulator>
  inline void AddRegularizationResiduals(
      const JointOptimizationState& state,
      int camera_index,
      const Image<LineTangents>& tangents_image,
      Accumulator* accumulator) const {
    CameraModel* model = state.intrinsics[camera_index].get();
    if (CentralGenericModel* cg_model = dynamic_cast<CentralGenericModel*>(model)) {
      usize model_jac_start_index = state.intrinsics_offset(camera_index);
      
      for (u32 x = 1; x < cg_model->grid().width() - 1; ++ x) {
        // Top row
        AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ false>(
            cg_model->grid(), tangents_image,
            x, 0,
            x, 1,
            x, 2,
            model_jac_start_index, accumulator);
        
        // Bottom row
        AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ true>(
            cg_model->grid(), tangents_image,
            x, cg_model->grid().height() - 1,
            x, cg_model->grid().height() - 2,
            x, cg_model->grid().height() - 3,
            model_jac_start_index, accumulator);
      }
      
      for (u32 y = 1; y < cg_model->grid().height() - 1; ++ y) {
        // Left column
        AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ false>(
            cg_model->grid(), tangents_image,
            0, y,
            1, y,
            2, y,
            model_jac_start_index, accumulator);
        
        // Right column
        AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ true>(
            cg_model->grid(), tangents_image,
            cg_model->grid().width() - 1, y,
            cg_model->grid().width() - 2, y,
            cg_model->grid().width() - 3, y,
            model_jac_start_index, accumulator);
      }
      
      // Corners
      // top left
      AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ false>(
          cg_model->grid(), tangents_image,
          0, 0,
          1, 1,
          2, 2,
          model_jac_start_index, accumulator);
      // top right
      AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ false>(
          cg_model->grid(), tangents_image,
          cg_model->grid().width() - 1, 0,
          cg_model->grid().width() - 2, 1,
          cg_model->grid().width() - 3, 2,
          model_jac_start_index, accumulator);
      // bottom left
      AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ true>(
          cg_model->grid(), tangents_image,
          0, cg_model->grid().height() - 1,
          1, cg_model->grid().height() - 2,
          2, cg_model->grid().height() - 3,
          model_jac_start_index, accumulator);
      // bottom right
      AccumulateRegularizationJacobian<compute_jacobians, /*inverse_order*/ true>(
          cg_model->grid(), tangents_image,
          cg_model->grid().width() - 1, cg_model->grid().height() - 1,
          cg_model->grid().width() - 2, cg_model->grid().height() - 2,
          cg_model->grid().width() - 3, cg_model->grid().height() - 3,
          model_jac_start_index, accumulator);
    } else {
      LOG(ERROR) << "Regularization is not implemented for this model.";
    }
  }
  
  template <bool compute_jacobians, bool inverse_order, class Accumulator>
  void AccumulateRegularizationJacobian(
      const Image<Vec3d>& grid,
      const Image<LineTangents>& tangents_image,
      int outer_x, int outer_y,
      int inner1_x, int inner1_y,
      int inner2_x, int inner2_y,
      usize model_jac_start_index,
      Accumulator* accumulator
  ) const {
    const Vec3d& outer = grid.at(outer_x, outer_y);
    const Vec3d& inner1 = grid.at(inner1_x, inner1_y);
    const Vec3d& inner2 = grid.at(inner2_x, inner2_y);
    Vec3d proj = inner1.dot(inner2) * inner1;
    Vec3d mirror = proj + (proj - inner2);
    
    if (!compute_jacobians) {
      accumulator->AddResidual(regularization_weight * (mirror.x() - outer.x()));
      accumulator->AddResidual(regularization_weight * (mirror.y() - outer.y()));
      accumulator->AddResidual(regularization_weight * (mirror.z() - outer.z()));
    } else {
      Matrix<int, 1, 3 * 2> jacobian_indices;
      
      jacobian_indices(inverse_order ? 4 : 0) = model_jac_start_index + 2 * (outer_x + outer_y * grid.width()) + 0;
      jacobian_indices(inverse_order ? 5 : 1) = model_jac_start_index + 2 * (outer_x + outer_y * grid.width()) + 1;
      
      jacobian_indices(2) = model_jac_start_index + 2 * (inner1_x + inner1_y * grid.width()) + 0;
      jacobian_indices(3) = model_jac_start_index + 2 * (inner1_x + inner1_y * grid.width()) + 1;
      
      jacobian_indices(inverse_order ? 0 : 4) = model_jac_start_index + 2 * (inner2_x + inner2_y * grid.width()) + 0;
      jacobian_indices(inverse_order ? 1 : 5) = model_jac_start_index + 2 * (inner2_x + inner2_y * grid.width()) + 1;
      
      Matrix<double, 3, 3 * 3, RowMajor> raw_jacobian;
      
      // raw_jacobian.block<3, 3>(0, 0) = -regularization_weight * Mat3d::Identity();
      ComputeBorderRegularizationJacobian(
          inner1.x(), inner1.y(), inner1.z(),
          inner2.x(), inner2.y(), inner2.z(),
          raw_jacobian.row(0).data(), raw_jacobian.row(1).data(), raw_jacobian.row(2).data());
      
      Matrix<double, 3, 3 * 2> jacobian;
      Matrix<double, 3, 2> direction_wrt_update;
      
      DirectionJacobianWrtLocalUpdate(
          inverse_order ? tangents_image(inner2_x, inner2_y) : tangents_image(outer_x, outer_y),
          &direction_wrt_update);
      jacobian.block<3, 2>(0, 0) = raw_jacobian.block<3, 3>(0, inverse_order ? 6 : 0) * direction_wrt_update;
      
      DirectionJacobianWrtLocalUpdate(
          tangents_image(inner1_x, inner1_y),
          &direction_wrt_update);
      jacobian.block<3, 2>(0, 2) = raw_jacobian.block<3, 3>(0, 3) * direction_wrt_update;
      
      DirectionJacobianWrtLocalUpdate(
          inverse_order ? tangents_image(outer_x, outer_y) : tangents_image(inner2_x, inner2_y),
          &direction_wrt_update);
      jacobian.block<3, 2>(0, 4) = raw_jacobian.block<3, 3>(0, inverse_order ? 0 : 6) * direction_wrt_update;
      
      accumulator->AddResidualWithJacobian(
          regularization_weight * (mirror.x() - outer.x()),
          jacobian_indices,
          regularization_weight * jacobian.row(0));
      accumulator->AddResidualWithJacobian(
          regularization_weight * (mirror.y() - outer.y()),
          jacobian_indices,
          regularization_weight * jacobian.row(1));
      accumulator->AddResidualWithJacobian(
          regularization_weight * (mirror.z() - outer.z()),
          jacobian_indices,
          regularization_weight * jacobian.row(2));
    }
  }
  
  Dataset* dataset;
  vector<int> seq_to_original_index;
  double numerical_diff_delta;
  double regularization_weight;
  bool localize_only;
  bool eliminate_points;
  bool on_the_fly_block_processing;
};

double OptimizeJointly(
    Dataset& dataset,
    BAState* state,
    int max_iteration_count,
    double init_lambda,
    double numerical_diff_delta,
    double regularization_weight,
    bool localize_only,
    bool eliminate_points,
    SchurMode schur_mode,
    double* final_lambda,
    bool* performed_an_iteration,
    bool debug_verify_cost,
    bool debug_fix_points,
    bool debug_fix_poses,
    bool debug_fix_rig_poses,
    bool debug_fix_intrinsics,
    bool print_progress) {
  if (performed_an_iteration) {
    *performed_an_iteration = false;
  }
  
  // Define cost function
  JointOptimizationCostFunction cost_function;
  cost_function.dataset = &dataset;
  cost_function.numerical_diff_delta = numerical_diff_delta;
  cost_function.regularization_weight = regularization_weight;
  cost_function.localize_only = localize_only;
  cost_function.eliminate_points = eliminate_points;
  
  // Create sequential indexing for poses for the state
  vector<int> original_to_seq_index;
  JointOptimizationState opt_state(*state, &original_to_seq_index, &cost_function.seq_to_original_index, localize_only, eliminate_points);
  
  // Perform optimization.
  LMOptimizer<double> optimizer;
  
  int block_size;
  int num_blocks;
  if (eliminate_points) {
    block_size = 3;
    num_blocks = opt_state.points.size();
    cost_function.on_the_fly_block_processing = false;  // this would require to adapt the iteration order over the residuals
  } else {  // eliminate_imageset_poses
    block_size = SE3d::DoF;
    num_blocks = opt_state.rig_tr_global.size();
    cost_function.on_the_fly_block_processing = (schur_mode == SchurMode::DenseOnTheFly || schur_mode == SchurMode::SparseOnTheFly);
  }
  if (cost_function.on_the_fly_block_processing && init_lambda < 0) {
    // TODO: Always initialize lambda to this fixed value, also if not using on-the-fly block processing?
    init_lambda = 0.0001f;
  }
  optimizer.UseBlockDiagonalStructureForSchurComplement(
      block_size, num_blocks,
      /*sparse_storage_for_off_diag_H*/ (schur_mode == SchurMode::Sparse || schur_mode == SchurMode::SparseOnTheFly),
      cost_function.on_the_fly_block_processing,
      /*block_batch_size*/ 1024,
      /*compute_schur_complement_with_cuda*/ schur_mode == SchurMode::DenseCUDA);
  
  int rank_deficiency = 0;
  if (opt_state.intrinsics.size() > 1) {
    // If there is more than one camera, all camera_tr_rig transformations are
    // being optimized, which adds additional rank deficiency
    rank_deficiency += SE3d::DoF;
  }
  for (int camera_index = 0; camera_index < state->num_cameras(); ++ camera_index) {
    CameraModel* model = state->intrinsics[camera_index].get();
    if (dynamic_cast<CentralGenericModel*>(model)) {
      // - 3 for global rotation
      // - 3 for global translation
      // - 1 for global scaling
      // - 3 for rotating all camera poses in one direction and all calibrated
      //     observation directions in the opposite direction
      rank_deficiency += 10;
    } else if (dynamic_cast<NoncentralGenericModel*>(model)) {
      // - 3 for global rotation
      // - 3 for global translation
      // - 1 for global scaling
      // - 3 for rotating all camera poses in one direction and all calibrated
      //     lines in the opposite direction
      // - 3 for moving all camera poses in one direction and all calibrated
      //     lines in the opposite direction
      // - more directions in cases where the line origin points can be anywhere
      //     on the line (this is if the lines are (nearly-)parallel).
      rank_deficiency += 13;
    } else if (dynamic_cast<CentralThinPrismFisheyeModel*>(model) ||
               dynamic_cast<CentralOpenCVModel*>(model)) {
      // - 3 for global rotation
      // - 3 for global translation
      // - 1 for global scaling
      rank_deficiency += 7;
    } else if (dynamic_cast<CentralRadialModel*>(model)) {
      // - 3 for global rotation
      // - 3 for global translation
      // - 1 for global scaling
      // - TODO: Is there also an ambiguity between scaling the radial factors
      //         and scaling the focal length fx and fy?
      rank_deficiency += 7;
    } else {
      LOG(ERROR) << "Rank deficiency unknown for this camera model, assuming 0";
    }
  }
  if (rank_deficiency > 0) {
    // TODO: Not using AccountForRankDeficiency() at the moment! It does not
    //       seem to bring any benefit to use completeOrthogonalDecomposition().
    (void) rank_deficiency;
    // optimizer.AccountForRankDeficiency(rank_deficiency);
  }
  
  if (debug_verify_cost) {
    // Verify that the cost function evaluation
    // 1) yields the same result when computing Jacobians and when computing
    //    residuals only
    // 2) does not incorrectly change the state (therefore, call VerifyCost() twice)
    // NOTE: Since the cost computation caches the last projection results and
    //       re-uses them as initialization for the next projection, the results
    //       are not expected to be numerically exactly the same.
    double cost1 = optimizer.VerifyCost(&opt_state, cost_function);
    double cost2 = optimizer.VerifyCost(&opt_state, cost_function);
    CHECK_LE(fabs(cost1 - cost2), 1e-3f);
  }
  
  if (debug_fix_points) {
    int first_points_offset = opt_state.first_points_offset();
    for (usize i = 0; i < opt_state.points.size() * 3; ++ i) {
      optimizer.FixVariable(first_points_offset + i);
    }
  }
  if (debug_fix_rig_poses && opt_state.are_camera_tr_rig_in_state()) {
    int first_camera_tr_rig_offset = opt_state.first_camera_tr_rig_offset();
    for (usize i = 0; i < opt_state.camera_tr_rig.size() * SE3d::DoF; ++ i) {
      optimizer.FixVariable(first_camera_tr_rig_offset + i);
    }
  }
  if (debug_fix_poses) {
    int first_rig_tr_global_offset = opt_state.first_rig_tr_global_offset();
    for (usize i = 0; i < opt_state.rig_tr_global.size() * SE3d::DoF; ++ i) {
      optimizer.FixVariable(first_rig_tr_global_offset + i);
    }
  }
  if (debug_fix_intrinsics) {
    int first_intrinsics_offset = opt_state.first_intrinsics_offset();
    int num_parameters = opt_state.degrees_of_freedom();
    for (usize i = first_intrinsics_offset; i < num_parameters; ++ i) {
      optimizer.FixVariable(i);
    }
  }
  
  double final_cost = -1;
  for (int iteration = 0; iteration < max_iteration_count; ++ iteration) {
    // For debugging:
    // optimizer.VerifyAnalyticalJacobian(
    //     &opt_state,
    //     /*step_size*/ 0.0001,
    //     /*error_threshold*/ 0.01,
    //     cost_function,
    //     /*first_dof*/ opt_state.image_tr_pattern.size() * SE3d::DoF + opt_state.points.size() * 3,
    //     /*last_dof*/ opt_state.degrees_of_freedom() - 1);
    
    OptimizationReport report = optimizer.Optimize(
        &opt_state,
        cost_function,
        /*max_iteration_count*/ 1,
        /*max_lm_attempts*/ 50,
        init_lambda,
        /*init_lambda_factor*/ 0.00001,  // TODO: tune?
        print_progress);
    final_cost = report.final_cost;
    init_lambda = optimizer.lambda();
    if (final_lambda) {
      *final_lambda = optimizer.lambda();
    }
    
    if (report.num_iterations_performed == 0) {
      break;
    } else if (performed_an_iteration) {
      *performed_an_iteration = true;
    }
    
    int key = PollKeyInput();
    if (key == 'q') {
      break;
    }
  }
  
  // Read back the opt_state content
  state->camera_tr_rig = opt_state.camera_tr_rig;
  for (int i = 0; i < cost_function.seq_to_original_index.size(); ++ i) {
    state->rig_tr_global[cost_function.seq_to_original_index[i]] = opt_state.rig_tr_global[i];
  }
  state->points = opt_state.points;
  for (int camera_index = 0; camera_index < state->num_cameras(); ++ camera_index) {
    state->intrinsics[camera_index].reset(opt_state.intrinsics[camera_index]->duplicate());
  }
  
  return final_cost;
}

}

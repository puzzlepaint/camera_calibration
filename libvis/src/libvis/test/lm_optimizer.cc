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


#include "libvis/logging.h"
#include <gtest/gtest.h>

#include "libvis/eigen.h"
#include "libvis/lm_optimizer.h"
#include "libvis/sophus.h"

using namespace vis;

namespace {

struct SimpleLineFittingCostFunction {
  aligned_vector<Vector2f> data_points;
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const Vector2f& state,
      Accumulator* accumulator) const {
    for (const Vector2f& data_point : data_points) {
      const float x = data_point.x();
      const float y_state = state(0) * x + state(1);
      const float y_data = data_point.y();
      
      // Residual: m * x + t - y .
      const float residual = y_state - y_data;
      if (compute_jacobians) {
        const float dr_dm = x;
        const float dr_dt = 1;
        accumulator->AddResidualWithJacobian(residual, 0, Matrix<float, 1, 2>(dr_dm, dr_dt));
      } else {
        accumulator->AddResidual(residual);
      }
    }
  }
};

}

// Tests (and demonstrates) simple line fitting.
TEST(LMOptimizer, SimpleLineFitting) {
  // Define the residuals.
  SimpleLineFittingCostFunction cost_function;
  const float kM = 3;
  const float kT = 2;
  cost_function.data_points.push_back(Vector2f(0, kM * 0 + kT));
  cost_function.data_points.push_back(Vector2f(1, kM * 1 + kT));
  cost_function.data_points.push_back(Vector2f(2, kM * 2 + kT));
  
  // Use a state consisting of 2 floats.
  LMOptimizer<float> optimizer;
  
  // Set the initial estimate.
  Vector2f state = Vector2f(kM - 1, kT + 1);
  
  // Verify that the analytical Jacobian at this state equals the numerical
  // Jacobian (with default step size and precision threshold for all state
  // components).
  EXPECT_TRUE(optimizer.VerifyAnalyticalJacobian(&state, 1.f, numeric_limits<float>::epsilon(), cost_function));
  
  // Run the optimization.
  optimizer.Optimize(&state, cost_function, /*max_iteration_count*/ 100);
  
  // Verify that the correct result is returned.
  EXPECT_FLOAT_EQ(kM, state(0));
  EXPECT_FLOAT_EQ(kT, state(1));
}

namespace {

/// Optimization state that contains an SE3f value itself.
struct SE3fState {
  /// Initializes the transformation to identity.
  inline SE3fState() {}
  
  inline SE3fState(const SE3fState& other)
      : dest_TR_src(other.dest_TR_src) {}
  
  inline int degrees_of_freedom() const {
    return SE3f::DoF;
  }
  
  static constexpr bool is_reversible() { return true; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    // Using minus delta here since we are subtracting. Using
    // left-multiplication of the delta must be consistent with the way the
    // Jacobian is computed.
    dest_TR_src = SE3f::exp(-delta) * dest_TR_src;
  }
  
  /// Convenience function; its only purpose is to homogenize access to
  /// dest_TR_src to be able to use SE3fState and ExternalSE3fState
  /// interchangeably in the CostFunction implementation.
  inline const SE3f& get_dest_TR_src() const {
    return dest_TR_src;
  }
  
  SE3f dest_TR_src;
};

/// Optimization state that demonstrates optimization of external values.
/// This state simply wraps an existing SE3f, avoiding to copy it. As a result,
/// the rest of the code (in particular, the cost and Jacobian computation) can
/// always consistently access that value in its original form and location
/// (which might for example be being a member of a larger class) both inside
/// and outside of the optimization process.
struct ExternalSE3fState {
  inline ExternalSE3fState(SE3f* dest_TR_src)
      : dest_TR_src(dest_TR_src) {}
  
  inline int degrees_of_freedom() const {
    return SE3f::DoF;
  }
  
  static constexpr bool is_reversible() { return true; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    // Using minus delta here since we are subtracting. Using
    // left-multiplication of the delta must be consistent with the way the
    // Jacobian is computed.
    *dest_TR_src = SE3f::exp(-delta) * (*dest_TR_src);
  }
  
  /// Convenience function; its only purpose is to homogenize access to
  /// dest_TR_src to be able to use SE3fState and ExternalSE3fState
  /// interchangeably in the CostFunction implementation.
  inline const SE3f& get_dest_TR_src() const {
    return *dest_TR_src;
  }
  
  SE3f* dest_TR_src;
};

// TODO: For SE3, it is not required to use the exp() function to do local
//       optimization updates. It is sufficient to use SO3's exp() and handle
//       the translation parameters normally, which is supposedly slightly
//       faster. Make an implementation of that as well.

struct MatchedPointsDistanceCostFunction {
  aligned_vector<Vec3f> src_points;
  aligned_vector<Vec3f> dest_points;
  
  template<bool compute_jacobians, class Accumulator, class State>
  inline void Compute(
      const State& state,
      Accumulator* accumulator) const {
    for (usize i = 0, size = src_points.size(); i < size; ++ i) {
      const Vec3f& src = src_points[i];
      const Vec3f& dest_data = dest_points[i];
      
      /// Transformed source point.
      const Vec3f dest_state = state.get_dest_TR_src() * src;
      
      /// Residuals: (T * src)[c] - dest[c] for each of the 3 vector components
      /// c. Together, this is the squared norm of the vector difference.
      const Vec3f residuals = dest_state - dest_data;
      if (compute_jacobians) {
        // Jacobian of: exp(hat(delta)) * T * src - dest , wrt. delta.
        // The derivation is in:
        // scripts/LMOptimizer SE3Optimization Test Jacobian derivation.ipynb.
        // The raw Jacobian is:
        // 1, 0, 0, 0, dest_state(2), -dest_state(1)
        // 0, 1, 0, -dest_state(2), 0, dest_state(0)
        // 0, 0, 1, dest_state(1), -dest_state(0), 0
        // The AddJacobian() calls avoid including the zeros.
        // Note that a one-element matrix constructor does not exist in Eigen,
        // thus we use (Matrix<...>() << element_value).finished().
        accumulator->AddResidualWithJacobian(residuals(0), 0, (Matrix<float, 1, 1>() << 1).finished(), 4, Matrix<float, 1, 2>(dest_state(2), -dest_state(1)));
        accumulator->AddResidualWithJacobian(residuals(1), 1, (Matrix<float, 1, 1>() << 1).finished(), 3, Matrix<float, 1, 3>(-dest_state(2), 0, dest_state(0)));
        accumulator->AddResidualWithJacobian(residuals(2), 2, Matrix<float, 1, 3>(1, dest_state(1), -dest_state(0)));
      } else {
        accumulator->AddResidual(residuals(0));
        accumulator->AddResidual(residuals(1));
        accumulator->AddResidual(residuals(2));
      }
    }
  }
};

}

// Tests (and demonstrates) SE3 pose optimization with point correspondences.
// NOTE: This implementation can only rotate around the coordinate system
//       origin, therefore the choice of origin is important! If the actual
//       rotation origin is hard to simulate via rotation around the origin and
//       translation, the optimization will likely not converge. To fix this,
//       the exp(hat(delta)) should be multiplied with T from the other side.
TEST(LMOptimizer, SE3Optimization) {
  // Define the residuals.
  MatchedPointsDistanceCostFunction cost_function;
  cost_function.src_points.push_back(Vec3f(1, 2, 3));
  cost_function.src_points.push_back(Vec3f(3, 2, 1));
  cost_function.src_points.push_back(Vec3f(1, 1, 2));
  cost_function.src_points.push_back(Vec3f(4, 2, 2));
  cost_function.src_points.push_back(Vec3f(2, 2, 1));
  
  SE3f ground_truth_dest_TR_src =
      SE3f(Quaternionf(AngleAxisf(0.42f, Vec3f(1, 3, 2).normalized())),
           Vec3f(0.5f, 0.6f, 0.7f));
  cost_function.dest_points.resize(cost_function.src_points.size());
  for (usize i = 0; i < cost_function.src_points.size(); ++ i) {
    cost_function.dest_points[i] = ground_truth_dest_TR_src * cost_function.src_points[i];
  }
  
  // Use a custom state for the pose.
  LMOptimizer<float> optimizer;
  
  // Set the initial estimate to identity.
  SE3fState state;
  state.dest_TR_src = SE3f();
  
  // Verify that the analytical Jacobian at this state equals the numerical
  // Jacobian (with default step size and precision threshold for all state
  // components).
  EXPECT_TRUE(optimizer.VerifyAnalyticalJacobian(&state, 1e-3f, 1.1e-2f, cost_function));
  
  // Run the optimization.
  optimizer.Optimize(
      &state,
      cost_function,
      /*max_iteration_count*/ 100);
  
  // Verify that the correct result is returned.
  constexpr float kErrorTolerance = 0.0001f;
  SE3f::Tangent error = SE3f::log(state.dest_TR_src.inverse() * ground_truth_dest_TR_src);
  EXPECT_LE(error(0), kErrorTolerance);
  EXPECT_LE(error(1), kErrorTolerance);
  EXPECT_LE(error(2), kErrorTolerance);
  EXPECT_LE(error(3), kErrorTolerance);
  EXPECT_LE(error(4), kErrorTolerance);
  EXPECT_LE(error(5), kErrorTolerance);
  
  // Do the same with the external SE3 state implementation.
  SE3f value = SE3f();
  ExternalSE3fState external_state(&value);
  optimizer.Optimize(
      &external_state,
      cost_function,
      /*max_iteration_count*/ 100);
  error = SE3f::log(value.inverse() * ground_truth_dest_TR_src);
  EXPECT_LE(error(0), kErrorTolerance);
  EXPECT_LE(error(1), kErrorTolerance);
  EXPECT_LE(error(2), kErrorTolerance);
  EXPECT_LE(error(3), kErrorTolerance);
  EXPECT_LE(error(4), kErrorTolerance);
  EXPECT_LE(error(5), kErrorTolerance);
}

namespace {

struct SchurComplementTestState {
  vector<Vec2f>* estimated_feature_positions;
  vector<Vec2f>* estimated_image_positions;
  
  SchurComplementTestState(
      vector<Vec2f>* estimated_feature_positions,
      vector<Vec2f>* estimated_image_positions)
      : estimated_feature_positions(estimated_feature_positions),
        estimated_image_positions(estimated_image_positions) {}
  
  inline int degrees_of_freedom() const {
    // The last image position remains fixed to fix the gauge freedom.
    return 2 * estimated_feature_positions->size() +
           2 * (estimated_image_positions->size() - 1);
  }
  
  static constexpr bool is_reversible() { return true; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    for (usize feature_index = 0; feature_index < estimated_feature_positions->size(); ++ feature_index) {
      estimated_feature_positions->at(feature_index) -= Vec2f(
          delta(2 * feature_index + 0),
          delta(2 * feature_index + 1));
    }
    for (usize image_index = 0; image_index < (estimated_image_positions->size() - 1); ++ image_index) {
      estimated_image_positions->at(image_index) -= Vec2f(
          delta(2 * estimated_feature_positions->size() + 2 * image_index + 0),
          delta(2 * estimated_feature_positions->size() + 2 * image_index + 1));
    }
  }
};

struct SchurComplementTestCostFunction {
  /// Indexed by: [image_index][feature_index] .
  vector<vector<Vec2f>>* observations;
  
  inline SchurComplementTestCostFunction(vector<vector<Vec2f>>* observations)
      : observations(observations) {}
  
  template<bool compute_jacobians, class Accumulator, class State>
  inline void Compute(
      const State& state,
      Accumulator* accumulator) const {
    for (usize image_index = 0, image_count = observations->size(); image_index < image_count; ++ image_index) {
      auto& image_observations = observations->at(image_index);
      for (usize feature_index = 0, feature_count = image_observations.size(); feature_index < feature_count; ++ feature_index) {
        const Vec2f& measurement = image_observations[feature_index];
        const Vec2f current = state.estimated_feature_positions->at(feature_index) -
                              state.estimated_image_positions->at(image_index);
        
        // We define the residuals as:
        // current.x/y - measurement.x/y
        if (compute_jacobians) {
          // NOTE: Since some entries of the Jacobian are always zero, we
          //       could narrow down the Jacobian vectors, but this is not
          //       done here for improved clarity.
          accumulator->AddResidualWithJacobian(
              current.x() - measurement.x(),
              2 * feature_index,
              Matrix<float, 1, 2>(1, 0),  // Jacobian wrt. feature position
              2 * feature_count + 2 * image_index,
              Matrix<float, 1, 2>(-1, 0),  // Jacobian wrt. image position
              true,
              image_index != image_count - 1);  // do not use the Jacobian for the fixed image
          accumulator->AddResidualWithJacobian(
              current.y() - measurement.y(),
              2 * feature_index,
              Matrix<float, 1, 2>(0, 1),  // Jacobian wrt. feature position
              2 * feature_count + 2 * image_index,
              Matrix<float, 1, 2>(0, -1),  // Jacobian wrt. image position
              true,
              image_index != image_count - 1);  // do not use the Jacobian for the fixed image
        } else {
          accumulator->AddResidual(current.x() - measurement.x());
          accumulator->AddResidual(current.y() - measurement.y());
        }
      }
    }
  }
};

}

/// Tests fast optimization using the Schur complement. Simulates a strongly
/// simplified bundle adjustment problem in 2D space, where images and point
/// features are defined by 2D positions, and observations are defined by
/// relative 2D vectors between an image and a point feature.
TEST(LMOptimizer, SchurComplement) {
  // Create random ground truth "images" (defined by a 2D position).
  constexpr int kNumImages = 10;
  vector<Vec2f> gt_image_positions(kNumImages);
  for (int image_index = 0; image_index < kNumImages; ++ image_index) {
    gt_image_positions[image_index] = Vec2f::Random();
  }
  
  // Create random ground truth "features" (defined by a 2D position).
  constexpr int kNumFeatures = 20;
  vector<Vec2f> gt_feature_positions(kNumFeatures);
  for (int feature_index = 0; feature_index < kNumFeatures; ++ feature_index) {
    gt_feature_positions[feature_index] = Vec2f::Random();
  }
  
  // Derive observations from the ground truth image and feature positions.
  // We make every image observe every feature for simplicity (so we do not need
  // to care about ensuring connectedness).
  /// Indexed by: [image_index][observation_index] . Stores: feature_position - image_position.
  vector<vector<Vec2f>> observations;
  
  observations.resize(kNumImages);
  for (int image_index = 0; image_index < kNumImages; ++ image_index) {
    observations[image_index].resize(kNumFeatures);
    for (int feature_index = 0; feature_index < kNumFeatures; ++ feature_index) {
      observations[image_index][feature_index] =
          gt_feature_positions[feature_index] - gt_image_positions[image_index];
    }
  }
  
  // Disturb all positions except the last image position (which anchors the
  // whole structure).
  vector<Vec2f> distorted_image_positions = gt_image_positions;
  vector<Vec2f> distorted_feature_positions = gt_feature_positions;
  
  for (int image_index = 0; image_index < kNumImages - 1; ++ image_index) {
    distorted_image_positions[image_index] += 0.1 * Vec2f::Random();
  }
  for (int feature_index = 0; feature_index < kNumFeatures; ++ feature_index) {
    distorted_feature_positions[feature_index] += 0.1 * Vec2f::Random();
  }
  
  // Define state and cost function to wrap the values defined above.
  // We use the following variable ordering in these wrappers:
  // First, the feature positions are listed, then the image positions.
  // This is because the first ones will be inverted quickly within the Schur
  // complement, and we expect to have more feature variables than image
  // variables.
  vector<Vec2f> estimated_image_positions = distorted_image_positions;
  vector<Vec2f> estimated_feature_positions = distorted_feature_positions;
  
  SchurComplementTestState state(&estimated_feature_positions, &estimated_image_positions);
  SchurComplementTestCostFunction cost_function(&observations);
  LMOptimizer<float> optimizer;
  
  // Run the optimization, first without using the Schur complement to confirm
  // that the formulation is sane in general.
  optimizer.Optimize(
      &state,
      cost_function,
      /*max_iteration_count*/ 100);
  
  // Verify that the correct result is returned.
  auto verify_result = [&](const char* message) {
    constexpr float kErrorTolerance = 0.0001f;
    for (int image_index = 0; image_index < kNumImages; ++ image_index) {
      EXPECT_NEAR(estimated_image_positions[image_index].x(),
                  gt_image_positions[image_index].x(),
                  kErrorTolerance) << message;
      EXPECT_NEAR(estimated_image_positions[image_index].y(),
                  gt_image_positions[image_index].y(),
                  kErrorTolerance) << message;
    }
    for (int feature_index = 0; feature_index < kNumFeatures; ++ feature_index) {
      EXPECT_NEAR(estimated_feature_positions[feature_index].x(),
                  gt_feature_positions[feature_index].x(),
                  kErrorTolerance) << message;
      EXPECT_NEAR(estimated_feature_positions[feature_index].y(),
                  gt_feature_positions[feature_index].y(),
                  kErrorTolerance) << message;
    }
  };
  verify_result("Error during initial run (without Schur complement).");
  
  // Now run the optimization again with the Schur complement to test it.
  estimated_image_positions = distorted_image_positions;
  estimated_feature_positions = distorted_feature_positions;
  
  optimizer.UseBlockDiagonalStructureForSchurComplement(
      2, estimated_feature_positions.size(), false, false, 1, false);  // TODO: Test with sparse off-diagonal storage as well
  
  optimizer.Optimize(
      &state,
      cost_function,
      /*max_iteration_count*/ 100);
  
  // Verify that the correct result is returned again.
  verify_result("Error during run with Schur complement.");
}

namespace vis {
class LMOptimizerTestHelper {
 public:
  LMOptimizerTestHelper(LMOptimizer<float>* optimizer)
      : optimizer(optimizer) {}
  
  // Derivation in octave / Matlab:
  // H = [1 5 0 0 3 4;
  //      5 6 0 0 7 8;
  //      0 0 9 5 7 6;
  //      0 0 5 4 3 2;
  //      3 7 7 3 1 4;
  //      4 8 6 2 4 7];
  // b = [1; 2; 3; 4; 5; 6];
  // 
  // H \ b
  // 
  // % Result:
  //     73.667
  //    171.667
  //    189.667
  //   -294.333
  //    465.667
  //   -582.000
  void Test() {
    double nan = numeric_limits<float>::quiet_NaN();
    
    optimizer->m_use_block_diagonal_structure = true;
    optimizer->m_block_size = 2;
    optimizer->m_num_blocks = 2;
    
    optimizer->m_block_diag_H.resize(2);
    optimizer->m_block_diag_H[0].resize(2, 2);
    optimizer->m_block_diag_H[0] <<
          1, 5,
        nan, 6;
    
    optimizer->m_block_diag_H[1].resize(2, 2);
    optimizer->m_block_diag_H[1] <<
          9, 5,
        nan, 4;
    
    optimizer->m_dense_H.resize(2, 2);
    optimizer->m_dense_H <<
          1, 4,
        nan, 7;
    
    optimizer->m_off_diag_H.resize(4, 2);
    optimizer->m_off_diag_H <<
        3, 4,
        7, 8,
        7, 6,
        3, 2;
    
    optimizer->m_block_diag_b.resize(4);
    optimizer->m_block_diag_b << 1, 2, 3, 4;
    optimizer->m_dense_b.resize(2);
    optimizer->m_dense_b << 5, 6;
    
    optimizer->m_x.resize(6);
    // This uses a dummy state and cost function. Those should not be used in
    // this call, since m_on_the_fly_block_processing == false.
    optimizer->SolveWithSchurComplementDenseOffDiag<Vec2f>(
        optimizer->m_block_size * optimizer->m_num_blocks,
        /*dense_degrees_of_freedom*/ 2,
        nullptr,
        SimpleLineFittingCostFunction());
    
    EXPECT_NEAR(optimizer->m_x(0),   73.667, 0.3);
    EXPECT_NEAR(optimizer->m_x(1),  171.667, 0.3);
    EXPECT_NEAR(optimizer->m_x(2),  189.667, 0.3);
    EXPECT_NEAR(optimizer->m_x(3), -294.333, 0.3);
    EXPECT_NEAR(optimizer->m_x(4),  465.667, 0.3);
    EXPECT_NEAR(optimizer->m_x(5), -582.000, 0.3);
  }
  
 private:
  LMOptimizer<float>* optimizer;
};
}

/// Tests that matrix solving using the Schur complement returns the correct
/// result vector for a randomly chosen example.
TEST(LMOptimizer, SchurComplement2) {
  LMOptimizer<float> optimizer;
  LMOptimizerTestHelper helper(&optimizer);
  helper.Test();
}

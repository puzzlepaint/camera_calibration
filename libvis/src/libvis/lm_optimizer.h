// Copyright 2017-2019 ETH Zürich, Thomas Schöps
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

#include <fstream>

// Only required for optional matrix multiplication on the GPU:
#include <cublasXt.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <Eigen/Sparse>

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/logging.h"
#include "libvis/loss_functions.h"
#include "libvis/lm_optimizer_back_substitution_accumulator.h"
#include "libvis/lm_optimizer_cost_accumulator.h"
#include "libvis/lm_optimizer_impl.h"
#include "libvis/lm_optimizer_residual_sum_and_jacobian_accumulator.h"
#include "libvis/lm_optimizer_update_accumulator.h"
#include "libvis/timing.h"

namespace vis {

/// Provides information about an optimization run by LMOptimizer.
struct OptimizationReport {
  void Print() {
    LOG(INFO) << "Cost: " << initial_cost << " --> " << final_cost;
    LOG(INFO) << "Iterations: " << num_iterations_performed;
    LOG(INFO) << "Time [s]: " << cost_and_jacobian_evaluation_time << " for cost & jacobian evaluation";
    LOG(INFO) << "          " << solve_time << " for solving";
  }
  
  /// Initial cost value before optimization.
  double initial_cost;
  
  /// Final cost value after optimization.
  double final_cost;
  
  /// Number of optimization iterations performed.
  int num_iterations_performed;
  
  /// Time in seconds taken by evaluating the cost function and its Jacobian.
  double cost_and_jacobian_evaluation_time;
  
  /// Time in seconds taken by solving for updates.
  double solve_time;
};


// Generic class for continuous non-linear optimization with the
// Levenberg-Marquardt method. See for example the PhD thesis of Thomas Schöps
// for an explanation and derivation. Below, only the usage of the class is
// explained.
// 
// Note that the update equation in general is:
//   dx = H^(-1) -b
// After solving this (either directly or indirectly with something like PCG
// (preconditioned conjugate gradients)), the state can be updated as follows:
//   x := x + dx
// However, this implementation here instead solves:
//   dx = H^(-1) b (note the missing minus in front of b)
// As a result, dx must be subtracted from x instead of being added it.
// 
// The user of the class defines the cost function. Optionally, analytical
// Jacobian computation can also be provided by the user for increased
// performance.
// 
// The template parameters must be given as follows:
// * Scalar should be float or double, determining the numerical precision used
//   for computing the cost and Jacobians.
// * State holds the optimization problem state and is typically an
//   Eigen::Vector. However, for optimization using Lie algebras of SE3 or SO3,
//   for example, one can provide a custom type which stores the rotation
//   part(s) of the state as quaternion while applying updates using the
//   exponential map on the corresponding Lie algebra element. It is also
//   possible to define the state to be a wrapper on some existing data
//   structure whose contents are optimized, for example, the pixels of an
//   image. This avoids the need to create multiple representations for it.
// * CostFunction computes the cost for a given state and can
//   optionally provide analytical Jacobians for increased optimization speed.
//   Analytical Jacobians should always be used if performance is of concern.
// 
// The class passed for State must provide the following:
// 
// class State {
//  public:
//   // Either a copy constructor and operator= must be provided that actually
//   // copy the data (i.e., if the state contains pointers to external data,
//   // the data pointed to must be copied!), as follows:
//   State(const State& other);
//   State& operator=(const State& other);
//   
//   // Or the state must be reversible (i.e., doing "state -= x" is always
//   // reverted to the original value by following it up with "state -= -x" for
//   // any state and any x), and declare this by defining the following member
//   // function:
//   static constexpr bool is_reversible() { return true; }
//   // NOTE: The optimizer also assumes classes to be reversible which define a
//   // function "rows()". This is a HACK to support Eigen::Matrix vectors as
//   // states. TODO: Couldn't we do better and detect Eigen::Matrix directly
//   // using partial template specializations?
//   
//   // Returns the number of variables in the optimization problem. For
//   // example, for fitting a 2D line represented by the equation m * x + t,
//   // this should return 2 (as the parameters are m and t). Note that as an
//   // exception, this function can also be named rows(), which makes it
//   // possible to use an Eigen::Matrix vector for the state. TODO: See above,
//   // couldn't we detect Eigen::Matrix more directly?
//   int degrees_of_freedom() const;
//   
//   // Subtracts a delta vector from the state. The delta is computed by
//   // LMOptimizer and its row count equals the return value of
//   // degrees_of_freedom(). In the simplest case, the State class will subtract
//   // the corresponding delta vector component from each state variable,
//   // however for cases such as optimization over Lie groups, the
//   // implementation can differ.
//   template <typename Derived>
//   void operator-=(const MatrixBase<Derived>& delta);
// };
// 
// The class passed for CostFunction must provide the following:
//
// class CostFunction {
//  public:
//   // Computes the cost for a given state by providing the values of all
//   // residuals, and optionally the Jacobians of the residuals wrt. the
//   // variables if supported.
//   template<bool compute_jacobians, class Accumulator>
//   inline void Compute(
//       const State& state,
//       Accumulator* accumulator) const {
//     for (residual : residuals) {  // loop over all residuals
//       // To add a residual (r_i in the generic cost term above) to the cost,
//       // call the following, depending on whether compute_jacobians is true.
//       // This expects the non-squared residual. Jacobian computations should
//       // be omitted if compute_jacobians is false.
//       // TODO: Support optionally using Gauss-Newton only. Will this remove the need to have the !compute_jacobians case?
//       Scalar residual_value = ...;  // compute residual
//       if (compute_jacobians) {
//         int index = ...;  // get / compute Jacobian indices
//         Matrix<Scalar, rows, 1> jacobian = ...;  // compute jacobian
//         accumulator->AddResidualWithJacobian(index, residual_value, jacobian);
//       } else {
//         accumulator->AddResidual(residual_value);
//       }
//     }
//   }
// };
// 
// Step-by-step instructions to set up an optimization process with this class:
// 
// 1. Think about the optimization state (i.e., the optimized variables) and the
//    cost function.
//    
//    For example, for simple affine function fitting of a function
//    y = m * x + t, the cost would be a sum over pow(m * x + t - y, 2) for all
//    data points (x, y), and the state can be defined as (m, t)^T (or
//    equivalently, (t, m)^T).
// 
// 2. Choose / implement a class to hold the state, and a sequential indexing of
//    all variables within the state, if needed.
//    
//    In the line fitting example, an Eigen::Vector2f would be well-suited to
//    hold the state and the variables can be enumerated as (0 : m, 1: t) (or
//    the other way round).
//    
//    If the state contains many variables from an existing data structure, it
//    might be beneficial to define the state class as a wrapper on this
//    existing data structure (following the State scheme given above). This
//    avoids the need to copy the variables into the state before optimization
//    and back into their original place after optimization. Furthermore, this
//    allows the code to consistently access these variables in their original
//    place and form, even the cost function (residual and Jacobian) computation
//    code.
//    In this case, for the manual implementation of operator -=, the sequential
//    indexing is required to map components from the parameter of this function
//    (the delta vector) to the variables in the state.
//    
//    The variable indexing might be able to account for some structure in the
//    variables. One might want to list some variables first that form a
//    block-diagonal sub-matrix in the Hessian. This later enables
//    to use the Schur complement to solve for the update much faster. (Having
//    the relevant variables first is an assumption that the implementation
//    makes without loss of generality.)
// 
// 3. Implement the residual and Jacobian calculation according to the scheme
//    given above (CostFunction). The Compute() function must loop over all
//    residuals and add their values (and their Jacobians if compute_jacobians
//    is true) to the accumulator object using the functions it provides
//    (AddResidual, AddResidualWithJacobian, ...). The Jacobian indexing refers
//    to the variable ordering defined above in step 2.
// 
// 4. To set up and solve a problem, proceed as follows:
//      
//      CostFunction cost_function;
//      // Set up the cost function (e.g., collect residuals) ...
//      
//      State state;
//      // Initialize the state ...
//      
//      LMOptimizer<Scalar> optimizer;  // choose float or double for Scalar
//      optimizer.Optimize(
//          &state,
//          cost_function,
//          max_iteration_count,
//          max_lm_attempts,  //  = 10
//          init_lambda,  //  = -1
//          init_lambda_factor,  // = static_cast<Scalar>(0.001),
//          print_progress);  // = false
//      
//      // Use the optimized state ...
// 
// 
// TODO:
// - Implement using numerical derivatives for optimization. How to efficiently
//   get the derivatives of individual residuals? Can we optionally require
//   additional methods for this? Is it possible to initiate the computations
//   in a variant of the AddJacobian() call?
// - Performance advice (from the LLT decomposition's documentation):
//   "For best performance, it is recommended to use a column-major storage
//   format with the Lower triangular part (the default), or, equivalently, a
//   row-major storage format with the Upper triangular part. Otherwise, you
//   might get a 20% slowdown for the full factorization step, and rank-updates
//   can be up to 3 times slower."
template<typename Scalar>
class LMOptimizer {
 public:
  /// Helper struct which creates a copy of the object passed in the constructor
  /// if DoCopy == true, or simply references the existing object if DoCopy ==
  /// false. The copy or reference can be gotten with GetObject().
  template <bool DoCopy, typename T>
  struct OptionalCopy {
    // Implements the case DoCopy == false
    inline OptionalCopy(T* object)
        : object(object) {}
    
    inline T* GetObject() {
      return object;
    }
    
    T* object;
  };
  
  template <typename T>
  struct OptionalCopy<true, T> {
    // Implements the case DoCopy == true
    inline OptionalCopy(T* object)
        : object(*object) {}
    
    inline T* GetObject() {
      return &object;
    }
    
    T object;
  };
  
  
  /// Tells the optimizer about the block-diagonal structure of the first
  /// (block_size * num_blocks) variables within the problem. This enables it to
  /// use the Schur complement when solving for the update to speed this step up
  /// greatly.
  /// 
  /// If sparse_storage_for_off_diag_H is true, the off-diagonal parts of H will
  /// be stored as a sparse (rather than dense) matrix. This can be faster in
  /// cases of high sparsity (which depends on the concrete problem).
  /// 
  /// If on_the_fly_block_processing is true, the cost function class must
  /// accumulate the residuals and Jacobians for the blocks on the block-diagonal
  /// part of H that will be used for the Schur complement in order, block by
  /// block. After finishing a block, it must call FinishedBlockForSchurComplement()
  /// on the accumulator. This allows the blocks (in the block-diagonal and
  /// off-diagonal parts of H, and in the corresponding part of b) to be processed
  /// on the fly, avoiding to store all of them at the same time, which saves memory.
  /// 
  /// If on_the_fly_block_processing is true, block_batch_size determines how
  /// many blocks are accumulated before being processed. If sparse_storage_for_off_diag_H
  /// is true, this value is ignored, since 1 is always used then.
  /// 
  /// If compute_schur_complement_with_cuda is true, a large matrix multiplication
  /// in the computation of the Schur complement is done with CUDA. This only takes effect
  /// if sparse_storage_for_off_diag_H == false and on_the_fly_block_processing == false.
  /// 
  /// TODO: Currently, the implementation requires that all blocks are of the
  /// same size.
  void UseBlockDiagonalStructureForSchurComplement(
      int block_size, int num_blocks, bool sparse_storage_for_off_diag_H,
      bool on_the_fly_block_processing, int block_batch_size,
      bool compute_schur_complement_with_cuda) {
    m_use_block_diagonal_structure = true;
    m_block_size = block_size;
    m_num_blocks = num_blocks;
    m_sparse_storage_for_off_diag_H = sparse_storage_for_off_diag_H;
    m_on_the_fly_block_processing = on_the_fly_block_processing;
    m_block_batch_size = sparse_storage_for_off_diag_H ? 1 : block_batch_size;
    m_compute_schur_complement_with_cuda = compute_schur_complement_with_cuda && !sparse_storage_for_off_diag_H && !on_the_fly_block_processing;
    
    LOG(1) << "Using the Schur complement (block_size: " << m_block_size << ", num_blocks: " << m_num_blocks << ")";
    if (m_sparse_storage_for_off_diag_H) {
      LOG(1) << "- with sparse storage for off-diagonal H";
    }
    if (m_on_the_fly_block_processing) {
      LOG(1) << "- with on-the-fly block processing (block_batch_size: " << m_block_batch_size << ")";
    }
    if (m_compute_schur_complement_with_cuda) {
      LOG(1) << "Using CUDA within the Schur complement computation";
    }
  }
  
  /// Sets the rank deficiency of the Hessian (e.g., due to gauge freedom) that
  /// should be accounted for in solving for state updates by setting the
  /// corresponding number of least-constrained variable updates to zero.
  /// TODO: Currently, this value is not used.
  void AccountForRankDeficiency(int rank_deficiency) {
    m_rank_deficiency = rank_deficiency;
  }
  
  /// Sets whether to use completeOrthogonalDecomposition() for matrix solving.
  /// This is slower than the default, but may perhaps behave better in case
  /// there is Gauge freedom.
  void UseCompleteOrthogonalDecomposition(bool enable) {
    m_use_complete_orthogonal_decomposition = enable;
  }
  
  /// Removes a variable from the optimization, fixing it to its initialization.
  /// NOTE: The current implementation of this is potentially very slow! It is
  ///       recommended to use this for debug purposes only. Notice that removing
  ///       a variable in this way will always be slower than setting up the
  ///       problem without this variable (treating it as a constant) in the
  ///       first place. This function is provided for convenience only.
  void FixVariable(int index) {
    if (fixed_variables.size() < index + 1) {
      fixed_variables.resize(index + 1, false);
    }
    if (!fixed_variables[index]) {
      ++ m_num_fixed_variables;
    }
    fixed_variables[index] = true;
  }
  
  /// Opposite of FixVariable().
  void FreeVariable(int index) {
    if (fixed_variables.size() < index + 1) {
      return;  // variables are free by default if there is no fixed_variables entry for them
    }
    if (fixed_variables[index]) {
      -- m_num_fixed_variables;
    }
    fixed_variables[index] = false;
  }
  
  /// Runs the optimization until convergence is assumed. The initial state is
  /// passed in as pointer "state". This state is modified by the function and
  /// set to the final state after it returns. The final cost value is given as
  /// function return value.
  /// TODO: Allow to specify the strategy for initialization and update of
  ///       lambda. Can also easily provide a Gauss-Newton implementation then by
  ///       checking for the special case lambda = 0 and not retrying the update
  ///       then.
  template <class State, class CostFunction>
  OptimizationReport Optimize(
      State* state,
      const CostFunction& cost_function,
      int max_iteration_count,
      int max_lm_attempts = 10,
      Scalar init_lambda = -1,
      Scalar init_lambda_factor = static_cast<Scalar>(0.001),
      bool print_progress = false) {
    constexpr bool is_reversible = IsReversibleGetter<State>::eval();
    return OptimizeImpl<State, CostFunction, is_reversible>(
        state,
        cost_function,
        max_iteration_count,
        max_lm_attempts,
        init_lambda,
        init_lambda_factor,
        print_progress);
  }
  
  /// Verifies the analytical cost Jacobian provided by CostFunction
  /// by comparing it to the numerically calculated value. This is done for
  /// the current state. NOTE: This refers to the Jacobian of the total cost wrt.
  /// the state variables. It does not check each residual's individual Jacobian.
  /// TODO: Allow setting step size and precision threshold for each state
  ///       component.
  template <class State, class CostFunction>
  bool VerifyAnalyticalJacobian(
      State* state,
      Scalar step_size,
      Scalar error_threshold,
      const CostFunction& cost_function,
      int first_dof = 0,
      int last_dof = -1) {
    // Determine the variable count of the optimization problem.
    const int degrees_of_freedom = DegreesOfFreedomGetter<State>::eval(*state);
    CHECK_GT(degrees_of_freedom, 0);
    
    if (last_dof < 0) {
      last_dof = degrees_of_freedom - 1;
    }
    
    // Determine cost at current state.
    ResidualSumAndJacobianAccumulator<Scalar> helper(degrees_of_freedom);
    cost_function.template Compute<true>(*state, &helper);
    const Scalar base_residual_sum = helper.residual_sum();
    const Scalar base_cost = helper.cost();
    
    // NOTE: Using forward differences only for now.
    bool have_error = false;
    for (int variable_index = first_dof; variable_index <= last_dof; ++ variable_index) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> delta;
      delta.resize(degrees_of_freedom, Eigen::NoChange);
      delta.setZero();
      // Using minus step size since the delta will be subtracted.
      delta(variable_index) = -step_size;
      
      State test_state(*state);
      test_state -= delta;
      ResidualSumAndJacobianAccumulator<Scalar> test_helper(degrees_of_freedom);
      cost_function.template Compute<false>(test_state, &test_helper);
      const Scalar test_residual_sum = test_helper.residual_sum();
      const Scalar test_cost = test_helper.cost();
      
      Scalar analytical_jacobian_component = helper.jacobian()(variable_index);
      Scalar numerical_jacobian_component = (test_residual_sum - base_residual_sum) / step_size;
      
      Scalar error = fabs(analytical_jacobian_component - numerical_jacobian_component);
      if (error <= error_threshold) {
        LOG(1) << "VerifyAnalyticalJacobian(): Component " << variable_index << " ok (diff: " << fabs(analytical_jacobian_component - numerical_jacobian_component) << ")";
      } else {
        LOG(ERROR) << "VerifyAnalyticalJacobian(): Component " << variable_index
                   << " differs (diff: " << fabs(analytical_jacobian_component - numerical_jacobian_component)
                   << "): Analytical: " << analytical_jacobian_component
                   << ", numerical: " << numerical_jacobian_component
                   << " (base_cost: " << base_cost << ", test_cost: "
                   << test_cost << ")";
        have_error = true;
      }
    }
    return !have_error;
  }
  
  /// Verifies the cost computed by the cost function.
  template <class State, class CostFunction>
  Scalar VerifyCost(
      State* state,
      const CostFunction& cost_function) {
    CostAccumulator<Scalar> helper1;
    cost_function.template Compute<false>(*state, &helper1);
    
    CostAccumulator<Scalar> helper2;
    cost_function.template Compute<true>(*state, &helper2);
    
    if (fabs(helper1.cost() - helper2.cost()) > 1e-3f) {
      LOG(ERROR) << "Cost differs when computed with or without Jacobians:";
      LOG(ERROR) << "Without Jacobians: " << helper1.cost();
      LOG(ERROR) << "With Jacobians: " << helper2.cost();
    }
    
    return helper1.cost();
  }
  
  /// Tests whether the given state is likely an optimum by comparing its cost
  /// to that of some samples around it. Returns true if no lower cost state is
  /// found, false otherwise.
  /// TODO: For reduce_cost_if_possible, could also try to escape from saddle
  ///       points right away if one is detected (cost goes down for both
  ///       opposite directions), or make a step as soon as one is detected that
  ///       can reduce the cost to 95% or less.
  template <class State, class CostFunction>
  bool VerifyOptimum(
      State* state,
      const CostFunction& cost_function,
      Scalar step_size,
      bool reduce_cost_if_possible,
      int first_dof = 0,
      int last_dof = -1,
      Scalar* final_cost = nullptr) {
    constexpr bool is_reversible = IsReversibleGetter<State>::eval();
    CHECK(!is_reversible) << "This is currently only implemented for states with copy constructors";
    
    CostAccumulator<Scalar> helper;
    bool lower_cost_found = false;
    
    cost_function.template Compute<false>(*state, &helper);
    Scalar center_cost = helper.cost();
    LOG(1) << "Center cost in VerifyOptimum(): " << center_cost;
    
    int dof = DegreesOfFreedomGetter<State>::eval(*state);
    if (last_dof < 0) {
      last_dof = dof - 1;
    }
    Matrix<Scalar, Eigen::Dynamic, 1> delta;
    delta.resize(dof);
    delta.setZero();
    
    int best_i = -1;  // initialized only to silence the warning
    int best_d = 0;  // initialized only to silence the warning
    Scalar best_factor = 0;  // initialized only to silence the warning
    Scalar best_cost = numeric_limits<Scalar>::infinity();
    
    for (int i = first_dof; i <= last_dof; ++ i) {
      bool cost_reduced_for_first_direction = false;
      Scalar cost_for_first_direction = -1;  // initialized only to silence the warning
      Scalar factor_for_first_direction = 0;  // initialized only to silence the warning
      bool exit_search = false;
      
      for (int d = -1; d <= 1; d += 2) {
        delta(i) = d * step_size;
        State offset_state(*state);
        offset_state -= delta;
        
        helper.Reset();
        cost_function.template Compute<false>(offset_state, &helper);
        Scalar offset_cost = helper.cost();
        
        bool lower_cost = offset_cost < center_cost;
        lower_cost_found |= lower_cost;
        
        if (lower_cost) {
          LOG(WARNING) << "[" << i << " / " << dof << "] Lower cost found: " << helper.cost() << " < " << center_cost;
          
          if (reduce_cost_if_possible) {
            Scalar factor = 1;
            for (int f = 0; f < 10; ++ f) {
              Scalar new_factor = 2 * factor;
              State new_offset_state(*state);
              new_offset_state -= new_factor * delta;
              helper.Reset();
              cost_function.template Compute<false>(new_offset_state, &helper);
              if (helper.cost() >= offset_cost) {
                break;
              }
              
              offset_cost = helper.cost();
              factor = new_factor;
            }
            
            if (offset_cost < best_cost) {
              best_cost = offset_cost;
              best_i = i;
              best_d = d;
              best_factor = factor;
            }
            
            if (!cost_reduced_for_first_direction) {
              cost_for_first_direction = offset_cost;
              factor_for_first_direction = factor;
            } else {  // if (cost_reduced_for_first_direction)
              // Detected a saddle point (if it is not a maximum): the cost goes
              // down in two opposite directions. Try to escape this saddle point
              // by going in one of these directions.
              if (cost_for_first_direction < offset_cost) {
                best_cost = cost_for_first_direction;
                best_i = i;
                best_d = -1 * d;
                best_factor = factor_for_first_direction;
              } else {
                best_cost = offset_cost;
                best_i = i;
                best_d = d;
                best_factor = factor;
              }
              LOG(WARNING) << "Saddle point (or maximum) detected.";
              exit_search = true;
              break;
            }
            cost_reduced_for_first_direction = true;
          }
        } else {
          LOG(1) << "[" << i << " / " << dof << "] Higher cost: " << helper.cost();
        }
      }
      
      delta(i) = 0;
      if (exit_search) {
        break;
      }
    }
    
    if (reduce_cost_if_possible && !std::isinf(best_cost)) {
      delta(best_i) = best_d * step_size;
      *state -= best_factor * delta;
      LOG(WARNING) << "Cost reduced from " << center_cost << " to " << best_cost;
      *final_cost = best_cost;
      return false;
    }
    
    *final_cost = center_cost;
    return !lower_cost_found;
  }
  
  /// Returns the final value of lambda (of the Levenberg-Marquardt algorithm)
  /// after the last optimization.
  inline Scalar lambda() const { return m_lambda; }
  
  
 private:
  template <class State, class CostFunction, bool IsReversible>
  OptimizationReport OptimizeImpl(
      State* state,
      const CostFunction& cost_function,
      int max_iteration_count,
      int max_lm_attempts,
      Scalar init_lambda,
      Scalar init_lambda_factor,
      bool print_progress) {
    constexpr const char* kDebugWriteMatricesPath = nullptr;  // TODO: Make that accessible from the outside?
    constexpr bool kDebugDisplayHGraphically = false;  // TODO: Make that accessible from the outside?
    
    CHECK_GE(init_lambda_factor, 0);
    
    OptimizationReport report;
    report.num_iterations_performed = 0;
    report.cost_and_jacobian_evaluation_time = 0;
    report.solve_time = 0;
    
    // Determine the variable count of the optimization problem, distributed
    // over the block-diagonal and dense parts of the Hessian.
    const int degrees_of_freedom = DegreesOfFreedomGetter<State>::eval(*state);
    CHECK_GT(degrees_of_freedom, 0);
    
    const int block_diagonal_degrees_of_freedom =
        m_use_block_diagonal_structure ? (m_block_size * m_num_blocks) : 0;
    
    const int dense_degrees_of_freedom = degrees_of_freedom - block_diagonal_degrees_of_freedom;
    CHECK_GE(dense_degrees_of_freedom, 0);
    
    // Allocate space for H, x, and b.
    // 
    // The structure of the update equation is as follows (both the top-left and
    // bottom-right blocks in H, and thus also the off-diagonal blocks and
    // corresponding blocks in the x and b vectors, could have size zero though):
    // +----------------+--------------+   +----------------+   +----------------+
    // | m_block_diag_H | m_off_diag_H |   |                |   | m_block_diag_b |
    // +----------------+--------------+ * | m_x            | = +----------------+
    // | m_off_diag_H^T | m_dense_H    |   |                |   | m_dense_b      |
    // +----------------+--------------+   +----------------+   +----------------+
    m_dense_H.resize(dense_degrees_of_freedom, dense_degrees_of_freedom);
    if (m_sparse_storage_for_off_diag_H) {
      m_off_diag_H_sparse.resize(m_on_the_fly_block_processing ? m_block_batch_size : m_num_blocks);
    } else {
      m_off_diag_H.resize(m_on_the_fly_block_processing ? (m_block_batch_size * m_block_size) : block_diagonal_degrees_of_freedom, dense_degrees_of_freedom);
    }
    m_block_diag_H.resize(m_on_the_fly_block_processing ? m_block_batch_size : m_num_blocks);
    for (int i = 0; i < m_block_diag_H.size(); ++ i) {
      m_block_diag_H[i].resize(m_block_size, m_block_size);
      if (m_sparse_storage_for_off_diag_H) {
        m_off_diag_H_sparse[i].resizeHeight(m_block_size);
      }
    }
    
    m_x.resize(degrees_of_freedom);
    
    m_dense_b.resize(dense_degrees_of_freedom);
    m_block_diag_b.resize(m_on_the_fly_block_processing ? (m_block_batch_size * m_block_size) : block_diagonal_degrees_of_freedom);
    
    // Perform optimization iterations up to max_iteration_count.
    Scalar last_cost = numeric_limits<Scalar>::quiet_NaN();
    bool applied_update = true;
    int iteration;
    for (iteration = 0; iteration < max_iteration_count; ++ iteration) {
      // Compute cost and Jacobians (which get accumulated on H and b).
      // TODO: Support numerical Jacobian.
      // TODO: We can template OptimizeImpl() with a special type of accumulator
      //       to use. This way, we can make a special case if the Schur
      //       complement is not used to use a simpler (faster) accumulator.
      vector<Scalar> residual_cost_vector;
      residual_cost_vector.reserve(m_expected_residual_count);
      
      if (m_on_the_fly_block_processing) {
        // TODO: Support initializing lambda via H, as when !m_on_the_fly_block_processing?
        CHECK(init_lambda > 0) << "Automatic init_lambda initialization is not implemented for m_on_the_fly_block_processing";
        
        m_lambda = init_lambda;
      } else {
        UpdateEquationAccumulator<Scalar> update_eq(
            block_diagonal_degrees_of_freedom, &m_dense_H,
            m_sparse_storage_for_off_diag_H ? nullptr : &m_off_diag_H,
            m_sparse_storage_for_off_diag_H ? &m_off_diag_H_sparse : nullptr,
            &m_block_diag_H, &m_dense_b, &m_block_diag_b, &residual_cost_vector,
            m_on_the_fly_block_processing ? m_block_batch_size : 0,
            m_lambda);
        Timer update_eq_timer("");
        cost_function.template Compute<true>(*state, &update_eq);
        update_eq.AccumulateFinishedBlocks();
        report.cost_and_jacobian_evaluation_time += update_eq_timer.Stop(/*add_to_statistics*/ false);
        last_cost = update_eq.cost();
        if (iteration == 0) {
          report.initial_cost = last_cost;
        }
        
        // Debug: Output H matrix and b vector if requested
        if (kDebugWriteMatricesPath) {
          LOG(INFO) << "degrees_of_freedom: " << degrees_of_freedom;
          LOG(INFO) << "block_diagonal_degrees_of_freedom: " << block_diagonal_degrees_of_freedom;
          LOG(INFO) << "dense_degrees_of_freedom: " << dense_degrees_of_freedom;
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H;
          Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
          GetHandB(&H, &b);
          ofstream H_stream(string(kDebugWriteMatricesPath) + "/debug_H.txt", std::ios::out);
          H_stream << H << std::endl;
          H_stream.close();
          ofstream b_stream(string(kDebugWriteMatricesPath) + "/debug_b.txt", std::ios::out);
          b_stream << b << std::endl;
          b_stream.close();
        }
        
        // If using sparse storage for m_off_diag_H, output sparseness
        if (m_use_block_diagonal_structure && m_sparse_storage_for_off_diag_H) {
          float sparseness_sum = 0;
          for (usize i = 0; i < m_off_diag_H_sparse.size(); ++ i) {
            sparseness_sum += m_off_diag_H_sparse[i].column_indices.size() / static_cast<float>(dense_degrees_of_freedom);
          }
          LOG(1) << "Sparseness for off_diag_H: " << (100 * sparseness_sum / m_off_diag_H_sparse.size()) << "%";
        }
        
        if (print_progress) {
          if (iteration == 0) {
            LOG(INFO) << "LMOptimizer: [0] Initial cost: " << update_eq.cost();
          } else {
            LOG(1) << "LMOptimizer: [" << iteration << "] cost: " << update_eq.cost();
          }
        }
        
        if (update_eq.cost() == 0) {
          if (print_progress) {
            LOG(INFO) << "LMOptimizer: Cost is zero, stopping.";
          }
          break;
        }
        
        // Initialize lambda based on the average diagonal element size in H.
        // TODO: make the strategy for initializing m_lambda configurable?
        // TODO: We do not consider the variable fixing for initializing lambda
        //       at the moment.
        if (iteration == 0) {
          if (init_lambda >= 0) {
            m_lambda = init_lambda;
          } else {
            m_lambda = 0;
            for (usize i = 0; i < m_block_diag_H.size(); ++ i) {
              for (int k = 0; k < m_block_diag_H[i].rows(); ++ k) {
                m_lambda += m_block_diag_H[i](k, k);
              }
            }
            for (int i = 0; i < dense_degrees_of_freedom; ++ i) {
              m_lambda += m_dense_H(i, i);
            }
            m_lambda = static_cast<Scalar>(init_lambda_factor) * m_lambda / degrees_of_freedom;
          }
        }
        
        // Cache the original diagonal of H.
        m_original_diagonal.resize(degrees_of_freedom);
        int diagonal_index = 0;
        for (usize i = 0; i < m_block_diag_H.size(); ++ i) {
          for (int k = 0; k < m_block_diag_H[i].rows(); ++ k) {
            m_original_diagonal[diagonal_index] = m_block_diag_H[i](k, k);
            ++ diagonal_index;
          }
        }
        for (int i = 0; i < dense_degrees_of_freedom; ++ i) {
          m_original_diagonal[diagonal_index] = m_dense_H(i, i);
          ++ diagonal_index;
        }
        CHECK_EQ(diagonal_index, degrees_of_freedom);
      }
      
      // Do Levenberg-Marquardt iterations until finding an update that decreases
      // the cost, or until max_lm_attempts are reached.
      applied_update = false;
      for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++ lm_iteration) {
        Timer solve_timer("");
        
        // If m_on_the_fly_block_processing is used, we need to iterate over the
        // residuals again for every different value of m_lambda that is used.
        if (m_on_the_fly_block_processing) {
          residual_cost_vector.clear();
          UpdateEquationAccumulator<Scalar> update_eq(
              block_diagonal_degrees_of_freedom, &m_dense_H,
              m_sparse_storage_for_off_diag_H ? nullptr : &m_off_diag_H,
              m_sparse_storage_for_off_diag_H ? &m_off_diag_H_sparse : nullptr,
              &m_block_diag_H, &m_dense_b, &m_block_diag_b, &residual_cost_vector,
              m_on_the_fly_block_processing ? m_block_batch_size : 0,
              m_lambda);
          Timer update_eq_timer("");
          cost_function.template Compute<true>(*state, &update_eq);
          update_eq.AccumulateFinishedBlocks();
          report.cost_and_jacobian_evaluation_time += update_eq_timer.Stop(/*add_to_statistics*/ false);
          last_cost = update_eq.cost();
          if (iteration == 0) {
            report.initial_cost = last_cost;
          }
        }
        
        // Solve the update equation H * x = b. Depending on different settings,
        // this may be solved in different ways.
        if (m_num_fixed_variables > 0) {
          // Create a new H and b with the fixed variables removed.
          // Note that this might be very inefficient compared to the standard
          // code path if the latter uses the Schur complement, since here, the
          // Schur complement is not implemented.
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> full_H;
          Eigen::Matrix<Scalar, Eigen::Dynamic, 1> full_b;
          GetHandB(&full_H, &full_b);
          SolveWithFixedVariables(full_H, full_b);
        } else {
          // Add to the diagonal of H according to the Levenberg-Marquardt method.
          if (!m_on_the_fly_block_processing) {
            int diagonal_index = 0;
            for (usize i = 0; i < m_block_diag_H.size(); ++ i) {
              for (int k = 0; k < m_block_diag_H[i].rows(); ++ k) {
                m_block_diag_H[i](k, k) = m_original_diagonal[diagonal_index] + m_lambda;
                ++ diagonal_index;
              }
            }
            for (int i = 0; i < dense_degrees_of_freedom; ++ i) {
              m_dense_H(i, i) = m_original_diagonal[diagonal_index] + m_lambda;
              ++ diagonal_index;
            }
            CHECK_EQ(diagonal_index, degrees_of_freedom);
          }
          
          if (block_diagonal_degrees_of_freedom > 0) {
            if (m_sparse_storage_for_off_diag_H) {
              SolveWithSchurComplementSparseOffDiag(
                  block_diagonal_degrees_of_freedom,
                  dense_degrees_of_freedom,
                  state,
                  cost_function);
            } else {
              SolveWithSchurComplementDenseOffDiag(
                  block_diagonal_degrees_of_freedom,
                  dense_degrees_of_freedom,
                  state,
                  cost_function);
            }
          } else {
            SolveDensely(m_dense_H, m_dense_b, &m_x);
          }
        }
        
        report.solve_time += solve_timer.Stop(/*add_to_statistics*/ false);
        
        // Debug: Output x vector
        if (kDebugWriteMatricesPath) {
          ofstream x_stream(string(kDebugWriteMatricesPath) + "/debug_x.txt", std::ios::out);
          x_stream << m_x << std::endl;
          x_stream.close();
        }
        
        // Debug: Graphical display of H
        if (kDebugDisplayHGraphically) {
          LOG(ERROR) << "This requires to #include \"libvis/image_display.h\"";
          // static ImageDisplay H_display;
          // 
          // Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H;
          // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
          // update_eq.GetHandB(&H, &b);
          // 
          // Image<float> H_image(H.cols(), H.rows());
          // for (int y = 0; y < H.rows(); ++ y) {
          //   for (int x = 0; x < H.cols(); ++ x) {
          //     H_image(x, y) = H(y, x);
          //   }
          // }
          // 
          // H_display.Update(H_image, "LMOptimizer: H", 0, H_image.CalcMax());
        }
        
        // Reject NaN updates
        // TODO: Getting a NaN update likely indicates a bug in the application.
        //       Provide some debug output here, saying whether there are NaNs
        //       in H and/or b?
        if (std::isnan(m_x(0))) {  // NOTE: for comprehensive check, use: m_x.hasNaN()
          m_lambda = 2.f * m_lambda;
          if (print_progress) {
            LOG(1) << "LMOptimizer:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << max_lm_attempts
                   << "] update rejected (NaN update"
                   << "), new lambda: " << m_lambda;
          }
          continue;
        }
        
        // Apply the update to create a temporary state.
        // Note the inversion of the delta here (subtracting m_x instead of adding it).
        OptionalCopy<!IsReversible, State> optional_state_copy(state);
        State* updated_state = optional_state_copy.GetObject();
        *updated_state -= m_x;
        
        // Compute the cost of the temporary state, and
        // test whether taking over this state will decrease the cost.
        vector<Scalar> test_residual_cost_vector;
        test_residual_cost_vector.reserve(m_expected_residual_count);
        UpdateEquationAccumulator<Scalar> test_cost(block_diagonal_degrees_of_freedom, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &test_residual_cost_vector, 0, m_lambda);
        Timer cost_timer("");
        cost_function.template Compute<false>(*updated_state, &test_cost);
        report.cost_and_jacobian_evaluation_time += cost_timer.Stop(/*add_to_statistics*/ false);
        
        if (/*TODO: If there are no invalid residuals, use: test_cost.cost() < update_eq.cost()*/
            CostIsSmallerThan(
                test_residual_cost_vector,
                residual_cost_vector)) {
          // Take over the update.
          if (print_progress && lm_iteration > 0) {
            LOG(1) << "LMOptimizer:   [" << (iteration + 1) << "] update accepted";
          }
          if (IsReversible) {
            // no action required, keep the updated state
          } else {
            *state = *updated_state;
          }
          m_lambda = 0.5f * m_lambda;
          applied_update = true;
          report.num_iterations_performed += 1;
          last_cost = test_cost.cost();
          break;
        } else {
          if (IsReversible) {
            // Undo update. This may cause slight numerical inaccuracies.
            // TODO: Would it be better to combine undoing the old update and
            //       applying the new update (if not giving up) into a single step?
            *state -= -m_x;
          } else {
            // no action required, drop the updated state copy
          }
          
          m_lambda = 2.f * m_lambda;
          if (print_progress) {
            LOG(1) << "LMOptimizer:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << max_lm_attempts
                   << "] update rejected (bad cost: " << test_cost.cost()
                   << "), new lambda: " << m_lambda;
          }
        }
      }
      
      if (!applied_update || last_cost == 0) {
        if (print_progress) {
          if (last_cost == 0) {
            LOG(INFO) << "LMOptimizer: Reached zero cost, stopping.";
          } else {
            LOG(INFO) << "LMOptimizer: Cannot find an update which decreases the cost, aborting.";
          }
        }
        iteration += 1;  // For correct display only.
        break;
      }
    }
    
    if (print_progress) {
      if (applied_update) {
        LOG(INFO) << "LMOptimizer: Maximum iteration count reached, stopping.";
      }
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Cost / Jacobian computation time: " << report.cost_and_jacobian_evaluation_time << " seconds";
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Solve time: " << report.solve_time << " seconds";
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Final cost:   " << last_cost;  // length matches with "Initial cost: "
    }
    
    report.final_cost = last_cost;
    return report;
  }
  
  bool CostIsSmallerThan(
      vector<Scalar> left_residual_cost_vector,
      vector<Scalar> right_residual_cost_vector) {
    const usize size = left_residual_cost_vector.size();
    CHECK_EQ(size, right_residual_cost_vector.size());
    
    Scalar left_sum = 0;
    Scalar right_sum = 0;
    usize count = 0;
    for (usize i = 0; i < size; ++ i) {
      if (left_residual_cost_vector[i] >= 0 && right_residual_cost_vector[i] >= 0) {
        left_sum += left_residual_cost_vector[i];
        right_sum += right_residual_cost_vector[i];
        ++ count;
      }
    }
    
    return count > 0 && left_sum < right_sum;
  }
  
  template <typename DerivedA, typename DerivedB>
  void SolveDensely(
      MatrixBase<DerivedA>& H,
      const MatrixBase<DerivedB>& b,
      Matrix<Scalar, Eigen::Dynamic, 1>* x) {
    if (m_use_complete_orthogonal_decomposition) {
      H.template triangularView<Eigen::Lower>() = H.template triangularView<Eigen::Upper>().transpose();
      *x = H.completeOrthogonalDecomposition().solve(b);
    } else {
      typedef LDLT<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Lower> LDLT_T;
      *x = LDLT_T(H.template selfadjointView<Eigen::Upper>()).solve(b);
    }
  }
  
  void GetHandB(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H,
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b) const {
    if (m_block_diag_b.rows() == 0) {
      *H = m_dense_H;
      *b = m_dense_b;
    } else {
      int H_size = m_block_diag_b.rows() + m_dense_b.rows();
      H->resize(H_size, H_size);
      int block_size = m_block_size;
      int block_diagonal_size = block_size * m_block_diag_H.size();
      
      H->topLeftCorner(block_diagonal_size, block_diagonal_size).setZero();
      for (usize i = 0; i < m_block_diag_H.size(); ++ i) {
        H->block(i * block_size, i * block_size, block_size, block_size) =
            m_block_diag_H.at(i);
      }
      
      if (!m_sparse_storage_for_off_diag_H) {
        H->topRightCorner(m_off_diag_H.rows(), m_off_diag_H.cols()) = m_off_diag_H;
      } else {
        H->topRightCorner(block_diagonal_size, H->cols() - block_diagonal_size).setZero();
        
        for (usize i = 0; i < m_off_diag_H_sparse.size(); ++ i) {
          const SparseColumnMatrix<Scalar>& sparse_matrix = m_off_diag_H_sparse.at(i);
          for (int stored_column = 0; stored_column < sparse_matrix.column_indices.size(); ++ stored_column) {
            for (int stored_row = 0; stored_row < sparse_matrix.columns.rows(); ++ stored_row) {
              (*H)(stored_row + i * block_size, sparse_matrix.column_indices[stored_column] + block_diagonal_size) = sparse_matrix.columns(stored_row, stored_column);
            }
          }
        }
      }
      
      H->bottomRightCorner(m_dense_H.rows(), m_dense_H.cols()) = m_dense_H;
      
      b->resize(H_size);
      b->topRows(m_block_diag_b.rows()) = m_block_diag_b;
      b->bottomRows(m_dense_b.rows()) = m_dense_b;
    }
    
    H->template triangularView<Eigen::Lower>() = H->template triangularView<Eigen::Upper>().transpose();
  }
  
  void SolveWithFixedVariables(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& full_H,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& full_b) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> thinned_H;
    thinned_H.resize(full_H.rows() - m_num_fixed_variables,
                     full_H.cols() - m_num_fixed_variables);
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> thinned_b;
    thinned_b.resize(full_b.size() - m_num_fixed_variables);
    
    int output_row = 0;
    for (int row = 0; row < full_H.rows(); ++ row) {
      if (row < static_cast<int>(fixed_variables.size()) && fixed_variables[row]) {
        continue;
      }
      
      int output_col = 0;
      for (int col = 0; col < full_H.cols(); ++ col) {
        if (col < static_cast<int>(fixed_variables.size()) && fixed_variables[col]) {
          continue;
        }
        
        thinned_H(output_row, output_col) = full_H(row, col);
        
        ++ output_col;
      }
      CHECK_EQ(output_col, thinned_H.cols());
      
      thinned_b(output_row) = full_b(row);
      
      ++ output_row;
    }
    CHECK_EQ(output_row, thinned_H.rows());
    
    thinned_H.diagonal().array() += m_lambda;
    
    Matrix<Scalar, Eigen::Dynamic, 1> thinned_x;
    SolveDensely(thinned_H, thinned_b, &thinned_x);
    
    output_row = 0;
    for (int row = 0; row < full_H.rows(); ++ row) {
      if (row < static_cast<int>(fixed_variables.size()) && fixed_variables[row]) {
        m_x(row) = 0;
      } else {
        m_x(row) = thinned_x(output_row);
        ++ output_row;
      }
    }
  }
  
  template <class State, class CostFunction>
  void SolveWithSchurComplementSparseOffDiag(
      int block_diagonal_degrees_of_freedom,
      int dense_degrees_of_freedom,
      State* state,
      const CostFunction& cost_function) {
    // Compute off-diagonal^T * block_diagonal^(-1) * b_block_diagonal, subtract from schur_b.
    // Compute off-diagonal^T * block_diagonal^(-1) * off-diagonal, subtract from schur_M.
    // NOTE: This block of code is relatively slow.
    // NOTE: If m_on_the_fly_block_processing is true, we would not need to copy the two matrices below.
    Matrix<Scalar, Eigen::Dynamic, 1> schur_b = m_dense_b;
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> schur_M = m_dense_H;
    
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block_I;
    block_I.resize(m_block_size, m_block_size);
    block_I.setIdentity();
    
    if (!m_on_the_fly_block_processing) {
      // This way of computation would allow to only store one block-row of
      // m_off_diag_H_sparse at a time, saving lots of memory.
      // NOTE: This is relatively slow. Perhaps due to adding up intermediate
      //       results all over the place in schur_M often, and thus having a
      //       bad memory access pattern?
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverted_diag_block;
      Matrix<Scalar, 1, Eigen::Dynamic> left;
      
      for (int block = 0; block < m_block_diag_H.size(); ++ block) {
        inverted_diag_block = m_block_diag_H[block].template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        auto& sparse_off_diag_block = m_off_diag_H_sparse[block];
        
        for (int stored_row_index = 0; stored_row_index < sparse_off_diag_block.column_indices.size(); ++ stored_row_index) {
          int actual_row_index = sparse_off_diag_block.column_indices[stored_row_index];
          
          left =
              sparse_off_diag_block.columns.col(stored_row_index).transpose() *
              inverted_diag_block;
          
          schur_b(actual_row_index) -=
              left *
              m_block_diag_b.segment(block * m_block_size, m_block_size);
          
          for (int stored_col_index = 0; stored_col_index < sparse_off_diag_block.column_indices.size(); ++ stored_col_index) {
            int actual_col_index = sparse_off_diag_block.column_indices[stored_col_index];
            
            if (actual_col_index >= actual_row_index) {
              schur_M(actual_row_index, actual_col_index) -=
                  left *
                  sparse_off_diag_block.columns.col(stored_col_index);
            }
          }
        }
      }
    }
    
    // Debug: Output M matrix and b vector if requested
    constexpr const char* kDebugWriteMatricesPath = nullptr;
    static bool debug = true;
    if (kDebugWriteMatricesPath && debug) {
      schur_M.template triangularView<Eigen::Lower>() = schur_M.template triangularView<Eigen::Upper>().transpose();
      
      ofstream M_stream(string(kDebugWriteMatricesPath) + "/debug_Schur_M.txt", std::ios::out);
      M_stream << schur_M << std::endl;
      M_stream.close();
      ofstream b_stream(string(kDebugWriteMatricesPath) + "/debug_Schur_b.txt", std::ios::out);
      b_stream << schur_b << std::endl;
      b_stream.close();
    }
    
    // Solve the equation system to get the bottom rows of x (for the dense part)
    // NOTE: This is the same as in SolveWithSchurComplementDenseOffDiag().
    if (m_use_complete_orthogonal_decomposition) {
      // Use the pseudoinverse as a means of Gauge fixing (it will take the shortest update).
      // Attention, in "BA - a modern synthesis" (Sec. 9.3), also a weight
      // matrix is used that might be useful here.
      schur_M.template triangularView<Eigen::Lower>() = schur_M.template triangularView<Eigen::Upper>().transpose();
      m_x.bottomRows(dense_degrees_of_freedom) =
          schur_M.completeOrthogonalDecomposition().solve(schur_b);
      // TODO: Try schur_M.bdcSvd(ComputeThinU|ComputeThinV).solve(schur_b)?
    } else {
      m_x.bottomRows(dense_degrees_of_freedom) = schur_M.template selfadjointView<Eigen::Upper>().ldlt().solve(schur_b);
    }
    
    // Use the partial solution to get the top rows of x (for the block-diagonal part) as:
    // block_diagonal^(-1) * (b_block_diagonal - off-diagonal * x_dense)
    if (m_on_the_fly_block_processing) {
      BackSubstitutionAccumulator<Scalar> accumulator(
          block_diagonal_degrees_of_freedom, &m_block_diag_H, nullptr, &m_off_diag_H_sparse, &m_block_diag_b, &m_x, m_block_batch_size);
      cost_function.template Compute<true>(*state, &accumulator);
      accumulator.AccumulateFinishedBlocks();
    } else {
      Matrix<Scalar, Eigen::Dynamic, 1> segment;
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverted_diag_block;
      
      for (int block = 0; block < m_block_diag_H.size(); ++ block) {
        auto& original_diag_block = m_block_diag_H[block];
        inverted_diag_block = original_diag_block.template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        auto& sparse_off_diag_block = m_off_diag_H_sparse[block];
        
        segment = m_block_diag_b.segment(block * m_block_size, m_block_size);
        
        for (int stored_col_index = 0; stored_col_index < sparse_off_diag_block.column_indices.size(); ++ stored_col_index) {
          int actual_col_index = sparse_off_diag_block.column_indices[stored_col_index];
          
          segment -= sparse_off_diag_block.columns.col(stored_col_index) * m_x(block_diagonal_degrees_of_freedom + actual_col_index);
        }
        
        m_x.segment(block * m_block_size, m_block_size) = inverted_diag_block * segment;
      }
    }
    
    if (kDebugWriteMatricesPath && debug) {
      ofstream b_stream(string(kDebugWriteMatricesPath) + "/debug_x.txt", std::ios::out);
      b_stream << m_x << std::endl;
      b_stream.close();
      
      debug = false;
    }
  }
  
  template <class State, class CostFunction>
  void SolveWithSchurComplementDenseOffDiag(
      int block_diagonal_degrees_of_freedom,
      int dense_degrees_of_freedom,
      State* state,
      const CostFunction& cost_function) {
    if (m_on_the_fly_block_processing) {
      // Solve the equation system to get the bottom rows of x (for the dense part)
      // NOTE: There are several copies of this piece of code.
      if (m_use_complete_orthogonal_decomposition) {
        // Use the pseudoinverse as a means of Gauge fixing (it will take the shortest update).
        // Attention, in "BA - a modern synthesis" (Sec. 9.3), also a weight
        // matrix is used that might be useful here.
        m_dense_H.template triangularView<Eigen::Lower>() = m_dense_H.template triangularView<Eigen::Upper>().transpose();
        m_x.bottomRows(dense_degrees_of_freedom) =
            m_dense_H.completeOrthogonalDecomposition().solve(m_dense_b);
        // TODO: Try m_dense_H.bdcSvd(ComputeThinU|ComputeThinV).solve(m_dense_b)?
      } else {
        m_x.bottomRows(dense_degrees_of_freedom) = m_dense_H.template selfadjointView<Eigen::Upper>().ldlt().solve(m_dense_b);
      }
      
      BackSubstitutionAccumulator<Scalar> accumulator(
          block_diagonal_degrees_of_freedom, &m_block_diag_H, &m_off_diag_H, nullptr, &m_block_diag_b, &m_x, m_block_batch_size);
      cost_function.template Compute<true>(*state, &accumulator);
      accumulator.AccumulateFinishedBlocks();
    } else {
      // Using the Schur complement, compute the small dense matrix to solve.
      // Compute block_diagonal^(-1) * off-diagonal, and
      // compute block_diagonal^(-1) * b_block_diagonal.
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> D_inv_B;
      D_inv_B.resize(block_diagonal_degrees_of_freedom, m_dense_H.cols());
      
      Matrix<Scalar, Eigen::Dynamic, 1> D_inv_b1;
      D_inv_b1.resize(block_diagonal_degrees_of_freedom);
      
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block_I;
      block_I.resize(m_block_size, m_block_size);
      block_I.setIdentity();
      
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H_block;
      
      for (usize block_index = 0; block_index < m_block_diag_H.size(); ++ block_index) {
        int base_index = block_index * m_block_size;
        H_block = m_block_diag_H[block_index].template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        for (int row = 0; row < m_block_size; ++ row) {
          Scalar result = 0;
          for (int k = 0; k < H_block.cols(); ++ k) {
            result += H_block(row, k) *
                      m_block_diag_b(base_index + k);
          }
          D_inv_b1(base_index + row) = result;
          
          for (int col = 0; col < m_dense_H.cols(); ++ col) {
            // We only need to take the non-zero diagonal block into account
            Scalar result = 0;
            for (int k = 0; k < H_block.cols(); ++ k) {
              result += H_block(row, k) *
                        m_off_diag_H(base_index + k, col);
            }
            D_inv_B(base_index + row, col) = result;
          }
        }
      }
      
      // Compute off-diagonal^T * block_diagonal^(-1) * b_block_diagonal
      // Note that the first part is the transpose of: block_diagonal^(-1) * off-diagonal,
      // which is already stored in D_inv_B. However, we anyway need D_inv_b1 at the end again.
      Matrix<Scalar, Eigen::Dynamic, 1> B_T_D_inv_b;
      B_T_D_inv_b.resize(m_dense_H.rows());
      B_T_D_inv_b = m_off_diag_H.transpose() * D_inv_b1;
      
      // Compute off-diagonal^T * block_diagonal^(-1) * off-diagonal
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B_T_D_inv_B;
      B_T_D_inv_B.resize(m_dense_H.rows(), m_dense_H.cols());
      
      if (m_compute_schur_complement_with_cuda) {
        MultiplyMatricesWithCuBLAS(D_inv_B, &B_T_D_inv_B);
      } else {
        B_T_D_inv_B.template triangularView<Eigen::Upper>() = m_off_diag_H.transpose() * D_inv_B;
      }
      
      // Compute the equation system to solve
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> schur_M;
      schur_M.resize(m_dense_H.rows(), m_dense_H.cols());
      schur_M.template triangularView<Eigen::Upper>() = m_dense_H - B_T_D_inv_B;
      Matrix<Scalar, Eigen::Dynamic, 1> schur_b = m_dense_b - B_T_D_inv_b;
      
      // Solve the equation system to get the bottom rows of x (for the dense part)
      if (m_use_complete_orthogonal_decomposition) {
        // Use the pseudoinverse as a means of Gauge fixing (it will take the shortest update).
        // Attention, in "BA - a modern synthesis" (Sec. 9.3), also a weight
        // matrix is used that might be useful here.
        schur_M.template triangularView<Eigen::Lower>() = schur_M.template triangularView<Eigen::Upper>().transpose();
        m_x.bottomRows(dense_degrees_of_freedom) =
            schur_M.completeOrthogonalDecomposition().solve(schur_b);
        // TODO: Try schur_M.bdcSvd(ComputeThinU|ComputeThinV).solve(schur_b)?
      } else {
        if (false /*kRunOnGPU*/) {
          // TODO: This unfortunately seems to strictly require the matrix to be
          //       positive-definite, and very often fails, while Eigen's
          //       robust Cholesky decomposition implementation just works.
          //       Maybe attempt to use cusolverDnSsyevd() to compute the eigenvalues
          //       and set the negative/zero ones to a very small positive value to
          //       get a psd matrix? Unfortunately, multiplying the resulting matrices
          //       together again is likely very instable. See:
          //       https://math.stackexchange.com/questions/423138/cholesky-for-non-positive-definite-matrices
          //       NOTE: If kUseFloatOnGPU above is set to false, it behaves better
          //             for a bit longer.
          LOG(ERROR) << "This is currently disabled.";
          // SolveWithCuSolver(dense_degrees_of_freedom, schur_M, schur_b);
        } else {
          m_x.bottomRows(dense_degrees_of_freedom) = schur_M.template selfadjointView<Eigen::Upper>().ldlt().solve(schur_b);
        }
      }
      
      // Use the partial solution to get the top rows of x (for the block-diagonal part)
      m_x.topRows(block_diagonal_degrees_of_freedom) =
          D_inv_b1 - D_inv_B * m_x.bottomRows(dense_degrees_of_freedom);
    }
  }
  
  void MultiplyMatricesWithCuBLAS(
      const Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& D_inv_B,
      Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* B_T_D_inv_B) {
    cublasXtHandle_t cublasxthandle;
    CHECK_EQ(cublasXtCreate(&cublasxthandle), CUBLAS_STATUS_SUCCESS);
    
    int device_ids[1] = {0};  // Simply use the first device
    CHECK_EQ(cublasXtDeviceSelect(cublasxthandle, 1, device_ids), CUBLAS_STATUS_SUCCESS);
    
    constexpr bool kUseFloatOnGPU = false;  // TODO: Make configurable
    if (kUseFloatOnGPU) {
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> left = m_off_diag_H.transpose().template cast<float>();
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> right = D_inv_B.template cast<float>();
      
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> result;
      result.resize(left.rows(), right.cols());
      
      float alpha = 1;
      float beta = 0;
      CHECK_EQ(cublasXtSgemm(
          cublasxthandle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          left.rows(), right.cols(), left.cols(),
          &alpha,
          left.data(), left.rows(),
          right.data(), right.rows(),
          &beta,
          result.data(), result.rows()), CUBLAS_STATUS_SUCCESS);
      
      CHECK_EQ(cublasXtDestroy(cublasxthandle), CUBLAS_STATUS_SUCCESS);
      
      CHECK_EQ(B_T_D_inv_B->rows(), result.rows());
      CHECK_EQ(B_T_D_inv_B->cols(), result.cols());
      *B_T_D_inv_B = result.cast<Scalar>();
    } else {
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> left = m_off_diag_H.transpose().template cast<double>();
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> right = D_inv_B.template cast<double>();
      
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> result;
      result.resize(left.rows(), right.cols());
      
      double alpha = 1;
      double beta = 0;
      CHECK_EQ(cublasXtDgemm(
          cublasxthandle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          left.rows(), right.cols(), left.cols(),
          &alpha,
          left.data(), left.rows(),
          right.data(), right.rows(),
          &beta,
          result.data(), result.rows()), CUBLAS_STATUS_SUCCESS);
      
      CHECK_EQ(cublasXtDestroy(cublasxthandle), CUBLAS_STATUS_SUCCESS);
      
      CHECK_EQ(B_T_D_inv_B->rows(), result.rows());
      CHECK_EQ(B_T_D_inv_B->cols(), result.cols());
      *B_T_D_inv_B = result.cast<Scalar>();
    }
  }
  
  // void SolveWithCuSolver(
  //     int dense_degrees_of_freedom,
  //     const Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& schur_M,
  //     const Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& schur_b) {
  //   cusolverDnHandle_t cusolver_handle;
  //   CHECK_EQ(cusolverDnCreate(&cusolver_handle), CUSOLVER_STATUS_SUCCESS);
  //   
  //   // Transfer schur_M to the GPU
  //   float* schur_M_gpu;
  //   cudaMalloc(&schur_M_gpu, schur_M.rows() * schur_M.cols() * sizeof(float));
  //   {
  //     Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> schur_M_float = schur_M.template cast<float>();  // TODO: This copy is unnecessary if schur_M is already using float
  //     cudaMemcpy(schur_M_gpu, schur_M_float.data(), schur_M.rows() * schur_M.cols() * sizeof(float), cudaMemcpyHostToDevice);
  //   }
  //   
  //   // Allocate workspace and devInfo
  //   int workspace_size = 0;
  //   cusolverStatus_t status = cusolverDnSpotrf_bufferSize(
  //       cusolver_handle,
  //       CUBLAS_FILL_MODE_UPPER,
  //       schur_M.rows(),
  //       schur_M_gpu,
  //       schur_M.cols(),
  //       &workspace_size);
  //   if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
  //     LOG(FATAL) << "CUSOLVER_STATUS_NOT_INITIALIZED";
  //   } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
  //     LOG(FATAL) << "CUSOLVER_STATUS_INVALID_VALUE";
  //   } else if (status == CUSOLVER_STATUS_ARCH_MISMATCH) {
  //     LOG(FATAL) << "CUSOLVER_STATUS_ARCH_MISMATCH";
  //   } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
  //     LOG(FATAL) << "CUSOLVER_STATUS_INTERNAL_ERROR";
  //   }
  //   
  //   float* workspace;
  //   cudaMalloc(&workspace, workspace_size * sizeof(float));
  //   int* dev_info;
  //   cudaMalloc(&dev_info, sizeof(int));
  //   
  //   // Do in-place Cholesky decomposition of schur_M(_gpu)
  //   status = cusolverDnSpotrf(
  //       cusolver_handle,
  //       CUBLAS_FILL_MODE_UPPER,
  //       schur_M.rows(),
  //       schur_M_gpu,
  //       schur_M.cols(),
  //       workspace,
  //       workspace_size,
  //       dev_info);
  //   int dev_info_cpu;
  //   cudaMemcpy(&dev_info_cpu, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  //   if (dev_info_cpu < 0) {
  //     LOG(FATAL) << "cusolverDN Cholesky decomposition failed. DevInfo (indicating the wrong parameter, not including the handle) is: " << dev_info_cpu;
  //   } else if (dev_info_cpu > 0) {
  //     LOG(WARNING) << "cusolverDN Cholesky decomposition failed because the leading minor of order " << dev_info_cpu << " is not positive definite. Re-doing with Eigen implementation.";
  //     cudaFree(schur_M_gpu);
  //     cudaFree(workspace);
  //     cudaFree(dev_info);
  //     CHECK_EQ(cusolverDnDestroy(cusolver_handle), CUSOLVER_STATUS_SUCCESS);
  //     
  //     m_x.bottomRows(dense_degrees_of_freedom) = schur_M.template selfadjointView<Eigen::Upper>().ldlt().solve(schur_b);
  //   } else {
  //     if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
  //       LOG(FATAL) << "CUSOLVER_STATUS_NOT_INITIALIZED";
  //     } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
  //       LOG(FATAL) << "CUSOLVER_STATUS_INVALID_VALUE";
  //     } else if (status == CUSOLVER_STATUS_ARCH_MISMATCH) {
  //       LOG(FATAL) << "CUSOLVER_STATUS_ARCH_MISMATCH";
  //     } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
  //       LOG(FATAL) << "CUSOLVER_STATUS_INTERNAL_ERROR";
  //     }
  //     
  //     // Free the workspace, and transfer schur_b to the GPU.
  //     cudaFree(workspace);
  //     
  //     float* schur_b_gpu;
  //     cudaMalloc(&schur_b_gpu, schur_b.rows() * sizeof(float));
  //     {
  //       Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor> schur_b_float = schur_b.template cast<float>();  // TODO: This copy is unnecessary if schur_b is already using float
  //       cudaMemcpy(schur_b_gpu, schur_b_float.data(), schur_b.rows() * sizeof(float), cudaMemcpyHostToDevice);
  //     }
  //     
  //     // Solve the system of equations using the decomposed schur_M(_gpu).
  //     // The result is written to schur_b_gpu in-place.
  //     status = cusolverDnSpotrs(
  //         cusolver_handle,
  //         CUBLAS_FILL_MODE_UPPER,
  //         schur_M.rows(),
  //         1,
  //         schur_M_gpu,
  //         schur_M.rows(),
  //         schur_b_gpu,
  //         schur_M.rows(),
  //         dev_info);
  //     cudaMemcpy(&dev_info_cpu, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  //     cudaFree(dev_info);
  //     if (dev_info_cpu != 0) {
  //       LOG(FATAL) << "cusolverDN Cholesky solve failed. DevInfo (indicating the wrong parameter, not including the handle) is: " << dev_info_cpu;
  //     }
  //     
  //     // Read back the result and free GPU memory.
  //     cudaFree(schur_M_gpu);
  //     
  //     {
  //       Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor> x_bottom_float;
  //       x_bottom_float.resize(schur_b.rows(), Eigen::NoChange);
  //       cudaMemcpy(x_bottom_float.data(), schur_b_gpu, x_bottom_float.rows() * sizeof(float), cudaMemcpyDeviceToHost);
  //       m_x.bottomRows(dense_degrees_of_freedom) = x_bottom_float.cast<Scalar>();
  //     }
  //     cudaFree(schur_b_gpu);
  //     
  //     CHECK_EQ(cusolverDnDestroy(cusolver_handle), CUSOLVER_STATUS_SUCCESS);
  //   }
  // }
  
  
  /// Expected residual count, used to pre-allocate space.
  int m_expected_residual_count = 10000;  // TODO: Make configurable
  
  /// Whether to use a block-diagonal structure for the Schur complement.
  bool m_use_block_diagonal_structure = false;
  
  /// Block size within block-diagonal part at the top-left of the Hessian.
  int m_block_size = 0;
  
  /// Number of blocks within block-diagonal part at the top-left of the Hessian.
  int m_num_blocks = 0;
  
  /// If using the Schur complement, determines whether to store the
  /// off-diagonal sub-matrix of H sparsely (true) or densely (false).
  bool m_sparse_storage_for_off_diag_H = true;
  
  /// If true, the cost function class must accumulate the residuals and
  /// Jacobians for the blocks on the block-diagonal part of H that will be used
  /// for the Schur complement in order, block by block. After finishing a block,
  /// it must call FinishedBlockForSchurComplement() on the accumulator. This
  /// allows the blocks to be processed on the fly, avoiding to store all of
  /// them at the same time, which saves memory.
  bool m_on_the_fly_block_processing = false;
  
  /// If m_on_the_fly_block_processing is true, determines how many blocks are
  /// accumulated before being processed.
  int m_block_batch_size = 0;
  
  /// Whether to use CUDA for computing a large matrix multiplication in the
  /// computation of the Schur complement.
  bool m_compute_schur_complement_with_cuda = false;
  
  /// Rank deficiency of the Hessian (e.g., due to gauge freedom) that should
  /// be accounted for in solving for state updates by setting the
  /// least-constrained variable updates to zero.
  int m_rank_deficiency = 0;
  
  /// Whether to use completeOrthogonalDecomposition() for matrix solving.
  bool m_use_complete_orthogonal_decomposition = false;
  
  /// The last value of lambda used in Levenberg-Marquardt.
  Scalar m_lambda = -1;
  
  /// Original diagonal values of H before adding lambda. This is used in case
  /// lambda is updated to compute the updated diagonal values quickly and with
  /// high precision (better than adding the lambda difference values onto the
  /// diagonal multiple times).
  vector<Scalar> m_original_diagonal;
  
  /// Dense part of the Gauss-Newton Hessian approximation.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_dense_H;
  
  /// Block-diagonal part of the Gauss-Newton Hessian approximation.
  /// Used if m_use_block_diagonal_structure.
  vector<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> m_block_diag_H;
  
  /// Off-diagonal part of the Gauss-Newton Hessian approximation (dense).
  /// Used if m_use_block_diagonal_structure && !m_sparse_storage_for_off_diag_H.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_off_diag_H;
  
  /// Off-diagonal part of the Gauss-Newton Hessian approximation (sparse).
  /// Used if m_use_block_diagonal_structure && m_sparse_storage_for_off_diag_H.
  /// The entries in the vector correspond to block-rows. Each entry corresponds
  /// to the entry with the same index in m_block_diag_H.
  vector<SparseColumnMatrix<Scalar>> m_off_diag_H_sparse;
  
  /// Solution to the update equation.
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_x;
  
  /// Vector for the right hand side of the update linear equation system,
  /// corresponding to the dense part of H, m_dense_H.
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_dense_b;
  
  /// Part of b corresponding to m_block_diag_H.
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_block_diag_b;
  
  int m_num_fixed_variables = 0;
  vector<bool> fixed_variables;
  
 friend class LMOptimizerTestHelper;
};

}

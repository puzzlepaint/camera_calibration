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

// TODO: This file contains commented-out gradient descent code for LMOptimizer.

// In Optimize():
//     constexpr bool kUseGradientDescent = false;  // TODO
//     if (kUseGradientDescent) {
//       return OptimizeWithGradientDescentImpl<State, CostFunction, is_reversible>(
//           state,
//           cost_function,
//           max_iteration_count,
//           max_lm_attempts,
//           init_lambda,
//           init_lambda_factor,
//           print_progress);
//     } else {
//       // Use other type of solver ...
//     }


//   Eigen::Matrix<Scalar, 1, Eigen::Dynamic> step_scaling;  // TODO: TEST

//   template <class State, class CostFunction, bool IsReversible>
//   OptimizationReport OptimizeWithGradientDescentImpl(
//       State* state,
//       const CostFunction& cost_function,
//       int max_iteration_count,
//       int max_lm_attempts,
//       Scalar /*init_lambda*/,
//       Scalar /*init_lambda_factor*/,
//       bool print_progress) {
//     OptimizationReport report;
//     report.num_iterations_performed = 0;
//     report.cost_and_jacobian_evaluation_time = numeric_limits<double>::quiet_NaN();  // TODO: not supported yet!
//     report.solve_time = 0;
//     
//     const int degrees_of_freedom = DegreesOfFreedomGetter<State>::eval(*state);
//     CHECK_GT(degrees_of_freedom, 0);
//     
//     Scalar last_cost = numeric_limits<Scalar>::quiet_NaN();
//     bool applied_update = false;
//     int iteration;
//     for (iteration = 0; iteration < max_iteration_count; ++ iteration) {
//       ResidualSumAndJacobianAccumulator<Scalar> jacobian_accumulator(degrees_of_freedom);
//       cost_function.template Compute<true>(*state, &jacobian_accumulator);
//       last_cost = jacobian_accumulator.cost();
//       if (iteration == 0) {
//         report.initial_cost = last_cost;
//       }
//       
//       const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& jacobian = jacobian_accumulator.jacobian();
//       CHECK_EQ(jacobian.cols(), step_scaling.cols());
//       Eigen::Matrix<Scalar, Eigen::Dynamic, 1> step = step_scaling.cwiseProduct(jacobian.normalized()).transpose();
//       
//       Scalar step_factor = 1;
//       applied_update = false;
//       for (int lm_iteration = 0; lm_iteration < max_lm_attempts; ++ lm_iteration) {
//         OptionalCopy<!IsReversible, State> optional_state_copy(state);
//         State* updated_state = optional_state_copy.GetObject();
//         m_x = step_factor * step;
//         *updated_state -= m_x;
//         
//         // Test whether taking over the update will decrease the cost.
//         UpdateEquationAccumulator<Scalar> test_cost(nullptr, nullptr, nullptr, nullptr, nullptr);
//         cost_function.template Compute<false>(*updated_state, &test_cost);
//         
//         if (test_cost.cost() < jacobian_accumulator.cost()) {
//           // Take over the update.
//           if (IsReversible) {
//             // no action required, keep the updated state
//           } else {
//             *state = *updated_state;
//           }
//           last_cost = test_cost.cost();
//           applied_update = true;
//           report.num_iterations_performed += 1;
//           LOG(1) << "GDOptimizer: update accepted, new cost: " << last_cost;
//           break;
//         } else {
//           if (IsReversible) {
//             // Undo update. This may cause slight numerical inaccuracies.
//             // TODO: Would it be better to combine undoing the old update and
//             //       applying the new update (if not giving up) into a single step?
//             *state -= -m_x;
//           }
//           
//           step_factor *= 0.8;  // TODO: tune? Make configurable.
//           LOG(1) << "GDOptimizer:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << max_lm_attempts
//                  << "] update rejected (bad cost: " << test_cost.cost()
//                  << "), new step_factor: " << step_factor;
//         }
//       }
//       
//       if (!applied_update) {
//         if (print_progress) {
//           if (last_cost == 0) {
//             LOG(INFO) << "GDOptimizer: Reached zero cost, stopping.";
//           } else {
//             LOG(INFO) << "GDOptimizer: Cannot find an update which decreases the cost, aborting.";
//           }
//         }
//         iteration += 1;  // For correct display only.
//         break;
//       }
//     }
//     
//     if (print_progress) {
//       if (applied_update) {
//         LOG(INFO) << "GDOptimizer: Maximum iteration count reached, stopping.";
//       }
// //       LOG(INFO) << "GDOptimizer: [" << iteration << "] Cost / Jacobian computation time: " << cost_and_jacobian_time_in_seconds << " seconds";
// //       LOG(INFO) << "GDOptimizer: [" << iteration << "] Solve time: " << solve_time_in_seconds << " seconds";
//       LOG(INFO) << "GDOptimizer: [" << iteration << "] Final cost:   " << last_cost;  // length matches with "Initial cost: "
//     }
//     
//     report.final_cost = last_cost;
//     return report;
//   }

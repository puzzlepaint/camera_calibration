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


#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/lm_optimizer.h>
#include <libvis/logging.h>
#include <libvis/sophus.h>
#include <gtest/gtest.h>

using namespace vis;

template <int N>
class ToyCostFunction {
 public:
  ToyCostFunction(bool use_gauge_fixing_residuals)
      : use_gauge_fixing_residuals(use_gauge_fixing_residuals) {}
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const Matrix<double, N, 1>& state,
      Accumulator* accumulator) const {
    float residual = 5;
    for (int i = 0; i < N; ++ i) {
      residual += state(i);
    }
    
    if (compute_jacobians) {
      Matrix<double, 1, N> jacobian = Matrix<double, 1, N>::Constant(1);
      accumulator->AddResidualWithJacobian(residual, 0, jacobian);
    } else {
      accumulator->AddResidual(residual);
    }
    
    if (use_gauge_fixing_residuals) {
      // Push all variables except the first one towards their current value.
      for (int i = 1; i < N; ++ i) {
        float jacobian = 1;
        if (compute_jacobians) {
          accumulator->AddResidualWithJacobian(0, i, Matrix<double, 1, 1>(jacobian));
        } else {
          accumulator->AddResidual(0);
        }
      }
    }
  }
  
 private:
  bool use_gauge_fixing_residuals;
};

namespace {
template <int N>
void TestToyProblem(bool set_rank_deficiency, bool use_gauge_fixing_residuals) {
  ToyCostFunction<N> cost_function(use_gauge_fixing_residuals);
  Matrix<double, N, 1> state = Matrix<double, N, 1>::Zero();
  LMOptimizer<double> optimizer;
  if (set_rank_deficiency) {
    optimizer.AccountForRankDeficiency(N - 1);
  }
  
  OptimizationReport report = optimizer.Optimize(
      &state,
      cost_function,
      /*max_iteration_count*/ 100,
      /*max_lm_attempts*/ 10,
      /*init_lambda*/ -1,
      /*init_lambda_factor*/ 0.001,
      /*print_progress*/ false);
  EXPECT_LE(report.final_cost, 1e-6f);
  
  LOG(INFO) << "N = " << N << ":";
  report.Print();
  LOG(INFO) << "";
}
}

TEST(Optimization, ToyProblems) {
  // Try how different optimization methods perform on the problem:
  //   5 + sum_i v_i = 0
  // where v_i are the variables, which has lots of Gauge freedom, starting from
  // v_i = 0.
  for (int i = 0; i < 3; ++ i) {
    LOG(INFO) << "======================================================";
    if (i == 0) {
      LOG(INFO) << "Test with Gauge freedom and default parameters";
    } else if (i == 1) {
      LOG(INFO) << "Test with Gauge freedom and AccountForRankDeficiency()";
    } else if (i == 2) {
      LOG(INFO) << "Test with Gauge fixing residuals";
    }
    LOG(INFO) << "";
    
    bool set_rank_deficiency = (i == 1);
    bool use_gauge_fixing_residuals = (i == 2);
    TestToyProblem<1>(set_rank_deficiency, use_gauge_fixing_residuals);
    TestToyProblem<2>(set_rank_deficiency, use_gauge_fixing_residuals);
    TestToyProblem<3>(set_rank_deficiency, use_gauge_fixing_residuals);
    TestToyProblem<50>(set_rank_deficiency, use_gauge_fixing_residuals);
    TestToyProblem<100>(set_rank_deficiency, use_gauge_fixing_residuals);
  }
}

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
#include "libvis/loss_functions.h"

using namespace vis;

namespace {
template <typename LossT>
void Check(float residual, const LossT& loss) {
  constexpr float kEpsilon = 1e-8;
  CHECK_LE(fabs(loss.ComputeCost(residual) - loss.ComputeCostFromSquaredResidual(residual * residual)), kEpsilon);
  CHECK_LE(fabs(loss.ComputeWeight(residual) - loss.ComputeWeightFromSquaredResidual(residual * residual)), kEpsilon);
}
}

// Checks that the Compute...() and the Compute...FromSquaredResidual() functions
// return consistent results.
TEST(LossFunctions, QuadraticLoss) {
  QuadraticLoss loss;
  Check(-2., loss);
  Check(-1., loss);
  Check(0., loss);
  Check(1., loss);
  Check(2., loss);
}

// Checks that the Compute...() and the Compute...FromSquaredResidual() functions
// return consistent results.
TEST(LossFunctions, HuberLoss) {
  HuberLoss<double> loss(1.4);
  Check(-2., loss);
  Check(-1., loss);
  Check(0., loss);
  Check(1., loss);
  Check(2., loss);
}

// Checks that the Compute...() and the Compute...FromSquaredResidual() functions
// return consistent results.
TEST(LossFunctions, TukeyBiweightLoss) {
  TukeyBiweightLoss<double> loss(1.4);
  Check(-2., loss);
  Check(-1., loss);
  Check(0., loss);
  Check(1., loss);
  Check(2., loss);
}

// Checks that the Compute...() and the Compute...FromSquaredResidual() functions
// return consistent results.
TEST(LossFunctions, CauchyLoss) {
  CauchyLoss<double> loss(1.4);
  Check(-2., loss);
  Check(-1., loss);
  Check(0., loss);
  Check(1., loss);
  Check(2., loss);
}

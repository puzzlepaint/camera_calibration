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

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/loss_functions.h"

namespace vis {

template <typename Scalar>
class CostAccumulator {
 public:
  inline void AddInvalidResidual() {
    // no-op
  }
  
  template <typename LossFunctionT = QuadraticLoss>
  inline void AddResidual(
      Scalar residual,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived>
  inline void AddResidual(
      const MatrixBase<Derived>& residual,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived>
  inline void AddResidualWithJacobian(
      Scalar residual,
      u32 /*index*/,
      const MatrixBase<Derived>& /*jacobian*/,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
  inline void AddResidualWithJacobian(
      const MatrixBase<Derived0>& residual,
      u32 /*index*/,
      const MatrixBase<Derived1>& /*jacobian*/,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
  inline void AddResidualWithJacobian(
      Scalar residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2>
  inline void AddResidualWithJacobian(
      const MatrixBase<Derived2>& residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2>
  inline void AddResidualWithJacobian(
      Scalar residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      u32 /*index2*/,
      const MatrixBase<Derived2>& /*jacobian2*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      bool /*enable2*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3>
  inline void AddResidualWithJacobian(
      const MatrixBase<Derived3>& residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      u32 /*index2*/,
      const MatrixBase<Derived2>& /*jacobian2*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      bool /*enable2*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1>
  inline void AddResidualWithJacobian(
      Scalar residual,
      const MatrixBase<Derived0>& /*indices*/,
      const MatrixBase<Derived1>& /*jacobian*/,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2>
  inline void AddResidualWithJacobian(
      const MatrixBase<Derived2>& residual,
      const MatrixBase<Derived0>& /*indices*/,
      const MatrixBase<Derived1>& /*jacobian*/,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3>
  inline void AddResidualWithJacobian(
      Scalar residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      const MatrixBase<Derived2>& /*indices2*/,
      const MatrixBase<Derived3>& /*jacobian2*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3,
            typename Derived4>
  inline void AddResidualWithJacobian(
      const MatrixBase<Derived4>& residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      const MatrixBase<Derived2>& /*indices2*/,
      const MatrixBase<Derived3>& /*jacobian2*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3,
            typename Derived4>
  inline void AddResidualWithJacobian(
      Scalar residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      u32 /*index2*/,
      const MatrixBase<Derived2>& /*jacobian2*/,
      const MatrixBase<Derived3>& /*indices3*/,
      const MatrixBase<Derived4>& /*jacobian3*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      bool /*enable2*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    cost_ += loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3,
            typename Derived4,
            typename Derived5>
  inline void AddResidualWithJacobian(
      const MatrixBase<Derived5>& residual,
      u32 /*index0*/,
      const MatrixBase<Derived0>& /*jacobian0*/,
      u32 /*index1*/,
      const MatrixBase<Derived1>& /*jacobian1*/,
      u32 /*index2*/,
      const MatrixBase<Derived2>& /*jacobian2*/,
      const MatrixBase<Derived3>& /*indices3*/,
      const MatrixBase<Derived4>& /*jacobian3*/,
      bool /*enable0*/ = true,
      bool /*enable1*/ = true,
      bool /*enable2*/ = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar squared_residual = residual.squaredNorm();
    cost_ += loss_function.ComputeCostFromSquaredResidual(squared_residual);
  }
  
  inline void FinishedBlockForSchurComplement() const {
    // no-op
  }
  
  inline void Reset() { cost_ = 0; }
  
  inline Scalar cost() const { return cost_; }
  
  private:
  Scalar cost_ = 0;
};

}

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

/// This helper class routes different AddResidualWithJacobian() function calls
/// to simpler variants that work on parts of the full Jacobian only, reducing
/// the effort to implement all variants. It is meant for accumulators that
/// multiply the transposed Jacobian with itself (J^T * J).
/// 
/// Derived classes need to implement the following functions:
/// 
///   // Residuals
///   
///   inline void AddInvalidResidual();
///   
///   template <typename ScalarOrVector, typename LossFunctionT = QuadraticLoss>
///   inline void AddResidual(
///        const ScalarOrVector& residual,
///        const LossFunctionT& loss_function = LossFunctionT());
///   
///   
///   // Jacobians
///   
///   // It always holds: indices_row[i] <= indices_col[j] for all i and j
///   template <bool on_diagonal, typename ScalarOrVector, typename DerivedIRow, typename DerivedJRow, typename DerivedICol, typename DerivedJCol, typename LossFunctionT = QuadraticLoss>
///   inline void AddJacobianII(
///        const ScalarOrVector& residual,
///        const MatrixBase<DerivedIRow>& indices_row,
///        const MatrixBase<DerivedJRow>& jacobian_weighted_row,
///        const MatrixBase<DerivedICol>& indices_col,
///        const MatrixBase<DerivedJCol>& jacobian_col,
///        const LossFunctionT& loss_function = LossFunctionT());
///   
///   
///   // -----------------------------------------------------------------------
///   // Implementing the following ones is optional. They may be implemented
///   // for potentially improved performance (in case the cost function class
///   // uses the corresponding AddResidualWithJacobian() variants).
///   // NOTE: In one test, performance was actually significantly worse when
///   //       implementing these, despite the fact that this should have meant
///   //       less computations. I am not sure what the cause was. Can it
///   //       happen that Eigen's vectorization can actually be detrimental?
///   
///   // It always holds: index_row < index_col
///   template <bool on_diagonal, typename ScalarOrVector, typename DerivedJRow, typename DerivedJCol, typename LossFunctionT>
///   inline void AddJacobianBB(
///        const ScalarOrVector& residual,
///        u32 index_row,
///        const MatrixBase<DerivedJRow>& jacobian_weighted_row,
///        u32 index_col,
///        const MatrixBase<DerivedJCol>& jacobian_col,
///        const LossFunctionT& loss_function);
///   
///   // It always holds: index_row < indices_col[i] for all i. on_diagonal is always false for this version.
///   template <typename ScalarOrVector, typename DerivedJRow, typename DerivedICol, typename DerivedJCol, typename LossFunctionT>
///   inline void AddJacobianBI(
///        const ScalarOrVector& residual,
///        u32 index_row,
///        const MatrixBase<DerivedJRow>& jacobian_weighted_row,
///        const MatrixBase<DerivedICol>& indices_col,
///        const MatrixBase<DerivedJCol>& jacobian_col,
///        const LossFunctionT& loss_function);
///   
///   // It always holds: indices_row[i] < index_col for all i. on_diagonal is always false for this version.
///   template <typename ScalarOrVector, typename DerivedIRow, typename DerivedJRow, typename DerivedJCol, typename LossFunctionT>
///   inline void AddJacobianIB(
///        const ScalarOrVector& residual,
///        const MatrixBase<DerivedIRow>& indices_row,
///        const MatrixBase<DerivedJRow>& jacobian_weighted_row,
///        u32 index_col,
///        const MatrixBase<DerivedJCol>& jacobian_col,
///        const LossFunctionT& loss_function);
template <typename Scalar, typename DerivedClass>
class LMOptimizerJTJAccumulatorBase {
 public:
  // ---------------------------------------------------------------------------
  // Default implementations of some functions that route the inputs to others.
  // For best performance, those functions should be hidden with direct
  // implementations in the derived class.
  
  template <bool on_diagonal, typename ScalarOrVector, typename DerivedJRow, typename DerivedJCol, typename LossFunctionT>
  inline void AddJacobianBB(
       const ScalarOrVector& residual,
       u32 index_row,
       const MatrixBase<DerivedJRow>& jacobian_weighted_row,
       u32 index_col,
       const MatrixBase<DerivedJCol>& jacobian_col,
       const LossFunctionT& loss_function) {
    Matrix<u32, Dynamic, 1> indices_row;
    indices_row.resize(jacobian_weighted_row.cols());
    for (int i = 0, cols = jacobian_weighted_row.cols(); i < cols; ++ i) {
      indices_row[i] = index_row + i;
    }
    
    Matrix<u32, Dynamic, 1> indices_col;
    indices_col.resize(jacobian_col.cols());
    for (int i = 0, cols = jacobian_col.cols(); i < cols; ++ i) {
      indices_col[i] = index_col + i;
    }
    
    static_cast<DerivedClass*>(this)->template AddJacobianII<on_diagonal>(residual, indices_row, jacobian_weighted_row, indices_col, jacobian_col, loss_function);
  }
  
  template <typename ScalarOrVector, typename DerivedJRow, typename DerivedICol, typename DerivedJCol, typename LossFunctionT>
  inline void AddJacobianBI(
       const ScalarOrVector& residual,
       u32 index_row,
       const MatrixBase<DerivedJRow>& jacobian_weighted_row,
       const MatrixBase<DerivedICol>& indices_col,
       const MatrixBase<DerivedJCol>& jacobian_col,
       const LossFunctionT& loss_function) {
    Matrix<u32, Dynamic, 1> indices_row;
    indices_row.resize(jacobian_weighted_row.cols());
    for (int i = 0, cols = jacobian_weighted_row.cols(); i < cols; ++ i) {
      indices_row[i] = index_row + i;
    }
    
    static_cast<DerivedClass*>(this)->template AddJacobianII<false>(residual, indices_row, jacobian_weighted_row, indices_col, jacobian_col, loss_function);
  }
  
  template <typename ScalarOrVector, typename DerivedIRow, typename DerivedJRow, typename DerivedJCol, typename LossFunctionT>
  inline void AddJacobianIB(
       const ScalarOrVector& residual,
       const MatrixBase<DerivedIRow>& indices_row,
       const MatrixBase<DerivedJRow>& jacobian_weighted_row,
       u32 index_col,
       const MatrixBase<DerivedJCol>& jacobian_col,
       const LossFunctionT& loss_function) {
    Matrix<u32, Dynamic, 1> indices_col;
    indices_col.resize(jacobian_col.cols());
    for (int i = 0, cols = jacobian_col.cols(); i < cols; ++ i) {
      indices_col[i] = index_col + i;
    }
    
    static_cast<DerivedClass*>(this)->template AddJacobianII<false>(residual, indices_row, jacobian_weighted_row, indices_col, jacobian_col, loss_function);
  }
  
  
  // ---------------------------------------------------------------------------
  // Implementations of functions that can be called by cost function classes
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived,
            typename ScalarOrVector>
  inline void AddResidualWithJacobian(
      const ScalarOrVector& residual,
      u32 index,
      const MatrixBase<Derived>& jacobian,
      const LossFunctionT& loss_function = LossFunctionT()) {
    static_cast<DerivedClass*>(this)->AddResidual(residual, loss_function);
    
    const Scalar weight = ComputeWeight(residual, loss_function);
    Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> jacobian_weighted = weight * jacobian;
    
    static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index, jacobian_weighted, index, jacobian, loss_function);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename ScalarOrVector>
  inline void AddResidualWithJacobian(
      const ScalarOrVector& residual,
      const MatrixBase<Derived0>& indices,
      const MatrixBase<Derived1>& jacobian,
      const LossFunctionT& loss_function = LossFunctionT()) {
    static_cast<DerivedClass*>(this)->AddResidual(residual, loss_function);
    
    const Scalar weight = ComputeWeight(residual, loss_function);
    Matrix<Scalar, Derived1::RowsAtCompileTime, Derived1::ColsAtCompileTime> jacobian_weighted = weight * jacobian;
    
    static_cast<DerivedClass*>(this)->template AddJacobianII<true>(residual, indices, jacobian_weighted, indices, jacobian, loss_function);
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename ScalarOrVector>
  inline void AddResidualWithJacobian(
      const ScalarOrVector& residual,
      u32 index0,
      const MatrixBase<Derived0>& jacobian0,
      u32 index1,
      const MatrixBase<Derived1>& jacobian1,
      bool enable0 = true,
      bool enable1 = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    static_cast<DerivedClass*>(this)->AddResidual(residual, loss_function);
    
    const Scalar weight = ComputeWeight(residual, loss_function);
    
    if (enable0) {
      Matrix<Scalar, Derived0::RowsAtCompileTime, Derived0::ColsAtCompileTime> jacobian0_weighted = weight * jacobian0;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index0, jacobian0_weighted, index0, jacobian0, loss_function);
      if (enable1) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index0, jacobian0_weighted, index1, jacobian1, loss_function);
      }
    }
    if (enable1) {
      Matrix<Scalar, Derived1::RowsAtCompileTime, Derived1::ColsAtCompileTime> jacobian1_weighted = weight * jacobian1;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index1, jacobian1_weighted, index1, jacobian1, loss_function);
    }
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename ScalarOrVector>
  inline void AddResidualWithJacobian(
      const ScalarOrVector& residual,
      u32 index0,
      const MatrixBase<Derived0>& jacobian0,
      u32 index1,
      const MatrixBase<Derived1>& jacobian1,
      u32 index2,
      const MatrixBase<Derived2>& jacobian2,
      bool enable0 = true,
      bool enable1 = true,
      bool enable2 = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    static_cast<DerivedClass*>(this)->AddResidual(residual, loss_function);
    
    const Scalar weight = ComputeWeight(residual, loss_function);
    
    if (enable0) {
      Matrix<Scalar, Derived0::RowsAtCompileTime, Derived0::ColsAtCompileTime> jacobian0_weighted = weight * jacobian0;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index0, jacobian0_weighted, index0, jacobian0, loss_function);
      if (enable1) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index0, jacobian0_weighted, index1, jacobian1, loss_function);
      }
      if (enable2) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index0, jacobian0_weighted, index2, jacobian2, loss_function);
      }
    }
    if (enable1) {
      Matrix<Scalar, Derived1::RowsAtCompileTime, Derived1::ColsAtCompileTime> jacobian1_weighted = weight * jacobian1;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index1, jacobian1_weighted, index1, jacobian1, loss_function);
      if (enable2) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index1, jacobian1_weighted, index2, jacobian2, loss_function);
      }
    }
    if (enable2) {
      Matrix<Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime> jacobian2_weighted = weight * jacobian2;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index2, jacobian2_weighted, index2, jacobian2, loss_function);
    }
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3,
            typename ScalarOrVector>
  inline void AddResidualWithJacobian(
      const ScalarOrVector& residual,
      u32 index0,
      const MatrixBase<Derived0>& jacobian0,
      u32 index1,
      const MatrixBase<Derived1>& jacobian1,
      const MatrixBase<Derived2>& indices2,
      const MatrixBase<Derived3>& jacobian2,
      bool enable0 = true,
      bool enable1 = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    constexpr bool enable2 = true;  // TODO: Include this as a parameter?
    
    static_cast<DerivedClass*>(this)->AddResidual(residual, loss_function);
    
    const Scalar weight = ComputeWeight(residual, loss_function);
    
    if (enable0) {
      Matrix<Scalar, Derived0::RowsAtCompileTime, Derived0::ColsAtCompileTime> jacobian0_weighted = weight * jacobian0;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index0, jacobian0_weighted, index0, jacobian0, loss_function);
      if (enable1) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index0, jacobian0_weighted, index1, jacobian1, loss_function);
      }
      if (enable2) {
        static_cast<DerivedClass*>(this)->template AddJacobianBI(residual, index0, jacobian0_weighted, indices2, jacobian2, loss_function);
      }
    }
    if (enable1) {
      Matrix<Scalar, Derived1::RowsAtCompileTime, Derived1::ColsAtCompileTime> jacobian1_weighted = weight * jacobian1;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index1, jacobian1_weighted, index1, jacobian1, loss_function);
      if (enable2) {
        static_cast<DerivedClass*>(this)->template AddJacobianBI(residual, index1, jacobian1_weighted, indices2, jacobian2, loss_function);
      }
    }
    if (enable2) {
      Matrix<Scalar, Derived3::RowsAtCompileTime, Derived3::ColsAtCompileTime> jacobian2_weighted = weight * jacobian2;
      
      static_cast<DerivedClass*>(this)->template AddJacobianII<true>(residual, indices2, jacobian2_weighted, indices2, jacobian2, loss_function);
    }
  }
  
  template <typename LossFunctionT = QuadraticLoss,
            typename Derived0,
            typename Derived1,
            typename Derived2,
            typename Derived3,
            typename Derived4,
            typename ScalarOrVector>
  inline void AddResidualWithJacobian(
      const ScalarOrVector& residual,
      u32 index0,
      const MatrixBase<Derived0>& jacobian0,
      u32 index1,
      const MatrixBase<Derived1>& jacobian1,
      u32 index2,
      const MatrixBase<Derived2>& jacobian2,
      const MatrixBase<Derived3>& indices3,
      const MatrixBase<Derived4>& jacobian3,
      bool enable0 = true,
      bool enable1 = true,
      bool enable2 = true,
      const LossFunctionT& loss_function = LossFunctionT()) {
    constexpr bool enable3 = true;  // TODO: Include this as a parameter?
    
    static_cast<DerivedClass*>(this)->AddResidual(residual, loss_function);
    
    const Scalar weight = ComputeWeight(residual, loss_function);
    
    if (enable0) {
      Matrix<Scalar, Derived0::RowsAtCompileTime, Derived0::ColsAtCompileTime> jacobian0_weighted = weight * jacobian0;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index0, jacobian0_weighted, index0, jacobian0, loss_function);
      if (enable1) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index0, jacobian0_weighted, index1, jacobian1, loss_function);
      }
      if (enable2) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index0, jacobian0_weighted, index2, jacobian2, loss_function);
      }
      if (enable3) {
        static_cast<DerivedClass*>(this)->template AddJacobianBI(residual, index0, jacobian0_weighted, indices3, jacobian3, loss_function);
      }
    }
    if (enable1) {
      Matrix<Scalar, Derived1::RowsAtCompileTime, Derived1::ColsAtCompileTime> jacobian1_weighted = weight * jacobian1;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index1, jacobian1_weighted, index1, jacobian1, loss_function);
      if (enable2) {
        static_cast<DerivedClass*>(this)->template AddJacobianBB<false>(residual, index1, jacobian1_weighted, index2, jacobian2, loss_function);
      }
      if (enable3) {
        static_cast<DerivedClass*>(this)->template AddJacobianBI(residual, index1, jacobian1_weighted, indices3, jacobian3, loss_function);
      }
    }
    if (enable2) {
      Matrix<Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime> jacobian2_weighted = weight * jacobian2;
      
      static_cast<DerivedClass*>(this)->template AddJacobianBB<true>(residual, index2, jacobian2_weighted, index2, jacobian2, loss_function);
      if (enable3) {
        static_cast<DerivedClass*>(this)->template AddJacobianBI(residual, index2, jacobian2_weighted, indices3, jacobian3, loss_function);
      }
    }
    if (enable3) {
      Matrix<Scalar, Derived4::RowsAtCompileTime, Derived4::ColsAtCompileTime> jacobian3_weighted = weight * jacobian3;
      
      static_cast<DerivedClass*>(this)->template AddJacobianII<true>(residual, indices3, jacobian3_weighted, indices3, jacobian3, loss_function);
    }
  }
  
 private:
  template <typename LossFunctionT>
  inline Scalar ComputeWeight(Scalar residual, const LossFunctionT& loss_function) const {
    return loss_function.ComputeWeight(residual);
  }
  
  template <typename LossFunctionT, typename Derived>
  inline Scalar ComputeWeight(const MatrixBase<Derived>& residual, const LossFunctionT& loss_function) const {
    return loss_function.ComputeWeightFromSquaredResidual(residual.squaredNorm());
  }
};

}

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

#include <unordered_map>

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/lm_optimizer_jtj_accumulator_base.h"
#include "libvis/lm_optimizer_update_accumulator.h"
#include "libvis/logging.h"
#include "libvis/loss_functions.h"

namespace vis {

/// This LMOptimizer accumulator computes the matrix-vector multiplication:
/// block_diagonal^(-1) * (b_block_diagonal - off-diagonal * x_dense)
/// This is the last step in solving for the update x with the Schur complement.
/// The computation is implemented while never storing the matrices (apart from
/// the vector x) in memory fully.
template <typename Scalar>
class BackSubstitutionAccumulator : public LMOptimizerJTJAccumulatorBase<Scalar, BackSubstitutionAccumulator<Scalar>> {
 public:
  BackSubstitutionAccumulator(
      int block_diagonal_degrees_of_freedom,
      vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>* block_diag_H,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* off_diag_H,
      vector<SparseColumnMatrix<Scalar>>* off_diag_H_sparse,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* block_diag_b,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* x,
      int block_batch_size)
      : block_batch_size(block_batch_size),
        block_diagonal_degrees_of_freedom_(block_diagonal_degrees_of_freedom),
        block_diag_H_(block_diag_H),
        off_diag_H_(off_diag_H),
        off_diag_H_sparse_(off_diag_H_sparse),
        block_diag_b_(block_diag_b),
        x_(x) {
    if (block_diag_H) {
      for (auto& matrix : *block_diag_H) {
        matrix.setZero();
      }
    }
    if (off_diag_H) {
      off_diag_H->setZero();
    }
    if (off_diag_H_sparse) {
      for (auto& matrix : *off_diag_H_sparse) {
        matrix.setZero();
      }
    }
    if (block_diag_b_) {
      block_diag_b_->setZero();
    }
  }
  
  inline void AddInvalidResidual() {
    // no-op
  }
  
  /// To be called by CostFunction to add a residual to the cost.
  template <typename LossFunctionT = QuadraticLoss,
            typename ScalarOrVector>
  inline void AddResidual(
      const ScalarOrVector& /*residual*/,
      const LossFunctionT& /*loss_function*/ = LossFunctionT()) {
    // no-op
  }
  
  // It always holds: indices_row[i] <= indices_col[j] for all i and j
  template <bool on_diagonal, typename ScalarOrVector, typename DerivedIRow, typename DerivedJRow, typename DerivedICol, typename DerivedJCol, typename LossFunctionT = QuadraticLoss>
  inline void AddJacobianII(
      const ScalarOrVector& residual,
      const MatrixBase<DerivedIRow>& indices_row,
      const MatrixBase<DerivedJRow>& jacobian_weighted_row,
      const MatrixBase<DerivedICol>& indices_col,
      const MatrixBase<DerivedJCol>& jacobian_col,
      const LossFunctionT& /*loss_function*/ = LossFunctionT()) {
    if (indices_row[0] >= block_diagonal_degrees_of_freedom_) {
      return;
    }
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    int row_offset;
    int col_offset;
    int row_offset_b;
    GetHAndOffsets(indices_row[0], indices_col[0], &H, &row_offset, &col_offset, &row_offset_b);
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      // TODO: assert(!on_diagonal);
      int block_index = (indices_row[0] + row_offset) / block_size();
      int index_offset = -on_the_fly_base_row - block_index * block_size();
      for (int k = 0; k < indices_col.size(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(indices_col[k] + col_offset);
        for (int i = 0; i < indices_row.size(); ++ i) {
          column(static_cast<unsigned int>(indices_row[i] + index_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      for (int i = 0; i < indices_row.size(); ++ i) {
        for (int k = on_diagonal ? i : 0; k < indices_col.size(); ++ k) {
          (*H)(static_cast<unsigned int>(indices_row[i] + row_offset), static_cast<unsigned int>(indices_col[k] + col_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
        
        if (on_diagonal) {
          (*block_diag_b_)(static_cast<unsigned int>(indices_row[i] + row_offset_b)) += DotOrMultiply(residual, jacobian_weighted_row.col(i));
        }
      }
    }
  }
  
  // It always holds: index_row < index_col
  template <bool on_diagonal, typename ScalarOrVector, typename DerivedJRow, typename DerivedJCol, typename LossFunctionT = QuadraticLoss>
  inline void AddJacobianBB(
       const ScalarOrVector& residual,
       u32 index_row,
       const MatrixBase<DerivedJRow>& jacobian_weighted_row,
       u32 index_col,
       const MatrixBase<DerivedJCol>& jacobian_col,
       const LossFunctionT& /*loss_function*/ = LossFunctionT()) {
    if (index_row >= block_diagonal_degrees_of_freedom_) {
      return;
    }
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    int row_offset;
    int col_offset;
    int row_offset_b;
    GetHAndOffsets(index_row, index_col, &H, &row_offset, &col_offset, &row_offset_b);
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      // TODO: assert(!on_diagonal);
      int block_index = (index_row + row_offset) / block_size();
      int index_offset = -on_the_fly_base_row - block_index * block_size();
      for (int k = 0; k < jacobian_col.cols(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(index_col + k + col_offset);
        for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
          column(static_cast<unsigned int>(index_row + i + index_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
        for (int k = on_diagonal ? i : 0; k < jacobian_col.cols(); ++ k) {
          (*H)(static_cast<unsigned int>(index_row + i + row_offset), static_cast<unsigned int>(index_col + k + col_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
        
        if (on_diagonal) {
          (*block_diag_b_)(static_cast<unsigned int>(index_row + i + row_offset_b)) += DotOrMultiply(residual, jacobian_weighted_row.col(i));
        }
      }
    }
  }
  
  // It always holds: index_row < indices_col[i] for all i
  template <typename ScalarOrVector, typename DerivedJRow, typename DerivedICol, typename DerivedJCol, typename LossFunctionT = QuadraticLoss>
  inline void AddJacobianBI(
       const ScalarOrVector& /*residual*/,
       u32 index_row,
       const MatrixBase<DerivedJRow>& jacobian_weighted_row,
       const MatrixBase<DerivedICol>& indices_col,
       const MatrixBase<DerivedJCol>& jacobian_col,
       const LossFunctionT& /*loss_function*/ = LossFunctionT()) {
    if (index_row >= block_diagonal_degrees_of_freedom_) {
      return;
    }
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    int row_offset;
    int col_offset;
    int row_offset_b;
    GetHAndOffsets(index_row, indices_col[0], &H, &row_offset, &col_offset, &row_offset_b);
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      int block_index = (index_row + row_offset) / block_size();
      int index_offset = -on_the_fly_base_row - block_index * block_size();
      for (int k = 0; k < indices_col.size(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(indices_col[k] + col_offset);
        for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
          column(static_cast<unsigned int>(index_row + i + index_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
        for (int k = 0; k < indices_col.size(); ++ k) {
          (*H)(static_cast<unsigned int>(index_row + i + row_offset), static_cast<unsigned int>(indices_col[k] + col_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    }
  }
  
  // It always holds: indices_row[i] < index_col for all i
  template <typename ScalarOrVector, typename DerivedIRow, typename DerivedJRow, typename DerivedJCol, typename LossFunctionT = QuadraticLoss>
  inline void AddJacobianIB(
       const ScalarOrVector& /*residual*/,
       const MatrixBase<DerivedIRow>& indices_row,
       const MatrixBase<DerivedJRow>& jacobian_weighted_row,
       u32 index_col,
       const MatrixBase<DerivedJCol>& jacobian_col,
       const LossFunctionT& /*loss_function*/ = LossFunctionT()) {
    if (indices_row[0] >= block_diagonal_degrees_of_freedom_) {
      return;
    }
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    int row_offset;
    int col_offset;
    int row_offset_b;
    GetHAndOffsets(indices_row[0], index_col, &H, &row_offset, &col_offset, &row_offset_b);
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      int block_index = (indices_row[0] + row_offset) / block_size();
      int index_offset = -on_the_fly_base_row - block_index * block_size();
      for (int k = 0; k < jacobian_col.cols(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(index_col + k + col_offset);
        for (int i = 0; i < indices_row.size(); ++ i) {
          column(static_cast<unsigned int>(indices_row[i] + index_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      for (int i = 0; i < indices_row.size(); ++ i) {
        for (int k = 0; k < jacobian_col.cols(); ++ k) {
          (*H)(static_cast<unsigned int>(indices_row[i] + row_offset), static_cast<unsigned int>(index_col + k + col_offset)) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    }
  }
  
  inline void FinishedBlockForSchurComplement() {
    ++ blocks_accumulated;
    if (blocks_accumulated < block_batch_size) {
      return;
    }
    
    AccumulateFinishedBlocks();
    
    // Reset matrices for the next block-row
    if (block_diag_H_) {
      for (auto& matrix : *block_diag_H_) {
        matrix.setZero();
      }
    }
    if (off_diag_H_) {
      off_diag_H_->setZero();
    }
    if (off_diag_H_sparse_) {
      for (auto& matrix : *off_diag_H_sparse_) {
        matrix.setZero();
      }
    }
    if (block_diag_b_) {
      block_diag_b_->setZero();
    }
  }
  
  void AccumulateFinishedBlocks() {
    if (blocks_accumulated == 0) {
      return;
    }
    
    Matrix<Scalar, Eigen::Dynamic, 1> segment;
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverted_diag_block;
    
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block_I;
    block_I.resize(block_size(), block_size());
    block_I.setIdentity();
    
    if (off_diag_H_) {
      // NOTE: This corresponds to the last part in LMOptimizer::SolveWithSchurComplementDenseOffDiag().
      
      Matrix<Scalar, Eigen::Dynamic, 1> result =
          block_diag_b_->topRows(blocks_accumulated * block_size()) -
          off_diag_H_->block(0, 0, blocks_accumulated * block_size(), off_diag_H_->cols()) *
              x_->bottomRows(x_->size() - block_diagonal_degrees_of_freedom_);
      
      for (int block = 0; block < blocks_accumulated; ++ block) {
        auto& original_diag_block = block_diag_H_->at(block);
        inverted_diag_block = original_diag_block.template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        x_->segment(on_the_fly_base_row + block * block_size(), block_size()) = inverted_diag_block * result.segment(block * block_size(), block_size());
      }
    } else {
      // NOTE: This duplicates code in LMOptimizer::SolveWithSchurComplementSparseOffDiag().
      
      for (int block = 0; block < blocks_accumulated; ++ block) {
        auto& original_diag_block = block_diag_H_->at(block);
        inverted_diag_block = original_diag_block.template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        auto& sparse_off_diag_block = (*off_diag_H_sparse_)[block];
        
        segment = block_diag_b_->segment(block * block_size(), block_size());
        
        for (int stored_col_index = 0; stored_col_index < sparse_off_diag_block.column_indices.size(); ++ stored_col_index) {
          int actual_col_index = sparse_off_diag_block.column_indices[stored_col_index];
          
          segment -= sparse_off_diag_block.columns.col(stored_col_index) * (*x_)(block_diagonal_degrees_of_freedom_ + actual_col_index);
        }
        
        x_->segment(on_the_fly_base_row + block * block_size(), block_size()) = inverted_diag_block * segment;
      }
    }
    
    on_the_fly_base_row += blocks_accumulated * block_size();
    blocks_accumulated = 0;
  }
  
  
 private:
  inline void GetHAndOffsets(int row, int col, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>** H, int* row_offset, int* col_offset, int* row_offset_b) const {
    if (col < block_diagonal_degrees_of_freedom_) {
      int block_index = (row - on_the_fly_base_row) / block_size();
      *H = &block_diag_H_->at(block_index);
      *row_offset = -on_the_fly_base_row - block_index * block_size();
      *col_offset = *row_offset;
    } else {
      *H = off_diag_H_;
      *row_offset = -on_the_fly_base_row;
      *col_offset = -block_diagonal_degrees_of_freedom_;
    }
    *row_offset_b = -on_the_fly_base_row;
  }
  
  template <typename DerivedB>
  inline Scalar DotOrMultiply(Scalar a, const MatrixBase<DerivedB>& b) const {
    static_assert(DerivedB::RowsAtCompileTime == 1 && DerivedB::ColsAtCompileTime == 1, "This function overload should only be chosen if b represents a single value");
    return a * b(0);
  }
  
  template <typename DerivedA, typename DerivedB>
  inline Scalar DotOrMultiply(const MatrixBase<DerivedA>& a, const MatrixBase<DerivedB>& b) const {
    return a.dot(b);
  }
  
  inline int block_size() const {
    return block_diag_H_->at(0).rows();
  }
  
  
  int on_the_fly_base_row = 0;
  int blocks_accumulated = 0;
  int block_batch_size = 1;
  
  int block_diagonal_degrees_of_freedom_;
  vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>* block_diag_H_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* off_diag_H_;
  vector<SparseColumnMatrix<Scalar>>* off_diag_H_sparse_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* block_diag_b_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* x_;
};

}

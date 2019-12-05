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
#include "libvis/logging.h"
#include "libvis/loss_functions.h"

namespace vis {

/// Internal storage type for m_off_diag_H_sparse.
template <typename Scalar>
struct SparseColumnMatrix {
  typedef Block<Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Dynamic, 1, true> ColXpr;
  
  void setZero() {
    column_indices.clear();
    original_to_stored_index.clear();
    // TODO: Make the minimum width configurable
    constexpr int kMinimumWidth = 16;
    if (columns.cols() > 4 * kMinimumWidth) {
      columns.resize(NoChange, kMinimumWidth);
    }
    columns.setZero();
  }
  
  void resizeHeight(int height) {
    // TODO: Make the minimum width configurable
    constexpr int kMinimumWidth = 16;
    columns.resize(height, std::max<int>(kMinimumWidth, columns.cols()));
  }
  
  ColXpr GetColumn(int column) {
//     // Look for existing columns
//     for (int i = 0, size = column_indices.size(); i < size; ++ i) {
//       if (column == column_indices[i]) {
//         return columns.col(i);
//       }
//     }
    
    if (column < original_to_stored_index.size() &&
        original_to_stored_index[column] >= 0) {
      return columns.col(original_to_stored_index[column]);
    }
    
    // Create new column
    column_indices.push_back(column);
    int new_column_stored_index = column_indices.size() - 1;
    if (column_indices.size() > columns.cols()) {
      int old_cols = columns.cols();
      columns.conservativeResize(NoChange, 2 * old_cols);  // TODO: Make the growing factor configurable?
      columns.rightCols(old_cols).setZero();
    }
    
    if (column >= original_to_stored_index.size()) {
      original_to_stored_index.resize(column + 1, -1);
    }
    original_to_stored_index[column] = new_column_stored_index;
    
    return columns.col(new_column_stored_index);
  }
  
  /// Stores all column indices with non-zeros.
  vector<int> column_indices;
  
  /// Maps original column indices to stored column indices. The column data is
  /// thus in: columns.col(original_to_stored_index.at(original_column_index)).
  vector<int> original_to_stored_index;
  
  /// Column i in this matrix holds the entries for column column_indices[i]
  /// of the complete represented matrix. Only columns with non-zeros are
  /// stored.
  Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> columns;
};


template <typename Scalar>
class UpdateEquationAccumulator : public LMOptimizerJTJAccumulatorBase<Scalar, UpdateEquationAccumulator<Scalar>> {
 public:
  UpdateEquationAccumulator(
      int block_diagonal_degrees_of_freedom,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* dense_H,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* off_diag_H,
      vector<SparseColumnMatrix<Scalar>>* off_diag_H_sparse,
      vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>* block_diag_H,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* dense_b,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* block_diag_b,
      vector<Scalar>* residual_cost_vector,
      int block_batch_size,  // set block_batch_size to 0 to disable on-the-fly block processing
      Scalar lambda)
      : cost_(0),
        block_diagonal_degrees_of_freedom_(block_diagonal_degrees_of_freedom),
        dense_H_(dense_H),
        off_diag_H_(off_diag_H),
        off_diag_H_sparse_(off_diag_H_sparse),
        block_diag_H_(block_diag_H),
        dense_b_(dense_b),
        block_diag_b_(block_diag_b),
        residual_cost_vector_(residual_cost_vector),
        block_batch_size(block_batch_size),
        lambda(lambda) {
    if (dense_H_) {
      dense_H_->setZero();
    }
    if (off_diag_H) {
      off_diag_H->setZero();
    }
    if (off_diag_H_sparse) {
      for (auto& matrix : *off_diag_H_sparse) {
        matrix.setZero();
      }
    }
    if (block_diag_H) {
      for (auto& matrix : *block_diag_H) {
        matrix.setZero();
      }
    }
    if (dense_b_) {
      dense_b_->setZero();
    }
    if (block_diag_b_) {
      block_diag_b_->setZero();
    }
  }
  
  
  inline void AddInvalidResidual() {
    residual_cost_vector_->emplace_back(-1);
  }
  
  /// To be called by CostFunction to add a residual to the cost.
  template <typename LossFunctionT = QuadraticLoss,
            typename ScalarOrVector>
  inline void AddResidual(
      const ScalarOrVector& residual,
      const LossFunctionT& loss_function = LossFunctionT()) {
    Scalar residual_cost = ComputeResidualCost(residual, loss_function);
    cost_ += residual_cost;
    if (residual_cost_vector_) {
      // Verify that the cost is non-negative, since we use negative costs to
      // indicate invalid residuals.
      CHECK_GE(residual_cost, 0);  // TODO: Do this in debug mode only
      residual_cost_vector_->emplace_back(residual_cost);
    }
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
    // All of the entries must be in the same part of H and b, so get the parts first.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b;
    int local_row;
    int local_col;
    int local_row_b;
    GetPartOfHAndB(indices_row[0], indices_col[0], &H, &b, &local_row, &local_col, &local_row_b);
    int row_offset = local_row - indices_row[0];
    int col_offset = local_col - indices_col[0];
    int row_offset_b = local_row_b - indices_row[0];
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      // TODO: assert(!on_diagonal);
      // TODO: assert(row_offset == 0);
      int block_index = local_row / block_size();
      int index_offset = -block_index * block_size();
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
          (*b)(static_cast<unsigned int>(indices_row[i] + row_offset_b)) += DotOrMultiply(residual, jacobian_weighted_row.col(i));
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
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b;
    int local_row;
    int local_col;
    int local_row_b;
    GetPartOfHAndB(index_row, index_col, &H, &b, &local_row, &local_col, &local_row_b);
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      // TODO: assert(!on_diagonal);
      // TODO: assert(row_offset == 0);
      int block_index = local_row / block_size();
      int base_index = local_row - block_index * block_size();
      for (int k = 0; k < jacobian_col.cols(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(local_col + k);
        for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
          column(base_index + i) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      // TODO: For a specific test case, this implementation here was MASSIVELY
      //       faster than the more "Eigen-native" implementation below. Not sure
      //       whether that is generally the case. --> Try this on more real-world
      //       examples that use this code.
      for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
        for (int k = on_diagonal ? i : 0; k < jacobian_col.cols(); ++ k) {
          (*H)(local_row + i, local_col + k) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
        
        if (on_diagonal) {
          (*b)(local_row_b + i) += DotOrMultiply(residual, jacobian_weighted_row.col(i));
        }
      }
    }
    
    // NOTE: This is the slower implementation (judging from a single test case only).
    // NOTE: This does not work on dynamically-sized matrices.
//     if (on_diagonal) {
//       // TODO: These kinds of checks (static checks / in debug mode only?) would be helpful. This applies to all variants of this function, not only this one.
// //       CHECK_GT(Derived1::ColsAtCompileTime, 0);
// //       CHECK_LE(index_in_H + Derived1::ColsAtCompileTime, H->rows());
// //       CHECK_LE(index_in_b + Derived1::ColsAtCompileTime, b->rows());
//       
//       H->template block<DerivedJRow::ColsAtCompileTime, DerivedJCol::ColsAtCompileTime>(local_row, local_col)
//           .template triangularView<Eigen::Upper>() +=
//               ((jacobian_weighted_row.transpose()) * jacobian_col).template cast<Scalar>();
//       
//       b->template segment<DerivedJRow::ColsAtCompileTime>(local_row_b) +=
//           (jacobian_weighted_row.transpose() * residual).template cast<Scalar>();
//     } else {
//       H->template block<DerivedJRow::ColsAtCompileTime, DerivedJCol::ColsAtCompileTime>(local_row, local_col) +=
//           (jacobian_weighted_row.transpose() * jacobian_col).template cast<Scalar>();
//     }
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
    // All of the entries must be in the same part of H and b, so get the parts first.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b;
    int local_row;
    int local_col;
    int local_row_b;
    GetPartOfHAndB(index_row, indices_col[0], &H, &b, &local_row, &local_col, &local_row_b);
    int col_offset = local_col - indices_col[0];
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      // TODO: assert(!on_diagonal);
      // TODO: assert(row_offset == 0);
      int block_index = local_row / block_size();
      int base_index = local_row - block_index * block_size();
      for (int k = 0; k < jacobian_col.cols(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(indices_col[k] + col_offset);
        for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
          column(base_index + i) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
        for (int k = 0; k < indices_col.size(); ++ k) {
          (*H)(local_row + i, indices_col[k] + col_offset) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
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
    // All of the entries must be in the same part of H and b, so get the parts first.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b;
    int local_row;
    int local_col;
    int local_row_b;
    GetPartOfHAndB(indices_row[0], index_col, &H, &b, &local_row, &local_col, &local_row_b);
    int row_offset = local_row - indices_row[0];
    
    if (off_diag_H_sparse_ && H == off_diag_H_) {
      // TODO: assert(!on_diagonal);
      // TODO: assert(row_offset == 0);
      int block_index = local_row / block_size();
      int index_offset = -block_index * block_size();
      for (int k = 0; k < jacobian_col.cols(); ++ k) {
        auto column = off_diag_H_sparse_->at(block_index).GetColumn(local_col + k);
        for (int i = 0; i < jacobian_weighted_row.cols(); ++ i) {
          column(indices_row[i] + index_offset) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    } else {
      for (int i = 0; i < indices_row.size(); ++ i) {
        for (int k = 0; k < jacobian_col.cols(); ++ k) {
          (*H)(indices_row[i] + row_offset, local_col + k) += (jacobian_weighted_row.col(i).dot(jacobian_col.col(k)));
        }
      }
    }
  }
  
  
  inline void FinishedBlockForSchurComplement() {
    if (!block_diag_H_) {
      return;
    }
    
    ++ blocks_accumulated;
    if (blocks_accumulated < block_batch_size) {
      return;
    }
    
    AccumulateFinishedBlocks();
    
    // Reset the matrices for the next batch of blocks
    if (off_diag_H_) {
      off_diag_H_->setZero();
    }
    if (off_diag_H_sparse_) {
      for (auto& matrix : *off_diag_H_sparse_) {
        matrix.setZero();
      }
    }
    if (block_diag_H_) {
      for (auto& matrix : *block_diag_H_) {
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
    
    // Add lambda on diagonal
    for (int i = 0; i < blocks_accumulated; ++ i) {
      block_diag_H_->at(i).diagonal().array() += lambda;
    }
    
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block_I;
    block_I.resize(block_size(), block_size());
    block_I.setIdentity();
    
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverted_diag_block;
    
    // Accumulated enough blocks for a batch. Process the batch.
    if (off_diag_H_) {
      // Dense processing.
      // NOTE: This corresponds to the first part in LMOptimizer::SolveWithSchurComplementDenseOffDiag().
      
      Matrix<Scalar, Dynamic, Dynamic, RowMajor> BT_DInv(dense_b_->size(), blocks_accumulated * block_size());
      
      for (int block = 0; block < blocks_accumulated; ++ block) {
        inverted_diag_block = (*block_diag_H_)[block].template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        BT_DInv.block(0, block * block_size(), dense_b_->size(), block_size()) =
            off_diag_H_->block(block * block_size(), 0, block_size(), dense_b_->size()).transpose() *
            inverted_diag_block;
      }
      
      *dense_b_ -=
          BT_DInv * block_diag_b_->segment(0, blocks_accumulated * block_size());
      
      *dense_H_ -= BT_DInv * off_diag_H_->block(0, 0, blocks_accumulated * block_size(), dense_b_->size());
    } else {
      // Sparse processing
      // NOTE: This duplicates computations from LMOptimizer::SolveWithSchurComplementSparseOffDiag().
      
      Matrix<Scalar, 1, Eigen::Dynamic> left;
      
      for (int block = 0; block < blocks_accumulated; ++ block) {
        inverted_diag_block = block_diag_H_->at(block).template selfadjointView<Eigen::Upper>().ldlt().solve(block_I);  // (hopefully) fast SPD matrix inversion
        // TODO: Profile this compared to inversion as below:
        // matrix.template triangularView<Eigen::Lower>() = matrix.template triangularView<Eigen::Upper>().transpose();
        // matrix = matrix.inverse();
        
        auto& sparse_off_diag_block = (*off_diag_H_sparse_)[block];
        
        for (int stored_row_index = 0; stored_row_index < sparse_off_diag_block.column_indices.size(); ++ stored_row_index) {
          int actual_row_index = sparse_off_diag_block.column_indices[stored_row_index];
          
          left =
              sparse_off_diag_block.columns.col(stored_row_index).transpose() *
              inverted_diag_block;
          
          (*dense_b_)(actual_row_index) -=
              left *
              block_diag_b_->segment(block * block_size(), block_size());
          
          for (int stored_col_index = 0; stored_col_index < sparse_off_diag_block.column_indices.size(); ++ stored_col_index) {
            int actual_col_index = sparse_off_diag_block.column_indices[stored_col_index];
            
            if (actual_col_index >= actual_row_index) {
              (*dense_H_)(actual_row_index, actual_col_index) -=
                  left *
                  sparse_off_diag_block.columns.col(stored_col_index);
            }
          }
        }
      }
    }
    
    on_the_fly_base_row += blocks_accumulated * block_size();
    blocks_accumulated = 0;
  }
  
  
  inline Scalar cost() const { return cost_; }
  
 private:
  inline void GetPartOfHAndB(int row, int col, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>** H, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>** b, int* local_row, int* local_col, int* local_row_b) const {
    if (row < block_diagonal_degrees_of_freedom_) {
      if (on_the_fly_processing_enabled()) {
        row -= on_the_fly_base_row;
      }
      
      *b = block_diag_b_;
      *local_row_b = row;
      
      if (col < block_diagonal_degrees_of_freedom_) {
        int block_index = row / block_size();
        *H = &block_diag_H_->at(block_index);
        u32 base_index = block_index * block_size();
        *local_row = row - base_index;
        *local_col = col - on_the_fly_base_row - base_index;
      } else {
        *H = off_diag_H_;
        *local_row = row;
        *local_col = col - block_diagonal_degrees_of_freedom_;
      }
    } else {
      *H = dense_H_;
      *b = dense_b_;
      *local_row = row - block_diagonal_degrees_of_freedom_;
      *local_col = col - block_diagonal_degrees_of_freedom_;
      *local_row_b = *local_row;
    }
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
  
  template <typename LossFunctionT>
  inline Scalar ComputeResidualCost(Scalar residual, const LossFunctionT& loss_function) const {
    return loss_function.ComputeCost(residual);
  }
  
  template <typename LossFunctionT, typename DerivedA>
  inline Scalar ComputeResidualCost(const MatrixBase<DerivedA>& residual, const LossFunctionT& loss_function) const {
    return loss_function.ComputeCostFromSquaredResidual(residual.squaredNorm());
  }
  
  inline int block_size() const {
    return block_diag_H_->at(0).rows();
  }
  
  inline bool on_the_fly_processing_enabled() const {
    return block_batch_size > 0;
  }
  
  
  Scalar cost_;
  int block_diagonal_degrees_of_freedom_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* dense_H_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* off_diag_H_;
  vector<SparseColumnMatrix<Scalar>>* off_diag_H_sparse_;
  vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>* block_diag_H_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* dense_b_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* block_diag_b_;
  
  // Since the cost can never be negative, we indicate invalid residuals with
  // negative costs (in particular, -1).
  vector<Scalar>* residual_cost_vector_;
  
  // For on-the-fly block processing
  int on_the_fly_base_row = 0;
  int blocks_accumulated = 0;
  int block_batch_size = 0;
  Scalar lambda;
};

}

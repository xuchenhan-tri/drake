#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/fixed_fem/dev/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Implements SymmetricBlockSparseMatrix with fixed size data.
 @tparam_nonsymbolic_scalar T.
 @tparam block_size Rows and columns of each block in the matrix. */
template <typename T, int block_size>
class SymmetricBlockSparseMatrixImpl : public SymmetricBlockSparseMatrix<T> {
 public:
  using Base = SymmetricBlockSparseMatrix<T>;
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SymmetricBlockSparseMatrixImpl);

  /* Constructs a SymmetricBlockSparseMatrixImpl with the given sparsity
   pattern.
   @param[in] row_blocks  Characterize the sparsity pattern of the matrix being
   constructed. The number of column blocks is given by `row_blocks.size()`.
   `row_blocks[c][r]` gives the block row index of the r-th row block in the
   c-th column. If `row_blocks[c]` is empty, then the c-th column block is
   empty.
   @pre row_blocks[c][r] <= c. In other words, only the upper triangular part of
   the symmetric block sparse matrix is stored. */
  explicit SymmetricBlockSparseMatrixImpl(
      std::vector<std::vector<int>> row_blocks);

  /* Adds Aij to the ij-th block of this matrix.
   @pre Aij = Aij.transpose() if i==j.
   @warn The precondition above is not verified for performance reasons. */
  void AddToBlock(
      int i, int j,
      const Eigen::Ref<const Eigen::Matrix<T, block_size, block_size>>& Aij);

  int cols() const final { return block_size * num_column_blocks_; }

 private:
  void DoSetZero() final;
  void DoMultiply(const VectorX<T>& x, VectorX<T>* y) const final;
  MatrixX<T> DoMakeDenseMatrix() const final;
  Eigen::SparseMatrix<T> DoMakeEigenSparseMatrix() const final;

  /* row_blocks[c][r] gives the r-th row block in the c-th column block. */
  std::vector<std::vector<int>> row_blocks_;
  int num_column_blocks_;
  int num_blocks_;
  std::vector<std::vector<Eigen::Matrix<T, block_size, block_size>>> data_;
  /* num_row_blocks[c] gives the number of row blocks in the c-th column block.
   */
  std::vector<int> num_row_blocks_;
  /* Mapping from row to index for each column; i.e.,
   row_blocks[c][row_blocks_to_index[c][r]] == r.
   row_blocks_to_index_[c][r] == -1 if the implied block is empty. */
  std::vector<std::vector<int>> row_blocks_to_index_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

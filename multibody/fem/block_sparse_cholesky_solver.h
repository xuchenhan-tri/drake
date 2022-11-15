#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

#pragma once

#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

std::vector<std::unordered_set<int>> BuildAdjacencyGraph(
    int num_verts, const std::vector<Vector4<int>>& elements);

std::vector<std::vector<int>> GetFillInGraph(
    int num_verts, const std::vector<Vector4<int>>& cliques);

/* Sparse cholesky solver where the blocks are of size 3x3. */
class BlockSparseCholeskySolver {
 public:
  /* @param row_blocks Specifies the sparsity pattern of the matrix. */
  explicit BlockSparseCholeskySolver(const std::vector<Vector4<int>>& cliques,
                                     int block_cols);

  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BlockSparseCholeskySolver);

  SymmetricBlockSparseMatrix<double>& GetMutableMatrix() { return L_; }

  MatrixX<double> CalcSchurComplement(int num_eliminated_blocks);

  void Factor() { FactorImpl(block_cols_); }

  int size() const { return block_cols_ * 3; }

  void SolveInPlace(VectorX<double>* y) const;

  VectorX<double> Solve(const VectorX<double>& y) const;

 private:
  void FactorImpl(int block_cols_to_factorize);

  /* Performs L(j+1:, j+1:) -= L(j+1:,j) * L(j+1:,j).transpose().
   @pre 0 <= j < block_cols_. */
  void RightLookingSymmetricRank1Update(int j);

  int block_cols_{0};
  SymmetricBlockSparseMatrix<double> L_;
  std::vector<Matrix3<double>> L_diag_;
  bool is_factored_{false};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

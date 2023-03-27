#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Returns the column-wise sparsity pattern of L given the adjacency graph of A
 and the elimination ordering.
 Note that the elimination ordering is a mapping from new index to old index. */
std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::vector<int>>& adjacency_graph,
    const std::vector<int>& elimination_ordering);

/* Given an input matrix M, and a permutation mapping e, sets the resulting
 matrix M̃ such that M̃(i, j) = M(e(i), e(j)). */
void PermuteSymmetricBlockSparseMatrix(
    const SymmetricBlockSparseMatrix<double>& input,
    const std::vector<int>& permutation,
    SymmetricBlockSparseMatrix<double>* result);

/* Sparse cholesky solver where the blocks are of size 3x3. */
class BlockSparseCholeskySolver {
 public:
  /* Constructs a solver. */
  BlockSparseCholeskySolver() = default;

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BlockSparseCholeskySolver);

  /* Sets the matrix to be factored and uses the AMD elimination ordering that
   to reduce fill-ins. */
  void SetMatrix(const SymmetricBlockSparseMatrix<double>& A);

  /* Updates the matrix to be factored. This is useful for solving a series of
   matrices with the same sparsity pattern using the same elimination ordering.
   For example, with matrices A, B, and C with the same sparisty pattern. It's
   more efficient to call
     solver.SetMatrix(A);
     solver.UpdateMatrix(B);
     solver.UpdateMatrix(C);
   than to call
     solver.SetMatrix(A);
     solver.SetMatrix(B);
     solver.SetMatrix(C); */
  void UpdateMatrix(const SymmetricBlockSparseMatrix<double>& A);

  void Factor();

  void SolveInPlace(VectorX<double>* y) const;

  VectorX<double> Solve(const VectorX<double>& y) const;

 private:
  /* Performs L(j+1:, j+1:) -= L(j+1:,j) * L(j+1:,j).transpose().
   @pre 0 <= j < block_cols_. */
  void RightLookingSymmetricRank1Update(int j);

  int block_cols_{0};
  SymmetricBlockSparseMatrix<double> L_{{}};
  std::vector<MatrixX<double>> L_diag_;
  /* The mapping from the internal block indices (i.e, the indices for L_) to
   the block indices of the matrix supplied in SetMatrix(). */
  contact_solvers::internal::PartialPermutation internal_to_original_;
  contact_solvers::internal::PartialPermutation internal_to_original_scalar_;
  bool is_factored_{false};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

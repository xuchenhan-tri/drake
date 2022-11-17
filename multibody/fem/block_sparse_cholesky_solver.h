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

/* result[j] is the set of i>=j such that the i,j block of the matrix is
 nonzero.*/
std::vector<std::set<int>> BuildAdjacencyGraph(
    int num_verts, const std::vector<Vector4<int>>& elements);

/* Gives the permutation ordering for elimination of the matrix with the given
 `adjacency_graph`.
 For example if the result is [1, 3, 0, 2], it means that we should first
 eliminate vertex 1, then 3, 0, and 2. In other words, this is a permutation
 mapping from new vertex indices to old indices. */
std::vector<int> CalcPermutationFromCholmod(
    const std::vector<std::set<int>>& adjacency_graph);

/* Returns an ordering such that everything in `D_indices` come before
 everything else. Within elements in `D_indices` and elements not in `D_indices,
 the ordering in `perfect_ordering` is preserved.
 @param[in] perfect_ordering  The result of CalcPermutationFromCholmod.
 @param[in] D_indices         Nonparticipating vertices that need to be
                              eliminated first. */
std::vector<int> CalcPermutationForSchurComplement(
    const std::vector<int>& perfect_ordering,
    const std::vector<int>& D_indices);

/* Returns the column-wise sparsity pattern of L given the adjacency graph of A
 and the elimination ordering.
 Note that the elimination ordering is a mapping from new index to old index. */
std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::set<int>>& adjacency_graph,
    std::vector<int> elimination_ordering);

std::vector<std::vector<int>> GetFillInGraph(
    int num_verts, const std::vector<Vector4<int>>& cliques);

/* Sparse cholesky solver where the blocks are of size 3x3. */
class BlockSparseCholeskySolver {
 public:
  /* @param sparsity_pattern Specifies the sparsity pattern of the matrix. */
  explicit BlockSparseCholeskySolver(
      std::vector<std::vector<int>> sparsity_pattern);

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

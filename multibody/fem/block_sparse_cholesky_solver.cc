#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

using std::set;
using Vector4i = Vector4<int>;
using contact_solvers::internal::PartialPermutation;
using std::vector;

/* Computes the elimination of the matrix with the given `adjacency_graph` that
 CHOLMOD thinks is the best. For example if the result is [1, 3, 0, 2], it means
 that we should first eliminate vertex 1, then 3, 0, and 2. In other words, this
 is a permutation mapping from new vertex indices to old indices. */
std::vector<int> CalcEliminationOrdering(
    const BlockSparsityPattern& block_sparsity_pattern) {
  // TODO(xuchenhan-tri): Use AMD ordering.
  std::vector<int> result(block_sparsity_pattern.diagonals.size());
  std::iota(result.begin(), result.end(), 0);
  return result;
}

std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::vector<int>>& sparsity_pattern,
    const std::vector<int>& elimination_ordering) {
  int N = elimination_ordering.size();
  DRAKE_DEMAND(static_cast<int>(sparsity_pattern.size()) == N);
  std::vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    old_to_new[elimination_ordering[i]] = i;
  }

  /* Computes the adjacency graph for the new ordering. */
  std::vector<std::set<int>> new_graph(N);
  for (int i = 0; i < N; ++i) {
    for (int v : sparsity_pattern[i]) {
      int a = old_to_new[i];
      int b = old_to_new[v];
      if (a >= b) {
        new_graph[b].insert(a);
      } else {
        new_graph[a].insert(b);
      }
    }
  }

  /* children[p] is a vector of sort children of p. */
  std::vector<std::vector<int>> children(N);
  std::vector<std::vector<int>> result(N);
  for (int i = 0; i < N; ++i) {
    /* Turn set into vector. */
    const std::set<int>& neighbor_i = new_graph[i];
    result[i].reserve(N);
    for (int n : neighbor_i) result[i].emplace_back(n);

    /* Merge the neighbors of i and all neighbors of children of i. */
    const auto& children_i = children[i];
    std::vector<int> result_i;  // Temp variable to hold result[i] as we
                                // accumulate all values.
    result_i.reserve(N);
    for (int c : children_i) {
      const auto& neighbor_c = result[c];
      std::set_union(result[i].begin(), result[i].end(), neighbor_c.begin() + 2,
                     neighbor_c.end(), std::back_inserter(result_i));
      result_i.swap(result[i]);
      result_i.clear();
    }
    /* Record the parent of i if i isn't already the root. */
    if (result[i].size() > 1) {
      const int p = result[i][1];
      children[p].emplace_back(i);
    }
  }
  return result;
}

/* Given an input matrix M, and a permutation mapping e, sets the resulting
 matrix M̃ such that M̃(i, j) = M(e(i), e(j)). */
void PermuteSymmetricBlockSparseMatrix(
    const SymmetricBlockSparseMatrix<double>& input,
    const std::vector<int>& permutation,
    SymmetricBlockSparseMatrix<double>* result) {
  DRAKE_DEMAND(result != nullptr);
  const int N = permutation.size();
  DRAKE_DEMAND(N == input.block_cols());
  result->SetZero();

  /* Construct the inverse mapping of e, f. */
  std::vector<int> inverse_permutation(permutation.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    inverse_permutation[permutation[i]] = i;
  }
  /* M̃(i, j) = M(e(i), e(j)) is equivalent to M̃(f(i), f(j)) = M(i, j). */
  for (int j = 0; j < N; ++j) {
    const std::vector<int>& row_indices = input.get_row_indices_in_col(j);
    for (int i : row_indices) {
      const MatrixX<double>& block = input.get_block(i, j);
      const int fi = inverse_permutation[i];
      const int fj = inverse_permutation[j];
      if (fi >= fj) {
        result->SetBlock(fi, fj, block);
      } else {
        result->SetBlock(fj, fi, block.transpose());
      }
    }
  }
}

void BlockSparseCholeskySolver::SetMatrix(
    const SymmetricBlockSparseMatrix<double>& A) {
  const BlockSparsityPattern& A_block_pattern = A.block_sparsity_pattern();
  const std::vector<int> elimination_ordering =
      CalcEliminationOrdering(A_block_pattern);
  block_cols_ = elimination_ordering.size();
  vector<int> scalar_permutation(A.cols());
  const std::vector<int>& A_block_sizes = A_block_pattern.diagonals;
  const std::vector<int>& starting_indices = A.starting_cols();
  int new_scalar_index = 0;
  for (int i = 0; i < static_cast<int>(elimination_ordering.size()); ++i) {
    const int block = elimination_ordering[i];
    const int start = starting_indices[block];
    const int size = A_block_sizes[block];
    for (int s = start; s < start + size; ++s) {
      scalar_permutation[new_scalar_index++] = s;
    }
  }
  internal_to_original_ = PartialPermutation(elimination_ordering);
  internal_to_original_scalar_ = PartialPermutation(move(scalar_permutation));

  L_diag_.resize(block_cols_);

  std::vector<std::vector<int>> L_sparsity_pattern = CalcSparsityPattern(
      A_block_pattern.sparsity_pattern, elimination_ordering);
  std::vector<int> L_block_sizes(A.block_cols());
  internal_to_original_.ApplyInverse(A_block_sizes, &L_block_sizes);
  BlockSparsityPattern L_block_pattern = {
      .diagonals = std::move(L_block_sizes),
      .sparsity_pattern = std::move(L_sparsity_pattern)};
  L_ = SymmetricBlockSparseMatrix<double>(std::move(L_block_pattern));
  UpdateMatrix(A);
}

void BlockSparseCholeskySolver::UpdateMatrix(
    const SymmetricBlockSparseMatrix<double>& A) {
  PermuteSymmetricBlockSparseMatrix(A, internal_to_original_.permutation(),
                                    &L_);
  is_factored_ = false;
}

void BlockSparseCholeskySolver::SolveInPlace(VectorX<double>* y) const {
  DRAKE_DEMAND(is_factored_);
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(y->size() == L_.cols());
  VectorX<double> permuted_y(*y);
  internal_to_original_scalar_.ApplyInverse(*y, &permuted_y);

  const BlockSparsityPattern& block_sparsity_pattern =
      L_.block_sparsity_pattern();
  const std::vector<int>& block_sizes = block_sparsity_pattern.diagonals;
  const std::vector<int>& starting_cols = L_.starting_cols();

  /* Solve Lz = y in place. */
  for (int j = 0; j < block_cols_; ++j) {
    const int block_size = block_sizes[j];
    const int offset = starting_cols[j];
    /* Solve for the j-th block entry. */
    permuted_y.segment(offset, block_size) =
        L_diag_[j].triangularView<Eigen::Lower>().solve(
            permuted_y.segment(offset, block_size));
    const auto& yj = permuted_y.segment(offset, block_size);
    /* Eliminate for the j-th block entry from the system. */
    const auto& blocks_in_col_j = L_.get_row_indices_in_col(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_y.segment(starting_cols[i], block_sizes[i]) -=
          L_.get_block(i, j) * yj;
    }
  }

  VectorX<double>& permuted_z = permuted_y;
  /* Solve Lᵀx = z in place. */
  for (int j = block_cols_ - 1; j >= 0; --j) {
    /* Eliminate all solved variables. */
    const auto& blocks_in_col_j = L_.get_row_indices_in_col(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_z.segment(starting_cols[j], block_sizes[j]) -=
          L_.get_block(i, j).transpose() *
          permuted_z.segment(starting_cols[i], block_sizes[i]);
    }
    /* Solve for the j-th block entry. */
    permuted_z.segment(starting_cols[j], block_sizes[j]) =
        L_diag_[j].transpose().triangularView<Eigen::Upper>().solve(
            permuted_z.segment(starting_cols[j], block_sizes[j]));
  }
  internal_to_original_scalar_.Apply(permuted_z, y);
}

VectorX<double> BlockSparseCholeskySolver::Solve(
    const VectorX<double>& y) const {
  VectorX<double> x(y);
  SolveInPlace(&x);
  return x;
}

void BlockSparseCholeskySolver::Factor() {
  for (int j = 0; j < L_.block_cols(); ++j) {
    /* Update diagonal. */
    const MatrixX<double>& Ajj = L_.get_diagonal_block(j);
    const auto llt = Eigen::LLT<MatrixX<double>>(Ajj);
    DRAKE_DEMAND(llt.info() == Eigen::Success);
    L_diag_[j] = llt.matrixL();
    /* Technically, there's no need to spell out the diagonal block of the L
     matrix, but we do it here for easy debugging. */
    L_.SetBlockFlat(0, j, L_diag_[j]);

    /* Update column.
     | a₁₁  *  | = | λ₁₁  0 | * | λ₁₁ᵀ L₂₁ᵀ |
     | a₂₁ a₂₂ |   | L₂₁ L₂₂|   |  0   L₂₂ᵀ |
     So we have
      L₂₁λ₁₁ᵀ = a₂₁, and thus
      λ₁₁L₂₁ᵀ = a₂₁ᵀ */
    const std::vector<int>& row_blocks = L_.get_row_indices_in_col(j);
    for (int a = 0; a < static_cast<int>(row_blocks.size()) - 1; ++a) {
      const int flat = a + 1;
      const auto& L_diag_j = L_diag_[j].triangularView<Eigen::Lower>();
      const MatrixX<double>& Aij = L_.get_block_flat(flat, j);
      MatrixX<double>& Lij = L_.get_mutable_block_flat(flat, j);
      Lij = L_diag_j.solve(Aij.transpose()).transpose();
    }
    RightLookingSymmetricRank1Update(j);
  }
  is_factored_ = true;
}

void BlockSparseCholeskySolver::RightLookingSymmetricRank1Update(int j) {
  const std::vector<int>& blocks_in_col_j = L_.get_row_indices_in_col(j);
  const int N = blocks_in_col_j.size();
  /* The following omp parallel for loop is equivalent to this easier to read
   non-openmp compliant for loop. */
  /*
   // We start from f1 = 1 here to skip the j,j entry.
   for (int f1 = 1; f1 < N; ++f1) {
     const int col = blocks_in_col_j[f1];
     const Matrix3<double>& B = L_.get_block_flat(f1, j);
     for (int f2 = f1; f2 < N; ++f2) {
       const int row = blocks_in_col_j[f2];
       const Matrix3<double>& A = L_.get_block_flat(f2, j);
       L_.SubtractProductFromBlock(row, col, A, B);
     }
   }
  */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(12)
#endif
  for (int a = 0; a < N - 1; ++a) {
    const int f1 = a + 1;
    const int col = blocks_in_col_j[f1];
    const MatrixX<double>& B = L_.get_block_flat(f1, j);
    for (int f2 = f1; f2 < N; ++f2) {
      const int row = blocks_in_col_j[f2];
      const MatrixX<double>& A = L_.get_block_flat(f2, j);
      L_.SubtractProductFromBlock(row, col, A, B);
    }
  }
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

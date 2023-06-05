#include "drake/multibody/contact_solvers/block_sparse_cholesky_solver.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "drake/multibody/contact_solvers/minimum_degree_ordering.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

/* Makes a new ordering of elements such that the priority elements come
 first, and the rest follow. The relative orderings within the priority and
 non-priority elements stay the same. This is best illustrated with an example.
 Suppose the input ordering is [8, 4, 7, 6, 3, 2, 9, 1, 5, 0] and the priority
 elements are the odd numbers [1, 3, 5, 7, 9], then output should be [7, 3, 9,
 1, 5, 8, 4, 6, 2, 0]. Notice that the odd numbers come before even numbers and
 the ordering with odd and even numbers stay unchanged.
 @pre ordering is a permutation on {0, 1, ..., ordering.size() - 1}.
 @pre elements in `priority_elements` in [0, ordering.size()).  */
std::vector<int> MakeNewOrderingWithPriorityElements(
    const std::vector<int>& ordering,
    const std::unordered_set<int>& priority_elements) {
  const int num_elements = ordering.size();

  /* priority_index and non_priority_index are the new indices for priority and
   non-priority elements. */
  int priority_index = 0;
  int non_priority_index = priority_elements.size();

  /* The inverse of the result, mapping the elements to indices. */
  std::vector<int> inverse_ordering(num_elements);
  for (int element : ordering) {
    if (priority_elements.count(element) > 0) {
      inverse_ordering[element] = priority_index++;
    } else {
      inverse_ordering[element] = non_priority_index++;
    }
  }

  /* Invert and return result. */
  std::vector<int> result(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    result[inverse_ordering[i]] = i;
  }

  return result;
}

}  // namespace

std::vector<int> ConcatenateMdOrderingWithinGroup(
    const BlockSparsityPattern& global_pattern,
    const std::unordered_set<int>& V1) {
  /* Sizes of V, V1, and V2. */
  const int N = global_pattern.block_sizes().size();
  const int N1 = V1.size();
  const int N2 = N - N1;
  /* Mapping from V to V1 and V2 and the inverse mappings. */
  std::vector<int> global_to_local(N);
  std::vector<int> v1_to_global(N1);
  std::vector<int> v2_to_global(N2);
  /* How many scalar variables in each vertex (as needed for the block sparsity
   pattern for G1 and G2). */
  std::vector<int> v1_block_sizes(N1);
  std::vector<int> v2_block_sizes(N2);
  /* Map from global index to whether the vertex is in V1. For better cache
   consistency, we put this information in a vector. */
  std::vector<bool> in_v1(N);
  int v1_index = 0;
  int v2_index = 0;
  const std::vector<int>& global_block_sizes = global_pattern.block_sizes();
  for (int i = 0; i < N; ++i) {
    if (V1.count(i) > 0) {
      global_to_local[i] = v1_index;
      v1_block_sizes[v1_index] = global_block_sizes[i];
      in_v1[i] = true;
      v1_to_global[v1_index] = i;
      ++v1_index;
    } else {
      global_to_local[i] = v2_index;
      v2_block_sizes[v2_index] = global_block_sizes[i];
      in_v1[i] = false;
      v2_to_global[v2_index] = i;
      ++v2_index;
    }
  }

  /* Build the induced graphs G1 and G2 from the global graph G. */
  const std::vector<std::vector<int>>& G = global_pattern.neighbors();
  std::vector<std::vector<int>> G1(N1);
  std::vector<std::vector<int>> G2(N2);
  for (int a = 0; a < N; ++a) {
    for (int b : G[a]) {
      if (in_v1[a] != in_v1[b]) {
        // One of a and b is in V1 and the other is in V2, so the edge ab is not
        // in either of the induced graph.
        continue;
      }
      const int j = std::min(global_to_local[a], global_to_local[b]);
      const int i = std::max(global_to_local[a], global_to_local[b]);
      if (in_v1[b]) {
        G1[j].emplace_back(i);
      } else {
        G2[j].emplace_back(i);
      }
    }
  }

  const std::vector<int> v1_ordering = ComputeMinimumDegreeOrdering(
      BlockSparsityPattern(std::move(v1_block_sizes), std::move(G1)));
  const std::vector<int> v2_ordering = ComputeMinimumDegreeOrdering(
      BlockSparsityPattern(std::move(v2_block_sizes), std::move(G2)));

  std::vector<int> result;
  result.reserve(N);
  /* The V1 vertices come first. */
  for (int v : v1_ordering) {
    result.emplace_back(v1_to_global[v]);
  }
  /* The V2 vertices follow. */
  for (int v : v2_ordering) {
    result.emplace_back(v2_to_global[v]);
  }
  return result;
}

template <typename BlockType>
BlockSparseCholeskySolver<BlockType>::~BlockSparseCholeskySolver() = default;

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::SetMatrix(const SymmetricMatrix& A) {
  const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
  /* Compute the elimination ordering using Minimum Degree algorithm. */
  const std::vector<int> elimination_ordering =
      ComputeMinimumDegreeOrdering(A_block_pattern);
  BlockSparsityPattern L_block_pattern =
      SymbolicFactor(A, elimination_ordering);
  SetMatrixImpl(A, elimination_ordering, std::move(L_block_pattern));
}

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::UpdateMatrix(
    const SymmetricMatrix& A) {
  PermuteAndCopyToL(A);
  solver_mode_ = SolverMode::kAnalyzed;
}

template <typename BlockType>
bool BlockSparseCholeskySolver<BlockType>::Factor() {
  DRAKE_THROW_UNLESS(solver_mode_ == SolverMode::kAnalyzed);
  const bool success = FactorImpl(0, L_->block_cols());
  solver_mode_ = success ? SolverMode::kFactored : SolverMode::kEmpty;
  return success;
}

// TODO(xuchenhan-tri): return a pair of success and schur complement.
template <typename BlockType>
bool BlockSparseCholeskySolver<BlockType>::CalcSchurComplementAndFactor(
    const SymmetricMatrix& M,
    const std::unordered_set<int>& nonparticipating_blocks,
    MatrixX<double>* schur_complement) {
  const int num_nonparticipating_blocks = nonparticipating_blocks.size();
  const int num_total_blocks = M.block_cols();
  const int num_participating_blocks =
      num_total_blocks - num_nonparticipating_blocks;
  DRAKE_DEMAND(num_participating_blocks >= 0);
  if (num_participating_blocks == 0) {
    *schur_complement = MatrixX<double>::Zero(0, 0);
    SetMatrix(M);
    return Factor();
  }

  /* The Schur complement of the `nonparticipating_blocks` appear on the bottom
   right corner of the matrix under Cholesky factorization when the
   `nonparticipating_blocks` have been factorized first with a right-looking
   Cholesky. So we need an elimination ordering that ensures that the blocks
   associated with `nonparticipating_blocks` are eliminated first. There are
   many ordering that satisfy this requirement, and we look for one that reduces
   fill-in. Here we compare the results of two similar heuristics:
    1. Taking Minimum Degree (MD) ordering, and restricting it that the
       nonparticipating block indices appear first, and
    2. Extract the adjacency of the the nonparticipating blocks G1 as well as
   the adjancency graph of the participating blocks G2 and perform MD on each
       and then concatenate the orderings.

   We count the number of fill-ins resulting from these orderings and pick the
   ordering that has the fewest. */
  const std::vector<int> elimination_ordering =
      PickEliminationOrderingAndSetMatrix(M, nonparticipating_blocks);
  if (!FactorImpl(0, num_nonparticipating_blocks)) {
    schur_complement = nullptr;
    solver_mode_ = SolverMode::kEmpty;
    return false;
  }
  const MatrixX<double> permuted_S =
      L_->MakeDenseBottomRightCorner(num_participating_blocks)
          .template selfadjointView<Eigen::Lower>();
  /* The Schur complement computed here is permuted according to the elimination
   ordering. We need to invert this permutation to obtain the original Schur
   complement. To do that, we need to find the _scalar_ permutation for the
   participating blocks, and for that we need the _block_ permutation of the
   participating blocks as well as the sizes of those blocks. */

  /* The block elimination ordering implies a permutation of the
   participating blocks. For example, suppose the elimination ordering is
   [0, 5, 2, 4, 6, 3, 1], non-participating blocks are {0, 2, 5}, and the
   participating blocks are {1, 3, 4, 6} (notice how the non-participating
   blocks come before participating blocks). Then the induced elimination
   ordering on the participating blocks is [2, 3, 1, 0]. To see that, note
   that 1, 3, 4, 6 are the 0th, 1st, 2nd, 3rd elements in the participating
   group; the order that they are eliminated is

     4 (2nd element in participating blocks).
     6 (3rd element in participating blocks).
     3 (1st element in participating blocks).
     1 (0th element in participating blocks).

   We name this elimination ordering on the participating blocks
   `participating_ordering`. We also store the mapping from global indices to
   participating indices in `global_to_participating`. */
  /* A value of -1 indicates that the element is non-participating. */
  std::vector<int> global_to_participating(num_total_blocks, -1);
  int participating_index = 0;
  for (int i = 0; i < num_total_blocks; ++i) {
    if (nonparticipating_blocks.count(i) == 0) {
      global_to_participating[i] = participating_index++;
    }
  }
  const std::vector<int>& block_sizes = M.sparsity_pattern().block_sizes();
  std::vector<int> participating_ordering;
  std::vector<int> participating_block_sizes;
  participating_ordering.reserve(num_participating_blocks);
  participating_block_sizes.reserve(num_participating_blocks);
  /* We start with i = ssize(nonparticipating_blocks) to skip all
   nonparticipating blocks. */
  for (int i = ssize(nonparticipating_blocks); i < ssize(elimination_ordering);
       ++i) {
    const int element = elimination_ordering[i];
    DRAKE_DEMAND(global_to_participating[element] >= 0);
    participating_ordering.emplace_back(global_to_participating[element]);
    participating_block_sizes.emplace_back(block_sizes[element]);
  }
  /* The scalar index of the first scalar in each participating block. */
  std::vector<int> participating_starting_indices(num_participating_blocks);
  participating_starting_indices[0] = 0;
  for (int i = 1; i < num_participating_blocks; ++i) {
    participating_starting_indices[i] = participating_starting_indices[i - 1] +
                                        participating_block_sizes[i - 1];
  }
  const int num_participating_scalars =
      participating_starting_indices.back() + participating_block_sizes.back();
  /* Build the scalar permutation. */
  VectorX<int> scalar_permutation(num_participating_scalars);
  int new_scalar_index = 0;
  for (int i = 0; i < ssize(participating_ordering); ++i) {
    const int block = participating_ordering[i];
    const int start = participating_starting_indices[block];
    const int size = participating_block_sizes[block];
    for (int s = start; s < start + size; ++s) {
      scalar_permutation(new_scalar_index++) = s;
    }
  }
  auto& S = *schur_complement;
  S.resizeLike(permuted_S);
  Eigen::PermutationMatrix<Eigen::Dynamic> P(scalar_permutation);
  S = P.transpose() * permuted_S * P;

  /* Finish the factorization. */
  bool success = FactorImpl(num_nonparticipating_blocks, L_->block_cols());
  solver_mode_ = success ? SolverMode::kFactored : SolverMode::kEmpty;
  return success;
}

template <typename BlockType>
VectorX<double> BlockSparseCholeskySolver<BlockType>::Solve(
    const Eigen::Ref<const VectorX<double>>& b) const {
  VectorX<double> x(b);
  SolveInPlace(&x);
  return x;
}

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::SolveInPlace(
    VectorX<double>* b) const {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kFactored);
  DRAKE_THROW_UNLESS(b != nullptr);
  DRAKE_THROW_UNLESS(b->size() == L_->cols());
  VectorX<double> permuted_b(*b);
  scalar_permutation_.Apply(*b, &permuted_b);

  const BlockSparsityPattern& block_sparsity_pattern = L_->sparsity_pattern();
  const std::vector<int>& block_sizes = block_sparsity_pattern.block_sizes();
  const std::vector<int>& starting_cols = L_->starting_cols();

  /* Solve Lz = b in place. */
  for (int j = 0; j < L_->block_cols(); ++j) {
    const int block_size = block_sizes[j];
    const int offset = starting_cols[j];
    /* Solve for the j-th block entry. */
    const VectorX<double> bj =
        L_diag_[j].matrixL().solve(permuted_b.segment(offset, block_size));
    permuted_b.segment(offset, block_size) = bj;
    /* Eliminate for the j-th block entry from the system. */
    const auto& blocks_in_col_j = L_->block_row_indices(j);
    for (int flat = 1; flat < ssize(blocks_in_col_j); ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_b.segment(starting_cols[i], block_sizes[i]).noalias() -=
          L_->block_flat(flat, j) * bj;
    }
  }

  VectorX<double>& permuted_z = permuted_b;
  /* Solve Lᵀx = z in place. */
  for (int j = L_->block_cols() - 1; j >= 0; --j) {
    /* Eliminate all solved variables. */
    const auto& blocks_in_col_j = L_->block_row_indices(j);
    for (int flat = 1; flat < ssize(blocks_in_col_j); ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_z.segment(starting_cols[j], block_sizes[j]).noalias() -=
          L_->block_flat(flat, j).transpose() *
          permuted_z.segment(starting_cols[i], block_sizes[i]);
    }
    /* Solve for the j-th block entry. */
    const VectorX<double> zj = L_diag_[j].matrixU().solve(
        permuted_z.segment(starting_cols[j], block_sizes[j]));
    permuted_z.segment(starting_cols[j], block_sizes[j]) = zj;
  }
  scalar_permutation_.ApplyInverse(permuted_z, b);
}

template <typename BlockType>
typename BlockSparseCholeskySolver<BlockType>::LowerTriangularMatrix
BlockSparseCholeskySolver<BlockType>::L() const {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kFactored);
  return *L_;
}

template <typename BlockType>
Eigen::PermutationMatrix<Eigen::Dynamic>
BlockSparseCholeskySolver<BlockType>::CalcPermutationMatrix() const {
  DRAKE_THROW_UNLESS(solver_mode() != SolverMode::kEmpty);
  const std::vector<int>& p = scalar_permutation_.permutation();
  return Eigen::PermutationMatrix<Eigen::Dynamic>(
      Eigen::Map<const VectorX<int>>(p.data(), p.size()));
}

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::SetMatrixImpl(
    const SymmetricMatrix& A, const std::vector<int>& elimination_ordering,
    BlockSparsityPattern&& L_pattern) {
  /* First documented responsibility: set `block_permutation_`. */
  /* Construct the inverse of the elimination ordering, which permutes the
   original indices to new indices. */
  std::vector<int> permutation(elimination_ordering.size());
  for (int i = 0; i < ssize(permutation); ++i) {
    permutation[elimination_ordering[i]] = i;
  }
  block_permutation_ = PartialPermutation(std::move(permutation));
  /* Second documented responsibility: set `scalar_permutation_`. */
  SetScalarPermutation(A, elimination_ordering);
  /* Third documented responsibility: allocate for `L_` and `L_diag_`. */
  L_ = std::make_unique<LowerTriangularMatrix>(std::move(L_pattern));
  L_diag_.resize(A.block_cols());
  /* Fourth documented responsibility: UpdateMatrix. */
  UpdateMatrix(A);
}

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::SetScalarPermutation(
    const SymmetricMatrix& A, const std::vector<int>& elimination_ordering) {
  /* It's easier to build the scalar elimination ordering first from block
   elimination ordering and then convert it to the scalar permutation (the
   inverse of the scalar elimination ordering) that induces the permutation P
   such that L⋅Lᵀ = P⋅A⋅Pᵀ.
   More specificially, Pᵢⱼ = 1 for j = scalar_elimination_ordering[i] (or
   equivalently i = scalar_permutation_[j]) and Pᵢⱼ = 0 otherwise. See
   CalcPermutationMatrix(). */
  std::vector<int> scalar_elimination_ordering(A.cols());
  {
    const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
    const std::vector<int>& A_block_sizes = A_block_pattern.block_sizes();
    const std::vector<int>& starting_indices = A.starting_cols();
    int i_permuted = 0;
    for (int block_permuted = 0; block_permuted < ssize(elimination_ordering);
         ++block_permuted) {
      const int block = elimination_ordering[block_permuted];
      const int start = starting_indices[block];
      const int size = A_block_sizes[block];
      for (int i = start; i < start + size; ++i) {
        scalar_elimination_ordering[i_permuted++] = i;
      }
    }
  }
  /* Invert the elimination ordering to get the permutation. */
  std::vector<int> scalar_permutation(scalar_elimination_ordering.size());
  for (int i_permuted = 0; i_permuted < ssize(scalar_permutation);
       ++i_permuted) {
    scalar_permutation[scalar_elimination_ordering[i_permuted]] = i_permuted;
  }
  scalar_permutation_ = PartialPermutation(std::move(scalar_permutation));
}

template <typename BlockType>
BlockSparsityPattern BlockSparseCholeskySolver<BlockType>::SymbolicFactor(
    const SymmetricMatrix& A, const std::vector<int>& elimination_ordering) {
  /* 1. Compute the block permutation as well as the scalar permutation. */
  const int n = elimination_ordering.size();
  /* Construct the inverse of the elimination ordering, which permutes the
   original indices to new indices. */
  std::vector<int> permutation(n);
  for (int i = 0; i < n; ++i) {
    permutation[elimination_ordering[i]] = i;
  }
  const PartialPermutation block_permutation(std::move(permutation));

  /* Find the sparsity pattern of the permuted A (under the permutation induced
   by the elimination ordering). */
  const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
  const std::vector<int>& A_block_sizes = A_block_pattern.block_sizes();
  const std::vector<std::vector<int>>& sparsity_pattern =
      A_block_pattern.neighbors();
  std::vector<std::vector<int>> permuted_sparsity_pattern(
      sparsity_pattern.size());
  for (int i = 0; i < ssize(sparsity_pattern); ++i) {
    const int pi = block_permutation.permuted_index(i);
    for (int j : sparsity_pattern[i]) {
      const int pj = block_permutation.permuted_index(j);
      permuted_sparsity_pattern[std::min(pi, pj)].emplace_back(
          std::max(pi, pj));
    }
  }
  std::vector<int> permuted_block_sizes(A.block_cols());
  block_permutation.Apply(A_block_sizes, &permuted_block_sizes);

  /* Compute the sparsity pattern of L given the sparsity pattern of A in the
   new ordering. */
  return contact_solvers::internal::SymbolicCholeskyFactor(
      BlockSparsityPattern(permuted_block_sizes, permuted_sparsity_pattern));
}

template <typename BlockType>
std::vector<int>
BlockSparseCholeskySolver<BlockType>::PickEliminationOrderingAndSetMatrix(
    const SymmetricMatrix& M,
    const std::unordered_set<int>& eliminated_blocks) {
  /* Option 1: Take the MD ordering of the entire matrix and restrict it so
   that eliminated blocks come first. */
  const std::vector<int> ordering1 = MakeNewOrderingWithPriorityElements(
      ComputeMinimumDegreeOrdering(M.sparsity_pattern()), eliminated_blocks);
  /* Option 2: Get the induced graphs of the adjacency graph by the eliminated
   and the non-eliminated blocks; perform MD on each and then concatenate the
   orderings. */
  const std::vector<int> ordering2 =
      ConcatenateMdOrderingWithinGroup(M.sparsity_pattern(), eliminated_blocks);
  BlockSparsityPattern L_sparsity_pattern1 = SymbolicFactor(M, ordering1);
  BlockSparsityPattern L_sparsity_pattern2 = SymbolicFactor(M, ordering2);
  if (L_sparsity_pattern1.CalcNumNonzeros() <
      L_sparsity_pattern2.CalcNumNonzeros()) {
    SetMatrixImpl(M, ordering1, std::move(L_sparsity_pattern1));
    return ordering1;
  } else {
    SetMatrixImpl(M, ordering2, std::move(L_sparsity_pattern2));
    return ordering2;
  }
}

template <typename BlockType>
bool BlockSparseCholeskySolver<BlockType>::FactorImpl(int starting_col_block,
                                                      int ending_col_block) {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kAnalyzed);
  for (int j = starting_col_block; j < ending_col_block; ++j) {
    /* Update diagonal. */
    const BlockType& Ajj = L_->diagonal_block(j);
    L_diag_[j].compute(Ajj);
    if (L_diag_[j].info() != Eigen::Success) {
      solver_mode_ = SolverMode::kEmpty;
      return false;
    }
    L_->SetBlockFlat(0, j, L_diag_[j].matrixL());
    /* Update L₂₁ column.
     | a₁₁  *  | = | λ₁₁  0 | * | λ₁₁ᵀ L₂₁ᵀ |
     | a₂₁ a₂₂ |   | L₂₁ L₂₂|   |  0   L₂₂ᵀ |
     So we have
      L₂₁λ₁₁ᵀ = a₂₁, and thus
      λ₁₁L₂₁ᵀ = a₂₁ᵀ */
    const std::vector<int>& row_blocks = L_->block_row_indices(j);
    const auto Ljj = L_diag_[j].matrixL();
    /* We start from flat = 1 here to skip the j,j diagonal entry. */
    for (int flat = 1; flat < ssize(row_blocks); ++flat) {
      const BlockType& Aij = L_->block_flat(flat, j);
      BlockType Lij = Ljj.solve(Aij.transpose()).transpose();
      L_->SetBlockFlat(flat, j, std::move(Lij));
    }
    /* Update L₂₂ according to L₂₂ = a₂₂ - L₂₁⋅L₂₁ᵀ. */
    RightLookingSymmetricRank1Update(j);
  }
  return true;
}

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::RightLookingSymmetricRank1Update(
    int j) {
  const std::vector<int>& blocks_in_col_j = L_->block_row_indices(j);
  const int n = blocks_in_col_j.size();
  /* We start from k = 1 here to skip the j,j diagonal entry. */
  for (int k = 1; k < n; ++k) {
    const int col = blocks_in_col_j[k];
    const BlockType& B = L_->block_flat(k, j);
    for (int l = k; l < n; ++l) {
      const int row = blocks_in_col_j[l];
      const BlockType& A = L_->block_flat(l, j);
      L_->AddToBlock(row, col, -A * B.transpose());
    }
  }
}

template <typename BlockType>
void BlockSparseCholeskySolver<BlockType>::PermuteAndCopyToL(
    const SymmetricMatrix& A) {
  const int n = A.block_cols();
  DRAKE_DEMAND(n == block_permutation_.domain_size());
  DRAKE_DEMAND(n == block_permutation_.permuted_domain_size());
  L_->SetZero();
  for (int j = 0; j < n; ++j) {
    const std::vector<int>& row_indices = A.block_row_indices(j);
    for (int i : row_indices) {
      const BlockType& block = A.block(i, j);
      const int pi = block_permutation_.permuted_index(i);
      const int pj = block_permutation_.permuted_index(j);
      if (pi >= pj) {
        L_->SetBlock(pi, pj, block);
      } else {
        L_->SetBlock(pj, pi, block.transpose());
      }
    }
  }
}

template class BlockSparseCholeskySolver<MatrixX<double>>;
template class BlockSparseCholeskySolver<Matrix3<double>>;

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* Given a block sparsity pattern G on vertices V = {0, 1, ..., N-1} and a
 partition on V = V1 ∪ V2 (such that V1 ∩ V2 = ∅), computes an elimination
 ordering on B in the following way:
  1. Generate the V1-induced graph G1 and the V2-induced graph G2.
  2. Compute the Minimum Degree ordering on G1 and G2 respectively.
  3. Concatenate the orderings so that all vertices in V1 come before vertices
     in V2.
 @param[in] global_pattern  The block sparsity pattern G.
 @param[in] V1              The vertices in the set V1.
 @returns  The elimination ordering obtained by following the algorithm
 described above. */
std::vector<int> ConcatenateMdOrderingWithinGroup(
    const BlockSparsityPattern& global_pattern,
    const std::unordered_set<int>& V1);

/* A supernodal Cholesky solver for solving the symmetric positive definite
 system
   A⋅x = b
 Example use case:

  BlockSparseCholeskySolver<MatrixX<double>> solver;
  // Sets the matrix A.
  solver.SetMatrix(A);
  // Factorize the matrix.
  solver.Factor();
  // Solve A⋅x1 = b1.
  x1 = solver.Solve(b1);
  // Solve A⋅x2 = b2. This reuses the factorization (important for speed!).
  x2 = solver.Solve(b2);
  // Update the numerical values in the matrix but the sparsity pattern doesn't
  // change.
  solver.UpdateMatrix(A2);
  // Need to refactor after the matrix has changed.
  solver.Factor();
  // Solve A2⋅x = b using updated factorization.
  x = solver.Solve(b);

 @tparam BlockType The matrix type for individual block matrices, usually
                   MatrixX<double>, but fixed size matrices are preferred if you
                   know the sizes of blocks are uniform and fixed. */
template <typename BlockType>
class BlockSparseCholeskySolver {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BlockSparseCholeskySolver);

  using SymmetricMatrix =
      BlockSparseLowerTriangularOrSymmetricMatrix<BlockType, true>;

  /* Constructs a solver. */
  BlockSparseCholeskySolver() = default;

  /* Sets the matrix to be factored and computes the elimination ordering using
   minimum degree algorithm to prepare for the factorization. */
  void SetMatrix(const SymmetricMatrix& A);

  /* Updates the matrix to be factored. This is useful for solving a series of
   matrices with the same sparsity pattern using the same elimination ordering.
   For example, with matrices A, B, and C with the same sparisty pattern. It's
   more efficient to call
     solver.SetMatrix(A);
     ...
     solver.UpdateMatrix(B);
     ...
   than to call
     solver.SetMatrix(A);
     ...
     solver.SetMatrix(B);
     ...
   If called before any calls to SetMatrix(), the identity elimination ordering
   is used. */
  void UpdateMatrix(const SymmetricMatrix& A);

  /* Computes the supernodal LLT factorization. Returns true if factorization
   succeeds, otherwise returns false. Failure is triggered by an internal
   failure of Eigen::LLT.  This can fail if, for instance, the input matrix set
   in SetMatrix() or UpdateMatrix() is not positive definite. If failure is
   encountered, the user should verify that the specified matrix is positive
   definite and not poorly conditioned.
   @throws std::exception if matrix_set() returns false. */
  bool Factor();

  /* Computes the supernodal LLT factorization of the given matrix `M`. Returns
   true if factorization succeeds, otherwise returns false. Failure is triggered
   by an internal failure of Eigen::LLT.  This can fail if, for instance, the
   input matrix set in SetMatrix() or UpdateMatrix() is not positive definite.
   If failure is encountered, the user should verify that the specified matrix
   is positive definite and not poorly conditioned.

   In addition, this function computes the Schur complement matrix of the input
   matrix M in the following sense:

   Let E be the set of `eliminated_blocks`. We define permutation p such
   that p(i) < p(j) iff
    (1) i ∈ E and j ∉ E or
    (2) i ∈ E and j ∈ E and i < j or
    (3) i ∉ E and j ∉ E and i < j.

   We then define M̂ = P*M*Pᵀ, i.e, M̂ is stably permuted from M so that blocks
   with indices in `eliminated_blocks` appear on the top left corner of the
   matrix M̂ and all other blocks appear on the bottom right corner of the
   matrix. The matrix M̂ can be written in block form as M̂ = [D B; Bᵀ A] where D
   corresponds to the blocks with indices in E and A corresponds to the blocks
   with indices outside of E. On output, S = A - BᵀD⁻¹B is written to
   `schur_complement`.
   @pre schur_complement != nullptr.
   @pre `eliminated_blocks` has all its entries in [0, M.block_cols()).
   @post is_factored() == true.
   @post matrix_set() == false. */
  bool CalcSchurComplementAndFactor(
      const SymmetricMatrix& M,
      const std::unordered_set<int>& eliminated_blocks,
      MatrixX<double>* schur_complement);

  /* Solves the system A⋅x = b and returns x.
   @throws std::exception if is_factored() returns false. */
  VectorX<double> Solve(const Eigen::Ref<const VectorX<double>>& b) const;

  /* Solves the system A⋅x = b and writes the result in b.
   @throws std::exception if is_factored() returns false. */
  void SolveInPlace(VectorX<double>* y) const;

  /* Returns true iff the matrix is set via SetMatrix() or UpdateMatrix() but
   the matrix hasn't been factorized yet (via Factor() or
   CalcSchurComplementAndFactor()) since the matrix is set. */
  bool matrix_set() const { return matrix_set_; }

  /* Returns true iff the matrix has been factorized (via Factor() or
   CalcSchurComplementAndFactor()) and is ready for back solve. */
  bool is_factored() const { return is_factored_; }

 private:
  using LowerTriangularMatrix =
      BlockSparseLowerTriangularOrSymmetricMatrix<BlockType, false>;

  /* Helper for SetMatrix() and CalcSchurComplementAndFactor() to set the matrix
   given its elimination ordering and the sparsity pattern of its Cholesky
   factorization. It performs the following:
    1. sets `block_permutation_`;
    2. sets `scalar_permutation_`;
    3. allocates for L_ and L_diag_;
    4. calls UpdateMatrix(A) to copy the numeric values of A to L. */
  void SetMatrixImpl(const SymmetricMatrix& A,
                     const std::vector<int>& elimination_ordering,
                     BlockSparsityPattern&& L_pattern);

  /* Sets `scalar_permutation_` given the matrix A and a prescribed elimination
   ordering. */
  void SetScalarPermutation(const SymmetricMatrix& A,
                            const std::vector<int>& elimination_ordering);

  /* Returns the block sparsity pattern of the L matrix from a block sparse
   Cholesky factorization following the prescribed elimination ordering. */
  BlockSparsityPattern SymbolicFactor(
      const SymmetricMatrix& A, const std::vector<int>& elimination_ordering);

  // TODO(xuchenhan-tri) Document this better.
  /* Helper for CalcSchurComplementAndFactor(). */
  std::vector<int> PickEliminationOrderingAndSetMatrix(
      const SymmetricMatrix& M,
      const std::unordered_set<int>& eliminated_blocks);

  /* Helper for Factor() and CalcSchurComplementAndFactor() to factorize part of
   the matrix. Unless [starting_col_block, ending_col_block) is equal to
   [0, L.block_cols()), this may leave L in an intermediate state. Call this
   function with care.
   @pre 0 <= starting_col_block <= ending_col_block <= L.block_cols(). */
  bool FactorImpl(int starting_col_block, int ending_col_block);

  /* Performs L(j+1:, j+1:) -= L(j+1:, j) * L(j+1:, j).transpose().
   @pre 0 <= j < L.block_cols(). */
  void RightLookingSymmetricRank1Update(int j);

  /* Permutes the given matrix A with `block_permutation_` p and set L such that
   the lower triangular part of L satisfies L(p(i), p(j)) = A(i, j).
   @pre SetMarix() has been called. */
  void PermuteAndCopyToL(const SymmetricMatrix& A);

  copyable_unique_ptr<LowerTriangularMatrix> L_;
  std::vector<Eigen::LLT<BlockType>> L_diag_;
  /* The mapping from the internal indices (i.e, the indices for L_) to
   the indices of the original matrix supplied in SetMatrix(). */
  PartialPermutation block_permutation_;  // permutation for block indices, same
                                          // size as A.block_cols().
  PartialPermutation scalar_permutation_;  // permutation for block indices,
                                           // same size as A.cols().
  bool is_factored_{false};
  bool matrix_set_{false};
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

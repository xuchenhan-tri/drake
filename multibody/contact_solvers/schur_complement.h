#pragma once

#include <unordered_set>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_sparse_cholesky_solver.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* Given a symmetric linear system of equations Mz = c in block form as:
     Dx  + By = 0     (1)
     Bᵀx + Ay = a     (2)
 where M = [D B; Bᵀ A], z = [x; y], c = [0; a], if A (size p-by-p),
 D (size q-by-q) and M (size p+q-by-p+q) are positive definite, one can solve
 the system using Schur complement. Specifically, using equation (1), we get
     x = -D⁻¹By       (3)
 Plugging (3) in (1), we get
    (A - BᵀD⁻¹B)y = a.
 After a solution for y is obtained, we can use (3) to recover the solution for
 x. The matrix S = A - BᵀD⁻¹B is the Schur complement of the block D of the
 matrix M. Since M is positive definite, so is the S.

 Given the symmetric matrix M and information on how to decompose M into
 submatrices A, B, and D, this class computes the Schur complement of the block
 D of M. The matrix is factorized in the process and consequently, one can solve
 the system Mz = c efficiently once the Schur complement is computed. */
class SchurComplement {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SchurComplement);

  /* Constructs an empty SchurComplement, i.e one that corresponds
   corresponds to a linear system with no equations.*/
  SchurComplement() : S_(0, 0) {}

  /* Constructs a SchurComplement for the block sparse matrix M of size
  3*N-by-3*N consisting of blocks of size 3x3. `D_indices` and `A_indices`
  determine which block rows/columns make up the D and A diagonal blocks.
  For example, if matrix M is given by
     x x x u u u w w w
     x x x u u u w w w
     x x x u u u w w w
     u u u y y y v v v
     u u u y y y v v v
     u u u y y y v v v
     w w w v v v z z z
     w w w v v v z z z
     w w w v v v z z z
  and D_indices = {0, 2} and A_indices = {1}, then submatrix D is given by
     x x x w w w
     x x x w w w
     x x x w w w
     w w w z z z
     w w w z z z
     w w w z z z;
  submatrix A is given by
     z z z
     z z z
     z z z;
  submatrix B is given by
     u u u
     u u u
     u u u
     v v v
     v v v
     v v v
  @pre D_indices and A_indices are sorted and disjoint and their union equals
  the set {0, ..., N-1}. */
  SchurComplement(const Block3x3SparseSymmetricMatrix& M,
                  std::unordered_set<int> D_indices);

  /* Returns the Schur complement for the block D of the matrix M,
   S = A - BD⁻¹Bᵀ. */
  const MatrixX<double>& get_D_complement() const { return S_; }

  /* Given a value of y, solves for x in the equation Dx + By = 0.
   @pre The size of y is equal to the number of columns of B, which is equal
   to the number of rows/columns of A implied at construction. */
  VectorX<double> SolveForX(const Eigen::Ref<const VectorX<double>>& y) const;

  /* Given a right hand side vector c with the same dimension as the input
   matrix M provided at construction, solve solves for M*z = c.
   @pre The size of c is compatible with the input matrix M provided at
   construction. */
  VectorX<double> Solve(const Eigen::Ref<const VectorX<double>>& c) const;

 private:
  /* Sorted block row/column indices for 3x3 blocks that belong to submatrix D.
   */
  std::vector<int> D_indices_;
  /* Sorted block row/column indices for 3x3 blocks that belong to submatrix A.
   */
  std::vector<int> A_indices_;
  /* Cholesky solver that factorizes the matrix and stores the factorization. */
  BlockSparseCholeskySolver<Matrix3<double>> solver_;
  /* The Schur complement of block D: S = A - BᵀD⁻¹B. */
  MatrixX<double> S_;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

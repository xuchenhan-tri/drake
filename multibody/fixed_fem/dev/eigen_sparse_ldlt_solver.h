#pragma once

#include <Eigen/SparseCholesky>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/linear_system_solver.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
/* Implements LinearSystemSolver with an underlying Eigen::SimplicialLDLT
 solver. The matrix A in the system Ax = b must be symmetric positive definite
 (SPD). This solver, however, will not verify that A is SPD as the verification
 process is usually expensive. The user of this class therefore has to be
 careful in passing in a SPD matrix. Failure to do so may lead to a throw.
 @tparam_nonsymbolic_scalar T */
template <typename T>
class EigenSparseLdltSolver : public LinearSystemSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EigenSparseLdltSolver);

  /* Constructs an EigenSparseLdltSolver. The given `A` must be SPD and outlive
   `this` object. */
  explicit EigenSparseLdltSolver(
      const contact_solvers::internal::SparseLinearOperator<T>* A)
      : LinearSystemSolver<T>(A), A_(A->sparse_matrix()) {}

  ~EigenSparseLdltSolver() {}

  /* Implements LinearSystemSolver::Solve(). */
  void Solve(const Eigen::Ref<const VectorX<T>>& b,
             EigenPtr<VectorX<T>> x) const final {
    *x = ldlt_.solve(b);
  }

 private:
  void DoCompute() const final {
    ldlt_.compute(*A_);
    if (ldlt_.info() != Eigen::Success) {
      throw std::runtime_error(
          fmt::format("Cholesky decomposition failed in {}, make sure the "
                      "system is symmetric and positive definite.",
                      __func__));
    }
  }

  const Eigen::SparseMatrix<T>* A_;
  /* TODO(xuchenhan-tri): Experiment with the ordering method of the solver.
   */
  mutable Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower,
                               Eigen::NaturalOrdering<int>>
      ldlt_;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake

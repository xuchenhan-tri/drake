#pragma once

#include <Eigen/Cholesky>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sparse_linear_operator.h"
#include "drake/multibody/fixed_fem/dev/linear_system_solver.h"
namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
/* Implements LinearSystemSolver with an underlying Eigen::LLT (Cholesky)
 solver. The matrix A in the system Ax = b must be symmetric positive definite
 (SPD). This solver, however, will not verify that A is SPD as the verification
 process is usually expensive. The user of this class therefore has to be
 careful in passing in a SPD matrix. Failure to do so will lead to a throw when
 `Compute()` is invoked.
 @tparam_nonsymbolic_scalar T */
template <typename T>
class EigenCholeskySolver : public LinearSystemSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EigenCholeskySolver);

  /* Constructs an EigenCholeskySolver. The given `A` must be SPD and outlive
   `this` object. */
  explicit EigenCholeskySolver(
      const contact_solvers::internal::SparseLinearOperator<T>* A)
      : LinearSystemSolver<T>(A) {
    Eigen::SparseMatrix<T> temp(A->rows(), A->cols());
    A->AssembleMatrix(&temp);
    A_ = temp;
  }

  ~EigenCholeskySolver() {}

  /* Implements LinearSystemSolver::Solve(). */
  void Solve(const Eigen::Ref<const VectorX<T>>& b,
             EigenPtr<VectorX<T>> x) const final {
    *x = llt_.solve(b);
  }

 private:
  void DoCompute() const final {
    llt_.compute(A_);
    if (llt_.info() != Eigen::Success) {
      throw std::runtime_error(
          fmt::format("Cholesky decomposition failed in {}, make sure the "
                      "system is symmetric and positive definite.",
                      __func__));
    }
  }

  MatrixX<T> A_;
  mutable Eigen::LLT<MatrixX<T>, Eigen::Lower> llt_;
};
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake

#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/eigen_sparse_matrix.h"
#include "drake/fem/linear_system_solver.h"

namespace drake {
namespace fem {
// TODO(xuchenhan-tri): This can be made more general to handle all Eigen
// iterative solvers.
template <typename T>
class EigenConjugateGradientSolver : public LinearSystemSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EigenConjugateGradientSolver)

  explicit EigenConjugateGradientSolver(
      const BackwardEulerObjective<T>& objective)
      : matrix_(objective) {}

  virtual ~EigenConjugateGradientSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(const Eigen::Ref<const VectorX<T>>& rhs,
                     EigenPtr<VectorX<T>> x) {
    *x = cg_.solve(rhs);
    std::cout << "CG iterations = " << cg_.iterations() << std::endl;
  }

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() {
    cg_.setTolerance(1e-5);
    cg_.setMaxIterations(300);
    matrix_.BuildMatrix();
    cg_.compute(matrix_);
  }

  bool is_matrix_free() const { return matrix_.is_matrix_free(); }

  void set_matrix_free(bool matrix_free) {
    matrix_.set_matrix_free(matrix_free);
  }

 private:
  EigenSparseMatrix<T> matrix_;
  Eigen::ConjugateGradient<EigenSparseMatrix<T>, Eigen::Lower | Eigen::Upper,
                           Eigen::IdentityPreconditioner>
      cg_;
};
}  // namespace fem
}  // namespace drake

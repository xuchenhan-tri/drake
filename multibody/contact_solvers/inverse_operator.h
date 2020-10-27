#pragma once

#include <string>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/linear_system_solver.h"
#include "drake/multibody/contact_solvers/linear_operator.h"
#include "drake/fem/eigen_conjugate_gradient_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
    namespace internal {

template <typename T>
class InverseOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(InverseOperator)

  /// Constructs an operator with given `name` implementing the LinearOperator
  /// interface for inverse of a linear operator `lop`.
  /// This class keeps a reference to `lop` and the solver that inverts `lop` and therefore they are
  /// required that it outlives this InverseOperator.
  InverseOperator(const std::string& name,
                  drake::fem::LinearSystemSolver<T>* solver, const LinearOperator<T>& lop)
      : LinearOperator<T>(name), solver_(solver), lop_(lop) {
    DRAKE_DEMAND(solver_ != nullptr);
    solver_->SetUp(lop);
  }

  ~InverseOperator() = default;

  int rows() const final { return lop_.rows(); }
  int cols() const final { return lop_.cols(); }

 protected:
  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    solver_->Solve(x, y);
  };

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    VectorX<T> dense_x(x);
    VectorX<T> dense_y(dense_x.size());
    solver_->Solve(dense_x, &dense_y);
    *y = dense_y.sparseView();
  }

 private:
  drake::fem::LinearSystemSolver<T>* solver_{nullptr};
  const LinearOperator<T>& lop_{nullptr};
};

    }  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::InverseOperator)

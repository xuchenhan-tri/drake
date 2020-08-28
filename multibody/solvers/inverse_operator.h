#pragma once

#include <string>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/linear_system_solver.h"
#include "drake/multibody/solvers/linear_operator.h"

namespace drake {
namespace multibody {
namespace solvers {

/// A LinearOperator that wraps an existing Eigen::SparseMatrix.
///
/// @tparam_nonsymbolic_scalar
template <typename T>
class InverseOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(InverseOperator)

  /// Constructs an operator with given `name` implementing the LinearOperator
  /// interface for inverse of a linear operator `Ainv`.
  /// This class keeps a reference to inverse `Ainv` and therefore it is
  /// required that it outlives this object.
  InverseOperator(const std::string& name,
                  drake::fem::LinearSystemSolver<T>* Ainv)
      : LinearOperator<T>(name), Ainv_(Ainv) {
    DRAKE_DEMAND(Ainv != nullptr);
  }

  ~InverseOperator() = default;

  int rows() const final { return Ainv_->rows(); }
  int cols() const final { return Ainv_->cols(); }

 protected:
  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    Ainv_->Solve(x, y);
  };

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    VectorX<T> dense_x(x);
    VectorX<T> dense_y(dense_x.size());
    Ainv_->Solve(dense_x, &dense_y);
    *y = dense_y.sparseView();
  }

 private:
  drake::fem::LinearSystemSolver<T>* Ainv_{nullptr};
};

}  // namespace solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::solvers::InverseOperator)

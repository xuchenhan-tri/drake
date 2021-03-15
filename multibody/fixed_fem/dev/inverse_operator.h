#pragma once

#include <string>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/linear_operator.h"
#include "drake/multibody/fixed_fem/dev/linear_system_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* A LinearOperator that wraps an LinearSystemSolver to provide the inverse
operator for a matrix/operator.
@tparam_nonsymbolic_scalar T. */
template <typename T>
class InverseOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(InverseOperator)

  /* Constructs an operator with given `name` implementing the LinearOperator
   interface for inverse of a linear operator `Ainv`.
   This class keeps a reference to inverse `Ainv` and therefore it is
   required that it outlives this object. */
  InverseOperator(const std::string& name,
                  const fixed_fem::internal::LinearSystemSolver<T>* Ainv)
      : LinearOperator<T>(name), Ainv_(Ainv) {
    /* Factorize the matrix at construction so that calls to Multiply won't need
     to factorize again. */
    Ainv->Compute();
    DRAKE_DEMAND(Ainv != nullptr);
  }

  ~InverseOperator() = default;

  int rows() const final { return Ainv_->size(); }
  int cols() const final { return Ainv_->size(); }

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
  const fixed_fem::internal::LinearSystemSolver<T>* Ainv_{nullptr};
};
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::InverseOperator)

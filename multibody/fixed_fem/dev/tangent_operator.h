#pragma once

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/linear_operator.h"
#include "drake/multibody/fixed_fem/dev/fem_model_base.h"
#include "drake/multibody/fixed_fem/dev/fem_state_base.h"

namespace drake {
namespace multibody {
namespace fem {

template <typename T>
class TangentOperator : public contact_solvers::internal::LinearOperator<T> {
 public:
  TangentOperator(const FemModelBase<T>* model, const FemStateBase<T>* state)
      : contact_solvers::internal::LinearOperator<T>("Tangent operator"),
        model_(model),
        state_(state) {}

  int rows() const final { return model_->num_dofs(); }

  int cols() const final { return model_->num_dofs(); }

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    const VectorX<T> x_dense = x;
    VectorX<T> y_dense(y->size());
    this->Multiply(x_dense, &y_dense);
    *y = y_dense.sparseView();
  }

  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    DRAKE_DEMAND(model_ != nullptr);
    DRAKE_DEMAND(state_ != nullptr);
    model_->CalcDifferential(*state_, x, y);
  }

  void set_state(const FemStateBase<T>* state) { state_ = state; }

 private:
  const FemModelBase<T>* model_;
  const FemStateBase<T>* state_;
};

}  // namespace fem
}  // namespace multibody
}  // namespace drake

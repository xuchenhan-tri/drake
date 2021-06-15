#pragma once

#include <string>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/linear_operator.h"

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
   interface for inverse of the given matirx `A`. */
  InverseOperator(const std::string& name, const MatrixX<T>& A)
      : LinearOperator<T>(name) {
    DRAKE_DEMAND(A.rows() == A.cols());
    size_ = A.rows();
    /* Factorize the matrix at construction so that calls to Multiply won't need
     to factorize again. */
    A_llt_ = A.llt();
  }

  ~InverseOperator() = default;

  int rows() const final { return size_; }
  int cols() const final { return size_; }

 private:
  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    VectorX<T> x_tmp(x);
    *y = A_llt_.solve(x_tmp).sparseView();
  }

  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    *y = A_llt_.solve(x);
  }

  int size_;
  Eigen::LLT<MatrixX<T>> A_llt_;
};
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::InverseOperator)

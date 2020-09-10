#pragma once

#include <string>

#include <Eigen/SparseCore>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/linear_operator.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/// A LinearOperator that wraps an existing Eigen::SparseMatrix.
///
/// @tparam_nonsymbolic_scalar
template <typename T>
class SparseLinearOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SparseLinearOperator)

  /// Constructs an operator with given `name` implementing the LinearOperator
  /// interface for matrix `A`.
  /// This class keeps a reference to input matrix `A` and therefore it is
  /// required that it outlives this object.
  SparseLinearOperator(const std::string& name, const Eigen::SparseMatrix<T>* A)
      : LinearOperator<T>(name),
        A_(A),
        multiply_(nullptr),
        matrix_free_(false),
        rows_(A->rows()),
        cols_(A->cols()) {
    DRAKE_DEMAND(A != nullptr);
  }

  /// Constructs an operator with given `name` implementing the LinearOperator
  /// interface for a matrix-free operator `multiply`.
  SparseLinearOperator(
      const std::string& name,
      const std::function<void(const Eigen::Ref<const VectorX<T>>&,
                               EigenPtr<VectorX<T>>)>& multiply,
      int rows, int cols)
      : LinearOperator<T>(name),
        A_(nullptr),
        multiply_(multiply),
        matrix_free_(true),
        rows_(rows),
        cols_(cols) {}

  ~SparseLinearOperator() = default;

  int rows() const final { return rows_; }
  int cols() const final { return cols_; }

 protected:
  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    if (!matrix_free_) {
      *y = *A_ * x;
    } else {
      multiply_(x, y);
    }
  };

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    if (!matrix_free_) {
      *y = *A_ * x;
    } else {
      VectorX<T> dense_x(x);
      VectorX<T> dense_y(y->size());
      multiply_(dense_x, &dense_y);
      *y = dense_y.sparseView();
    }
  }

  void DoMultiplyByTranspose(const VectorX<T>& x, VectorX<T>* y) const final {
    if (!matrix_free_) {
      *y = A_->transpose() * x;
    } else {
      throw std::runtime_error(
          "Matrix free sparse linear operator does not support "
          "MultiplyByTranspose.");
    }
  }

  void DoMultiplyByTranspose(const Eigen::SparseVector<T>& x,
                             Eigen::SparseVector<T>* y) const final {
    if (!matrix_free_) {
      *y = A_->transpose() * x;
    } else {
      throw std::runtime_error(
          "Matrix free sparse linear operator does not support "
          "MultiplyByTranspose.");
    }
  }

  void DoAssembleMatrix(Eigen::SparseMatrix<T>* A) const final {
    if (!matrix_free_) {
      *A = *A_;
    } else {
      throw std::runtime_error(
          "Matrix free sparse linear operator does not support "
          "AssembleMatrix.");
    }
  }

 private:
  const Eigen::SparseMatrix<T>* A_{nullptr};
  std::function<void(const Eigen::Ref<const VectorX<T>>&, EigenPtr<VectorX<T>>)>
      multiply_{nullptr};
  bool matrix_free_{false};
  int rows_{0};
  int cols_{0};
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::SparseLinearOperator)

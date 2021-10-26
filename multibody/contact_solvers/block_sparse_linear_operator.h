#pragma once

#include <string>

#include <Eigen/SparseCore>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/linear_operator.h"
#include "drake/multibody/contact_solvers/block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

// A LinearOperator that wraps an existing BlockSparseMatrix.
//
// @tparam_nonsymbolic_scalar
template <typename T>
class BlockSparseLinearOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BlockSparseLinearOperator)

  // Constructs an operator with given `name` implementing the LinearOperator
  // interface for matrix `A`.
  // This class keeps a reference to input matrix `A` and therefore it is
  // required that it outlives this object.
  BlockSparseLinearOperator(const std::string& name,
                            const BlockSparseMatrix<T>* A)
      : LinearOperator<T>(name), A_(A) {
    DRAKE_DEMAND(A != nullptr);
    row_vec_.resize(A->rows());
    col_vec_.resize(A->cols());
  }

  ~BlockSparseLinearOperator() = default;

  int rows() const final { return A_->rows(); }
  int cols() const final { return A_->cols(); }

 private:
  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    A_->Multiply(x, y);
  };

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    col_vec_ = x;
    A_->Multiply(col_vec_, &row_vec_);
    *y = row_vec_.sparseView();
  }

  void DoMultiplyByTranspose(const Eigen::Ref<const VectorX<T>>& x,
                             VectorX<T>* y) const final {
    A_->MultiplyByTranspose(x, y);
  }

  void DoMultiplyByTranspose(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                             Eigen::SparseVector<T>* y) const final {
    row_vec_ = x;
    A_->MultiplyByTranspose(row_vec_, &col_vec_);
    *y = col_vec_.sparseView();
  }

  // We want to inherit all public overrides of DoAssembleMatrix().
  // The way to accomplish this is to use the "using" keyword.
  // Then overriding a signature does not hide the others.
  //using LinearOperator<T>::DoAssembleMatrix;
  using LinearOperator<T>::DoAssembleMatrix;

#if 0
  void DoAssembleMatrix(Eigen::SparseMatrix<T>* A) const final {
    //*A = *A_;
  }  

  void DoAssembleMatrix(MatrixX<T>* A) const final {
    //*A = MatrixX<T>(*A_);
  }  
#endif  

  void DoAssembleMatrix(BlockSparseMatrix<T>* A) const final {
    *A = *A_;
  }  

  void DoAssembleMatrix(MatrixX<T>* A) const final {
    A->resize(A_->rows(), A_->cols());
    *A = A_->MakeDenseMatrix();
  }  

 private:
  const BlockSparseMatrix<T>* A_{nullptr};
  // Scratch space to "emulate" multiplication by Eigen::SparseVector.
  mutable VectorX<T> row_vec_;
  mutable VectorX<T> col_vec_;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::BlockSparseLinearOperator)

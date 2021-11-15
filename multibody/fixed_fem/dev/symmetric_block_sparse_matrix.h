#pragma once

#include <vector>

#include <Eigen/SparseCore>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* This class provides a representation for sparse matrices with a structure
 consisting of dense blocks of `block_size`x`block_size` submatrices. It
 is similar to contact_solvers::internal::BlockSparseMatrix in that it enables
 efficient algorithms capable of exploiting highly optimized operations with
 dense blocks. It differs from BlockSparseMatrix in a few aspects:

  1. It is tailored to sparse matrices with a particular structure
     (`block_size`x`block_size` blocks).
  2. It is tailored to symmetric matrices and only stores the upper triangular
     part of the matrix.
  3. It allows modification to the data (but not the sparsity pattern) after
     construction. Therefore, it is suitable for storing matrices with constant
     sparsity pattern and mutable data.

 In particular, these features make SymmetricBlockSparseMatrix suitable for
 storing the stiffness/damping/tangent matrix of an FEM model, where the matrix
 has constant sparsity pattern, has `block_size`x`block_size` block structure,
 and is symmetric.
 @tparam_nonsymbolic_scalar */
template <typename T>
class SymmetricBlockSparseMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SymmetricBlockSparseMatrix);
  virtual ~SymmetricBlockSparseMatrix() = default;

  /* Sets all blocks to zeros while maintaining the sparsity pattern. */
  void SetZero();

  /* For this matrix A, performs the operation y = A⋅x. x must be of size
   cols() and y must be a non-nullptr to a vector of size rows(). */
  void Multiply(const VectorX<T>& x, VectorX<T>* y) const;

  /* Makes a dense matrix representation of this block-sparse matrix. */
  MatrixX<T> MakeDenseMatrix() const;

  /* Makes a Eigen::SparseMatrix<T> representation of this block-sparse matrix.
   */
  Eigen::SparseMatrix<T> MakeEigenSparseMatrix() const;

  int rows() const { return cols(); }

  virtual int cols() const = 0;

 protected:
  SymmetricBlockSparseMatrix() = default;
  /* Derived classes must override these methods to provide an implementation
   for the NVI. */
  virtual void DoSetZero() = 0;
  virtual void DoMultiply(const VectorX<T>& x, VectorX<T>* y) const = 0;
  virtual MatrixX<T> DoMakeDenseMatrix() const = 0;
  virtual Eigen::SparseMatrix<T> DoMakeEigenSparseMatrix() const = 0;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::SymmetricBlockSparseMatrix);

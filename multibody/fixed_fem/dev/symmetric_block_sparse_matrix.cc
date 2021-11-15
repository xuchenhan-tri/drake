#include "drake/multibody/fixed_fem/dev/symmetric_block_sparse_matrix.h"

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
void SymmetricBlockSparseMatrix<T>::SetZero() {
  DoSetZero();
}

template <typename T>
void SymmetricBlockSparseMatrix<T>::Multiply(const VectorX<T>& x,
                                             VectorX<T>* y) const {
  DRAKE_DEMAND(x.size() == cols());
  DRAKE_DEMAND(y->size() == rows());
  DoMultiply(x, y);
}

template <typename T>
MatrixX<T> SymmetricBlockSparseMatrix<T>::MakeDenseMatrix() const {
  return DoMakeDenseMatrix();
}

template <typename T>
Eigen::SparseMatrix<T> SymmetricBlockSparseMatrix<T>::MakeEigenSparseMatrix()
    const {
  return DoMakeEigenSparseMatrix();
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::SymmetricBlockSparseMatrix);
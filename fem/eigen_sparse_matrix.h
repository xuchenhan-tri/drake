#pragma once

// #include <Eigen/Core>
// #include <Eigen/Sparse>

#include "drake/fem/backward_euler_objective.h"

//#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
template <typename T>
class EigenSparseMatrix;
}
}  // namespace drake

namespace Eigen {
namespace internal {
template <typename T>
struct traits<drake::fem::EigenSparseMatrix<T>> {
  typedef T Scalar;
  typedef typename SparseMatrix<T>::StorageIndex StorageIndex;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = ColMajor | NestByRefBit | LvalueBit | CompressedAccessBit,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};
}  // namespace internal
}  // namespace Eigen

namespace drake {
namespace fem {
/** A wrapper around eigen sparse matrix that supports matrix-free operations.
 */
template <typename T>
class EigenSparseMatrix : public Eigen::EigenBase<EigenSparseMatrix<T>> {
 public:
  // Required typedefs, constants, and method:
  typedef T Scalar;
  typedef T RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  StorageIndex rows() const { return objective_.get_num_dofs(); }

  StorageIndex cols() const { return objective_.get_num_dofs(); }

  template <typename Rhs>
  Eigen::Product<EigenSparseMatrix<T>, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<EigenSparseMatrix<T>, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

  // Custom API:
  EigenSparseMatrix(const BackwardEulerObjective<T>& objective)
      : objective_(objective) {}

  void Multiply(const Eigen::Ref<const VectorX<T>>& x, EigenPtr<VectorX<T>> b) const {
    return objective_.Multiply(x, b);
  }

  VectorX<T> Multiply(const Eigen::Ref<const VectorX<T>>& x) const {
      VectorX<T> b(x.size());
      Multiply(x, &b);
      return b;
  }

  void set_matrix_free(bool matrix_free) { objective_.set_matrix_free(matrix_free); }

  bool is_matrix_free() const { return objective_.is_matrix_free(); }

  // For non-matrix free operations only. Resize the matrix according to
  // information provided by the objective and allocate memory for the matrix.
  // Only needs to be called when the number of dofs change.
  void Reinitialize() {
    // TODO
  }

  // For non-matrix free operations only. Fill the matrix using Jacobian from
  // the objective. Needs to be called every time step.
  void BuildMatrix() {
    objective_.BuildJacobian();
  }

 private:
  const BackwardEulerObjective<T>& objective_;
};
}  // namespace fem
}  // namespace drake

// Implementation of EigenSparseMatrix * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
template <typename Rhs, typename T>
struct generic_product_impl<drake::fem::EigenSparseMatrix<T>, Rhs, SparseShape,
                            DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<
          drake::fem::EigenSparseMatrix<T>, Rhs,
          generic_product_impl<drake::fem::EigenSparseMatrix<T>, Rhs>> {
  typedef
      typename Product<drake::fem::EigenSparseMatrix<T>, Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst,
                            const drake::fem::EigenSparseMatrix<T>& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not
    // bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);
    dst.noalias() += lhs.Multiply(rhs);
  }
};

}  // namespace internal
}  // namespace Eigen
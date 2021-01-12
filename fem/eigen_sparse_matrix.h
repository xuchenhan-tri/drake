#pragma once

// #include <Eigen/Core>
// #include <Eigen/Sparse>

#include <iostream>

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
  typedef typename Eigen::SparseMatrix<T>::InnerIterator InnerIterator;
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
  EigenSparseMatrix(const BackwardEulerObjective<T>& objective,
                    bool matrix_free)
      : objective_(objective), matrix_free_(matrix_free) {
    if (matrix_free) {
      matrix_.resize(objective.get_num_dofs(), objective.get_num_dofs());
      matrix_.setZero();
    }
  }

  void Multiply(const Eigen::Ref<const VectorX<T>>& x,
                EigenPtr<Matrix3X<T>> b) const {
    DRAKE_DEMAND(matrix_free_);
    const Matrix3X<T>& tmp_x =
        Eigen::Map<const Matrix3X<T>>(x.data(), 3, x.size() / 3);
    return objective_.Multiply(tmp_x, b);
  }

  VectorX<T> Multiply(const Eigen::Ref<const VectorX<T>>& x) const {
    if (matrix_free_) {
      Matrix3X<T> b(3, x.size() / 3);
      Multiply(x, &b);
      return Eigen::Map<VectorX<T>>(b.data(), b.size());
    }
    VectorX<T> b = matrix_ * x;
    Eigen::Ref<Matrix3X<T>> tmp_b =
        Eigen::Map<Matrix3X<T>>(b.data(), 3, b.size() / 3);
    objective_.Project(&tmp_b);
    return Eigen::Map<VectorX<T>>(tmp_b.data(), tmp_b.size());
  }

  void set_matrix_free(bool matrix_free) { matrix_free_ = matrix_free; }

  bool is_matrix_free() const { return matrix_free_; }

  // For non-matrix free operations only. Resize the matrix according to
  // information provided by the objective and allocate memory for the matrix.
  // This is expensive but only needs to be called when the number of dofs
  // change, or when the sparsity pattern changes. Currently, the sparsity
  // pattern will remain unchanged throughout the simulation. In the future,
  // that might not be true (e.g. when we support fracture).
  void Reinitialize() {
    if (!matrix_free_) {
      int matrix_size = objective_.get_num_dofs();
      if (matrix_.cols() != matrix_size) {
        matrix_.resize(matrix_size, matrix_size);
        objective_.SetSparsityPattern(&matrix_);
      }
    }
  }

  const BackwardEulerObjective<T>& get_objective() const { return objective_; }

  const Eigen::SparseMatrix<T>& get_matrix() const { return matrix_; }

  // For non-matrix free operations only. Fill the matrix using Jacobian from
  // the objective. Needs to be called every time step.
  void BuildMatrix() {
    if (!matrix_free_) {
      objective_.BuildJacobian(&matrix_);
    }
  }

 private:
  const BackwardEulerObjective<T>& objective_;
  bool matrix_free_;
  Eigen::SparseMatrix<T> matrix_;
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
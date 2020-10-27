#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/linear_system_solver.h"
#include "drake/multibody/contact_solvers/linear_operator.h"

namespace drake {
namespace fem {
namespace internal {
template <typename T>
class EigenMatrixReplacement;
}
}  // namespace fem
}  // namespace drake

namespace Eigen {
namespace internal {
/* Minimum required trait for a custom sparse matrix to be used in a Eigen
  sparse iterative solver. */
template <typename T>
struct traits<drake::fem::internal::EigenMatrixReplacement<T>> {
  typedef T Scalar;
  typedef typename SparseMatrix<T>::StorageIndex StorageIndex;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = 0
  };
};
}  // namespace internal
}  // namespace Eigen

namespace drake {
namespace fem {
namespace internal {
/** A wrapper around Eigen:SparseMatrix<T> that supports matrix-free operations.
 */
template <typename T>
class EigenMatrixReplacement
    : public Eigen::EigenBase<EigenMatrixReplacement<T>> {
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

  StorageIndex rows() const { return linear_operator_->rows(); }

  StorageIndex cols() const { return linear_operator_->cols(); }

  template <typename Rhs>
  Eigen::Product<EigenMatrixReplacement<T>, Rhs, Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<EigenMatrixReplacement<T>, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

  // Custom APIs:
  EigenMatrixReplacement() = default;
  ~EigenMatrixReplacement() = default;

  // TODO(xuchenhan-tri): linear_operator_ may dangle here if the input lop goes out of scope.
  void SetUp(const multibody::contact_solvers::internal::LinearOperator<T>& lop) {
    linear_operator_ = &lop;
  }

  VectorX<T> Multiply(const Eigen::Ref<const VectorX<T>>& x) const {
    VectorX<T> y(rows());
    linear_operator_->Multiply(x, &y);
    return y;
  }

 private:
  const multibody::contact_solvers::internal::LinearOperator<T>* linear_operator_{nullptr};
};
}  // namespace internal
}  // namespace fem
}  // namespace drake

// Implementation of EigenSparseMatrix * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
template <typename Rhs, typename T>
struct generic_product_impl<drake::fem::internal::EigenMatrixReplacement<T>,
                            Rhs, SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<
          drake::fem::internal::EigenMatrixReplacement<T>, Rhs,
          generic_product_impl<drake::fem::internal::EigenMatrixReplacement<T>,
                               Rhs>> {
  typedef typename Product<drake::fem::internal::EigenMatrixReplacement<T>,
                           Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(
      Dest& dst, const drake::fem::internal::EigenMatrixReplacement<T>& lhs,
      const Rhs& rhs, const Scalar& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    dst.noalias() += alpha * lhs.Multiply(rhs);
  }
};

}  // namespace internal
}  // namespace Eigen

namespace Eigen {
// TODO(xuchenhan-tri): Properly implement both a mass preconditioner and a
// Jacobi preconditioner for *both* matrixed solver and the matrix-free solver.
// Currently, mass preconditioning is used for matrix-free solver and Jacobi
// preconditioning is used for matrixed solver.
template <typename T>
class MassPreconditioner {
  typedef T Scalar;
  typedef drake::VectorX<T> Vector;

 public:
  typedef typename Vector::StorageIndex StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic
  };

  MassPreconditioner() : initialized_(false) {}

  template <typename MatType>
  explicit MassPreconditioner(const MatType& mat) : inv_mass_(mat.cols()) {
    compute(mat);
  }

  Index rows() const { return inv_mass_.size(); }
  Index cols() const { return inv_mass_.size(); }

  template <typename MatType>
  MassPreconditioner& analyzePattern(const MatType&) {
    return *this;
  }

  template <typename MatType>
  MassPreconditioner& factorize(const MatType& mat) {
    return compute(mat);
  }

  template <typename MatType>
  MassPreconditioner& compute(const MatType& mat) {
    // TODO (xuchenhan-tri): implement me.
    inv_mass_ = drake::VectorX<T>::Ones(mat.cols());
    initialized_ = true;
    return *this;
  }

  template <typename Rhs, typename Dest>
  void _solve_impl(const Rhs& b, Dest& x) const {
    x = inv_mass_.array() * b.array();
  }

  template <typename Rhs>
  inline const Solve<MassPreconditioner, Rhs> solve(
      const MatrixBase<Rhs>& b) const {
    eigen_assert(initialized_ && "MassPreconditioner is not initialized.");
    eigen_assert(inv_mass_.size() == b.rows() &&
                 "MassPreconditioner::solve(): invalid number of rows of the "
                 "right hand side matrix b");
    return Solve<MassPreconditioner, Rhs>(*this, b.derived());
  }

  Eigen::ComputationInfo info() { return Eigen::Success; }

 protected:
  drake::VectorX<T> inv_mass_;
  bool initialized_;
};
}  // namespace Eigen

namespace drake {
namespace fem {
// TODO(xuchenhan-tri): This can be made more general to handle all Eigen
// iterative solvers.
template <typename T>
class EigenConjugateGradientSolver : public LinearSystemSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EigenConjugateGradientSolver)

  EigenConjugateGradientSolver() { cg_.setTolerance(1e-3); }

  virtual ~EigenConjugateGradientSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(const Eigen::Ref<const VectorX<T>>& rhs,
                     EigenPtr<VectorX<T>> x) {
    *x = cg_.solve(rhs);
  }

  /** Set up the equation A*x = rhs. */
  virtual void SetUp(const multibody::contact_solvers::internal::LinearOperator<T>& lop) {
    matrix_.SetUp(lop);
    cg_.compute(matrix_);
  }

  void get_max_iterations() const { cg_.maxIterations(); }

  void set_max_iterations(int max_iterations) {
    cg_.setMaxIterations(max_iterations);
  }

  int rows() const { return matrix_.rows(); }
  int cols() const { return matrix_.cols(); }

  T get_accuracy() const { return cg_.tolerance(); }

  void set_accuracy(T tol) { cg_.setTolerance(tol); }

 private:
  internal::EigenMatrixReplacement<T> matrix_;
  Eigen::ConjugateGradient<internal::EigenMatrixReplacement<T>,
                           Eigen::Lower | Eigen::Upper,
                           Eigen::IdentityPreconditioner>
    cg_;
};
}  // namespace fem
}  // namespace drake

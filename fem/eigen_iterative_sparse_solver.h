#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/linear_system_solver.h"
namespace Eigen {
namespace internal {

    namespace drake{
        namespace fem
        {
            template <typename T>
            class MatrixReplacement;
        }
    }

// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <>
struct traits<MatrixReplacement>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

}  // namespace internal
}  // namespace Eigen

namespace drake {
namespace fem {
template <typename T>
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement<T>> {
 public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  StorageIndex rows() const { return objective_->num_dofs(); }
  StorageIndex cols() const { return objective_->num_dofs(); }

  template <typename Rhs>
  Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

  // Custom API:
  MatrixReplacement(const BackwardEulerObjective<double>& objective)
      : objective_(objective) {}
  VectorX<double> Multiply(const Eigen::Ref<VectorX<double>>& x) {
    return objective_.Multiply(x);
  }

 private:
  const BackwardEulerObjective<T>& objective_;
};

template <typename T>
class EigenIterativeSparseSolver : public LinearSystemSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EigenIterativeSparseSolver)

  EigenIterativeSparseSolver(const BackwardEulerObjective<T>& objective)
      : LinearSystemSolver<T>(objective) {}

  virtual ~EigenIterativeSparseSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(VectorX<T>& rhs, VectorX<T>* x) {
    if (matrix_free_)
      &x = matrix_free_cg_.solve(rhs);
    else
      &x = cg_.solve(rhs);
  }

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() {
    if (matrix_free_) {
      matrix_free_cg_.compute(matrix_replacement_);
    } else {
      this->objective_.BuildJacobian(matrix_);
      cg_.compute(matrix_);
    }
  }

  bool is_matrix_free() const { return matrix_free_; }

 private:
  bool matrix_free_{true};
  MatrixReplacement<T> matrix_replacement_;
  Eigen::SparseMatrix<T> matrix_;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<T>>
      cg_;
  Eigen::ConjugateGradient<MatrixReplacement<T>, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<T>>
      matrix_free_cg_;
};
}  // namespace fem
}  // namespace drake

namespace Eigen {
namespace internal {
// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
template <typename Rhs>
struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<MatrixReplacement, Rhs,
                                generic_product_impl<MatrixReplacement, Rhs>> {
  typedef typename Product<MatrixReplacement, Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not
    // bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    dst += lhs.Multply(rhs);
  }
};
}  // namespace internal
}  // namespace Eigen

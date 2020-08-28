#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/eigen_sparse_matrix.h"
#include "drake/fem/linear_system_solver.h"

namespace Eigen {
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
    inv_mass_.resize(mat.cols());
    // if (mat.is_matrix_free()) {
    if (1) {
      const drake::VectorX<T>& mass = mat.get_objective().get_mass();
      for (int i = 0; i < static_cast<int>(mass.size()); ++i) {
        T one_over_mass = (mass(i) == static_cast<T>(0))
                              ? 1.0
                              : (static_cast<T>(1) / mass(i));
        for (int d = 0; d < 3; ++d) {
          inv_mass_(3 * i + d) = one_over_mass;
        }
      }
    } else {
      const auto& matrix = mat.get_matrix();
      for (int j = 0; j < matrix.outerSize(); ++j) {
        typename MatType::InnerIterator it(matrix, j);
        while (it && it.index() != j) ++it;
        if (it && it.index() == j && it.value() != Scalar(0))
          inv_mass_(j) = Scalar(1) / it.value();
        else
          inv_mass_(j) = Scalar(1);
      }
    }
    initialized_ = true;
    return *this;
  }

  /** \internal */
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

  explicit EigenConjugateGradientSolver(
      const BackwardEulerObjective<T>& objective)
      : matrix_(objective, false) {
    cg_.setTolerance(1e-3);
  }

  virtual ~EigenConjugateGradientSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(const Eigen::Ref<const VectorX<T>>& rhs,
                     EigenPtr<VectorX<T>> x) {
    *x = cg_.solve(rhs);
//    std::cout << "CG iterations = " << cg_.iterations() << std::endl;
  }

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() {
    matrix_.Reinitialize();
    matrix_.BuildMatrix();
    cg_.compute(matrix_);
  }

  bool is_matrix_free() const { return matrix_.is_matrix_free(); }

  void set_matrix_free(bool matrix_free) {
    matrix_.set_matrix_free(matrix_free);
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
  EigenSparseMatrix<T> matrix_;
  Eigen::ConjugateGradient<EigenSparseMatrix<T>, Eigen::Lower | Eigen::Upper,
                           Eigen::MassPreconditioner<T>>
      cg_;
};
}  // namespace fem
}  // namespace drake

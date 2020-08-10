#pragma once

#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/linear_system_solver.h"

namespace drake {
namespace fem {
template <typename T>
class ConjugateGradientSolver : public LinearSystemSolver<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConjugateGradientSolver)

  ConjugateGradientSolver(const BackwardEulerObjective<T>& objective): objective_(objective){}

  virtual ~ConjugateGradientSolver() {}

  /** Perform linear solve to get x = A⁻¹*rhs. */
  virtual void Solve(VectorX<T>& rhs, VectorX<T>* x) {
    DRAKE_DEMAND(x->size() == rhs.size());
    r.resizeLike(rhs);
    p.resizeLike(rhs);
    z.resizeLike(rhs);
    temp.resizeLike(rhs);
    p = rhs;
    Precondition(p, &z);
    T d0 = z.dot(p);
    // r = b - A * x --with assigned dof zeroed out
    Multiply(*x, &temp);
    r = rhs - temp;
    Project(&r);
    // z = M^(-1) * r
    Precondition(r, &z);
    Project(&z);
    T r_dot_z = r.dot(z);
    if (r_dot_z <= accuracy_ * d0) {
      return;
    }
    p = z;
    T r_dot_z_new = r_dot_z;
    for (int k = 1; k <= max_iterations_; k++) {
      // temp = A*p
      Multiply(p, &temp);
      Project(&temp);
      DRAKE_THROW_UNLESS(p.dot(temp) >= 0.0);
      // alpha = r^T * z / (p^T * A * p)
      T alpha = r_dot_z_new / p.dot(temp);
      *x += alpha * p;
      r -= alpha * temp;
      // z = M^(-1) * r
      Precondition(r, &z);
      r_dot_z = r_dot_z_new;
      r_dot_z_new = r.dot(z);
      if (r_dot_z_new < accuracy_ * d0) {
        return;
      }
      T beta = r_dot_z_new / r_dot_z;
      p = beta * p + z;
    }
    // Did not converge.
    return;
  }

  /** Set up the equation A*x = rhs. */
  virtual void SetUp() {
    objective_.BuildJacobian();
  }

  void Multiply(const VectorX<T>& x, VectorX<T>* b) const {
        Eigen::Map<Matrix3X<T>> reshaped_b(b->data(),3, b->size()/3);
        objective_.Multiply(Eigen::Map<const Matrix3X<T>>(x.data(), 3, x.size()/3), &reshaped_b);
  }

  bool is_matrix_free() const { return objective_.is_matrix_free(); }

  void set_matrix_free(bool matrix_free) { objective_.set_matrix_free(matrix_free); }

  int get_max_iterations() const { return max_iterations_; }

  void set_max_iterations(int max_iterations) {
    max_iterations_ = max_iterations;
  }

  T get_accuracy() const { return accuracy_; }

  void set_accuracy(T accuracy) { accuracy_ = accuracy; }

 private:
  const BackwardEulerObjective<T>& objective_;
  int max_iterations_{};
  T accuracy_{};
  VectorX<T> r, p, z, temp;

  // Calculates M^{-1} x = b where M is the preconditioning matrix.
  void Precondition(const Eigen::Ref<const VectorX<T>>& x,
                    EigenPtr<VectorX<T>> b) const {
    objective_.preconditioner_(x, b);
  }

  void Project(EigenPtr<VectorX<T>> b) const {
    objective_.projection_(b);
  }
};
}  // namespace fem
}  // namespace drake

#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/eigen_conjugate_gradient_solver.h"

namespace drake {
namespace fem {

/** A solver that solves G(x) = 0 using Newton's Method.
One iteration given by Newton's method is:

    x_n+1 = x_n + dx.
    dx = -dG/dx(x_n)^(-1) * G(x_n) and

The second equation can be reformulated as A * dx = b, where A = dG/dx(x_n) and
b = -G(x_n). This class delegates solving A * dx = b to an objective class and
handles the equation x_n+1 = x_n.
*/
template <typename T>
class NewtonSolver {
 public:
  enum NewtonSolverStatus { Success, NoConvergence };

  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(NewtonSolver);

  explicit NewtonSolver(BackwardEulerObjective<T>* objective)
      : objective_(*objective) {}

  /** Takes in an initial guess for the solution and overwrites it with the
      actual solution. */
  NewtonSolverStatus Solve(EigenPtr<VectorX<T>> x);

  /** Updates the `x`-dependent states and evaluates the residual -G(x). */
  bool UpdateAndEvalResidual(const Eigen::Ref<const VectorX<T>>& x);

  /** The norm calculation is delegated to the objective to support customized
   * norms. */
  T norm(const Eigen::Ref<const VectorX<T>>& x) { return objective_.norm(x); }

  int max_iteration() const { return max_iterations_; }

  void set_max_iterations(int max_iterations) {
    max_iterations_ = max_iterations;
  }

  T linear_solver_accuracy() const { return linear_solver_.get_accuracy(); }

  void set_linear_solver_accuracy(T tolerance) {
    linear_solver_.set_accuracy(tolerance);
  }

  // TODO(xuchenhan-tri): a nice doc about what this tolerance measures.
  T tolerance() const { return tolerance_; }

  void set_tolerance(T tolerance) { tolerance_ = tolerance; }

  EigenConjugateGradientSolver<T>& get_linear_solver() { return linear_solver_; }

 private:
  BackwardEulerObjective<T>& objective_;
  EigenConjugateGradientSolver<T> linear_solver_;
  int max_iterations_{
      20};  // If Newton's method does not converge in 20 iterations, you should
            // consider either using another method or come up with a better
            // initial guess.
  T tolerance_{1e-3};
  VectorX<T> dx_;
  VectorX<T> residual_;
};

}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::NewtonSolver)

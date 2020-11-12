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

  explicit NewtonSolver(const BackwardEulerObjective<T>& objective, const FemData<T>& data, FemState<T>* state)
      : objective_(objective), data_(data), state_(*state) {}

  /** Takes in an initial guess for the solution and overwrites it with the
      actual solution. */
  NewtonSolverStatus Solve(EigenPtr<VectorX<T>> z) const;

  /** Update the velocity of the vertices to v = v0 + dt * dv and position of the vertices to q = q0 + dt * v */
  void UpdateState() const{
      const auto& v0 = state_.get_v0();
      const auto& dv = state_.get_dv();
      auto& v = state_.get_mutable_v();
      v = v0 + dv;
      const auto& q0 = state_.get_q0();
      const auto& dt = data_.get_dt();
      auto& q = state_.get_mutable_q();
      q = q0 + dt * v;
  }

  /** Evaluates the residual -G(z). */
  bool EvalResidual() const;

  /** The norm calculation is delegated to the objective to support customized
   * norms. */
  T norm(const Eigen::Ref<const VectorX<T>>& z) const { return objective_.norm(z); }

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

  EigenConjugateGradientSolver<T>& get_linear_solver() const { return linear_solver_; }

 private:
  const BackwardEulerObjective<T>& objective_;
  mutable EigenConjugateGradientSolver<T> linear_solver_;
  int max_iterations_{
      20};  // If Newton's method does not converge in 20 iterations, you should
            // consider either using another method or come up with a better
            // initial guess.
  T tolerance_{1e-3};
  // Scratch space for Newton solver.
  mutable VectorX<T> dz_;
  mutable VectorX<T> residual_;
  const FemData<T>& data_;
  FemState<T>& state_;
};

}  // namespace fem
}  // namespace drake
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::drake::fem::NewtonSolver)

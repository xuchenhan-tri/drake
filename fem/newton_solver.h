#pragma once

#include "drake/fem/backward_euler_objective.h"
#include "drake/fem/eigen_conjugate_gradient_solver.h"

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

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

  NewtonSolver(BackwardEulerObjective<T>& objective) : objective_(objective), linear_solver_(objective) {}
  /** Takes in an initial guess for the solution and overwrites it with the
      actual solution. */
  NewtonSolverStatus Solve(EigenPtr<VectorX<T>> x) {
    dx_.resizeLike(*x);
    residual_.resizeLike(*x);
    // if (UpdateAndEvalResidual(*x)) return NewtonSolverStatus::Success;
    UpdateAndEvalResidual(*x);
    for (int i = 0; i < max_iterations_; ++i) {
        std::cout << "Newton iteration: " << i << std::endl;
      linear_solver_.SetUp();
      linear_solver_.Solve(residual_, &dx_);
      *x += dx_;
      if (UpdateAndEvalResidual(*x)) return NewtonSolverStatus::Success;
    }
    return NewtonSolverStatus::NoConvergence;
  }

  bool UpdateAndEvalResidual(const VectorX<T>& x) {
    objective_.Update(x);
    objective_.CalcResidual(&residual_);
    // Make sure there is no NAN in the residual.
    DRAKE_DEMAND(residual_ == residual_);
    return norm(residual_) < tolerance_;
  //  Eigen::JacobiSVD<Matrix3<T>> svd(R_, Eigen::ComputeFullU | Eigen::ComputeFullV);
//  R_ = svd.matrixU() * svd.matrixV().transpose();
}

  // TODO(xuchenhan-tri): provide more customized norm than l2.
  T norm(const VectorX<T>& x)
  {
      return x.template lpNorm<Eigen::Infinity>();
    // return x.norm();
  }

  int max_iteration() const { return max_iterations_; }

  void set_max_iterations(int max_iterations) {
    max_iterations_ = max_iterations;
  }

  // TODO(xuchenhan-tri): a nice doc about what this tolerance measures.
  T tolerance() const { return tolerance_; }

  void set_tolerance(T tolerance) { tolerance_ = tolerance; }

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

#include "drake/fem/newton_solver.h"
namespace drake {
namespace fem {

template <typename T>
typename NewtonSolver<T>::NewtonSolverStatus NewtonSolver<T>::Solve(
    EigenPtr<VectorX<T>> x) {
  dx_.resizeLike(*x);
  residual_.resizeLike(*x);
  UpdateAndEvalResidual(*x);
  for (int i = 0; i < max_iterations_; ++i) {
    std::unique_ptr<multibody::solvers::LinearOperator<T>> J = objective_.GetJacobian();
    linear_solver_.SetUp(*J);
    linear_solver_.Solve(residual_, &dx_);
    *x += dx_;
    if (UpdateAndEvalResidual(*x)) {
      std::cout << "Newton converged in " << i + 1 << " iteration(s)."
                << std::endl;
      return NewtonSolverStatus::Success;
    }
  }
  std::cout << "Newton did not converge in " << max_iterations_
            << " iteration(s)." << std::endl;
  return NewtonSolverStatus::NoConvergence;
}

template <typename T>
bool NewtonSolver<T>::UpdateAndEvalResidual(
    const Eigen::Ref<const VectorX<T>>& x) {
  objective_.Update(x);
  objective_.CalcResidual(&residual_);
  // Make sure there is no NAN in the residual.
  DRAKE_DEMAND(residual_ == residual_);
  return norm(residual_) < tolerance_;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::NewtonSolver)

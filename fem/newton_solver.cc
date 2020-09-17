#include "drake/fem/newton_solver.h"
namespace drake {
namespace fem {
//    template <typename T>
//    typename NewtonSolver<T>::NewtonSolverStatus NewtonSolver<T>::Solve(
//            EigenPtr<VectorX<T>> z) const {
//        dz_.resizeLike(*z);
//        residual_.resizeLike(*z);
//        UpdateAndEvalResidual(*z);
//        for (int i = 0; i < max_iterations_; ++i) {
//            std::unique_ptr<multibody::solvers::LinearOperator<T>> J = objective_.GetA();
//            linear_solver_.SetUp(*J);
//            linear_solver_.Solve(residual_, &dz_);
//            *z += dz_;
//            if (UpdateAndEvalResidual(*z)) {
//                std::cout << "Newton converged in " << i + 1 << " iteration(s)."
//                          << std::endl;
//                return NewtonSolverStatus::Success;
//            }
//        }
//        std::cout << "Newton did not converge in " << max_iterations_
//                  << " iteration(s)." << std::endl;
//        return NewtonSolverStatus::NoConvergence;
//    }
//
//    template <typename T>
//    bool NewtonSolver<T>::UpdateAndEvalResidual(
//            const Eigen::Ref<const VectorX<T>>& z) const {
//        objective_.UpdateState(z);
//        objective_.CalcResidual(z, &residual_);
//        // Make sure there is no NAN in the residual.
//        DRAKE_DEMAND(residual_ == residual_);
//        return norm(residual_) < tolerance_;
//    }

template <typename T>
typename NewtonSolver<T>::NewtonSolverStatus NewtonSolver<T>::Solve(EigenPtr<VectorX<T>> z) const {
  residual_.resizeLike(*z);
  dz_.resizeLike(*z);
  UpdateState();
  EvalResidual();
  for (int i = 0; i < max_iterations_; ++i) {
    std::unique_ptr<multibody::solvers::LinearOperator<T>> J = objective_.GetA(state_);
    linear_solver_.SetUp(*J);
    linear_solver_.Solve(residual_, &dz_);
    *z += dz_;
    UpdateState();
    if (EvalResidual()) {
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
bool NewtonSolver<T>::EvalResidual() const {
  objective_.CalcResidual(state_, &residual_);
  // Make sure there is no NAN in the residual.
  DRAKE_DEMAND(residual_ == residual_);
  return norm(residual_) < tolerance_;
}
}  // namespace fem
}  // namespace drake
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::fem::NewtonSolver)

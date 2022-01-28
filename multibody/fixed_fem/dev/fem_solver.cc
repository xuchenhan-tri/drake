#include "drake/multibody/fixed_fem/dev/fem_solver.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
FemSolver<T>::FemSolver(const FemModelBase<T>* model,
                        const DiscreteTimeIntegrator<T>* integrator)
    : model_(model), integrator_(integrator) {
  DRAKE_DEMAND(model_ != nullptr);
  DRAKE_DEMAND(integrator_ != nullptr);
  ResetScratchDataIfNecessary();
}

template <typename T>
void FemSolver<T>::AdvanceOneTimeStep(const FemStateBase<T>& prev_state,
                                      FemStateBase<T>* next_state) const {
  DRAKE_DEMAND(next_state != nullptr);
  model_->ThrowIfModelStateIncompatible(__func__, prev_state);
  model_->ThrowIfModelStateIncompatible(__func__, *next_state);
  /* Make initial guess of the unknown variable that it stays the same. */
  const VectorX<T>& unknown_variable = integrator_->GetUnknowns(prev_state);
  integrator_->AdvanceOneTimeStep(prev_state, unknown_variable, next_state);
  /* Run Newton-Raphson iterations. */
  SolveWithInitialGuess(next_state);
}

template <typename T>
void FemSolver<T>::set_linear_solve_tolerance(const T& tolerance) {
  linear_solve_tolerance_ = tolerance;
  if constexpr (std::is_same_v<T, double>) {
    if (tangent_matrix_petsc_ != nullptr) {
      tangent_matrix_petsc_->set_relative_tolerance(tolerance);
    }
  } else {
    eigen_tangent_matrix_solver_.setTolerance(tolerance);
  }
}

template <typename T>
int FemSolver<T>::SolveWithInitialGuess(FemStateBase<T>* state) const {
  /* Make sure the scratch quantities are of the correct sizes. */
  ResetScratchDataIfNecessary();
  model_->ApplyBoundaryCondition(state);
  model_->CalcResidual(*state, &b_);
  int iter = 0;
  /* Newton-Raphson iterations. We iterate until any of the following is true:
   1. The max number of allowed iterations is reached;
   2. The norm of the change in the state in a single iteration is smaller
      than the absolute tolerance.
   3. The relative error (the norm of the change in the state divided by the
      norm of the state) is smaller than the unitless relative tolerance. */
  do {
    /* Use PETSc matrix when scalar type is double. Otherwise, use Eigen
     matrix. */
    if constexpr (std::is_same_v<T, double>) {
      model_->CalcTangentMatrix(*state, integrator_->weights(),
                                tangent_matrix_petsc_.get());
      tangent_matrix_petsc_->AssembleIfNecessary();
      /* Solve for A * dz = -b, where A is the tangent matrix. */
      dz_ = tangent_matrix_petsc_->Solve(
          internal::PetscSymmetricBlockSparseMatrix::SolverType::
              kConjugateGradient,
          internal::PetscSymmetricBlockSparseMatrix::PreconditionerType::
              kIncompleteCholesky,
          -b_);
    } else {
      model_->CalcTangentMatrix(*state, integrator_->weights(),
                                &tangent_matrix_eigen_);
      /* Solve for A * dz = -b, where A is the tangent matrix. */
      eigen_tangent_matrix_solver_.compute(tangent_matrix_eigen_);
      dz_ = eigen_tangent_matrix_solver_.solve(-b_);
    }
    integrator_->UpdateStateFromChangeInUnknowns(dz_, state);
    model_->CalcResidual(*state, &b_);
    ++iter;
  } while (dz_.norm() > std::max(relative_tolerance_ *
                                     integrator_->GetUnknowns(*state).norm(),
                                 absolute_tolerance_) &&
           iter < kMaxIterations_);
  if (iter == kMaxIterations_) {
    throw std::runtime_error(fmt::format(
        "The solver did not converge in {} iterations. Please provide a "
        "better initial guess. Consider taking a smaller time step, "
        "especially if the constitutive model you are using is nonlinear "
        "(e.g. CorotatedModel).",
        kMaxIterations_));
  }
  return iter;
}

template <typename T>
void FemSolver<T>::ResetScratchDataIfNecessary() const {
  if (b_.size() != model_->num_dofs()) {
    b_.resize(model_->num_dofs());
    dz_.resize(model_->num_dofs());
    if constexpr (std::is_same_v<T, double>) {
      tangent_matrix_petsc_ =
          model_->MakePetscSymmetricBlockSparseTangentMatrix();
      tangent_matrix_petsc_->set_relative_tolerance(linear_solve_tolerance_);
    } else {
      tangent_matrix_eigen_ = model_->MakeEigenSparseTangentMatrix();
    }
  }
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::fem::internal::FemSolver);
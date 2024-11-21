#include "mpm_solver.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using LinearSolver =
    contact_solvers::internal::BlockSparseCholeskySolver<Matrix3<double>>;

template <typename T>
MpmSolver<T>::MpmSolver(MpmState<T>* state, SolverState<T>* solver_state)
    : mpm_state_(mpm_state), solver_state_(solver_state) {
  DRAKE_DEMAND(mpm_state != nullptr);
  DRAKE_DEMAND(solver_state != nullptr);
  DRAKE_DEMAND(solver_state->dv.size() == mpm_state->num_dofs());
  num_dofs_ = mpm_state->num_dofs();
  SolveFreeMotion();
}

template <typename T>
void AdvanceMpmState(const VectorX<double>& participating_v_next) {
  DRAKE_DEMAND(participating_v_next.size() == participating_v_star_.size());
  const VectorX<double> participating_dv =
      participating_v_next - participating_v_star_;
  const VectorX<double> nonparticipating_dv =
      schur_complement_.SolveForX(participating_dv);
  VectorX<double> permutated_dv(num_dofs());
  permuted_dv << participating_dv, nonparticipating_dv;
  VectorX<double> dv(num_dofs());
  mpm_state_->grid_dof_permutation().ApplyInverse(permuted_dv, &dv);
  mpm_state_->AdvanceToNextState(dv.cast<T>());
}

template <typename T>
void MpmSolver<T>::SolveFreeMotion() {
  VectorX<T> b = VectorX<T>::Zero(state_.num_dofs());
  mpm_state_->CalcResidual(*solver_state_, &b);
  T residual_norm = b.norm();
  if (residual_norm < abs_tolerance_) {
    return 0;
  }
  VectorX<T> ddv = VectorX<T>::Zero(num_dofs_);
  const T initial_residual_norm = residual_norm;
  Block3x3SparseSymmetricMatrix tangent_matrix =
      mpm_state_->MakeTangentMatrix();
  LinearSolver linear_solver;
  int iter = 0;

  while (iter < max_iterations &&
         /* On first iteration, this is equivalent to residual_norm <
            abs_tolerance_, which we have ruled out earlier. */
         !solver_converged(residual_norm, initial_residual_norm)) {
    mpm_state_.CalcTangentMatrix(*solver_state_, &tangent_matrix);
    if (iter == 0) {
      linear_solver.SetMatrix(tangent_matrix);
    } else {
      linear_solver.UpdateMatrix(tangent_matrix);
    }
    const bool factored = linear_solver.Factor();
    if (!factored) {
      throw std::runtime_error(
          "Tangent matrix factorization failed in MpmSolver because the MPM "
          "tangent matrix is not symmetric positive definite (SPD). This may "
          "be triggered by a combination of a stiff nonlinear constitutive "
          "model and a large time step.");
    }
    /* Solve for the change in unknowns. */
    ddv = linear_solver.Solve(-b);
    mpm_state_->UpdateSolverState(ddv, solver_state_);
    mpm_state_->CalcResidual(*solver_state_, &b);
    residual_norm = b.norm();
    ++iter;
  }
  if (!solver_converged(residual_norm, initial_residual_norm)) {
    /* Solver failed to converge with max number of Newton iterations. */
    throw std::runtime_error(
        "MpmSolver failed to converge with max number of Newton iterations.");
  }
  mpm_state_->CalcTangentMatrix(*solver_state_, &tangent_matrix);
  schur_complement_ = SchurComplement(
      tangent_matrix, mpm_state->GetNonParticipatingGridNodes());
  participating_v_star_ = mpm_state_->GetParticipatingVelocities();
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

#include "mpm_solver.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using LinearSolver =
    contact_solvers::internal::BlockSparseCholeskySolver<Matrix3<double>>;

template <typename T>
MpmSolver<T>::MpmSolver(MpmState<T>* state) : state_(state) {
  DRAKE_DEMAND(state != nullptr);
  SolveFreeMotion();
}

template <typename T>
void CalcNextState(const VectorX<double>& participating_v_next) {
  DRAKE_DEMAND(participating_v_next.size() == participating_v_star_.size());
  participating_dv = participating_v_next - participating_v_star_;
  VectorX<double> nonparticipating_dv =
      schur_complement_.SolveForX(participating_dv);
  VectorX<double> permutated_dv(state_.num_dofs());
  permuted_dv << participating_dv, nonparticipating_dv;
  VectorX<double> dv(state_.num_dofs());
  state_->grid_dof_permutation().ApplyInverse(permuted_dv, &dv);
  state_.CalcNextState(dv);
}

template <typename T>
void MpmSolver<T>::SolveFreeMotion() {
  VectorX<T> b = VectorX<T>::Zero(state_.num_dofs());
  state_.CalcResidual(&b);
  T residual_norm = b.norm();
  if (residual_norm < abs_tolerance_) {
    return 0;
  }
  VectorX<T> ddv = VectorX<T>::Zero(state_.num_dofs());
  const T initial_residual_norm = residual_norm;
  Block3x3SparseSymmetricMatrix tangent_matrix = state_.MakeTangentMatrix();
  LinearSolver linear_solver;
  int iter = 0;

  while (iter < max_iterations &&
         /* On first iteration, this is equivalent to residual_norm <
            abs_tolerance_, which we have ruled out earlier. */
         !solver_converged(residual_norm, initial_residual_norm)) {
    state_.CalcTangentMatrix(&tangent_matrix);
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
    state_.IncrementDv(ddv);
    state_.CalcResidual(&b);
    residual_norm = b.norm();
    ++iter;
  }
  if (!solver_converged(residual_norm, initial_residual_norm)) {
    /* Solver failed to converge with max number of Newton iterations. */
    throw std::runtime_error(
        "MpmSolver failed to converge with max number of Newton iterations.");
  }
  state_.CalcTangentMatrix(&tangent_matrix);
  schur_complement_ =
      SchurComplement(tangent_matrix, GetNonParticipatingGridNodes());
  participating_v_star_ = state_.GetParticipatingVelocities();
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

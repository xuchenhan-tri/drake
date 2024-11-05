#include "mpm_solver.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
MpmSolver<T>::MpmSolver(T dt, SparseGrid<T>* sparse_grid,
                        ParticleData<T>* particles, Parallelism parallelism)
    : dt_(dt), state_(sparse_grid, particles), parallelism_(parallelism) {}

template <typename T>
int MpmSolver<T>::SolveFreeMotion() {
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
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

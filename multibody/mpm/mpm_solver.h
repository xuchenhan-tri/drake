#pragma once

#include "mpm_state.h"
#include "solver_state.h"

#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/contact_solvers/schur_complement.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

struct MpmSolverParameters {
  int max_iterations{100};
  double absolute_tolerance{1e-6};
  double relative_tolerance{1e-4};
};

/* Newton-Raphson solver for implicit MPM. */
template <typename T>
class MpmSolver {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MpmSolver);

  /* Constructs an empty MpmSolver. */
  MpmSolver(MpmSolverParameters parameters = MpmSolverParameters());

  /* Given the mpm_state at the previous time step, computes the results
   reported by schur_complement() and participating_v_star(). */
  void ComputeFreeMotionState(const MpmState<T>& mpm_state);

  /* Given the next time step velocities for the participating grid nodes,
   computes the dv = v_next - v* for all active grid nodes in the internal
   MpmState.
   @param[in] The participating dofs in v_next.
   @param[in] The partial permutation that maps all active dofs to participating
   dofs.
   @pre ComputeFreeMotionState() has been called.
   @pre participating_v_next has the same size and ordering as
   participating_v_star(). */
  VectorX<T> CalcDv(
      const VectorX<double>& participating_v_next,
      const multibody::contact_solvers::internal::PartialPermutation&
          participating_dof_permutation) const;

  /* Returns the Schur complement of the tangent matrix of the MpmState
   evaluated at free motion velocity v*. */
  const MatrixX<double>& schur_complement() const {
    return schur_complement_.get_D_complement();
  }

  /* Returns the free motion velocities (v*) of the grid nodes participating in
   constraints. */
  const VectorX<double>& participating_v_star() const {
    return participating_v_star_;
  }

  /* The solver is considered as converged if ‖r‖ < max(εᵣ * ‖r₀‖, εₐ) where r
   and r₀ are `residual_norm` and `initial_residual_norm` respectively, and εᵣ
   and εₐ are relative and absolute tolerance respectively. */
  bool solver_converged(const T& residual_norm,
                        const T& initial_residual_norm) const {
    return residual_norm <
           std::max(parameters_.relative_tolerance * initial_residual_norm,
                    parameters_.absolute_tolerance);
  }

  void set_parameters(MpmSolverParameters parameters) {
    parameters_ = std::move(parameters);
  }

 private:
  SolverState<T> solver_state_;
  contact_solvers::internal::SchurComplement schur_complement_;
  VectorX<double> participating_v_star_{VectorX<double>::Zero(0)};
  MpmSolverParameters parameters_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
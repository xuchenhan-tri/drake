#pragma once

#include "mpm_state.h"
#include "transfer.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Newton-Raphson solver for implicit MPM. */
template <typename T>
class MpmSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmSolver);

  /* Solves the equation

     M*(v-v0)-f(x(v))*dt = 0 (1)

   with unknown variable v on the grid with a Newton-Raphson solver.
   @param [in, out] state  On input, `state` provides the MpmState evaluated at
   the previous time step.*/
  MpmSolver(MpmState<T>* state) : state_(state);

  /* Given the next time step velocities for the participating grid nodes, moves
   the internal MPM state to the next time step's state. */
  void CalcNextState(const VectorX<double>& participating_v_next);

  /* Returns the Schur complement of the tangent matrix of equation (1). */
  const MatrixX<double>& schur_complement() const {
    return schur_complement_.get_D_complement();
  }

  /* Returns the velocities of the grid nodes participating in constraints. */
  const VectorX<double>& participating_v_star() const {
    return participating_v_star;
  }

 private:
  void SolveFreeMotion();

  MpmState* state_{};
  contact_solvers::internal::SchurComplement schur_complement_;
  VectorX<double> participating_v_star_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
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

  /* Initializes an MpmSolver for an implicit MPM step.
   @param[in] dt          Time step.
   @param[in] sparse_grid The background Eulerian grid, must be non-null.
   @param[in] particles   The particles at the previous time step, must be
                          non-null. */
  MpmSolver(T dt, SparseGrid<T>* sparse_grid, ParticleData<T>* particles,
            Parallelism parallelism)
      : state_(dt, sparse_grid, particles, parallelism) {}

  /* Solves the equation

     M*dv-f(x(v+dv))*dt = 0

   with unknown variable dv on the grid with a Newton-Raphson solver. */
  int SolveFreeMotion();

 private:
  MpmState state_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
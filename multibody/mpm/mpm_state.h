#pragma once

#include "particles.h"
#include "sparse_grid.h"
#include "transfer.h"

#include "drake/common/parallelism.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
struct SolverState;

/* Newton-Raphson solver for implicit MPM. */
template <typename T, template <typename> class Grid = SparseGrid>
class MpmState {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmState);

  /* Creates a MpmState derived from the given particles.
   @param[in] dt         The time step used in this MpmState (in seconds).
   @param[in] dx         The grid spacing (in meters).
   @param[in] particles  The particle data.
   @pre dt > 0 and dx > 0.
   @pre particles and grid are non-null. */
  MpmState(T dt, double dx, Particles<T> particles,
           Parallelism parallelism = {});

  int num_dofs() const { return num_dofs_; }

  int num_particles() const { return particles_.data.x.size(); }

  /* Sets dv = dv + ddv. */
  void UpdateSolverState(const VectorX<T>& ddv,
                         SolverState<T>* solver_state) const;

  /* Computes the energy at the given solver state using the formula
    E = 1/2*dv*M*dv  + ∑ₚ Ψ(Fₚ)*volumeₚ*dt. */
  T CalcTotalEnergy(const SolverState<T>& solver_state) const ;

  /* Computes the residual vector

    b = M * dv - f(x(v+dv), v+dv) * dt,

   where M is the lumped mass matrix. */
  void CalcResidual(const SolverState<T>& solver_state, VectorX<T>* b) const;

  /* Makes a Block3x3SparseSymmetricMatrix that has the sparsity pattern of
   the grid induced by the particles. Each entry of the returned matrix is set
   to zero. Note that there exists a non-zero entry between grid node i and j
   iff there exists a particle that transfers to both i and j. With quadratic
   B-spline, a node can have up to 125 neighbors, including itself.
   @pre SetNodeIndices() has been called on the grid referenced by this
   transfer. */
  multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix
  MakeTangentMatrix() const;

  /* Computes the tangent matrix of the residual vector
   b = M * dv - f(x(v+dv), v+dv) * dt. */
  void CalcTangentMatrix(
      const SolverState<T>& solver_state,
      multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix*
          tangent_matrix) const;

  const Grid<T>& grid() const { return grid_; }
  const Particles<T>& particles() const { return particles_; }

  const contact_solvers::internal::PartialPermutation& grid_dof_permutation()
      const {
    return dof_permutation_;
  }

  const contact_solvers::internal::PartialPermutation& grid_node_permutation()
      const {
    return node_permutation_;
  }

  /* Given `dv`, which contains the v_next - v0 for grid nodes in the order of
   grid indices, moves `this` MpmState to the next time step. */
  void AdvanceToNextState(const VectorX<T>& dv);

 private:
  /* Computes the particle deformation gradient, stress, and stress
   derivatives based on grid data and dv. */
  void UpdateSolverParticleState(SolverState<T>* solver_state) const;
  void UpdateSolverParticleStateSimd(SolverState<T>* solver_state) const;

  /* Updates the grid indices, node permutation and dof permutation after new
   grid data has been transferred from particles. This function must be called
   after every each time a P2G transfer has happened. */
  void UpdateGrid();

  T dt_{};
  double dx_{};
  Particles<T> particles_{};
  mutable Grid<T> grid_;
  std::unique_ptr<Transfer<T, Grid>> transfer_{};
  Parallelism parallelism_{};
  int num_dofs_{};
  T D_inverse_{};
  /* Partial permutation that maps all grid node indices to participating node
   indices. */
  multibody::contact_solvers::internal::PartialPermutation node_permutation_;
  /* Partial permutation that maps all grid dofs to participating dofs. */
  multibody::contact_solvers::internal::PartialPermutation dof_permutation_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
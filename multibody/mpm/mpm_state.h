#pragma once

#include "particles.h"
#include "sparse_grid.h"
#include "transfer.h"

#include "drake/common/parallelism.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Newton-Raphson solver for implicit MPM. */
template <typename T, template <typename> class Grid = SparseGrid>
class MpmState {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmState);

  /* State-dependent data. */
  struct Data {
    Data(const ParticleData<T>& particles)
        : F(particles.F), tau_v0(particles.tau_v0) {
      volume_scaled_stress_derivatives.resize(F.size());
    }

    std::vector<Matrix3<T>> F;
    std::vector<Matrix3<T>> tau_v0;
    std::vector<math::FourthOrderTensor<T>> volume_scaled_stress_derivatives;
  };

  /* Creates a MpmState derived from the given particles.
   @param[in] dt         The time step used in this MpmState.
   @param[in] particles  The particle data.
   @param[in] grid       The grid data structure used to store the state.
   @pre dt > 0.
   @pre particles and grid are non-null. */
  MpmState(T dt, Grid<T>* grid, Particles<T>* particles,
           Parallelism parallelism = {});

  int num_dofs() const { return dv_.size(); }

  /* Sets dv = dv + ddv. */
  void IncrementDv(const VectorX<T>& ddv);

  const VectorX<T>& dv() { return dv_; }

  /* Computes the energy 1/2*dv*M*dv + ∑ₚ Ψ(Fₚ)*volumeₚ*dt. */
  T CalcTotalEnergy();

  /* Computes the residual vector b = M * dv - f(x(v+dv), v+dv) * dt where M is
   the lumped mass matrix. */
  void CalcResidual(VectorX<T>* b);

  /* Makes a Block3x3SparseSymmetricMatrix that has the sparsity pattern of
   the grid induced by the particles. Each entry of the returned matrix is set
   to zero. Note that there exists a non-zero entry between grid node i and j
   iff there exists a particle that transfers to both i and j. With quadratic
   B-spline, a node can have up to 125 neighbors, including itself.
   @pre SetNodeIndices() has been called on the grid referenced by this
   transfer. */
  multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix
  MakeTangentMatrix() const;

  /* Computes the tangent matrix of the residual vector b = M * dv -
   f(x(v+dv), v+dv) * dt. */
  void CalcTangentMatrix(
      multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix*
          tangent_matrix);

  /* Testing only */
  const Data& data() const { return data_; }

 private:
  /* Computes the particle deformation gradient, stress, and stress
   derivatives based on grid data and dv. */
  void UpdateParticleState();
  void UpdateParticleStateSimd();

  T dt_{};
  VectorX<T> dv_;
  Grid<T>* grid_{};
  Particles<T>* particles_{};
  Parallelism parallelism_{};
  T D_inverse_{};
  Data data_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
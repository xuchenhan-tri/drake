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
template <typename T>
class MpmState {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmState);
  /* Creates a MpmState derived from the given particles.
   @param[in] dt         The time step used in this MpmState.
   @param[in] particles  The particle data.
   @param[in] grid       The grid data structure used to store the state.
   @pre dt > 0.
   @pre particles and grid are non-null. */
  MpmState(T dt, SparseGrid<T>* grid, ParticleData<T>* particles,
           Parallelism parallelism = {});

  int num_dofs() const { return dv_.size(); }

  /* Sets dv = dv + ddv. */
  void IncrementDv(const VectorX<T>& ddv);

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

 private:
  /* State-dependent data. */
  struct Data {
    Data(const ParticleData<T>& particles)
        : F(particles.F), tau_v0(particles.tau_v0) {
      volume_scaled_stress_derivatives.resize(F.size());
    }

    std::vector<Matrix3<T>> F;
    std::vector<Matrix3<T>> tau_v0;
    std::vector<Eigen::Matrix<T, 9, 9>> volume_scaled_stress_derivatives;
  };

  /* Computes the particle deformation gradient, stress, and stress
   derivatives based on grid data and dv. */
  void UpdateParticleState();

  T dt_{};
  VectorX<T> dv_;
  SparseGrid<T>* grid_{};
  ParticleData<T>* particles_{};
  Parallelism parallelism_{};
  Data data_;
  T D_inverse_{};
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
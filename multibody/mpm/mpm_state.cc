#include "mpm_state.h"

#include <iostream>

#include "mock_sparse_grid.h"
#include "solver_state.h"

#include "drake/common/fmt_eigen.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using multibody::contact_solvers::internal::BlockSparsityPattern;

template <typename T, template <typename> class Grid>
MpmState<T, Grid>::MpmState(T dt, double dx, Particles<T> particles,
                            Parallelism parallelism)
    : dt_(dt),
      dx_(dx),
      particles_(std::move(particles)),
      grid_(dx, parallelism),
      parallelism_(parallelism),
      D_inverse_(4.0 / (grid_.dx() * grid_.dx())) {
  DRAKE_DEMAND(dt > 0);
  DRAKE_DEMAND(dx > 0);
  transfer_ = std::make_unique<Transfer<T, Grid>>(dt, &grid_, &particles_);
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    transfer_->SerialParticleToGrid();
  } else {
    transfer_->ParallelSimdParticleToGrid(parallelism);
  }
  UpdateGrid();
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::UpdateSolverState(const VectorX<T>& ddv,
                                          SolverState<T>* solver_state) {
  DRAKE_DEMAND(solver_state != nullptr);
  DRAKE_DEMAND(ddv.size() == num_dofs());
  DRAKE_DEMAND(solver_state->dv.size() == num_dofs());
  solver_state->dv += ddv;
  UpdateSolverParticleState(solver_state);
}

template <typename T, template <typename> class Grid>
T MpmState<T, Grid>::CalcTotalEnergy(const SolverState<T>& solver_state) {
  const int kDim = 3;
  const VectorX<T>& dv = solver_state.dv;
  const std::vector<Matrix3<T>>& F = solver_state.F;
  /* Potential energy from the particles. */
  T total_energy = particles_.data.ComputeTotalEnergy(F);
  /* The 1/2*dv*M*dv term. */
  grid_.IterateGrid([&](GridData<T>* node) {
    if (node->m > 0.0) {
      const int index = node->index;
      DRAKE_ASSERT(index >= 0 && index < num_dofs_ / 3);
      total_energy +=
          0.5 * node->m * dv.template segment<kDim>(index * kDim).squaredNorm();
    }
  });
  return total_energy;
}

// TODO(xuchenhan-tri): Implement a parallel + simd version of this.
template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::CalcResidual(const SolverState<T>& solver_state,
                                     VectorX<T>* b) {
  DRAKE_DEMAND(b != nullptr);
  b->resize(num_dofs());
  constexpr int kDim = 3;
  // TODO(xuchenhan-tri): Make sure the scratch is zeroed out before splating.
  using Scalar = decltype(grid_.dx());
  /* Splat temporary (negative) impulses to the grid data scratch and collect
   them into b. */
  auto splat_force_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                                Pad<GridData<T>>* grid_data,
                                ParticleData<T>* particle_data,
                                int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_.dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Scalar& w = bspline.weight(i, j, k);
          const Vector3<Scalar>& xi = grid_x[i][j][k];
          /* Note that we take the stress from the scratch data here instead
           directly from the particle because we are in the Newton loop. */
          const Matrix3<T>& tau_v0 = solver_state.tau_v0[data_index];
          /* For the elastic force from particles, we compute -∂E/∂xᵢ and
           get

             fᵢ = -∑ₚ Vₚ * Pₚ * Fₚⁿᵀ * D⁻¹ * (xᵢ − xₚ) * wᵢₚ

           with Pₚ = ∂Ψ/∂Fₚ. Noting that Pₚ * Fₚⁿᵀ is the Kirchhoff stress,
           we group Vₚ * Pₚ * Fₚⁿᵀ into a single term `tau_v0`. Rearranging
           terms reveals that - fᵢdt is given by the equation in the code
           below. */
          (*grid_data)[i][j][k].scratch +=
              tau_v0 * (xi - x) * D_inverse_ * dt_ * w;
        }
      }
    }
  };
  const ParticleSorter& sorter = particles_.sorter;
  ParticleData<T>& particle_data = particles_.data;
  sorter.Iterate(&grid_, &particle_data, true, std::move(splat_force_kernel));
  /* Collect from the scratch data, add in the M * dv term, and clear the
   scratch data. */
  const VectorX<T>& dv = solver_state.dv;
  grid_.IterateGrid([&](GridData<T>* node) {
    if (node->m > 0.0) {
      const int index = node->index;
      DRAKE_ASSERT(index >= 0 && index < b->size() / 3);
      b->template segment<kDim>(index * kDim) =
          node->m * dv.template segment<kDim>(index * kDim) + node->scratch;
      node->scratch.setZero();
    }
  });
}

// TODO(xuchenhan-tri): We should be able to make a single pass in this function
// to both index the grid and compute the connectivity (neighbors). This will
// reduce the number of duplicated neighbors.
template <typename T, template <typename> class Grid>
Block3x3SparseSymmetricMatrix MpmState<T, Grid>::MakeTangentMatrix() const {
  /* Here we loop over all particles; each particle creates an edge in the
   connectivity graph that connects all grid nodes in the pad (3x3x3 = 27 grid
   nodes) the particle transfers to. */
  const int num_active_nodes = grid_.num_active_nodes();
  std::vector<std::vector<int>> sparsity_pattern(num_active_nodes);
  for (auto& row : sparsity_pattern) {
    /* Each node i can have at most 125 nodes j such that a particle p splats to
     both i and j. */
    row.reserve(125);
  }
  using Scalar = decltype(grid_.dx());
  auto allocate_sparsity_kernel =
      [&](const Pad<Vector3<Scalar>>& grid_x, const Pad<GridData<T>>& grid_data,
          const ParticleData<T>& particle_data, int data_index) {
        for (int idx0 = 0; idx0 < 27; ++idx0) {
          const int i = idx0 / 9;
          const int j = (idx0 / 3) % 3;
          const int k = idx0 % 3;
          const int index0 = grid_data[i][j][k].index;
          DRAKE_DEMAND(index0 >= 0 && index0 < num_active_nodes);

          /* Process the lower triangle by skipping redundant pairs. */
          for (int idx1 = idx0; idx1 < 27; ++idx1) {
            const int ii = idx1 / 9;
            const int jj = (idx1 / 3) % 3;
            const int kk = idx1 % 3;

            const int index1 = grid_data[ii][jj][kk].index;
            DRAKE_DEMAND(index1 >= 0 && index1 < num_active_nodes);

            if (index0 <= index1) {
              sparsity_pattern[index0].push_back(index1);
            } else {
              sparsity_pattern[index1].push_back(index0);
            }
          }
        }
      };
  const ParticleSorter& sorter = particles_.sorter;
  const ParticleData<T>& particle_data = particles_.data;
  sorter.IterateOneParticlePerPad(grid_, particle_data,
                                  std::move(allocate_sparsity_kernel));

  const int kDim = 3;
  BlockSparsityPattern block_pattern(std::vector<int>(num_active_nodes, kDim),
                                     std::move(sparsity_pattern));
  return Block3x3SparseSymmetricMatrix(std::move(block_pattern));
}

template <>
void MpmState<AutoDiffXd, MockSparseGrid>::CalcTangentMatrix(
    const SolverState<AutoDiffXd>& solver_state,
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  throw std::runtime_error("UpdateSolverParticleStateSimd(): Not implemented");
}

template <>
void MpmState<float>::CalcTangentMatrix(
    const SolverState<float>& solver_state,
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  throw std::runtime_error("UpdateSolverParticleStateSimd(): Not implemented");
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::CalcTangentMatrix(
    const SolverState<T>& solver_state,
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  tangent_matrix->SetZero();
  using Scalar = decltype(grid_.dx());
  const T scale = dt_ * dt_ * D_inverse_ * D_inverse_;
  const int num_active_nodes = grid_.num_active_nodes();
  auto splat_force_derivatives_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                                            Pad<GridData<T>>* grid_data,
                                            ParticleData<T>* particle_data,
                                            int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_.dx());
    const auto& dPdF_v0 =
        solver_state.volume_scaled_stress_derivatives[data_index];
    Matrix3<T> hessian = Matrix3<T>::Zero();

    for (int idx0 = 0; idx0 < 27; ++idx0) {
      const int i = idx0 / 9;
      const int j = (idx0 / 3) % 3;
      const int k = idx0 % 3;
      const int index0 = (*grid_data)[i][j][k].index;
      DRAKE_ASSERT(index0 >= 0 && index0 < num_active_nodes);
      const Scalar& w0 = bspline.weight(i, j, k);
      const Vector3<Scalar>& x0 = grid_x[i][j][k];
      const Vector3<T> scaled_u0 =
          scale * w0 * particle_data->F[data_index].transpose() * (x0 - x);

      for (int idx1 = 0; idx1 <= idx0; ++idx1) {
        const int ii = idx1 / 9;
        const int jj = (idx1 / 3) % 3;
        const int kk = idx1 % 3;
        const int index1 = (*grid_data)[ii][jj][kk].index;
        DRAKE_DEMAND(index1 >= 0 && index1 < num_active_nodes);
        const Scalar& w1 = bspline.weight(ii, jj, kk);
        const Vector3<Scalar>& x1 = grid_x[ii][jj][kk];
        const Vector3<T> u1 =
            w1 * particle_data->F[data_index].transpose() * (x1 - x);
        dPdF_v0.ContractWithVectors(scaled_u0, u1, &hessian);
        if (index0 >= index1) {
          tangent_matrix->AddToBlock(index0, index1, hessian);
        } else {
          tangent_matrix->AddToBlock(index1, index0, hessian);
        }
      }
    }
  };
  const ParticleSorter& sorter = particles_.sorter;
  ParticleData<T>& particle_data = particles_.data;
  sorter.Iterate(&grid_, &particle_data, false,
                 std::move(splat_force_derivatives_kernel));
  /* Add in the mass terms. */
  grid_.IterateGrid([&](GridData<T>* node) {
    if (node->m > 0.0) {
      const int index = node->index;
      tangent_matrix->AddToBlock(index, index,
                                 node->m * Matrix3<T>::Identity());
    }
  });
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::AdvanceToNextState(const VectorX<T>& dv) {
  const int kDim = 3;
  DRAKE_DEMAND(dv.size() == num_dofs());
  auto upgdate_grid_velocity = [&](GridData<T>* node) {
    if (node->m > 0.0) {
      node->v += dv.template segment<kDim>(kDim * node->index);
    }
  };
  grid_.IterateGrid(upgdate_grid_velocity);
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    transfer_->SerialGridToParticle();
    transfer_.reset(new Transfer<T, Grid>(dt_, &grid_, &particles_));
    transfer_->SerialParticleToGrid();
  } else {
    transfer_->ParallelSimdGridToParticle(parallelism_);
    transfer_.reset(new Transfer<T, Grid>(dt_, &grid_, &particles_));
    transfer_->ParallelSimdParticleToGrid(parallelism_);
  }
  UpdateGrid();
}

template <>
void MpmState<AutoDiffXd, MockSparseGrid>::UpdateSolverParticleStateSimd(
    SolverState<AutoDiffXd>*) {
  throw std::runtime_error("UpdateSolverParticleStateSimd(): Not implemented");
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::UpdateSolverParticleState(
    SolverState<T>* solver_state) {
  const VectorX<T>& dv = solver_state->dv;
  using Scalar = decltype(grid_.dx());
  auto update_F_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                             Pad<GridData<T>>* grid_data,
                             ParticleData<T>* particle_data, int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    Matrix3<T> C = Matrix3<T>::Zero();
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_.dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const int grid_index = (*grid_data)[i][j][k].index;
          const Vector3<T>& vi =
              (*grid_data)[i][j][k].v + dv.template segment<3>(3 * grid_index);
          const Vector3<Scalar>& xi = grid_x[i][j][k];
          const Scalar w = bspline.weight(i, j, k);
          C += (w * vi) * (xi - x).transpose();
        }
      }
    }
    C *= D_inverse_;
    const Matrix3<T>& particle_F = particle_data->F[data_index];
    solver_state->F[data_index] = particle_F + C * dt_ * particle_F;
  };
  const ParticleSorter& sorter = particles_.sorter;
  sorter.Iterate(&grid_, &particles_.data, false, std::move(update_F_kernel));

  /* Then update stress and stress derivatives. */
  particles_.data.UpdateStress(&solver_state->F, &solver_state->tau_v0,
                               /* apply plasticity */ false, parallelism_);
  particles_.data.UpdateStressDerivatives(
      solver_state->F, &solver_state->volume_scaled_stress_derivatives);
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::UpdateSolverParticleStateSimd(
    SolverState<T>* solver_state) {
  const VectorX<T>& dv = solver_state->dv;
  auto update_F_kernel = [&](const Pad<Vector3<T>>& grid_x,
                             Pad<GridData<T>>* grid_data,
                             ParticleData<T>* particle_data,
                             const std::vector<int>& data_indices) {
    Matrix3<SimdScalar<T>> B = Matrix3<SimdScalar<T>>::Zero();
    Vector3<SimdScalar<T>> x = Load(particle_data->x, data_indices);
    const BsplineWeights<SimdScalar<T>> bspline =
        BsplineWeights<SimdScalar<T>>(x, grid_.dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const int grid_index = (*grid_data)[i][j][k].index;
          const Vector3<T>& vi =
              (*grid_data)[i][j][k].v + dv.template segment<3>(3 * grid_index);
          const Vector3<T>& xi = grid_x[i][j][k];
          const SimdScalar<T> w = bspline.weight(i, j, k);
          B += (w * vi) * (xi - x).transpose();
        }
      }
    }
    Matrix3<SimdScalar<T>> C = B * D_inverse_;
    Matrix3<SimdScalar<T>> F = Load(particle_data->F, data_indices);
    F += C * dt_ * F;
    Store(F, &solver_state->F, data_indices);
  };
  const ParticleSorter& sorter = particles_.sorter;
  sorter.IterateParallelSimd(&grid_, &particles_.data, false, parallelism_,
                             std::move(update_F_kernel));

  /* Then update stress and stress derivatives. */
  particles_.data.UpdateStress(&solver_state->F, &solver_state->tau_v0,
                               /* apply plasticity */ false, parallelism_);
  particles_.data.UpdateStressDerivatives(
      solver_state->F, &solver_state->volume_scaled_stress_derivatives,
      parallelism_);
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::UpdateGrid() {
  /* Index the grid and build participation permutations. */
  grid_.SetNodeIndices(&node_permutation_);
  constexpr int kDim = 3;
  num_dofs_ = grid_.num_active_nodes() * 3;

  int permuted_grid_index = 0;
  const int num_nodes = grid_.num_active_nodes();
  const int num_dofs = num_nodes * kDim;
  std::vector<int> permuted_dof_indices(num_dofs, -1);
  for (int n = 0; n < num_nodes; ++n) {
    if (n < node_permutation_.domain_size() &&
        node_permutation_.participates(n)) {
      for (int d = 0; d < kDim; ++d) {
        permuted_dof_indices[kDim * n + d] = kDim * permuted_grid_index + d;
      }
    }
  }
  dof_permutation_ = contact_solvers::internal::PartialPermutation(
      std::move(permuted_dof_indices));
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MpmState<double>;
template class drake::multibody::mpm::internal::MpmState<float>;
template class drake::multibody::mpm::internal::MpmState<
    drake::AutoDiffXd, drake::multibody::mpm::internal::MockSparseGrid>;
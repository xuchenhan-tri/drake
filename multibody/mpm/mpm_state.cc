#include "mpm_state.h"

#include <iostream>

#include "mock_sparse_grid.h"

#include "drake/common/fmt_eigen.h"
namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using multibody::contact_solvers::internal::BlockSparsityPattern;

template <typename T, template <typename> class Grid>
MpmState<T, Grid>::MpmState(T dt, Grid<T>* grid, Particles<T>* particles,
                            Parallelism parallelism)
    : dt_(dt),
      grid_(grid),
      particles_(particles),
      parallelism_(parallelism),
      D_inverse_(4.0 / (grid->dx() * grid->dx())),
      data_(particles->data) {
  DRAKE_DEMAND(dt > 0);
  DRAKE_DEMAND(grid != nullptr);
  DRAKE_DEMAND(particles != nullptr);
  Transfer<T, Grid> transfer(dt, grid, particles);
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    transfer.SerialParticleToGrid();
  } else {
    transfer.ParallelSimdParticleToGrid(parallelism);
  }
  grid->SetNodeIndices();
  constexpr int kSpatialDim = 3;
  dv_ = VectorX<T>::Zero(grid->num_active_nodes() * kSpatialDim);
  UpdateParticleState();
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::IncrementDv(const VectorX<T>& ddv) {
  DRAKE_DEMAND(ddv.size() == dv_.size());
  dv_ += ddv;
  UpdateParticleState();
}

template <typename T, template <typename> class Grid>
T MpmState<T, Grid>::CalcTotalEnergy() {
  const int kDim = 3;
  /* Potential energy from the particles. */
  T total_energy = particles_->data.ComputeTotalEnergy(data_.F);
  /* The 1/2*dv*M*dv term. */
  grid_->IterateGrid([&](GridData<T>* node) {
    if (node->m > 0.0) {
      const int index = node->index;
      DRAKE_ASSERT(index >= 0 && index < dv_.size() / 3);
      total_energy += 0.5 * node->m *
                      dv_.template segment<kDim>(index * kDim).squaredNorm();
    }
  });
  return total_energy;
}

// TODO(xuchenhan-tri): Implement a parallel + simd version of this.
template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::CalcResidual(VectorX<T>* b) {
  DRAKE_DEMAND(b != nullptr);
  b->resizeLike(dv_);
  constexpr int kDim = 3;
  // TODO(xuchenhan-tri): Make sure the scratch is zeroed out before splating.
  using Scalar = decltype(grid_->dx());
  /* Splat temporary (negative) impulses to the grid data scratch and collect
   them into b. */
  auto splat_force_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                                Pad<GridData<T>>* grid_data,
                                ParticleData<T>* particle_data,
                                int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const Scalar& w = bspline.weight(i, j, k);
          const Vector3<Scalar>& xi = grid_x[i][j][k];
          /* Note that we take the stress from the scratch data here instead
           directly from the particle because we are in the Newton loop. */
          const Matrix3<T>& tau_v0 = data_.tau_v0[data_index];
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
  const ParticleSorter& sorter = particles_->sorter;
  ParticleData<T>& particle_data = particles_->data;
  sorter.Iterate(grid_, &particle_data, true, std::move(splat_force_kernel));
  /* Collect from the scratch data, add in the M * dv term, and clear the
   scratch data. */
  grid_->IterateGrid([&](GridData<T>* node) {
    if (node->m > 0.0) {
      const int index = node->index;
      DRAKE_ASSERT(index >= 0 && index < b->size() / 3);
      b->template segment<kDim>(index * kDim) =
          node->m * dv_.template segment<kDim>(index * kDim) + node->scratch;
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
  const int num_active_nodes = grid_->num_active_nodes();
  std::vector<std::vector<int>> sparsity_pattern(num_active_nodes);
  for (auto& row : sparsity_pattern) {
    /* Each node i can have at most 125 nodes j such that a particle p splats to
     both i and j. */
    row.reserve(125);
  }
  using Scalar = decltype(grid_->dx());
  auto allocate_sparsity_kernel =
      [&](const Pad<Vector3<Scalar>>& grid_x, Pad<GridData<T>>* grid_data,
          ParticleData<T>* particle_data, int data_index) {
        for (int idx0 = 0; idx0 < 27; ++idx0) {
          const int i = idx0 / 9;
          const int j = (idx0 / 3) % 3;
          const int k = idx0 % 3;
          const int index0 = (*grid_data)[i][j][k].index;
          DRAKE_DEMAND(index0 >= 0 && index0 < num_active_nodes);

          /* Process the lower triangle by skipping redundant pairs. */
          for (int idx1 = idx0; idx1 < 27; ++idx1) {
            const int ii = idx1 / 9;
            const int jj = (idx1 / 3) % 3;
            const int kk = idx1 % 3;

            const int index1 = (*grid_data)[ii][jj][kk].index;
            DRAKE_DEMAND(index1 >= 0 && index1 < num_active_nodes);

            if (index0 <= index1) {
              sparsity_pattern[index0].push_back(index1);
            } else {
              sparsity_pattern[index1].push_back(index0);
            }
          }
        }
      };
  const ParticleSorter& sorter = particles_->sorter;
  ParticleData<T>& particle_data = particles_->data;
  sorter.IterateOneParticlePerPad(grid_, &particle_data, false,
                                  std::move(allocate_sparsity_kernel));

  const int kSpatialDim = 3;
  BlockSparsityPattern block_pattern(
      std::vector<int>(num_active_nodes, kSpatialDim),
      std::move(sparsity_pattern));
  return Block3x3SparseSymmetricMatrix(std::move(block_pattern));
}

template <>
void MpmState<AutoDiffXd, MockSparseGrid>::CalcTangentMatrix(
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  throw std::runtime_error("UpdateParticleStateSimd(): Not implemented");
}

template <>
void MpmState<float>::CalcTangentMatrix(
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  throw std::runtime_error("UpdateParticleStateSimd(): Not implemented");
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::CalcTangentMatrix(
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  tangent_matrix->SetZero();
  using Scalar = decltype(grid_->dx());
  const T scale = dt_ * dt_ * D_inverse_ * D_inverse_;
  const int num_active_nodes = grid_->num_active_nodes();
  auto splat_force_derivatives_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                                            Pad<GridData<T>>* grid_data,
                                            ParticleData<T>* particle_data,
                                            int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_->dx());
    const Eigen::Matrix<T, 9, 9> scaled_dPdF =
        scale * data_.volume_scaled_stress_derivatives[data_index];
    Matrix3<T> hessian = Matrix3<T>::Zero();

    for (int idx0 = 0; idx0 < 27; ++idx0) {
      const int i = idx0 / 9;
      const int j = (idx0 / 3) % 3;
      const int k = idx0 % 3;
      const int index0 = (*grid_data)[i][j][k].index;
      DRAKE_ASSERT(index0 >= 0 && index0 < num_active_nodes);
      const Scalar& w0 = bspline.weight(i, j, k);
      const Vector3<Scalar>& x0 = grid_x[i][j][k];
      const Vector3<T> u0 =
          w0 * particle_data->F[data_index].transpose() * (x0 - x);

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
        PerformDoubleTensorContraction<T>(scaled_dPdF, u0, u1, &hessian);
        if (index0 >= index1) {
          tangent_matrix->AddToBlock(index0, index1, hessian);
        } else {
          tangent_matrix->AddToBlock(index1, index0, hessian);
        }
      }
    }
  };
  const ParticleSorter& sorter = particles_->sorter;
  ParticleData<T>& particle_data = particles_->data;
  sorter.Iterate(grid_, &particle_data, false,
                 std::move(splat_force_derivatives_kernel));
  /* Add in the mass terms. */
  grid_->IterateGrid([&](GridData<T>* node) {
    if (node->m > 0.0) {
      const int index = node->index;
      tangent_matrix->AddToBlock(index, index,
                                 node->m * Matrix3<T>::Identity());
    }
  });
}

template <>
void MpmState<AutoDiffXd, MockSparseGrid>::UpdateParticleStateSimd() {
  throw std::runtime_error("UpdateParticleStateSimd(): Not implemented");
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::UpdateParticleState() {
  using Scalar = decltype(grid_->dx());
  auto update_F_kernel = [&](const Pad<Vector3<Scalar>>& grid_x,
                             Pad<GridData<T>>* grid_data,
                             ParticleData<T>* particle_data, int data_index) {
    const Vector3<T>& x = particle_data->x[data_index];
    Matrix3<T> C = Matrix3<T>::Zero();
    const BsplineWeights<Scalar> bspline = MakeBsplineWeights(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const int grid_index = (*grid_data)[i][j][k].index;
          const Vector3<T>& vi =
              (*grid_data)[i][j][k].v + dv_.template segment<3>(3 * grid_index);
          const Vector3<Scalar>& xi = grid_x[i][j][k];
          const Scalar w = bspline.weight(i, j, k);
          C += (w * vi) * (xi - x).transpose();
        }
      }
    }
    C *= D_inverse_;
    const Matrix3<T>& particle_F = particle_data->F[data_index];
    data_.F[data_index] = particle_F + C * dt_ * particle_F;
  };
  const ParticleSorter& sorter = particles_->sorter;
  sorter.Iterate(grid_, &particles_->data, false, std::move(update_F_kernel));

  /* Then update stress and stress derivatives. */
  particles_->data.UpdateStress(&data_.F, &data_.tau_v0,
                                /* apply plasticity */ false, parallelism_);
  particles_->data.UpdateStressDerivatives(
      data_.F, &data_.volume_scaled_stress_derivatives);
}

template <typename T, template <typename> class Grid>
void MpmState<T, Grid>::UpdateParticleStateSimd() {
  auto update_F_kernel = [&](const Pad<Vector3<T>>& grid_x,
                             Pad<GridData<T>>* grid_data,
                             ParticleData<T>* particle_data,
                             const std::vector<int>& data_indices) {
    Matrix3<SimdScalar<T>> B = Matrix3<SimdScalar<T>>::Zero();
    Vector3<SimdScalar<T>> x = Load(particle_data->x, data_indices);
    const BsplineWeights<SimdScalar<T>> bspline =
        BsplineWeights<SimdScalar<T>>(x, grid_->dx());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const int grid_index = (*grid_data)[i][j][k].index;
          const Vector3<T>& vi =
              (*grid_data)[i][j][k].v + dv_.template segment<3>(3 * grid_index);
          const Vector3<T>& xi = grid_x[i][j][k];
          const SimdScalar<T> w = bspline.weight(i, j, k);
          B += (w * vi) * (xi - x).transpose();
        }
      }
    }
    Matrix3<SimdScalar<T>> C = B * D_inverse_;
    Matrix3<SimdScalar<T>> F = Load(particle_data->F, data_indices);
    F += C * dt_ * F;
    Store(F, &data_.F, data_indices);
  };
  const ParticleSorter& sorter = particles_->sorter;
  sorter.IterateParallelSimd(grid_, &particles_->data, false, parallelism_,
                             std::move(update_F_kernel));

  /* Then update stress and stress derivatives. */
  particles_->data.UpdateStress(&data_.F, &data_.tau_v0,
                                /* apply plasticity */ false, parallelism_);
  particles_->data.UpdateStressDerivatives(
      data_.F, &data_.volume_scaled_stress_derivatives, parallelism_);
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MpmState<double>;
template class drake::multibody::mpm::internal::MpmState<float>;
template class drake::multibody::mpm::internal::MpmState<
    drake::AutoDiffXd, drake::multibody::mpm::internal::MockSparseGrid>;
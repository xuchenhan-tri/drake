#include "mpm_state.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

using multibody::contact_solvers::internal::Block3x3SparseSymmetricMatrix;
using multibody::contact_solvers::internal::BlockSparsityPattern;

template <typename T>
MpmState<T>::MpmState(T dt, SparseGrid<T>* grid, ParticleData<T>* particles,
                      Parallelism parallelism)
    : grid_(grid),
      particles_(particles),
      parallelism_(parallelism),
      data_(*particles),
      D_inverse_(4.0 / (grid->dx() * grid->dx())) {
  DRAKE_DEMAND(dt > 0);
  DRAKE_DEMAND(grid != nullptr);
  DRAKE_DEMAND(particles != nullptr);
  Transfer<T> transfer(dt, grid, particles);
  transfer.ParallelSimdParticleToGrid(parallelism);
  grid->SetNodeIndices();
  constexpr int kSpatialDim = 3;
  dv_ = VectorX<T>::Zero(grid->num_active_nodes() * kSpatialDim);
  UpdateParticleState();
}

template <typename T>
void MpmState<T>::IncrementDv(const VectorX<T>& ddv) {
  DRAKE_DEMAND(ddv.size() == dv_.size());
  dv_ += ddv;
  UpdateParticleState();
}

template <typename T>
void MpmState<T>::CalcResidual(VectorX<T>* b) {
  DRAKE_DEMAND(b != nullptr);
  b->resizeLike(dv_);
  constexpr int kDim = 3;
  /* Overwrite old values with M*dv term. */
  grid_->IterateGrid([&](const GridData<T>& node) {
    const int index = node.index;
    DRAKE_ASSERT(index >= 0 && index < b_->size());
    b->template segment<kDim>(index * kDim) =
        node.m * dv_.template segment<kDim>(index * kDim);
    node.scratch.setZero();
  });
  /* Splat forces to the grid cache and collect them into b. */
  particles_->IterateParticles([&](const ParticleData<T>& particles,
                                   int particle_data_index,
                                   const BsplineWeights<T>& weights,
                                   const Pad<Vector3>& grid_x,
                                   Pad<GridData<T>>* grid_data) {
    const int p = particle_data_index;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const T& w = bspline.weight(i, j, k);
          const Vector3<T>& xi = grid_x[i][j][k];
          /* For the elastic force from particles, we compute -∂E/∂xᵢ and
           get

             fᵢ = -∑ₚ Vₚ * Pₚ * Fₚⁿᵀ * D⁻¹ * (xᵢ − xₚ) * wᵢₚ

           with Pₚ = ∂Ψ/∂Fₚ. Noting that Pₚ * Fₚⁿᵀ is the Kirchhoff stress,
           we group Vₚ * Pₚ * Fₚⁿᵀ into a single term `tau_v0`. Rearranging
           terms reveals that - fᵢdt is given by the equation in the code
           below. */
          const auto& tau_v0 =
              data_.tau_v0[p];  // Note that we take the stress from the scratch
                                // data here instead directly from the particle
                                // because we are in the Newton loop.
          grid_data[i][j][k].scratch +=
              tau_v0 * (xi - particles.x[p]) * D_inverse * dt;
        }
      }
    }
  });
}

// TODO(xuchenhan-tri): We should be able to make a single pass in this function
// to both index the grid and compute the connectivity (neighbors). This will
// reduce the number of duplicated neighbors.
template <typename T>
Block3x3SparseSymmetricMatrix MpmState<T>::MakeTangentMatrix() const {
  /* Here we loop over all particles; each particle creates an edge in the
   connectivity graph that connects all grid nodes in the pad (3x3x3 = 27 grid
   nodes) the particle transfers to. */
  const std::vector<uint64_t>& base_node_offsets =
      particles_->base_node_offsets;
  const std::vector<int>& sentinel_particles = particles_->sentinel_particles;
  const int num_blocks = grid_->num_blocks();
  Pad<GridData<T>> grid_data;
  const int num_active_nodes = grid_->num_active_nodes();
  std::vector<std::vector<int>> sparsity_pattern(num_active_nodes);
  for (auto& row : sparsity_pattern) {
    /* Each node i can have at most 125 nodes j such that a particle p splats to
     both i and j. */
    row.reserve(125);
  }
  for (int b = 0; b < num_blocks; ++b) {
    const int particle_start = sentinel_particles[b];
    const int particle_end = sentinel_particles[b + 1];
    for (int p = particle_start; p < particle_end; ++p) {
      grid_data = grid_->GetPadData(base_node_offsets[p]);
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
      /* Skip over particles that belong to the same pad as the particle we just
       processed. */
      while (p + 1 != particle_end &&
             base_node_offsets[p] == base_node_offsets[p + 1]) {
        ++p;
      }
    }
  }
  const int kSpatialDim = 3;
  BlockSparsityPattern block_pattern(
      std::vector<int>(num_active_nodes, kSpatialDim),
      std::move(sparsity_pattern));
  return Block3x3SparseSymmetricMatrix(std::move(block_pattern));
}

template <typename T>
void MpmState<T>::CalcTangentMatrix(
    Block3x3SparseSymmetricMatrix* tangent_matrix) {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  // TODO(xuchenhan-tri): Implement this function.
}

template <typename T>
void MpmState<T>::UpdateParticleState() {
  /* First update the deformation gradient. */
  const int lanes = SimdScalar<T>::lanes();
  const std::vector<int>& sentinel_particles = particles_->sentinel_particles;
  const std::vector<int>& data_indices = particles_->data_indices;
  const std::vector<uint64_t>& base_node_offsets =
      particles_->base_node_offsets;
  const int num_blocks = grid_->num_blocks();
  [[maybe_unused]] const int num_threads = parallelism_.num_threads();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int b = 0; b < num_blocks; ++b) {
    bool need_new_pad = true;
    Pad<Vector3<T>> grid_x;
    Pad<GridData<T>> grid_data;
    const int particle_start = sentinel_particles[b];
    const int particle_end = sentinel_particles[b + 1];
    std::vector<int> indices;
    indices.reserve(lanes);
    int p = particle_start;
    while (p < particle_end) {
      int next_p = p + 1;
      while (next_p < particle_end &&
             base_node_offsets[next_p] == base_node_offsets[p] &&
             next_p - p < lanes) {
        ++next_p;
      }
      if (need_new_pad) {
        grid_data = grid_->GetPadData(base_node_offsets[p]);
        grid_x = grid_->GetPadNodes(particles_->x[data_indices[p]]);
      }
      indices.clear();
      for (int i = p; i < next_p; ++i) {
        indices.push_back(data_indices[i]);
      }
      Matrix3<SimdScalar<T>> B = Matrix3<SimdScalar<T>>::Zero();
      Vector3<SimdScalar<T>> x = Load(particles_->x, indices);
      const BsplineWeights<SimdScalar<T>> bspline =
          BsplineWeights<SimdScalar<T>>(x, grid_->dx());
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            const int grid_index = grid_data[i][j][k].index;
            const Vector3<T>& vi =
                grid_data[i][j][k].v + dv_.template segment<3>(3 * grid_index);
            const Vector3<T>& xi = grid_x[i][j][k];
            const SimdScalar<T> w = bspline.weight(i, j, k);
            B += (w * vi) * (xi - x).transpose();
          }
        }
      }
      Matrix3<SimdScalar<T>> C = B * D_inverse_;
      Matrix3<SimdScalar<T>> F = Load(data_.F, indices);
      F += C * dt_ * F;
      Store(F, &data_.F, indices);

      need_new_pad = (next_p == particle_end) ||
                     base_node_offsets[next_p] != base_node_offsets[p];
      p = next_p;
    }
  }
  /* Then update stress and stress derivatives. */
  particles_->UpdateStress(data_.F, &data_.tau_v0, parallelism_);
  particles_->UpdateStressDerivatives(
      data_.F, &data_.volume_scaled_stress_derivatives, parallelism_);
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::MpmState<double>;